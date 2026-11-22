from typing import Callable, Iterator, Literal, Sequence
import typing
import contextlib
import io
import os
import pathlib
import tarfile
import warnings
import dataclasses
import numpy as np
import scipy.ndimage
import matplotlib.axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
import astropy.time
import astropy.io.fits
import astropy.visualization
import named_arrays as na
from ._data import urls, download

__all__ = [
    "SpectrographObservation",
]

_dispersion_split = 0.03
"""
The size of a column of the detector, in angstroms, above which ``CDELT1``
is taken to be per angstrom of the first order.

A column of either detector is about 0.042 to 0.045 angstrom of the first
order and half that of the second, and a file gives ``CDELT1`` in the order
of the line at its reference wavelength, ``CRVAL1``, which may be either.
"""

_message_ascii = "non-ASCII characters are present in the FITS file header"
"""
The start of the warning which :mod:`astropy.io.fits` gives for the headers
of many level-1 files of SUMER, whose names of scientists have letters
outside of ASCII, and which says nothing about the data.
"""

_gap_max = 2
"""
The most columns without data which are bridged inside a run of read-out
columns.

A dead column of the detector holds no data, and splitting the run there
would lose the columns beyond it.
"""

_tolerance_window = 0.5
"""
How far apart, in columns, the middles of one window may be in two exposures
for them to be loaded together.

The level-1 files correct the geometry of each exposure on its own,
so the wavelength scale of a window moves a little from one exposure to the
next.
"""

_tolerance_dispersion = 1e-3
"""
How far apart, relative to it, the size of a column of one window may be in
two exposures for them to be loaded together.
"""

_tolerance_roll = 1
"""
How far, in degrees, the roll of the instrument in one exposure may be from
the mean roll of the exposures for them to be loaded together.

The roll drifts by up to about a tenth of a degree in a day,
and by a few tenths of a degree over the weeks of a campaign,
but it was 119 degrees on 1999 February 27.
"""

_tolerance_timedelta = 1e-3
"""
How far apart, relative to it, an exposure time may be from the one asked for
to be kept, since the files give, for example, 100.01 s for some exposures of
a sequence of 100 s.
"""

_velocity_search = u.Quantity(100, u.km / u.s)
"""
How far from the wavelength of the line the brightest part of the mean
profile is looked for, when the line is zeroed on its mean position.

SUMER's wavelength scale is off by up to about 50 km/s, and the limit keeps a
brighter line which shares the window from being taken for the line.
"""

_velocity_centroid = u.Quantity(100, u.km / u.s)
"""
How far on either side of the brightest part of the mean profile the
centroid of the line is found, when the line is zeroed on its mean position.
"""

_unit_radiance = u.W / (u.sr * u.m**2 * u.AA)
"""The unit of the level-1 radiances, which the files call ``W/sr/m^2/Angstroem``."""

_units_level1 = "W/sr/m^2/Angstroem"
"""How the level-1 files name the unit of their radiances, :data:`_unit_radiance`."""

# named-arrays takes the names of axes where numpy takes numbers,
# which the annotations of numpy do not allow.
_nanmean = typing.cast(Callable[..., na.ScalarArray], np.nanmean)
_nanpercentile = typing.cast(Callable[..., na.ScalarArray], np.nanpercentile)
_diff = typing.cast(Callable[..., na.ScalarArray], np.diff)
_where = typing.cast(Callable[..., na.ScalarArray], np.where)


@dataclasses.dataclass(frozen=True)
class _Window:
    """Where one spectral window lies on the detector, and its wavelengths."""

    start: int
    """The first column of the window in the image, counting from zero."""

    num: int
    """The number of columns of the window."""

    wavelength: float
    """The wavelength of the middle of the first column, in angstroms."""

    dispersion: float
    """The size of a column in wavelength, in angstroms."""

    @property
    def middle(self) -> float:
        """The wavelength of the middle of the window, in angstroms."""
        return self.wavelength + (self.num - 1) / 2 * self.dispersion

    def column(self, wavelength: float) -> int:
        """
        The column of the image which holds a wavelength.

        Parameters
        ----------
        wavelength
            The wavelength, in angstroms.
        """
        return self.start + int(
            np.rint((wavelength - self.wavelength) / self.dispersion)
        )


@dataclasses.dataclass
class _Exposure:
    """What is kept of one exposure while a sequence is loaded."""

    time: astropy.time.Time
    """The start of the exposure."""

    header: astropy.io.fits.Header
    """The primary header of the exposure."""

    window: _Window
    """The window which holds the line."""

    extent: None | tuple[int, int]
    """
    The first column, and the one after the last, of the run of read-out
    columns around the line, counted from the start of the window,
    or :obj:`None` if the line was not read out.
    """

    radiance: None | np.ndarray
    """
    The radiance of the columns of :attr:`extent`, along the slit and in
    wavelength, NaN in the columns which hold no data.
    """


def _members(path: str | os.PathLike) -> Iterator[bytes]:
    """
    The FITS files of the exposures in one level-1 file of SUMER,
    which is a gzipped tar archive of them.

    Parameters
    ----------
    path
        The level-1 file.
    """
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".fits"):
                continue
            content = tar.extractfile(member)
            if content is not None:
                yield content.read()


def _number(header: astropy.io.fits.Header, key: str, default: float = 0) -> float:
    """
    The value of a numeric card of a header.

    Parameters
    ----------
    header
        The header.
    key
        The keyword of the card.
    default
        The value if the header has no such card.
    """
    return float(typing.cast(float, header.get(key, default)))


def _text(header: astropy.io.fits.Header, key: str, default: str = "") -> str:
    """
    The value of a text card of a header, without its trailing spaces.

    Parameters
    ----------
    header
        The header.
    key
        The keyword of the card.
    default
        The value if the header has no such card.
    """
    return str(typing.cast(str, header.get(key, default))).strip()


def _window_spanning(
    column: int,
    reference: float,
    minimum: float,
    maximum: float,
    dispersion_first: float,
    order: int,
) -> _Window:
    """
    A window from its limits and its reference wavelength and column,
    with its wavelengths in the order of the line it is loaded for.

    The limits are the reference wavelength less and more half the width of
    the window, rounded down to whole columns,
    so the lower limit is the middle of the first column,
    and the upper one is the middle of the last column of a window with an
    odd number of columns but of the column after it otherwise.
    The window is taken to end before the upper limit,
    so that images of one window and images of several,
    which one sequence can mix, give it the same number of columns.

    Parameters
    ----------
    column
        The column of the reference wavelength, counting from zero.
    reference
        The reference wavelength, in angstroms of the first order,
        that is, its place on the detector.
    minimum
        The lower limit of the window, in angstroms of the first order.
    maximum
        The upper limit of the window, in angstroms of the first order.
    dispersion_first
        The size of a column, in angstroms of the first order.
    order
        The diffraction order of the line.
    """
    offset = int(np.rint((reference - minimum) / dispersion_first))
    return _Window(
        start=column - offset,
        num=int(np.rint((maximum - minimum) / dispersion_first)),
        wavelength=(reference - offset * dispersion_first) / order,
        dispersion=dispersion_first / order,
    )


def _window(
    header: astropy.io.fits.Header,
    wavelength: u.Quantity,
    order: int,
) -> None | _Window:
    """
    The spectral window of an exposure which holds a line,
    or :obj:`None` if no window holds it.

    An image of SUMER either holds several windows side by side
    (``PIMGTYP = 'MULTIPLE'``),
    each with its own reference wavelength, ``WAVELn``,
    at its own column, ``IREFPIXn``, counted from one,
    and limits ``WAVEMINn`` and ``WAVEMAXn``;
    or one window, the whole image (``PIMGTYP = 'SINGLE'``),
    described by ``CRVAL1``, ``CRPIX1``, ``WAVEMIN``, and ``WAVEMAX``.

    The wavelengths of a window are those of the line it was planned for,
    in the order of that line or in the first order,
    and the files do not say which.
    ``CRVAL1``, at ``CRPIX1``, is in the order of ``CDELT1``,
    which is plain from its size,
    and every column of the image lies on one wavelength scale of the first
    order through it,
    so the place on the detector of a window's reference column,
    divided by its reference wavelength, gives the order of its wavelengths.
    The window holds the line if its limits, in the first order,
    hold the place on the detector of the line.

    Parameters
    ----------
    header
        The primary header of an exposure.
    wavelength
        The wavelength of the line.
    order
        The diffraction order in which the line was observed.
    """
    cdelt = _number(header, "CDELT1")
    order_reference = 1 if cdelt > _dispersion_split else 2
    dispersion_first = cdelt * order_reference
    column_reference = int(_number(header, "CRPIX1")) - 1
    place_reference = _number(header, "CRVAL1") * order_reference
    place = float(typing.cast(float, wavelength.to_value(u.AA))) * order

    if _text(header, "PIMGTYP") == "MULTIPLE":
        for n in range(1, int(_number(header, "ONUMIMG")) + 1):
            column = int(_number(header, f"IREFPIX{n}")) - 1
            label = _number(header, f"WAVEL{n}")
            place_window = (
                place_reference + (column - column_reference) * dispersion_first
            )
            order_label = int(np.rint(place_window / label))
            if order_label not in (1, 2):
                continue
            minimum = _number(header, f"WAVEMIN{n}") * order_label
            maximum = _number(header, f"WAVEMAX{n}") * order_label
            if minimum <= place <= maximum:
                return _window_spanning(
                    column=column,
                    reference=label * order_label,
                    minimum=minimum,
                    maximum=maximum,
                    dispersion_first=dispersion_first,
                    order=order,
                )
        return None

    minimum = _number(header, "WAVEMIN") * order_reference
    maximum = _number(header, "WAVEMAX") * order_reference
    if not minimum <= place <= maximum:
        return None
    return _window_spanning(
        column=column_reference,
        reference=place_reference,
        minimum=minimum,
        maximum=maximum,
        dispersion_first=dispersion_first,
        order=order,
    )


def _extent(filled: np.ndarray, column: int) -> None | tuple[int, int]:
    """
    The first column, and the one after the last, of the run of columns of an
    image which hold data and include a given column,
    or :obj:`None` if no run reaches it.

    Gaps of up to :data:`_gap_max` columns without data are bridged,
    and a run reaches the column if it includes it or ends within that many
    columns of it.

    Parameters
    ----------
    filled
        Whether each column holds data.
    column
        The column the run must reach.
    """
    indices = np.flatnonzero(filled)
    if not indices.size:
        return None
    breaks = np.flatnonzero(np.diff(indices) > _gap_max + 1)
    starts = np.concatenate([indices[:1], indices[breaks + 1]])
    stops = np.concatenate([indices[breaks], indices[~0:]]) + 1
    for first, last in zip(starts, stops):
        if first - _gap_max <= column < last + _gap_max:
            return int(first), int(last)
    return None


def _exposure(
    hdul: astropy.io.fits.HDUList,
    time: astropy.time.Time,
    window: _Window,
    wavelength: u.Quantity,
    order: int,
) -> _Exposure:
    """
    The radiance of the run of read-out columns around a line in one
    exposure, keeping none of the rest of its image.

    Parameters
    ----------
    hdul
        The FITS file of the exposure.
    time
        The start of the exposure.
    window
        The window which holds the line.
    wavelength
        The wavelength of the line.
    order
        The diffraction order of the line.

    Raises
    ------
    ValueError
        If the radiances are not in the unit of the level-1 files.
    """
    hdu = typing.cast(astropy.io.fits.PrimaryHDU, hdul[0])
    header = hdu.header
    if _text(header, "IMGUNITS") != _units_level1:
        raise ValueError(f"Unexpected units {_text(header, 'IMGUNITS')!r}.")

    image = np.asarray(hdu.data, dtype=float)
    filled = np.any(np.isfinite(image) & (image != 0), axis=0)
    column = window.column(float(typing.cast(float, wavelength.to_value(u.AA))))
    extent = _extent(filled, column)
    if extent is None:
        return _Exposure(time, header, window, None, None)

    columns = slice(*extent)
    extension = typing.cast(astropy.io.fits.ImageHDU, hdul["SUMER-RAD-ARRAY"])
    radiometry = np.asarray(extension.data, dtype=float)
    row_order = int(_number(header, "RADORDER", 1)) - 1
    factor = radiometry[order - 1, columns] / radiometry[row_order, columns]
    radiance = image[:, columns] * factor
    radiance[:, ~filled[columns]] = np.nan
    return _Exposure(
        time=time,
        header=header,
        window=window,
        extent=(extent[0] - window.start, extent[1] - window.start),
        radiance=radiance,
    )


def _centroid(
    radiance: na.AbstractScalarArray,
    wavelengths: na.AbstractScalarArray,
    wavelength: u.Quantity,
    axis_wavelength: str,
) -> u.Quantity:
    """
    The mean position of a line in some exposures:
    the centroid, above its lowest value, of their mean profile near its
    brightest part, which is looked for near the wavelength of the line.

    Parameters
    ----------
    radiance
        The spectra.
    wavelengths
        The edges of the columns.
    wavelength
        The wavelength of the line.
    axis_wavelength
        The logical axis corresponding to changes in wavelength.

    Raises
    ------
    ValueError
        If the spectra have no data near the wavelength of the line.
    """
    # The mean over every axis but wavelength is a profile along wavelength
    # alone, which is searched as a sequence.
    axes = tuple(a for a in radiance.axes if a != axis_wavelength)
    with warnings.catch_warnings():
        # columns which no exposure read out have no mean
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = _nanmean(radiance, axis=axes)
    along = (axis_wavelength,)
    profile = np.asarray(
        typing.cast(u.Quantity, mean.ndarray_aligned(along)).to_value(_unit_radiance)
    )
    middles = typing.cast(na.ScalarArray, wavelengths.cell_centers(axis_wavelength))
    centers = typing.cast(u.Quantity, middles.ndarray_aligned(along)).to(u.AA)
    velocity = np.asarray(
        centers.to_value(u.km / u.s, equivalencies=u.doppler_optical(wavelength))
    )

    finite = np.isfinite(profile)
    candidates = finite & (np.abs(velocity) < _velocity_search.to_value(u.km / u.s))
    if not np.any(candidates):
        raise ValueError(
            f"The spectra have no data within {_velocity_search} of {wavelength}."
        )
    peak = velocity[candidates][np.argmax(profile[candidates])]

    width = _velocity_centroid.to_value(u.km / u.s)
    near = finite & (np.abs(velocity - peak) < width)
    excess = profile[near] - np.min(profile[near])
    if not np.sum(excess) > 0:
        return u.Quantity(centers[near][np.argmax(profile[near])])
    return u.Quantity(np.sum(excess * centers[near]) / np.sum(excess))


def _vertices(centers: np.ndarray, width: float) -> np.ndarray:
    """
    The edges between consecutive positions along a raster,
    halfway between them, with the outer edges as far beyond the first and
    last positions as the nearest edge is inside them.

    A sequence which holds the slit still has every edge in one place.

    Parameters
    ----------
    centers
        The positions, in order of time.
    width
        The width of a single exposure along the coordinate of `centers`,
        which is negative if the coordinate decreases from one side of the
        slit to the other,
        used for the edges if there is only one position.
    """
    if len(centers) == 1:
        return centers[0] + np.array([-width / 2, width / 2])
    middle = (centers[1:] + centers[:-1]) / 2
    first = centers[0] - (middle[0] - centers[0])
    last = centers[~0] + (centers[~0] - middle[~0])
    return np.concatenate([[first], middle, [last]])


@dataclasses.dataclass(eq=False, repr=False)
class SpectrographObservation(
    na.FunctionArray[
        na.TemporalDopplerPositionalVectorArray,
        na.ScalarArray,
    ],
):
    """
    A sequence of exposures of one spectral window of the
    Solar Ultraviolet Measurements of Emitted Radiation (SUMER) spectrograph,
    in the radiances of its level-1 files.

    Each exposure is the spectrum along the slit at one place on the Sun.
    A raster steps the slit across the Sun from one exposure to the next,
    so the exposures are arranged along :attr:`axis_detector_x`
    in order of time,
    with the slit along :attr:`axis_detector_y`
    and the spectrum along :attr:`axis_wavelength`.
    A sequence which sweeps the same field more than once
    goes back and forth along :attr:`axis_detector_x`,
    so select one sweep by its times to get a map of the field.
    A sequence which holds the slit still (``RASTYPE = 'TEMPORAL'``)
    has every edge along :attr:`axis_detector_x` in one place,
    so it is a series in time rather than a map.
    The detector has more rows than a slit of 120 or 300 arcseconds lights,
    and the rows beyond the ends of the slit hold no signal.

    The spectrum is that of the columns around the line which hold data in
    any exposure, and each exposure is not a number (NaN) in the columns
    outside its own data, and in its dead columns,
    so integrate over wavelength with :func:`numpy.nansum`.
    The windows of an image are the parts of the detector which were read out,
    and those around one line can lie side by side,
    so the spectrum can reach past the limits of the window the line was
    planned for, into its neighbours,
    and it can also start or end a few columns from those limits,
    where the files place the read-out data a few columns from the window.

    The coordinates are at the edges of each pixel, as in the other sun-data
    packages:
    the time of each edge along :attr:`axis_detector_x` is the start of an
    exposure, ``DATE_OBS``, and the last is the end of the last exposure;
    the positions are those of the coordinate system of each file,
    with each edge of a raster step halfway between the two exposures beside
    it, and with the slit turned from solar north by the roll of the
    instrument, which the files give as ``ANGLE``,
    and which is two degrees or less in most files but 119 degrees in some
    of 1999;
    and the wavelengths are those of the line which was observed,
    in :attr:`order`.

    SUMER has no absolute wavelength scale:
    the files place the wavelength a window was planned for at a reference
    column, and the line can lie several columns from it,
    tens of kilometers per second in Doppler velocity.
    So by default the wavelengths are shifted to put the mean position of the
    line in the loaded exposures at its rest wavelength,
    which removes any real net shift of the region too,
    and :attr:`shift_wavelength` records the shift.

    The radiances are those of the level-1 files, which are calibrated for the
    first order of the grating (``RADORDER = 1``) per angstrom.
    For a line in the second order they are multiplied by the ratio of the
    second-order to the first-order radiometry of each column of the detector,
    which each file holds in its ``SUMER-RAD-ARRAY`` extension,
    and are then per angstrom of the second order,
    which gives quiet-Sun radiances of O V 629.7 angstroms
    that agree with those of :cite:t:`Vernazza1978`.
    """

    timedelta: u.Quantity | na.AbstractScalar = dataclasses.field(
        default_factory=lambda: 0 * u.s,
    )
    """The exposure time of each exposure."""

    order: int = 1
    """The diffraction order of the wavelengths and the radiances."""

    detector: str = "A"
    """The detector which recorded the exposures, ``A`` or ``B``."""

    slit: str = ""
    """The slit, as the files name it, such as ``<2> 1.0" * 300" centered``."""

    shift_wavelength: u.Quantity = dataclasses.field(
        default_factory=lambda: 0 * u.AA,
    )
    """
    How far the wavelengths were moved from those the files give,
    to zero the line as :meth:`from_level1` was asked to.
    Add it to the wavelengths to get those of the files.
    """

    axis_wavelength: str = "wavelength"
    """The logical axis corresponding to changes in wavelength."""

    axis_detector_x: str = "detector_x"
    """
    The logical axis corresponding to changes in detector :math:`x`-coordinate,
    which is the exposures, in order of time.
    """

    axis_detector_y: str = "detector_y"
    """
    The logical axis corresponding to changes in detector :math:`y`-coordinate,
    which is the position along the slit.
    """

    @classmethod
    def from_level1(
        cls,
        path: str | os.PathLike | Sequence[str | os.PathLike] | na.AbstractScalarArray,
        wavelength: u.Quantity,
        order: int,
        time_start: None | str | astropy.time.Time = None,
        time_stop: None | str | astropy.time.Time = None,
        timedelta: None | u.Quantity = None,
        slit: None | str = None,
        velocity_zero: None | Literal["mean"] | u.Quantity = "mean",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> "SpectrographObservation":
        """
        Load the exposures of one spectral window from level-1 files of SUMER.

        An exposure is kept if it began in the time range,
        has a window which holds `wavelength`,
        and has the exposure time and slit asked for, if they are given.
        Interleaved sequences of one study can share a time range and a
        window, so give `timedelta` and `slit` to keep only one of them.

        Parameters
        ----------
        path
            One level-1 file, or several, such as those downloaded by
            :func:`soho.sumer.download`.
        wavelength
            A wavelength in the window to load,
            usually that of the line it was planned for,
            in the order given by `order`.
            It is also the rest wavelength of the result.
        order
            The diffraction order in which the window observed `wavelength`,
            1 or 2,
            which the files do not record for a window of its own.
            For example, O V 629.7 angstroms is observed in the second order.
        time_start
            The earliest start of the exposures.
            If :obj:`None`, there is no earliest start.
        time_stop
            The time before which the exposures must start.
            If :obj:`None`, there is no latest start.
        timedelta
            The exposure time of the exposures to keep,
            within a thousandth of it.
            If :obj:`None`, exposures of any length are kept.
        slit
            The slit of the exposures to keep, as the files name it.
            If :obj:`None`, exposures through any slit are kept.
        velocity_zero
            Where to put the zero of the Doppler velocity,
            since SUMER has no absolute wavelength scale.
            If ``"mean"`` (the default), the wavelengths are shifted to put the
            mean position of the line in the kept exposures at `wavelength`.
            If a velocity, they are shifted to put that velocity,
            on the scale the files give, at `wavelength`.
            If :obj:`None`, the wavelengths are those the files give.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis of the exposures.
        axis_detector_y
            The logical axis corresponding to the position along the slit.

        Raises
        ------
        ValueError
            If no exposure is kept,
            or if the kept exposures do not share one window, one detector,
            one slit, and one roll of the instrument, to within a degree.
        NotImplementedError
            If the kept exposures are binned.
        """
        if isinstance(path, na.AbstractScalarArray):
            paths = [str(p) for p in np.asarray(path.ndarray).flat]
        elif isinstance(path, (str, os.PathLike)):
            paths = [path]
        else:
            paths = list(path)

        start = None if time_start is None else astropy.time.Time(time_start)
        stop = None if time_stop is None else astropy.time.Time(time_stop)

        exposures: list[_Exposure] = []
        for p in paths:
            for content in _members(p):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=_message_ascii)
                    hdul = astropy.io.fits.open(io.BytesIO(content))
                    hdu = typing.cast(astropy.io.fits.PrimaryHDU, hdul[0])
                    header = hdu.header
                # Some exposures have a header but no image.
                if header.get("NAXIS", 0) != 2:
                    continue
                time = astropy.time.Time(_text(header, "DATE_OBS"))
                if start is not None and time < start:
                    continue
                if stop is not None and not time < stop:
                    continue
                if timedelta is not None:
                    exposure = _number(header, "EXPTIME") * u.s
                    if abs(exposure - timedelta) > _tolerance_timedelta * timedelta:
                        continue
                if slit is not None and _text(header, "SLIT") != slit.strip():
                    continue
                window = _window(header, wavelength, order)
                if window is None:
                    continue
                if (_number(header, "BINX", 1), _number(header, "BINY", 1)) != (1, 1):
                    raise NotImplementedError("Binned exposures are not supported yet.")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=_message_ascii)
                    exposures.append(_exposure(hdul, time, window, wavelength, order))

        if not exposures:
            raise ValueError(
                f"No exposure in {paths} has a window holding {wavelength}"
                " which matches the selection."
            )

        exposures.sort(key=lambda e: float(typing.cast(float, e.time.jd)))
        headers = [e.header for e in exposures]
        windows = [e.window for e in exposures]

        window = windows[0]
        for w in windows:
            num = abs(w.num - window.num) > 1
            shift = abs(w.middle - window.middle) / window.dispersion
            stretch = abs(w.dispersion / window.dispersion - 1)
            if num or shift > _tolerance_window or stretch > _tolerance_dispersion:
                raise ValueError(
                    f"The exposures differ in their window, {window} and {w}; "
                    "select one sequence with the time range, `timedelta`, or `slit`."
                )

        for key, values in (
            ("detector", [_text(h, "DETECTOR") for h in headers]),
            ("slit", [_text(h, "SLIT") for h in headers]),
            ("number of rows", [_number(h, "NAXIS2") for h in headers]),
            ("pixel size along the slit", [_number(h, "CDELT2") for h in headers]),
        ):
            if len(set(values)) > 1:
                raise ValueError(
                    f"The exposures differ in their {key}, {sorted(set(map(str, values)))}; "
                    "select one sequence with the time range, `timedelta`, or `slit`."
                )

        # The exposures are turned by their mean roll,
        # taken around the circle so that rolls on either side of 180 degrees
        # are near each other.
        rolls = np.deg2rad([_number(h, "ANGLE") for h in headers])
        angle = np.arctan2(np.mean(np.sin(rolls)), np.mean(np.cos(rolls)))
        spread = np.rad2deg(np.abs((rolls - angle + np.pi) % (2 * np.pi) - np.pi).max())
        if spread > _tolerance_roll:
            raise ValueError(
                f"The exposures differ in their roll, ANGLE, by up to {spread:.2f} "
                "degrees from their mean; "
                "select one sequence with the time range, `timedelta`, or `slit`."
            )

        header = headers[0]

        # The columns around the line which hold data in any exposure,
        # counted from the start of the window.
        # A sequence can mix images of one window with images of several,
        # which hold the same window in other columns.
        extents = [e.extent for e in exposures if e.extent is not None]
        lower = min(e[0] for e in extents) if extents else 0
        upper = max(e[1] for e in extents) if extents else window.num
        num_row = int(_number(header, "NAXIS2"))
        radiance = np.full((len(exposures), num_row, upper - lower), np.nan)
        for i, e in enumerate(exposures):
            if e.extent is not None and e.radiance is not None:
                index = slice(e.extent[0] - lower, e.extent[1] - lower)
                radiance[i, :, index] = e.radiance

        outputs = na.ScalarArray(
            ndarray=radiance << _unit_radiance,
            axes=(axis_detector_x, axis_detector_y, axis_wavelength),
        )

        # The wavelengths are the median of those of the exposures,
        # which differ by much less than a column.
        wavelength_first = np.median([w.wavelength for w in windows])
        dispersion = np.median([w.dispersion for w in windows])
        index_column = lower + np.arange(upper - lower + 1) - 0.5
        wavelengths = na.ScalarArray(
            ndarray=(wavelength_first + index_column * dispersion) << u.AA,
            axes=axis_wavelength,
        )

        rest = wavelength.to(u.AA)
        if velocity_zero is None:
            shift = u.Quantity(0, u.AA)
        elif isinstance(velocity_zero, str):
            if velocity_zero != "mean":
                raise ValueError(f"{velocity_zero=} not recognized.")
            centroid = _centroid(outputs, wavelengths, rest, axis_wavelength)
            shift = u.Quantity(centroid - rest)
        else:
            zero = velocity_zero.to(u.AA, equivalencies=u.doppler_optical(rest))
            shift = u.Quantity(zero - rest)
        wavelengths = wavelengths - shift

        index_row = np.arange(num_row + 1) + 0.5
        offset_y = (index_row - _number(header, "CRPIX2")) * _number(header, "CDELT2")

        # The slit lies along the y axis of the instrument,
        # and the width of the slit along its x axis,
        # which are turned from those of the files by the roll of the
        # instrument.
        cos = np.cos(angle)
        sin = np.sin(angle)

        width = _number(header, "IXWIDTH")
        x = _vertices(np.array([_number(h, "CRVAL3") for h in headers]), width * cos)
        y = _vertices(np.array([_number(h, "CRVAL2") for h in headers]), width * sin)

        time_vertices = astropy.time.Time(
            [e.time.isot for e in exposures] + [_text(headers[~0], "DATE_END")],
        )

        inputs = na.TemporalDopplerPositionalVectorArray(
            time=na.ScalarArray(
                typing.cast(np.ndarray, time_vertices),
                axes=axis_detector_x,
            ),
            wavelength=wavelengths,
            wavelength_rest=wavelength,
            position=na.Cartesian2dVectorArray(
                x=na.ScalarArray(x << u.arcsec, axes=axis_detector_x)
                - na.ScalarArray(offset_y * sin << u.arcsec, axes=axis_detector_y),
                y=na.ScalarArray(y << u.arcsec, axes=axis_detector_x)
                + na.ScalarArray(offset_y * cos << u.arcsec, axes=axis_detector_y),
            ),
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=na.ScalarArray(
                np.array([_number(h, "EXPTIME") for h in headers]) << u.s,
                axes=axis_detector_x,
            ),
            order=order,
            detector=_text(header, "DETECTOR"),
            slit=_text(header, "SLIT"),
            shift_wavelength=shift,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @classmethod
    def from_time_range(
        cls,
        time_start: str | astropy.time.Time,
        time_stop: str | astropy.time.Time,
        wavelength: u.Quantity,
        order: int,
        timedelta: None | u.Quantity = None,
        slit: None | str = None,
        velocity_zero: None | Literal["mean"] | u.Quantity = "mean",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        directory: None | pathlib.Path = None,
        overwrite: bool = False,
        num_retry: int = 5,
    ) -> "SpectrographObservation":
        """
        Find the level-1 files of SUMER which observed a time range in ESA's
        catalogue of SOHO, download them from the SOHO archive,
        and load the exposures of one spectral window which began in it.

        Parameters
        ----------
        time_start
            The earliest start of the exposures.
        time_stop
            The time before which the exposures must start.
        wavelength
            A wavelength in the window to load, in the order given by `order`.
        order
            The diffraction order in which the window observed `wavelength`.
        timedelta
            The exposure time of the exposures to keep,
            within a thousandth of it.
            If :obj:`None`, exposures of any length are kept.
        slit
            The slit of the exposures to keep, as the files name it.
            If :obj:`None`, exposures through any slit are kept.
        velocity_zero
            Where to put the zero of the Doppler velocity,
            since SUMER has no absolute wavelength scale.
            If ``"mean"`` (the default), the wavelengths are shifted to put the
            mean position of the line in the kept exposures at `wavelength`.
            If a velocity, they are shifted to put that velocity,
            on the scale the files give, at `wavelength`.
            If :obj:`None`, the wavelengths are those the files give.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis of the exposures.
        axis_detector_y
            The logical axis corresponding to the position along the slit.
        directory
            The directory to place the downloaded files in.
            If :obj:`None` (the default), :data:`soho.directory_default` is used.
        overwrite
            Boolean flag controlling whether to download files which are
            already in `directory`.
        num_retry
            The number of times to try to connect to the servers.
        """
        found = urls(time_start, time_stop, num_retry=num_retry)
        paths = download(
            found,
            directory=directory,
            overwrite=overwrite,
            num_retry=num_retry,
        )
        return cls.from_level1(
            path=paths,
            wavelength=wavelength,
            order=order,
            time_start=time_start,
            time_stop=time_stop,
            timedelta=timedelta,
            slit=slit,
            velocity_zero=velocity_zero,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    def _radiance_aligned(self) -> np.ndarray:
        """
        The radiances as a plain array, along the exposures, the slit, and
        wavelength, in the unit of the level-1 files.
        """
        axes = (self.axis_detector_x, self.axis_detector_y, self.axis_wavelength)
        outputs = typing.cast(na.ScalarArray, self.outputs)
        aligned = typing.cast(u.Quantity, outputs.ndarray_aligned(axes))
        return np.asarray(aligned.to_value(_unit_radiance), dtype=float)

    def _with_radiance(self, radiance: np.ndarray) -> "SpectrographObservation":
        """
        A copy with new radiances, given as by :meth:`_radiance_aligned`.

        Parameters
        ----------
        radiance
            The radiances, along the exposures, the slit, and wavelength.
        """
        axes = (self.axis_detector_x, self.axis_detector_y, self.axis_wavelength)
        outputs = na.ScalarArray(radiance << _unit_radiance, axes=axes)
        return dataclasses.replace(self, outputs=outputs)

    def remove_bad_steps(
        self,
        ratio_max: float = 2,
        ratio_min: float = 0.3,
        width: int = 9,
    ) -> "SpectrographObservation":
        """
        Replace each bad raster step by the mean of the nearest good steps on
        either side.

        A step is bad if the total radiance of its exposure is more than
        `ratio_max` times, or less than `ratio_min` times, the median of the
        `width` exposures around it,
        as when the detector loses part of an exposure.

        Parameters
        ----------
        ratio_max
            The largest ratio of a good step to the median around it.
        ratio_min
            The smallest ratio of a good step to the median around it.
        width
            The number of exposures the median is taken over.
        """
        radiance = self._radiance_aligned()
        with warnings.catch_warnings():
            # exposures with no data have no total
            warnings.simplefilter("ignore", RuntimeWarning)
            total = np.nansum(radiance, axis=(1, 2))
            half = width // 2
            reference = np.array(
                [
                    np.median(total[max(i - half, 0) : i + half + 1])
                    for i in range(len(total))
                ]
            )
        bad = (total > ratio_max * reference) | (total < ratio_min * reference)
        good = np.flatnonzero(~bad)
        repaired = radiance.copy()
        for i in np.flatnonzero(bad):
            near = good[good < i][-1:].tolist() + good[good > i][:1].tolist()
            if near:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    repaired[i] = np.nanmean(radiance[near], axis=0)
        return self._with_radiance(repaired)

    def trim_slit(self, fraction: float = 0.05) -> "SpectrographObservation":
        """
        Leave out the rows of the detector beyond the ends of the slit,
        keeping the longest run of rows whose total radiance is more than
        `fraction` of the median of the rows which have any.

        Parameters
        ----------
        fraction
            The least total radiance of a row which is kept,
            relative to the median of the rows.
        """
        radiance = self._radiance_aligned()
        total = np.nansum(radiance, axis=(0, 2))
        positive = total[total > 0]
        if not positive.size:
            return self
        lit = total > fraction * np.median(positive)
        best = (0, 0)
        start = None
        for i, value in enumerate(np.append(lit, False)):
            if value and start is None:
                start = i
            elif not value and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        trimmed = self[{self.axis_detector_y: slice(*best)}]
        return typing.cast("SpectrographObservation", trimmed)

    def remove_stripes(
        self,
        fraction_sun: float = 0.3,
        num_min: int = 20,
        size: int = 15,
        gain_min: float = 0.7,
        gain_max: float = 1.3,
    ) -> "SpectrographObservation":
        """
        Divide each row of the slit by its gain,
        which removes the stripes fixed to the position along the slit
        that some rasters have, such as the scans of the polar coronal holes.

        The gain of a row is the median radiance of the Sun in that row,
        over the exposures, relative to a running median of it along the
        slit.
        A pixel is on the Sun if its radiance is more than `fraction_sun` of
        the 95th percentile of the radiances.
        A row with too few pixels on the Sun keeps a gain of one, and if too
        few rows have enough, the radiances are left as they are.

        Parameters
        ----------
        fraction_sun
            The least radiance of a pixel on the Sun,
            relative to the 95th percentile of the radiances.
        num_min
            The least number of pixels on the Sun a row needs for its gain.
        size
            The number of rows the running median is taken over.
        gain_min
            The smallest gain of a row.
        gain_max
            The largest gain of a row.
        """
        radiance = self._radiance_aligned()
        total = np.nansum(radiance, axis=~0)
        sun = total > fraction_sun * np.percentile(total, 95)
        rows = np.full(total.shape[1], np.nan)
        for j in range(total.shape[1]):
            if sun[:, j].sum() >= num_min:
                rows[j] = np.median(total[sun[:, j], j])
        valid = np.isfinite(rows)
        if valid.sum() < 2 * size:
            return self
        filled = np.interp(np.arange(len(rows)), np.flatnonzero(valid), rows[valid])
        smooth = scipy.ndimage.median_filter(filled, size=size, mode="nearest")
        gain = np.where(valid, np.clip(filled / smooth, gain_min, gain_max), 1)
        return self._with_radiance(radiance / gain[np.newaxis, :, np.newaxis])

    def show(
        self,
        ax: None | matplotlib.axes.Axes = None,
        cax: None | matplotlib.axes.Axes = None,
        norm: None | Callable = None,
        vmin: None | float | u.Quantity | na.AbstractScalar = None,
        vmax: None | float | u.Quantity | na.AbstractScalar = None,
        velocity_min: u.Quantity = -100 * u.km / u.s,
        velocity_max: u.Quantity = +100 * u.km / u.s,
        cbar_fraction: float = 0.1,
    ) -> matplotlib.axes.Axes:
        """
        Display the exposures of a raster as a false-color image,
        whose hue is the Doppler velocity
        and whose brightness is the radiance,
        as :meth:`iris.sg.SpectrographObservation.show` does.

        Columns which an exposure did not read out are shown as zero.

        Parameters
        ----------
        ax
            The :mod:`matplotlib` axes on which to plot the image.
            If :obj:`None`, a new figure is created.
        cax
            The axes on which to plot the colorbar.
            If :obj:`None`, space is stolen from `ax` to create a new set of axes.
        norm
            The normalization method used to scale data into the range [0, 1]
            before mapping to colors.
        vmin
            The minimum value of the data range.
            If :obj:`None`, zero is used.
        vmax
            The maximum value of the data range.
            If :obj:`None`, the 99.5th percentile of each wavelength is used,
            or the largest of them for a wavelength with no data.
        velocity_min
            The minimum Doppler velocity of the data range.
        velocity_max
            The maximum Doppler velocity of the data range.
        cbar_fraction
            The fraction of the space to use for the colorbar axes if `cax`
            is :obj:`None`.

        Raises
        ------
        ValueError
            If the slit stood still between two exposures,
            as in a sequence which is a series in time rather than a raster,
            or moved along itself.
        """
        axis_wavelength = self.axis_wavelength
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y

        # Each step of a raster crosses the slit,
        # whichever way the roll of the instrument turned them both.
        position = self.inputs.position
        slit = position[{axis_x: 0, axis_y: 1}] - position[{axis_x: 0, axis_y: 0}]
        slit = typing.cast(na.Cartesian2dVectorArray, slit)
        step_x = _diff(position.x, axis=axis_x)
        step_y = _diff(position.y, axis=axis_x)
        across = step_x * slit.y - step_y * slit.x
        if np.any(across == 0 * u.arcsec**2):
            raise ValueError(
                "The slit stood still between exposures, or moved along itself, "
                "so they are not a raster."
            )

        outputs = self.outputs.copy()
        outputs[~np.isfinite(outputs)] = 0 * _unit_radiance

        wavelength_rest = na.as_named_array(self.inputs.wavelength_rest)
        wavelength_center = typing.cast(na.AbstractScalarArray, wavelength_rest).ndarray

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(8, 8),
                constrained_layout=True,
            )
        ax = typing.cast(matplotlib.axes.Axes, ax)

        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                position="right",
                size=f"{cbar_fraction * 100}%",
                pad=1,
            )
        cax = typing.cast(matplotlib.axes.Axes, cax)

        if vmin is None:
            vmin = 0 * _unit_radiance

        if vmax is None:
            percentile = _nanpercentile(outputs, q=99.5, axis=(axis_x, axis_y))
            # A wavelength with no data would be scaled by zero.
            zero = u.Quantity(0, _unit_radiance)
            largest = typing.cast(u.Quantity, percentile.ndarray).max()
            if not largest > zero:
                largest = u.Quantity(1, _unit_radiance)
            vmax = _where(percentile > zero, percentile, largest)

        # The velocities of the inputs are optical, c (w / w0 - 1).
        equivalencies = u.doppler_optical(wavelength_center)

        # Recent versions of astropy annotate what quantity_support returns as
        # something which cannot be used in a with statement, though it can.
        support = typing.cast(
            contextlib.AbstractContextManager,
            astropy.visualization.quantity_support(),
        )
        with support:
            cax_twin = cax.twinx()
            colorbar = na.plt.rgbmesh(
                self.inputs.velocity,
                self.inputs.position.x,
                self.inputs.position.y,
                C=outputs,
                axis_wavelength=axis_wavelength,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                wavelength_min=velocity_min,
                wavelength_max=velocity_max,
            )
            na.plt.pcolormesh(
                colorbar.inputs.x,
                colorbar.inputs.y.to(u.AA, equivalencies=equivalencies),
                C=colorbar.outputs,
                axis_rgb=axis_wavelength,
                ax=cax,
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb=axis_wavelength,
                ax=cax_twin,
            )

            time = typing.cast(
                np.ndarray, astropy.time.Time(self.inputs.time.ndarray).isot
            )
            ax.set_title(f"{time[0]} to {time[~0]}")
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
            # The limits are quantities,
            # which the support for quantities above lets matplotlib take.
            wavelength_min = velocity_min.to(u.AA, equivalencies=equivalencies)
            wavelength_max = velocity_max.to(u.AA, equivalencies=equivalencies)
            cax.set_ylim(
                typing.cast(float, wavelength_min),
                typing.cast(float, wavelength_max),
            )
            cax_twin.set_ylim(
                typing.cast(float, velocity_min),
                typing.cast(float, velocity_max),
            )

        return ax
