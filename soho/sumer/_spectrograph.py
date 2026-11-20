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

_tolerance_window = 0.1
"""
How far apart, in columns, the wavelengths of one window may be in two
exposures for them to be loaded together.

The level-1 files correct the geometry of each exposure on its own,
so the wavelength scale of a window changes by a few thousandths of a column
from one exposure to the next.
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


def _number(header: astropy.io.fits.Header, key: str) -> float:
    """
    The value of a numeric card of a header.

    Parameters
    ----------
    header
        The header.
    key
        The keyword of the card.
    """
    return float(typing.cast(float, header[key]))


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
    The limits are the reference wavelength less and more half the width of
    the window, rounded down to whole columns,
    so ``WAVEMIN`` is the middle of the first column,
    and ``WAVEMAX`` is the middle of the last column of a window with an odd
    number of columns but of the column after it otherwise.
    The window is taken to end before ``WAVEMAX``,
    so that both kinds of image, which one sequence can mix,
    give it the same number of columns.
    Which columns hold data is found from the images themselves,
    by :meth:`SpectrographObservation.from_level1`.

    The files do not say in which order the wavelengths of a window are given.
    Those of a single window are in the order of ``CDELT1``,
    which is plain from its size.
    Those of one of several windows are either in the order of the line it
    was planned for, such as 629.2 to 630.3 angstroms for O V 629.7 in the
    second order,
    or in the first order, the place of that line on the detector,
    such as 1258.4 to 1260.6 angstroms for the same window.
    So a window holds the line if either range holds it.
    Neither can be mistaken for the other, since a line in the second order
    lies at more than 660 angstroms of the first order,
    where no window labelled in the second order reaches.

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
    position = float(typing.cast(float, wavelength.to_value(u.AA))) * order

    if _text(header, "PIMGTYP") == "MULTIPLE":
        for n in range(1, int(_number(header, "ONUMIMG")) + 1):
            minimum = _number(header, f"WAVEMIN{n}")
            maximum = _number(header, f"WAVEMAX{n}")
            for order_label in dict.fromkeys((order, 1)):
                if minimum <= position / order_label <= maximum:
                    break
            else:
                continue
            dispersion = dispersion_first / order_label
            reference = _number(header, f"WAVEL{n}")
            offset = int(np.rint((reference - minimum) / dispersion))
            return _Window(
                start=int(_number(header, f"IREFPIX{n}")) - 1 - offset,
                num=int(np.rint((maximum - minimum) / dispersion)),
                wavelength=(reference - offset * dispersion) * order_label / order,
                dispersion=dispersion_first / order,
            )
        return None

    minimum = _number(header, "WAVEMIN")
    maximum = _number(header, "WAVEMAX")
    if not minimum <= position / order_reference <= maximum:
        return None
    dispersion = dispersion_first / order_reference
    reference = _number(header, "CRVAL1")
    offset = int(np.rint((reference - minimum) / dispersion))
    return _Window(
        start=int(_number(header, "CRPIX1")) - 1 - offset,
        num=int(np.rint((maximum - minimum) / dispersion)),
        wavelength=(reference - offset * dispersion) * order_reference / order,
        dispersion=dispersion_first / order,
    )


def _extent(data: np.ndarray, column: int) -> None | tuple[int, int]:
    """
    The first column, and the one after the last, of the run of columns of an
    image which hold data and include a given column,
    or :obj:`None` if that column holds none.

    The columns which were not read out hold zeros.

    Parameters
    ----------
    data
        The image, with the columns along its last axis.
    column
        The column the run must include.
    """
    filled = np.any(np.isfinite(data) & (data != 0), axis=0)
    if not 0 <= column < filled.size or not filled[column]:
        return None
    empty = np.flatnonzero(~filled)
    lower = empty[empty < column]
    upper = empty[empty > column]
    first = int(lower[~0]) + 1 if lower.size else 0
    last = int(upper[0]) if upper.size else filled.size
    return first, last


def _centroid(
    radiance: np.ndarray,
    wavelengths: np.ndarray,
    wavelength: float,
) -> float:
    """
    The mean position of a line in some exposures:
    the centroid, above its lowest value, of their mean profile near its
    brightest part, which is looked for near the wavelength of the line.

    Parameters
    ----------
    radiance
        The spectra, with wavelength along the last axis.
    wavelengths
        The edges of the columns, in angstroms.
    wavelength
        The wavelength of the line, in angstroms.

    Raises
    ------
    ValueError
        If the spectra have no data near the wavelength of the line.
    """
    with warnings.catch_warnings():
        # columns which no exposure read out have no mean
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(radiance.reshape(-1, radiance.shape[~0]), axis=0)
    centers = (wavelengths[1:] + wavelengths[:-1]) / 2
    velocity = np.asarray(
        u.Quantity(centers, u.AA).to_value(
            u.km / u.s,
            equivalencies=u.doppler_optical(wavelength * u.AA),
        )
    )
    search = _velocity_search.to_value(u.km / u.s)
    candidates = np.isfinite(profile) & (np.abs(velocity) < search)
    if not np.any(candidates):
        raise ValueError(
            f"The spectra have no data within {_velocity_search} of {wavelength} A."
        )
    peak = velocity[candidates][np.argmax(profile[candidates])]
    width = _velocity_centroid.to_value(u.km / u.s)
    near = np.isfinite(profile) & (np.abs(velocity - peak) < width)
    excess = profile[near] - np.min(profile[near])
    if not np.sum(excess) > 0:
        return float(centers[near][np.argmax(profile[near])])
    return float(np.sum(excess * centers[near]) / np.sum(excess))


def _vertices(centers: np.ndarray, width: float) -> np.ndarray:
    """
    The edges between consecutive positions along a raster,
    halfway between them, with the outer edges as far beyond the first and
    last positions as the nearest edge is inside them.

    Parameters
    ----------
    centers
        The positions, in order of time.
    width
        The width of a single exposure,
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
    The detector has more rows than a slit of 120 or 300 arcseconds lights,
    and the rows beyond the ends of the slit hold no signal.

    The spectrum is that of the columns around the window which hold data in
    any exposure, and each exposure is not a number (NaN) in the columns
    outside its own data,
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
    it, and they do not include the small roll of the instrument,
    which the files give as ``ANGLE``;
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
            The exposure time of the exposures to keep.
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
            and one slit.
        """
        if isinstance(path, na.AbstractScalarArray):
            paths = [str(p) for p in np.asarray(path.ndarray).flat]
        elif isinstance(path, (str, os.PathLike)):
            paths = [path]
        else:
            paths = list(path)

        start = None if time_start is None else astropy.time.Time(time_start)
        stop = None if time_stop is None else astropy.time.Time(time_stop)

        kept = []
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
                time = astropy.time.Time(header["DATE_OBS"])
                if start is not None and time < start:
                    continue
                if stop is not None and not time < stop:
                    continue
                if timedelta is not None and header["EXPTIME"] * u.s != timedelta:
                    continue
                if slit is not None and header["SLIT"] != slit:
                    continue
                window = _window(header, wavelength, order)
                if window is None:
                    continue
                kept.append((time, header, window, hdul))

        if not kept:
            raise ValueError(
                f"No exposure in {paths} has a window holding {wavelength}"
                " which matches the selection."
            )

        kept.sort(key=lambda k: k[0].jd)
        times, headers, windows, hduls = zip(*kept)

        window = windows[0]
        for w in windows:
            num = w.num != window.num
            shift = abs(w.wavelength - window.wavelength) / window.dispersion
            stretch = abs(w.dispersion - window.dispersion) * window.num
            if (
                num
                or shift > _tolerance_window
                or stretch > _tolerance_window * window.dispersion
            ):
                raise ValueError(
                    f"The exposures differ in their window, {window} and {w}; "
                    "select one sequence with the time range, `timedelta`, or `slit`."
                )

        for key, values in (
            ("detector", [h["DETECTOR"].strip() for h in headers]),
            ("slit", [h["SLIT"] for h in headers]),
            ("number of rows", [h["NAXIS2"] for h in headers]),
            ("binning", [(h.get("BINX", 1), h.get("BINY", 1)) for h in headers]),
        ):
            if len(set(values)) > 1:
                raise ValueError(
                    f"The exposures differ in their {key}, {sorted(set(map(str, values)))}; "
                    "select one sequence with the time range, `timedelta`, or `slit`."
                )

        header = headers[0]
        if (header.get("BINX", 1), header.get("BINY", 1)) != (1, 1):
            raise NotImplementedError("Binned exposures are not supported yet.")

        # The columns around the window which hold data in any exposure,
        # counted from the start of the window.
        # The windows of an image are the parts of the detector which were
        # read out, so the data around one can reach a few columns past the
        # limits its header gives, or fall a few short of them,
        # and run on into a neighbouring window.
        # Where they end moves by a column or two from one exposure to the
        # next, so each exposure is unobserved, NaN, outside its own run.
        extents = []
        for hdul, w in zip(hduls, windows):
            data = np.asarray(typing.cast(astropy.io.fits.PrimaryHDU, hdul[0]).data)
            extent = _extent(data, w.start + w.num // 2)
            if extent is not None:
                extent = (extent[0] - w.start, extent[1] - w.start)
            extents.append(extent)
        found = [e for e in extents if e is not None]
        lower = min(e[0] for e in found) if found else 0
        upper = max(e[1] for e in found) if found else window.num
        num = upper - lower

        radiance = []
        for h, hdul, w, extent in zip(headers, hduls, windows, extents):
            if h.get("IMGUNITS", "").strip() != "W/sr/m^2/Angstroem":
                raise ValueError(f"Unexpected units {h.get('IMGUNITS')!r}.")
            # A sequence can mix images of one window with images of
            # several, which hold the same window in other columns.
            image = np.asarray(hdul[0].data, dtype=float)
            radiometry = np.asarray(hdul["SUMER-RAD-ARRAY"].data, dtype=float)
            row_order = int(h.get("RADORDER", 1)) - 1
            data = np.full((image.shape[0], num), np.nan)
            if extent is not None:
                columns = slice(w.start + extent[0], w.start + extent[1])
                factor = radiometry[order - 1, columns] / radiometry[row_order, columns]
                index = slice(extent[0] - lower, extent[1] - lower)
                data[:, index] = image[:, columns] * factor
            radiance.append(data)

        outputs = na.ScalarArray(
            ndarray=np.stack(radiance) << _unit_radiance,
            axes=(axis_detector_x, axis_detector_y, axis_wavelength),
        )

        # The wavelengths are the median of those of the exposures,
        # which differ by much less than a column.
        wavelength_first = np.median([w.wavelength for w in windows])
        dispersion = np.median([w.dispersion for w in windows])
        index_column = lower + np.arange(num + 1) - 0.5
        wavelengths = (wavelength_first + index_column * dispersion) * u.AA

        rest = wavelength.to(u.AA)
        if velocity_zero is None:
            shift = 0 * u.AA
        elif isinstance(velocity_zero, str):
            if velocity_zero != "mean":
                raise ValueError(f"{velocity_zero=} not recognized.")
            centroid = _centroid(
                radiance=np.stack(radiance),
                wavelengths=wavelengths.to_value(u.AA),
                wavelength=rest.value,
            )
            shift = centroid * u.AA - rest
        else:
            zero = velocity_zero.to(u.AA, equivalencies=u.doppler_optical(rest))
            shift = zero - rest
        wavelengths = wavelengths - shift

        num_row = header["NAXIS2"]
        index_row = np.arange(num_row + 1) + 0.5
        cdelt2 = np.array([h["CDELT2"] for h in headers])
        if np.any(cdelt2 != cdelt2[0]):
            raise ValueError("The exposures differ in their pixel size along the slit.")
        offset_y = (index_row - header["CRPIX2"]) * cdelt2[0]

        width = header.get("IXWIDTH", 0)
        x = _vertices(np.array([h["CRVAL3"] for h in headers]), width)
        y = _vertices(np.array([h["CRVAL2"] for h in headers]), 0)

        time_vertices = astropy.time.Time(
            [t.isot for t in times] + [headers[~0]["DATE_END"]],
        )

        inputs = na.TemporalDopplerPositionalVectorArray(
            time=na.ScalarArray(
                typing.cast(np.ndarray, time_vertices),
                axes=axis_detector_x,
            ),
            wavelength=na.ScalarArray(wavelengths, axes=axis_wavelength),
            wavelength_rest=wavelength,
            position=na.Cartesian2dVectorArray(
                x=na.ScalarArray(x << u.arcsec, axes=axis_detector_x),
                y=na.ScalarArray(y << u.arcsec, axes=axis_detector_x)
                + na.ScalarArray(offset_y << u.arcsec, axes=axis_detector_y),
            ),
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=na.ScalarArray(
                np.array([h["EXPTIME"] for h in headers]) << u.s,
                axes=axis_detector_x,
            ),
            order=order,
            detector=header["DETECTOR"].strip(),
            shift_wavelength=shift,
            slit=header["SLIT"],
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
            The exposure time of the exposures to keep.
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
        Display the exposures as a false-color image,
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
            If :obj:`None`, the 99.5th percentile of each wavelength is used.
        velocity_min
            The minimum Doppler velocity of the data range.
        velocity_max
            The maximum Doppler velocity of the data range.
        cbar_fraction
            The fraction of the space to use for the colorbar axes if `cax`
            is :obj:`None`.
        """
        outputs = self.outputs.copy()
        outputs[~np.isfinite(outputs)] = 0 * _unit_radiance

        wavelength_rest = na.as_named_array(self.inputs.wavelength_rest)
        wavelength_center = typing.cast(na.AbstractScalarArray, wavelength_rest).ndarray

        axis_wavelength = self.axis_wavelength
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y

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
            # named-arrays takes the names of axes where numpy takes numbers
            percentile = typing.cast(Callable[..., na.AbstractScalar], np.nanpercentile)
            vmax = percentile(outputs, q=99.5, axis=(axis_x, axis_y))

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
                colorbar.inputs.y.to(
                    u.AA,
                    equivalencies=u.doppler_radio(wavelength_center),
                ),
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
            equivalencies = u.doppler_radio(wavelength_center)
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
