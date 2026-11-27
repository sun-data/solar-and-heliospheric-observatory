import io
import tarfile
import pathlib
import warnings
import dataclasses
import pytest
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.io.fits
import soho
import named_arrays as na
from soho.sumer._spectrograph import _window, _extent, _vertices
from soho.sumer._spectrograph import _centroid, _unit_radiance
from soho.sumer._spectrograph import _shift_rows, _centroids_rows, _slope
from soho.sumer._spectrograph import _fill_dead_columns
from soho.sumer._spectrograph import _shift_exposures, _centroids_exposures
from soho.sumer._spectrograph import _sinusoid, _sinusoid_coefficients

_wavelength = 629.73 * u.AA
"""The wavelength of O V."""


def _header(**cards: float | int | str) -> astropy.io.fits.Header:
    """A header made of the given cards."""
    header = astropy.io.fits.Header()
    for key, value in cards.items():
        header[key] = value
    return header


def _rolled(
    path: pathlib.Path,
    directory: pathlib.Path,
    angle: float,
) -> pathlib.Path:
    """
    A copy of a level-1 file whose exposures give another roll of the
    instrument, ``ANGLE``, with the same pointing.

    Parameters
    ----------
    path
        The level-1 file.
    directory
        The directory to write the copy in.
    angle
        The roll of the copy, in degrees.
    """
    result = directory / f"rolled_{angle}.tgz"
    with tarfile.open(path, "r:gz") as source, tarfile.open(result, "w:gz") as copy:
        for member in source:
            content = source.extractfile(member)
            if content is None:
                continue
            data = content.read()
            if member.name.endswith(".fits"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hdul = astropy.io.fits.open(io.BytesIO(data))
                    hdul[0].header["ANGLE"] = angle
                    buffer = io.BytesIO()
                    hdul.writeto(buffer, output_verify="ignore")
                data = buffer.getvalue()
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            copy.addfile(info, io.BytesIO(data))
    return result


@pytest.mark.parametrize(
    argnames="header,wavelength,order,expected",
    argvalues=[
        # several windows, labelled in the order of the line
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=2,
                CDELT1=0.0437,
                CRVAL1=1242.01,
                CRPIX1=463,
                WAVEL1=1242.01,
                WAVEMIN1=1240.92,
                WAVEMAX1=1243.10,
                IREFPIX1=463,
                WAVEL2=629.73,
                WAVEMIN2=629.183,
                WAVEMAX2=630.277,
                IREFPIX2=863,
            ),
            629.73 * u.AA,
            2,
            (837, 50, 629.73 - 25 * 0.0437 / 2, 0.0437 / 2),
        ),
        # several windows, the second labelled with its place in the first
        # order, and CDELT1 in the second order, that of the first window
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=2,
                CDELT1=0.0218,
                CRVAL1=624.94,
                CRPIX1=300,
                WAVEL1=624.94,
                WAVEMIN1=624.394,
                WAVEMAX1=625.486,
                IREFPIX1=300,
                WAVEL2=1259.46,
                WAVEMIN2=1258.37,
                WAVEMAX2=1260.55,
                IREFPIX2=520,
            ),
            629.73 * u.AA,
            2,
            (494, 50, (1259.46 - 25 * 0.0436) / 2, 0.0218),
        ),
        # a line of the first order in a window labelled in the second order,
        # as S II 1250.58 lies in the window of Mg X 625.0
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=2,
                CDELT1=0.0437241,
                CRVAL1=1242.01,
                CRPIX1=463,
                WAVEL1=1242.01,
                WAVEMIN1=1240.92,
                WAVEMAX1=1243.10,
                IREFPIX1=463,
                WAVEL2=625.0,
                WAVEMIN2=624.453,
                WAVEMAX2=625.547,
                IREFPIX2=646,
            ),
            1250.58 * u.AA,
            1,
            (620, 50, 1250.0 - 25 * 0.0437241, 0.0437241),
        ),
        # a window labelled in the first order between 660 and 805 angstroms,
        # which does not hold a line of the second order of the same number
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=1,
                CDELT1=0.0446,
                CRVAL1=770.42,
                CRPIX1=300,
                WAVEL1=770.42,
                WAVEMIN1=769.30,
                WAVEMAX1=771.53,
                IREFPIX1=300,
            ),
            770.42 * u.AA,
            2,
            None,
        ),
        # but does hold the line of the first order
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=1,
                CDELT1=0.0446,
                CRVAL1=770.42,
                CRPIX1=300,
                WAVEL1=770.42,
                WAVEMIN1=769.30,
                WAVEMAX1=771.53,
                IREFPIX1=300,
            ),
            770.42 * u.AA,
            1,
            (274, 50, 770.42 - 25 * 0.0446, 0.0446),
        ),
        # one window, the whole image
        (
            _header(
                PIMGTYP="SINGLE",
                NAXIS1=50,
                CDELT1=0.022036,
                CRVAL1=584.334,
                CRPIX1=26,
                WAVEMIN=583.783,
                WAVEMAX=584.885,
            ),
            584.334 * u.AA,
            2,
            (0, 50, 584.334 - 25 * 0.022036, 0.022036),
        ),
        # one window of an odd number of columns, whose last is left out
        (
            _header(
                PIMGTYP="SINGLE",
                NAXIS1=25,
                CDELT1=0.0218535,
                CRVAL1=629.73,
                CRPIX1=13,
                WAVEMIN=629.73 - 12 * 0.0218535,
                WAVEMAX=629.73 + 12 * 0.0218535,
            ),
            629.73 * u.AA,
            2,
            (0, 24, 629.73 - 12 * 0.0218535, 0.0218535),
        ),
        # one window, which does not hold the line
        (
            _header(
                PIMGTYP="SINGLE",
                NAXIS1=50,
                CDELT1=0.022036,
                CRVAL1=584.334,
                CRPIX1=26,
                WAVEMIN=583.783,
                WAVEMAX=584.885,
            ),
            629.73 * u.AA,
            2,
            None,
        ),
    ],
)
def test_window(
    header: astropy.io.fits.Header,
    wavelength: u.Quantity,
    order: int,
    expected: None | tuple[int, int, float, float],
):
    result = _window(header, wavelength, order)
    if expected is None:
        assert result is None
        return
    assert result is not None
    assert result.start == expected[0]
    assert result.num == expected[1]
    assert np.isclose(result.wavelength, expected[2], atol=1e-3)
    assert np.isclose(result.dispersion, expected[3], rtol=1e-3)


@pytest.mark.parametrize(
    argnames="filled,column,expected",
    argvalues=[
        # a dead column is bridged
        (np.array([0, 1, 1, 0, 1], dtype=bool), 1, (1, 5)),
        # a gap of three columns is not
        (np.array([1, 0, 0, 0, 1], dtype=bool), 0, (0, 1)),
        (np.array([1, 0, 0, 0, 1], dtype=bool), 4, (4, 5)),
        # a column without data next to a run is reached by it
        (np.array([1, 1, 0, 0, 0, 0], dtype=bool), 3, (0, 2)),
        # but not one further away
        (np.array([1, 1, 0, 0, 0, 0], dtype=bool), 4, None),
        (np.array([1, 1, 1], dtype=bool), 0, (0, 3)),
        (np.array([0, 0, 0], dtype=bool), 1, None),
    ],
)
def test_extent(
    filled: np.ndarray,
    column: int,
    expected: None | tuple[int, int],
):
    assert _extent(filled, column) == expected


@pytest.mark.parametrize(
    argnames="centers,width,expected",
    argvalues=[
        (np.array([0.0, 1.0, 2.0]), 0.5, np.array([-0.5, 0.5, 1.5, 2.5])),
        (np.array([3.0]), 1.0, np.array([2.5, 3.5])),
        # a sequence which holds the slit still
        (np.array([3.0, 3.0]), 1.0, np.array([3.0, 3.0, 3.0])),
    ],
)
def test_vertices(
    centers: np.ndarray,
    width: float,
    expected: np.ndarray,
):
    assert np.allclose(_vertices(centers, width), expected)


@pytest.mark.parametrize(
    argnames="offset,offset_other,height_other,velocity_search",
    argvalues=[
        # the line where the files put it, and a weaker one nearby
        (0, 17, 0.2, 100 * u.km / u.s),
        # a brighter line beyond the search is not taken for the line,
        # as N V 1242.8 angstroms is not for Fe XII 1242.0 angstroms
        (0, 18, 2, 100 * u.km / u.s),
        (0, -18, 2, 100 * u.km / u.s),
        # the line 17 columns from where the files put it,
        # and a weaker line there, as in some files of 2005 and 2007
        (-17, 0, 0.2, 200 * u.km / u.s),
        (17, 0, 0.2, 200 * u.km / u.s),
        (0, 30, 2, 200 * u.km / u.s),
        (0, -30, 2, 200 * u.km / u.s),
    ],
)
def test_centroid(
    offset: int,
    offset_other: int,
    height_other: float,
    velocity_search: u.Quantity,
):
    """The line is found where it is, not where the files put it."""
    dispersion = 0.0215 * u.AA
    num = 101
    axis = "wavelength"
    index = na.arange(0, num + 1, axis=axis)
    wavelengths = _wavelength + (index - num / 2) * dispersion
    column = (wavelengths.cell_centers(axis) - _wavelength) / dispersion
    column = column.to(u.dimensionless_unscaled).value
    line = np.exp(-np.square((column - offset) / 2.4) / 2)
    other = height_other * np.exp(-np.square((column - offset_other) / 1.7) / 2)
    exposures = na.ScalarArray(np.ones(3), axes=("detector_x",))
    radiance = (line + other + 0.05) * exposures * _unit_radiance

    result = _centroid(
        radiance=radiance,
        wavelengths=wavelengths,
        wavelength=_wavelength,
        axis_wavelength=axis,
        velocity_search=velocity_search,
    )

    expected = _wavelength + offset * dispersion
    assert np.isclose(result, expected, rtol=0, atol=1e-3 * dispersion)


def _lines(centers: np.ndarray, num_exposure: int = 3, num: int = 41) -> np.ndarray:
    """
    Gaussian lines of 2.4 columns, one row of the slit for each center,
    in columns from the middle of the spectrum, along the exposures, the slit,
    and wavelength.
    """
    column = np.arange(num) - num // 2
    rows = np.exp(-np.square((column - centers[:, np.newaxis]) / 2.4) / 2)
    return np.broadcast_to(rows, (num_exposure, *rows.shape)).copy()


def _columns(radiance: np.ndarray) -> np.ndarray:
    """The centroid of each spectrum, in columns from the middle of the spectrum."""
    column = np.arange(radiance.shape[~0]) - radiance.shape[~0] // 2
    return np.nansum(radiance * column, axis=~0) / np.nansum(radiance, axis=~0)


@pytest.mark.parametrize("shift", [0, 0.3, -1.5, 2, -0.75])
def test_shift_rows(shift: float):
    """A row moves toward the start of the spectrum, and keeps its radiance."""
    radiance = _lines(np.array([0.0, 1.0]))
    result = _shift_rows(radiance, np.array([shift, 0]))
    assert np.allclose(_columns(result)[:, 0], -shift)
    assert np.allclose(np.nansum(result, axis=~0), np.nansum(radiance, axis=~0))
    assert np.array_equal(result[:, 1], radiance[:, 1])


def test_shift_rows_ends():
    """A column which takes radiance from beyond the spectrum, or from a column
    with no data, has no data."""
    radiance = _lines(np.array([0.0]))
    radiance[:, :, 20] = np.nan
    result = _shift_rows(radiance, np.array([0.5]))
    assert np.all(np.isnan(result[:, :, [19, 20, ~0]]))
    assert np.all(np.isfinite(result[:, :, [18, 21, ~1]]))
    # a whole number of columns takes nothing from the next one
    whole = _shift_rows(radiance, np.array([1.0]))
    assert np.all(np.isnan(whole[:, :, [19, ~0]]))
    assert np.all(np.isfinite(whole[:, :, [18, 20, ~1]]))


def test_centroids_rows():
    """The median centroid of the bright pixels of each row."""
    step = 10
    radiance = _lines(np.array([0, 0.5, -1, 0.2]), num_exposure=12)
    velocity = step * (np.arange(radiance.shape[~0]) - radiance.shape[~0] // 2)
    # faint pixels are left out, and a row with too few bright pixels has
    # no median
    radiance[:4, 2] = 0.01 * radiance[:4, 2]
    radiance[:9, 3] = np.nan
    # a band wide enough that it cuts off none of the line
    result = _centroids_rows(
        radiance=radiance,
        velocity=velocity,
        velocity_band=100,
        fraction_bright=0.2,
        num_min=5,
    )
    assert np.allclose(result[:3], step * np.array([0, 0.5, -1]), atol=0.01)
    assert np.isnan(result[3])
    # no pixel with data over the whole band
    none = _centroids_rows(np.full_like(radiance, np.nan), velocity, 100, 0.2, 5)
    assert np.all(np.isnan(none))


def test_fill_dead_columns():
    """
    A column with no data in any pixel is filled between its neighbours.

    Columns with data in some pixels, and dead columns at the ends,
    are left as they are.
    """
    radiance = np.arange(4 * 3 * 8, dtype=float).reshape(4, 3, 8)
    radiance[..., [0, 3, 4]] = np.nan
    radiance[0, 0, 6] = np.nan
    result = _fill_dead_columns(radiance)
    assert np.allclose(result[..., 3], radiance[..., 2] + 1)
    assert np.allclose(result[..., 4], radiance[..., 2] + 2)
    assert np.all(np.isnan(result[..., 0]))
    assert np.isnan(result[0, 0, 6])
    assert np.array_equal(result[..., [1, 2, 5, 7]], radiance[..., [1, 2, 5, 7]])
    assert np.all(np.isnan(_fill_dead_columns(np.full((2, 2, 3), np.nan))))


def test_slope():
    """A straight line through points, with one far from it left out."""
    x = np.arange(20, dtype=float)
    y = 0.5 * x + 3
    y[7] = 100
    assert np.isclose(_slope(x, y), 0.5)
    assert np.isclose(_slope(x[:2], y[:2]), 0.5)


@pytest.mark.parametrize("shift", [0, 0.3, -1.5])
def test_shift_exposures(shift: float):
    """An exposure moves toward the start of the spectrum, and keeps its radiance."""
    radiance = _lines(np.array([0.0, 1.0]), num_exposure=2)
    result = _shift_exposures(radiance, np.array([shift, 0]))
    assert np.allclose(_columns(result)[0], -shift + np.array([0, 1]))
    assert np.allclose(np.nansum(result, axis=~0), np.nansum(radiance, axis=~0))
    assert np.array_equal(result[1], radiance[1])


def test_centroids_exposures():
    """The median centroid of the bright pixels of each exposure."""
    step = 10
    rows = _lines(np.array([0, 0.5, -1, 0.2]), num_exposure=12)
    radiance = np.swapaxes(rows, 0, 1).copy()
    velocity = step * (np.arange(radiance.shape[~0]) - radiance.shape[~0] // 2)
    # a faint exposure has too few bright pixels for its median
    radiance[3, :9] = np.nan
    result = _centroids_exposures(
        radiance=radiance,
        velocity=velocity,
        velocity_band=100,
        fraction_bright=0.2,
        num_min=5,
    )
    assert np.allclose(result[:3], step * np.array([0, 0.5, -1]), atol=0.01)
    assert np.isnan(result[3])


def test_sinusoid():
    """
    The period and the sinusoid which, with a straight line, fit some points,
    with one far from them left out.
    """
    time = np.linspace(0, 8, 200)
    phase = 2 * np.pi * time / 2.1
    values = 0.3 * time - 1 + 1.5 * np.cos(phase) - 0.5 * np.sin(phase)
    values[17] = 100
    periods = np.linspace(1.75, 2.5, 76)
    period, coefficients = _sinusoid(time, values, periods)
    assert np.isclose(period, 2.1)
    assert np.allclose(coefficients, [1.5, -0.5])
    # too few points to leave any out
    few = _sinusoid_coefficients(time[:5], values[:5], 2.1)
    assert np.allclose(few, [1.5, -0.5])


class TestSpectrographObservation:
    def test_from_level1(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        assert isinstance(result, soho.sumer.SpectrographObservation)
        shape = result.outputs.shape
        assert shape["detector_x"] == 76
        # The first column of the window was not read out.
        assert shape["wavelength"] == 49
        assert np.all(np.isfinite(result.outputs.ndarray))
        assert result.inputs.time.shape == {"detector_x": shape["detector_x"] + 1}
        assert result.inputs.wavelength.shape == {"wavelength": shape["wavelength"] + 1}
        position = result.inputs.position
        assert position.x.shape["detector_x"] == shape["detector_x"] + 1
        assert position.y.shape["detector_y"] == shape["detector_y"] + 1
        assert result.outputs.unit.is_equivalent(u.W / (u.sr * u.m**2 * u.AA))
        assert result.order == 2
        assert result.detector in ("A", "B")
        assert np.all(result.timedelta == 15 * u.s)
        assert result.inputs.wavelength_rest == _wavelength
        time = astropy.time.Time(result.inputs.time.ndarray)
        assert np.all(np.diff(time.jd) > 0)
        assert result.inputs.wavelength.min() < _wavelength
        assert result.inputs.wavelength.max() > _wavelength

        width = result.inputs.wavelength.volume_cell(result.axis_wavelength)
        radiance = (result.outputs * width).sum(result.axis_wavelength)
        mean = np.nanmean(radiance.ndarray)
        assert 0.01 * u.W / (u.sr * u.m**2) < mean < 10 * u.W / (u.sr * u.m**2)

    @pytest.mark.parametrize("angle", [-2.04, 118.86, 180])
    def test_from_level1_roll(
        self,
        path_test: pathlib.Path,
        tmp_path: pathlib.Path,
        angle: float,
    ):
        kwargs = dict(wavelength=_wavelength, order=2)
        # The roll of the test file is zero.
        files = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            **kwargs,
        )
        rolled = soho.sumer.SpectrographObservation.from_level1(
            path=_rolled(path_test, tmp_path, angle),
            **kwargs,
        )
        axis_y = rolled.axis_detector_y
        step = files.inputs.position[{axis_y: 1}] - files.inputs.position[{axis_y: 0}]
        step_rolled = (
            rolled.inputs.position[{axis_y: 1}] - rolled.inputs.position[{axis_y: 0}]
        )
        # the slit is turned by the roll
        radians = np.deg2rad(angle)
        assert np.all(step.x == 0 * u.arcsec)
        assert np.allclose(step_rolled.x, -np.sin(radians) * step.y)
        assert np.allclose(step_rolled.y, np.cos(radians) * step.y)
        # about the pointing of the files, at the reference row,
        # which is the middle of the slit of the test file
        middle = files.inputs.position.mean(axis_y)
        middle_rolled = rolled.inputs.position.mean(axis_y)
        assert np.allclose(middle_rolled.x, middle.x)
        assert np.allclose(middle_rolled.y, middle.y)

    @pytest.mark.parametrize("angle", [-2.04, 118.86])
    def test_from_level1_roll_single(
        self,
        path_test: pathlib.Path,
        tmp_path: pathlib.Path,
        angle: float,
    ):
        """The width of a single exposure is turned by the roll."""
        kwargs = dict(wavelength=_wavelength, order=2)
        files = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            **kwargs,
        )
        axis_x = files.axis_detector_x
        start = astropy.time.Time(files.inputs.time[{axis_x: 0}].ndarray)
        kwargs = kwargs | dict(time_start=start, time_stop=start + 1 * u.s)
        single = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            **kwargs,
        )
        rolled = soho.sumer.SpectrographObservation.from_level1(
            path=_rolled(path_test, tmp_path, angle),
            **kwargs,
        )
        assert single.outputs.shape[axis_x] == 1
        position = single.inputs.position
        width = position[{axis_x: 1}] - position[{axis_x: 0}]
        position_rolled = rolled.inputs.position
        width_rolled = position_rolled[{axis_x: 1}] - position_rolled[{axis_x: 0}]
        radians = np.deg2rad(angle)
        assert np.all(width.x > 0 * u.arcsec)
        assert np.all(width.y == 0 * u.arcsec)
        assert np.allclose(width_rolled.x, np.cos(radians) * width.x)
        assert np.allclose(width_rolled.y, np.sin(radians) * width.x)

    @pytest.mark.parametrize(
        argnames="angles,expected",
        argvalues=[
            ((-2.1, -1.9), -2),
            # on either side of 180 degrees
            ((179.5, -179.5), 180),
        ],
    )
    def test_from_level1_roll_mean(
        self,
        path_test: pathlib.Path,
        tmp_path: pathlib.Path,
        angles: tuple[float, float],
        expected: float,
    ):
        """Exposures of slightly different rolls are turned by their mean."""
        kwargs = dict(wavelength=_wavelength, order=2)
        files = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            **kwargs,
        )
        rolled = soho.sumer.SpectrographObservation.from_level1(
            path=[_rolled(path_test, tmp_path, angle) for angle in angles],
            **kwargs,
        )
        axis_x = rolled.axis_detector_x
        axis_y = rolled.axis_detector_y
        index = {axis_x: 0, axis_y: 1}
        index_other = {axis_x: 0, axis_y: 0}
        step = files.inputs.position[index] - files.inputs.position[index_other]
        position = rolled.inputs.position
        step_rolled = position[index] - position[index_other]
        radians = np.deg2rad(expected)
        assert np.allclose(step_rolled.x, -np.sin(radians) * step.y)
        assert np.allclose(step_rolled.y, np.cos(radians) * step.y)

    def test_from_level1_roll_differ(
        self,
        path_test: pathlib.Path,
        tmp_path: pathlib.Path,
    ):
        with pytest.raises(ValueError, match="differ in their roll"):
            soho.sumer.SpectrographObservation.from_level1(
                path=[path_test, _rolled(path_test, tmp_path, 118.86)],
                wavelength=_wavelength,
                order=2,
            )

    def test_from_level1_time_range(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=[path_test],
            wavelength=_wavelength,
            order=2,
            time_start="2005-04-15T15:50:00",
            time_stop="2005-04-15T16:00:00",
        )
        time = astropy.time.Time(result.inputs.time.ndarray)
        assert time[0] >= astropy.time.Time("2005-04-15T15:50:00")
        assert time[~1] < astropy.time.Time("2005-04-15T16:00:00")
        assert result.outputs.shape["detector_x"] < 76

    def test_from_level1_none(self, path_test: pathlib.Path):
        with pytest.raises(ValueError, match="No exposure"):
            soho.sumer.SpectrographObservation.from_level1(
                path=path_test,
                wavelength=1548.2 * u.AA,
                order=1,
            )

    def test_velocity_zero(self, path_test: pathlib.Path):
        kwargs = dict(path=path_test, wavelength=_wavelength, order=2)
        files = soho.sumer.SpectrographObservation.from_level1(
            velocity_zero=None,
            **kwargs,
        )
        assert files.shift_wavelength == 0 * u.AA

        zeroed = soho.sumer.SpectrographObservation.from_level1(**kwargs)
        wavelengths = zeroed.inputs.wavelength.ndarray
        centers = (wavelengths[1:] + wavelengths[:-1]) / 2
        profile = np.nanmean(zeroed.outputs.ndarray.value, axis=(0, 1))
        peak = centers[np.argmax(profile)]
        dispersion = np.diff(wavelengths).mean()
        assert abs(peak - _wavelength) < 2 * dispersion
        assert np.allclose(
            wavelengths + zeroed.shift_wavelength,
            files.inputs.wavelength.ndarray,
        )

        velocity = 30 * u.km / u.s
        shifted = soho.sumer.SpectrographObservation.from_level1(
            velocity_zero=velocity,
            **kwargs,
        )
        expected = velocity.to(u.AA, equivalencies=u.doppler_optical(_wavelength))
        assert np.isclose(shifted.shift_wavelength, expected - _wavelength)

    def test_velocity_search(self, path_test: pathlib.Path):
        kwargs = dict(path=path_test, wavelength=_wavelength, order=2)
        zeroed = soho.sumer.SpectrographObservation.from_level1(**kwargs)
        # The line of this file is 7 columns from where the file puts it,
        # and no brighter line is within 20.
        wide = soho.sumer.SpectrographObservation.from_level1(
            velocity_search=200 * u.km / u.s,
            **kwargs,
        )
        assert wide.shift_wavelength == zeroed.shift_wavelength
        with pytest.raises(ValueError, match="no data within"):
            soho.sumer.SpectrographObservation.from_level1(
                velocity_search=0 * u.km / u.s,
                **kwargs,
            )

    def test_show(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        ax = result.show()
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_remove_bad_steps(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        radiance = result._radiance_aligned()
        radiance[30] = 10 * radiance[30]
        repaired = result._with_radiance(radiance).remove_bad_steps()
        expected = (radiance[29] + radiance[31]) / 2
        assert np.allclose(repaired._radiance_aligned()[30], expected, equal_nan=True)
        assert np.allclose(
            repaired._radiance_aligned()[10],
            radiance[10],
            equal_nan=True,
        )

    def test_trim_slit(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        axis = result.axis_detector_y
        lit = result.trim_slit()
        num_row = lit.outputs.shape[axis]
        assert 0 < num_row < result.outputs.shape[axis]

        radiance = lit._radiance_aligned()
        radiance[:, :3] = 0
        trimmed = lit._with_radiance(radiance).trim_slit()
        assert trimmed.outputs.shape[axis] == num_row - 3
        assert trimmed.inputs.position.y.shape[axis] == num_row - 3 + 1

    def test_remove_stripes(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        radiance = result._radiance_aligned()
        num_row = radiance.shape[1]
        stripes = 1 + 0.2 * (np.arange(num_row) % 2 - 0.5)
        striped = result._with_radiance(radiance * stripes[np.newaxis, :, np.newaxis])
        removed = striped.remove_stripes(num_min=5)._radiance_aligned()

        def contrast(cube: np.ndarray) -> float:
            """How much alternate rows of the slit differ, on average."""
            rows = np.nanmedian(np.nansum(cube, axis=~0), axis=0)
            rows = rows[rows > 0]
            return float(np.nanmedian(np.abs(np.diff(rows)) / rows[1:]))

        assert contrast(removed) < contrast(
            striped.remove_stripes(gain_max=1, gain_min=1)._radiance_aligned()
        )

    @staticmethod
    def _tilted(
        path: pathlib.Path,
    ) -> tuple[
        soho.sumer.SpectrographObservation,
        soho.sumer.SpectrographObservation,
        np.ndarray,
    ]:
        """
        The test observation, the same with a tilt of two columns, about
        20 km/s, put in from one end of the slit to the other, and the
        velocity of the middle of each column.
        """
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path,
            wavelength=_wavelength,
            order=2,
        ).trim_slit()
        radiance = result._radiance_aligned()
        rows = np.arange(radiance.shape[1], dtype=float)
        tilt = 2 * (rows - rows.mean()) / (rows.size - 1)
        tilted = result._with_radiance(_shift_rows(radiance, tilt))
        edges = result.inputs.wavelength.ndarray_aligned((result.axis_wavelength,))
        middles = (edges[1:] + edges[:-1]) / 2
        velocity = middles.to_value(
            u.km / u.s,
            equivalencies=u.doppler_optical(_wavelength),
        )
        return result, tilted, velocity

    @staticmethod
    def _centroids(
        observation: soho.sumer.SpectrographObservation,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """The median centroid of each row within 40 km/s, in km/s."""
        return _centroids_rows(
            radiance=_fill_dead_columns(observation._radiance_aligned()),
            velocity=velocity,
            velocity_band=40,
            fraction_bright=0.2,
            num_min=10,
        )

    @classmethod
    def _change(
        cls,
        observation: soho.sumer.SpectrographObservation,
        velocity: np.ndarray,
    ) -> float:
        """The change of the centroid from one end of the slit to the other, km/s."""
        centroid = cls._centroids(observation, velocity)
        rows = np.arange(centroid.size, dtype=float)
        finite = np.isfinite(centroid)
        return _slope(rows[finite], centroid[finite]) * (rows.size - 1)

    @staticmethod
    def _zero(observation: soho.sumer.SpectrographObservation) -> u.Quantity:
        """The centroid of the mean profile, which from_level1 puts at zero, km/s."""
        wavelength = _centroid(
            radiance=observation.outputs,
            wavelengths=observation.inputs.wavelength,
            wavelength=_wavelength,
            axis_wavelength=observation.axis_wavelength,
        )
        return wavelength.to(u.km / u.s, equivalencies=u.doppler_optical(_wavelength))

    def test_remove_tilt(self, path_test: pathlib.Path):
        """A tilt put into the spectra is taken out, and the zero stays."""
        result, tilted, velocity = self._tilted(path_test)
        # the band leaves out the wings of the line, so the tilt measures
        # much less than it is
        change = self._change(tilted, velocity) - self._change(result, velocity)
        assert abs(change) > 3
        removed = tilted.remove_tilt()
        assert abs(self._change(removed, velocity)) < 0.05
        assert abs(self._zero(removed) - self._zero(tilted)) < 0.05 * u.km / u.s
        assert removed.inputs is tilted.inputs
        # all of the tilt put in is taken out, not only what the band measures
        expected = self._centroids(result.remove_tilt(), velocity)
        difference = self._centroids(removed, velocity) - expected
        assert np.sqrt(np.nanmean(np.square(difference))) < 0.5

    def test_remove_tilt_dead_column(self, path_test: pathlib.Path):
        """A column with no data in the band does not stop the measurement."""
        result, tilted, velocity = self._tilted(path_test)
        radiance = tilted._radiance_aligned()
        column = int(np.argmin(np.abs(velocity - 30)))
        radiance[..., column] = np.nan
        dead = tilted._with_radiance(radiance)
        removed = dead.remove_tilt()
        assert removed is not dead
        # the same tilt is found as without the dead column, so the spectra
        # agree away from it
        expected = tilted.remove_tilt()._radiance_aligned()
        away = np.abs(np.arange(velocity.size) - column) > 2
        difference = removed._radiance_aligned()[..., away] - expected[..., away]
        scale = np.nanmax(expected)
        assert np.nanmedian(np.abs(difference)) < 1e-3 * scale
        assert np.nanmax(np.abs(difference)) < 3e-2 * scale

    def test_remove_tilt_tolerance(self, path_test: pathlib.Path):
        """A tilt within the tolerance to begin with is left alone."""
        result, tilted, velocity = self._tilted(path_test)
        assert tilted.remove_tilt(tolerance=1000 * u.km / u.s) is tilted

    def test_remove_tilt_no_line(self, path_test: pathlib.Path):
        """A line far from zero velocity makes no peak in the band."""
        result, tilted, velocity = self._tilted(path_test)
        radiance = tilted._radiance_aligned()
        away = np.full(radiance.shape[1], -12.0)
        moved = tilted._with_radiance(_shift_rows(radiance, away))
        assert moved.remove_tilt() is moved

    @pytest.mark.parametrize("case", ["no column", "no data"])
    def test_remove_tilt_nothing(self, path_test: pathlib.Path, case: str):
        """With no column in the band, or no data, nothing is shifted."""
        result, tilted, velocity = self._tilted(path_test)
        if case == "no column":
            assert tilted.remove_tilt(velocity_band=1e-6 * u.km / u.s) is tilted
        else:
            radiance = np.full_like(tilted._radiance_aligned(), np.nan)
            empty = tilted._with_radiance(radiance)
            assert empty.remove_tilt() is empty

    @pytest.mark.parametrize("seed", [13, 20, 23, 27, 1])
    def test_remove_tilt_noise(self, path_test: pathlib.Path, seed: int):
        """Without a line in the band, the shift stays within its limit."""
        result, tilted, velocity = self._tilted(path_test)
        radiance = tilted._radiance_aligned()
        rng = np.random.default_rng(seed)
        noise = np.abs(1 + 0.3 * rng.standard_normal(radiance.shape))
        noise = np.where(np.isfinite(radiance), noise, np.nan)
        noisy = tilted._with_radiance(noise)
        removed = noisy.remove_tilt()._radiance_aligned()
        # a change of twice the band, 80 km/s, from one end of the slit to the
        # other moves either end by less than four columns
        lost = np.mean(np.isnan(removed)) - np.mean(np.isnan(noise))
        assert lost < 4 / radiance.shape[~0]

    @pytest.mark.parametrize(
        argnames="slopes,same",
        argvalues=[
            # the centroids can no longer be measured once shifted
            ([1, np.nan], True),
            # the centroids do not respond to the shift
            ([1, 1], True),
            # the step goes beyond the largest tilt
            ([100, 99], False),
        ],
    )
    def test_remove_tilt_stops(
        self,
        path_test: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        slopes: list[float],
        same: bool,
    ):
        """
        The search for the tilt stops where it can go no further.

        It then removes the tilt which measured least of those tried,
        which may be none.
        """
        result, tilted, velocity = self._tilted(path_test)
        calls = []

        def centroids_rows(
            radiance: np.ndarray,
            velocity: np.ndarray,
            velocity_band: float,
            fraction_bright: float,
            num_min: int,
        ) -> np.ndarray:
            """Centroids with the next of the given tilts, in km/s per row."""
            rows = np.arange(radiance.shape[1], dtype=float)
            calls.append(radiance.shape)
            return slopes[len(calls) - 1] * rows

        monkeypatch.setattr(soho.sumer._spectrograph, "_centroids_rows", centroids_rows)
        removed = tilted.remove_tilt()
        assert len(calls) == 2
        assert (removed is tilted) == same

    def test_remove_tilt_none(self, path_test: pathlib.Path):
        """With too few bright pixels in every row, nothing is shifted."""
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        assert result.remove_tilt(num_min=10_000) is result

    def test_hours(self, path_test: pathlib.Path):
        """The middle of each exposure, in hours from the middle of the first."""
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        hours = result._hours()
        assert hours.shape == (result.outputs.shape[result.axis_detector_x],)
        assert hours[0] == 0
        assert np.all(np.diff(hours) > 0)
        # the 76 exposures of the pass take about 22 minutes
        assert 0.3 < hours[~0] < 0.4

    @staticmethod
    def _drifted(
        path: pathlib.Path,
    ) -> tuple[
        soho.sumer.SpectrographObservation,
        soho.sumer.SpectrographObservation,
        np.ndarray,
    ]:
        """
        The test observation with its exposures spread over six hours,
        the same with a drift of half a column, about 5 km/s, put in as a
        sinusoid of two hours, and the velocity of the middle of each column.
        """
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path,
            wavelength=_wavelength,
            order=2,
        ).trim_slit()
        axis_x = result.axis_detector_x
        time = result.inputs.time
        num = time.shape[axis_x]
        vertices = time.ndarray[0] + np.linspace(0, 6, num) * u.hr
        time = na.ScalarArray(vertices, axes=axis_x)
        result = dataclasses.replace(
            result,
            inputs=dataclasses.replace(result.inputs, time=time),
        )
        drift = 0.5 * np.sin(2 * np.pi * result._hours() / 2 + 0.3)
        radiance = _shift_exposures(result._radiance_aligned(), drift)
        drifted = result._with_radiance(radiance)
        edges = result.inputs.wavelength.ndarray_aligned((result.axis_wavelength,))
        middles = (edges[1:] + edges[:-1]) / 2
        velocity = middles.to_value(
            u.km / u.s,
            equivalencies=u.doppler_optical(_wavelength),
        )
        return result, drifted, velocity

    @staticmethod
    def _centroids_drift(
        observation: soho.sumer.SpectrographObservation,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """The median centroid of each exposure within 40 km/s, in km/s."""
        return _centroids_exposures(
            radiance=_fill_dead_columns(observation._radiance_aligned()),
            velocity=velocity,
            velocity_band=40,
            fraction_bright=0.2,
            num_min=10,
        )

    @classmethod
    def _amplitude(
        cls,
        observation: soho.sumer.SpectrographObservation,
        velocity: np.ndarray,
    ) -> float:
        """The amplitude of a sinusoid of two hours in the centroids, in km/s."""
        centroid = cls._centroids_drift(observation, velocity)
        hours = observation._hours()
        finite = np.isfinite(centroid)
        coefficients = _sinusoid_coefficients(hours[finite], centroid[finite], 2)
        return float(np.hypot(*coefficients))

    def test_remove_drift(self, path_test: pathlib.Path):
        """A drift put into the spectra is taken out, and the zero stays."""
        result, drifted, velocity = self._drifted(path_test)
        # the band leaves out the wings of the line, so the drift measures
        # less than it is
        assert self._amplitude(drifted, velocity) > 1
        removed = drifted.remove_drift()
        assert self._amplitude(removed, velocity) < 0.05
        assert abs(self._zero(removed) - self._zero(drifted)) < 0.05 * u.km / u.s
        assert removed.inputs is drifted.inputs
        # all of the drift put in is taken out, not only what the band measures
        expected = self._centroids_drift(result, velocity)
        difference = self._centroids_drift(removed, velocity) - expected
        assert np.sqrt(np.nanmean(np.square(difference))) < 0.5

    def test_remove_drift_short(self, path_test: pathlib.Path):
        """Exposures which span less than `span_min` are left alone."""
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        assert result.remove_drift() is result

    def test_remove_drift_tolerance(self, path_test: pathlib.Path):
        """A drift within the tolerance to begin with is left alone."""
        result, drifted, velocity = self._drifted(path_test)
        assert drifted.remove_drift(tolerance=1000 * u.km / u.s) is drifted

    def test_remove_drift_no_line(self, path_test: pathlib.Path):
        """A line far from zero velocity makes no peak in the band."""
        result, drifted, velocity = self._drifted(path_test)
        radiance = drifted._radiance_aligned()
        away = np.full(radiance.shape[0], -12.0)
        moved = drifted._with_radiance(_shift_exposures(radiance, away))
        assert moved.remove_drift() is moved

    @pytest.mark.parametrize("case", ["no column", "no data"])
    def test_remove_drift_nothing(self, path_test: pathlib.Path, case: str):
        """With no column in the band, or no data, nothing is shifted."""
        result, drifted, velocity = self._drifted(path_test)
        if case == "no column":
            assert drifted.remove_drift(velocity_band=1e-6 * u.km / u.s) is drifted
        else:
            radiance = np.full_like(drifted._radiance_aligned(), np.nan)
            empty = drifted._with_radiance(radiance)
            assert empty.remove_drift() is empty

    def test_remove_drift_none(self, path_test: pathlib.Path):
        """With too few bright pixels in every exposure, nothing is shifted."""
        result, drifted, velocity = self._drifted(path_test)
        assert drifted.remove_drift(num_min=10_000) is drifted

    @pytest.mark.parametrize("seed", [13, 20, 23])
    def test_remove_drift_noise(self, path_test: pathlib.Path, seed: int):
        """Without a line in the band, the shift stays within its limit."""
        result, drifted, velocity = self._drifted(path_test)
        radiance = drifted._radiance_aligned()
        rng = np.random.default_rng(seed)
        noise = np.abs(1 + 0.3 * rng.standard_normal(radiance.shape))
        noise = np.where(np.isfinite(radiance), noise, np.nan)
        noisy = drifted._with_radiance(noise)
        removed = noisy.remove_drift()._radiance_aligned()
        # an amplitude of the band, 40 km/s, moves an exposure by no more than
        # twice the band, less than nine columns
        lost = np.mean(np.isnan(removed)) - np.mean(np.isnan(noise))
        assert lost < 9 / radiance.shape[~0]

    @pytest.mark.parametrize(
        argnames="amplitudes,same",
        argvalues=[
            # the centroids can no longer be measured once shifted
            ([1, np.nan], True),
            # the centroids do not respond to the shift
            ([1, 1], True),
            # the step goes beyond the largest amplitude
            ([100, 90], False),
        ],
    )
    def test_remove_drift_stops(
        self,
        path_test: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        amplitudes: list[float],
        same: bool,
    ):
        """
        The search for the drift stops where it can go no further.

        It then removes the drift which measured least of those tried,
        which may be none.
        """
        result, drifted, velocity = self._drifted(path_test)
        hours = drifted._hours()
        calls = []

        def centroids_exposures(
            radiance: np.ndarray,
            velocity: np.ndarray,
            velocity_band: float,
            fraction_bright: float,
            num_min: int,
        ) -> np.ndarray:
            """Centroids with the next of the given amplitudes, in km/s."""
            calls.append(radiance.shape)
            amplitude = amplitudes[len(calls) - 1]
            return amplitude * np.sin(2 * np.pi * hours / 2 + 0.3)

        monkeypatch.setattr(
            soho.sumer._spectrograph,
            "_centroids_exposures",
            centroids_exposures,
        )
        removed = drifted.remove_drift()
        assert len(calls) == 2
        assert (removed is drifted) == same

    def test_show_still(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        position = result.inputs.position
        position = 0 * position + position[{result.axis_detector_x: 0}]
        still = dataclasses.replace(
            result,
            inputs=dataclasses.replace(result.inputs, position=position),
        )
        with pytest.raises(ValueError, match="not a raster"):
            still.show()
        plt.close("all")

    def test_show_along_y(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        # a raster turned a quarter of a turn steps along y alone,
        # with the slit along x
        axis_x = result.axis_detector_x
        position = result.inputs.position
        position = na.Cartesian2dVectorArray(x=-position.y, y=position.x)
        assert np.all(np.diff(position.x, axis=axis_x) == 0 * u.arcsec)
        along_y = dataclasses.replace(
            result,
            inputs=dataclasses.replace(result.inputs, position=position),
        )
        assert isinstance(along_y.show(), matplotlib.axes.Axes)
        plt.close("all")

    def test_show_along_slit(self, path_test: pathlib.Path):
        """A sequence which moves the slit along itself is not a raster."""
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        axis_x = result.axis_detector_x
        position = result.inputs.position
        num = position.x.shape[axis_x]
        position = na.Cartesian2dVectorArray(
            x=0 * position.x + position.x[{axis_x: 0}],
            y=position.y + na.arange(0, num, axis=axis_x) * u.arcsec,
        )
        along_slit = dataclasses.replace(
            result,
            inputs=dataclasses.replace(result.inputs, position=position),
        )
        with pytest.raises(ValueError, match="not a raster"):
            along_slit.show()
        plt.close("all")

    def test_from_level1_timedelta_tolerance(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
            timedelta=15.01 * u.s,
        )
        assert result.outputs.shape["detector_x"] == 76

    def test_from_level1_timedelta(self, path_test: pathlib.Path):
        with pytest.raises(ValueError, match="No exposure"):
            soho.sumer.SpectrographObservation.from_level1(
                path=path_test,
                wavelength=_wavelength,
                order=2,
                timedelta=1 * u.s,
            )


def test_open():
    kwargs = dict(
        time_start="2005-04-15T15:50:00",
        time_stop="2005-04-15T16:00:00",
        wavelength=_wavelength,
        order=2,
    )
    result = soho.sumer.open(**kwargs)
    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.shape["detector_x"] > 0
    # the search is passed on
    with pytest.raises(ValueError, match="no data within"):
        soho.sumer.open(velocity_search=0 * u.km / u.s, **kwargs)
