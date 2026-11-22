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
    result = soho.sumer.open(
        time_start="2005-04-15T15:50:00",
        time_stop="2005-04-15T16:00:00",
        wavelength=_wavelength,
        order=2,
    )
    assert isinstance(result, soho.sumer.SpectrographObservation)
    assert result.outputs.shape["detector_x"] > 0
