import pathlib
import pytest
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import soho
from soho.sumer._data import _url
from soho.sumer._spectrograph import _window, _extent, _vertices

_name_test = "sum_20050415_153008_l1.tgz"
"""
A small level-1 file, whose window on O V 629.7 angstroms, in the second
order, is labelled with its place in the first order.
"""

_wavelength = 629.73 * u.AA
"""The wavelength of O V."""


@pytest.fixture(scope="module")
def path_test(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """The test file, downloaded once for the module."""
    directory = tmp_path_factory.mktemp("sumer")
    urls = na.ScalarArray(np.array([_url(_name_test)]), axes="time")
    return pathlib.Path(str(soho.sumer.download(urls, directory).ndarray[0]))


def _header(**cards: float | int | str) -> astropy.io.fits.Header:
    """A header made of the given cards."""
    header = astropy.io.fits.Header()
    for key, value in cards.items():
        header[key] = value
    return header


@pytest.mark.parametrize(
    argnames="header,wavelength,order,expected",
    argvalues=[
        # several windows, labelled in the order of the line
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=2,
                CDELT1=0.0437,
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
        # several windows, labelled with their place in the first order,
        # and CDELT1 in the second order
        (
            _header(
                PIMGTYP="MULTIPLE",
                ONUMIMG=1,
                CDELT1=0.0218,
                WAVEL1=1259.46,
                WAVEMIN1=1258.37,
                WAVEMAX1=1260.55,
                IREFPIX1=200,
            ),
            629.73 * u.AA,
            2,
            (174, 50, (1259.46 - 25 * 0.0436) / 2, 0.0218),
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
    argnames="data,column,expected",
    argvalues=[
        (np.array([[0, 1, 2, 0, 3]]), 1, (1, 3)),
        (np.array([[0, 1, 2, 0, 3]]), 4, (4, 5)),
        (np.array([[0, 1, 2, 0, 3]]), 3, None),
        (np.array([[1, 1, 1]]), 0, (0, 3)),
        (np.array([[np.nan, 1, 1]]), 2, (1, 3)),
    ],
)
def test_extent(
    data: np.ndarray,
    column: int,
    expected: None | tuple[int, int],
):
    assert _extent(data, column) == expected


@pytest.mark.parametrize(
    argnames="centers,width,expected",
    argvalues=[
        (np.array([0.0, 1.0, 2.0]), 0.5, np.array([-0.5, 0.5, 1.5, 2.5])),
        (np.array([3.0]), 1.0, np.array([2.5, 3.5])),
    ],
)
def test_vertices(
    centers: np.ndarray,
    width: float,
    expected: np.ndarray,
):
    assert np.allclose(_vertices(centers, width), expected)


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

    def test_show(self, path_test: pathlib.Path):
        result = soho.sumer.SpectrographObservation.from_level1(
            path=path_test,
            wavelength=_wavelength,
            order=2,
        )
        ax = result.show()
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

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
