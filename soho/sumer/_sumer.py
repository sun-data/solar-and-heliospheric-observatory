from typing import Literal
import pathlib
import astropy.units as u
import astropy.time
from ._spectrograph import SpectrographObservation

__all__ = [
    "open",
]


def open(
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
) -> SpectrographObservation:
    """
    Download the level-1 files of SUMER which observed a time range,
    and load the exposures of one spectral window which began in it
    as an instance of :class:`~soho.sumer.SpectrographObservation`.

    This is a shortcut for
    :meth:`soho.sumer.SpectrographObservation.from_time_range`.

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
        as in :meth:`~soho.sumer.SpectrographObservation.from_level1`.
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
        Boolean flag controlling whether to download files which are already
        in `directory`.
    num_retry
        The number of times to try to connect to the servers.
    """
    return SpectrographObservation.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        wavelength=wavelength,
        order=order,
        timedelta=timedelta,
        slit=slit,
        velocity_zero=velocity_zero,
        axis_wavelength=axis_wavelength,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        directory=directory,
        overwrite=overwrite,
        num_retry=num_retry,
    )
