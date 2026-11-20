"""
Download and analyze the spectra of the Solar Ultraviolet Measurements of
Emitted Radiation (SUMER) spectrograph.
"""

from ._data import (
    urls,
    download,
)
from ._spectrograph import SpectrographObservation
from ._sumer import open

__all__ = [
    "urls",
    "download",
    "SpectrographObservation",
    "open",
]
