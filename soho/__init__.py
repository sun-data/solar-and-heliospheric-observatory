"""
Download and analyze observations from the Solar and Heliospheric Observatory
(SOHO).
"""

import pathlib
import joblib

__all__ = [
    "directory_default",
    "memory",
    "sumer",
]

directory_default = pathlib.Path.home() / ".soho/cache"
"""The default directory for downloaded files."""

memory = joblib.Memory(location=directory_default / "joblib", verbose=0)
"""
A representation of the cache which stores intermediate results,
such as the answers of ESA's catalogue of SOHO.

The cache has a directory of its own, apart from the downloaded files,
so that :meth:`joblib.Memory.clear` does not delete them.
Assign another :class:`joblib.Memory` to this attribute to move the cache,
or one whose location is :obj:`None` to stop caching.
"""

# Imported after the attributes above, which the subpackages use
from . import sumer  # noqa: E402
