# solar-and-heliospheric-observatory

[![tests](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/solar-and-heliospheric-observatory/graph/badge.svg)](https://codecov.io/gh/sun-data/solar-and-heliospheric-observatory)
[![Black](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/ruff.yml)
[![typing](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/typing.yml/badge.svg)](https://github.com/sun-data/solar-and-heliospheric-observatory/actions/workflows/typing.yml)
[![Documentation Status](https://readthedocs.org/projects/solar-and-heliospheric-observatory/badge/?version=latest)](https://solar-and-heliospheric-observatory.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/solar-and-heliospheric-observatory.svg)](https://badge.fury.io/py/solar-and-heliospheric-observatory)

A Python library to download and analyze observations from the
ESA/NASA [Solar and Heliospheric Observatory (SOHO)](https://soho.nascom.nasa.gov/).

So far it supports the level-1 spectra of the
Solar Ultraviolet Measurements of Emitted Radiation (SUMER) spectrograph,
which it finds in ESA's catalogue of SOHO and downloads from the SOHO archive.
The spectra are not represented as instances of
[`ndcube.NDCube`](https://docs.sunpy.org/projects/ndcube/en/stable/),
but as instances of [`named_arrays.FunctionArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.FunctionArray.html),
which carry the time, sky position, and wavelength of every pixel.

## Installation

This package is published on PyPI and can be installed using pip
```bash
pip install solar-and-heliospheric-observatory
```
It is imported as `soho`.

## Example

Load the window of O V 629.7 Å, observed in the second order,
of the quiet-Sun raster which SUMER observed on 1996 September 16
```python
import astropy.units as u
import soho

obs = soho.sumer.open(
    time_start="1996-09-16T16:07",
    time_stop="1996-09-16T22:49",
    wavelength=629.73 * u.AA,
    order=2,
)
```

## Citation

If you use solar-and-heliospheric-observatory in your research, please cite it.
The citation metadata is kept in [`CITATION.cff`](https://github.com/sun-data/solar-and-heliospheric-observatory/blob/main/CITATION.cff),
which the "Cite this repository" button on GitHub can export as BibTeX or APA.
Please also acknowledge SUMER as the source of its data,
and cite [Wilhelm et al. (1995)](https://doi.org/10.1007/BF00733430).
