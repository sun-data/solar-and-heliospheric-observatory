Introduction
============

The `Solar and Heliospheric Observatory (SOHO) <https://soho.nascom.nasa.gov/>`_
is a project of international cooperation between ESA and NASA,
which has been observing the Sun since 1996 :cite:p:`Domingo1995`.

This Python package downloads SOHO observations and represents them
using :mod:`named_arrays`,
a named tensor implementation with :class:`astropy.units.Quantity` support.
So far it supports the level-1 spectra of the
Solar Ultraviolet Measurements of Emitted Radiation (SUMER) spectrograph
:cite:p:`Wilhelm1995`, in :mod:`soho.sumer`,
which it finds in ESA's catalogue of SOHO and downloads from the SOHO archive
:cite:p:`Curdt2014`.

Installation
============

This package is published to PyPI and can be installed using pip.

.. code-block::

    pip install solar-and-heliospheric-observatory

It is imported as ``soho``.

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    soho


Examples
========

Load the raster of a prominence above the northwest limb which SUMER observed
on 1996 April 10 in N V 1242.8 angstroms, in the first order of its grating,
and display it as a false-color image,
whose hue is the Doppler velocity and whose brightness is the radiance.

.. jupyter-execute::

    import astropy.units as u
    import soho

    # Download the raster and load the window of N V into memory
    obs = soho.sumer.open(
        time_start="1996-04-10T08:04",
        time_stop="1996-04-10T12:11",
        wavelength=1242.80 * u.AA,
        order=1,
    )

    # Display the prominence
    ax = obs.show(
        velocity_min=-60 * u.km / u.s,
        velocity_max=+60 * u.km / u.s,
    )
    ax.set_ylim(570, 790);

References
==========

.. bibliography::
