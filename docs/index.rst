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

Load the quiet-Sun raster which SUMER observed near disk center on
1996 September 16 in O V 629.7 angstroms, in the second order of its grating,
and display it as a false-color image,
whose hue is the Doppler velocity and whose brightness is the radiance.
The velocity is zero at the mean position of the line in the raster,
and the pairs of blue and red spots are explosive events.

.. jupyter-execute::

    import astropy.units as u
    import soho

    # Download the raster and load the window of O V into memory
    obs = soho.sumer.open(
        time_start="1996-09-16T16:07",
        time_stop="1996-09-16T22:49",
        wavelength=629.73 * u.AA,
        order=2,
    )

    # Display the raster, without the rows of the detector beyond the ends of
    # the slit
    ax = obs.show(
        velocity_min=-100 * u.km / u.s,
        velocity_max=+100 * u.km / u.s,
    )
    ax.set_ylim(-160, 170);

References
==========

.. bibliography::
