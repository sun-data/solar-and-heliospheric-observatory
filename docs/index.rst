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

Load the quiet-Sun raster of O V 629.7 angstroms which SUMER observed on
1996 September 16, in the second order of its grating,
and display its radiance.
The dark bands above and below are the rows of the detector beyond the ends
of the slit.

.. jupyter-execute::

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import named_arrays as na
    import soho

    # Download the raster and load the window of O V into memory
    obs = soho.sumer.open(
        time_start="1996-09-16T16:07",
        time_stop="1996-09-16T22:49",
        wavelength=629.73 * u.AA,
        order=2,
    )

    # Integrate over the line, leaving out the columns which an exposure
    # did not read out
    dw = obs.inputs.wavelength.volume_cell(obs.axis_wavelength)
    radiance = np.nansum(obs.outputs * dw, axis=obs.axis_wavelength)

    # Display the radiance
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        img = na.plt.pcolormesh(
            obs.inputs.position.x,
            obs.inputs.position.y,
            C=radiance,
            ax=ax,
            cmap="gray",
            norm=matplotlib.colors.PowerNorm(0.5),
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"helioprojective $x$ ({ax.get_xlabel()})")
        ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")

References
==========

.. bibliography::
