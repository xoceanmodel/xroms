xroms
=====

xroms analyses output of the ROMS family of ocean models (Rutgers ROMS, UCLA ROMS, CROCO and REMORA) with xarray. It
knows the staggered C-grid and the terrain-following vertical coordinate. It computes depths, grid metrics, derivatives
at constant depth, and derived quantities such as speed, vorticity, buoyancy frequency and mixed layer depth.

There is no setup step and nothing is stored. Open the output with xarray and call xroms; each call computes what it
needs from the data in hand, lazily under dask, so results always match the data you pass, however you subset, select
or edit it.

.. code-block:: python

   import xarray as xr
   import xroms

   ds = xr.open_dataset("ocean_his.nc", chunks={})
   ds.xroms.speed                                  # on rho points, in the Dataset's own naming
   ds.xroms.ddz("salt")                            # vertical derivative, on w levels
   xroms.zslice(ds.temp, [-10, -50], ds)           # temperature at 10 m and 50 m below mean sea level

Coming from xroms 0.6? See :doc:`migration`.

Installation
------------

From conda-forge:

.. code-block:: bash

   conda install -c conda-forge xroms

From PyPI:

.. code-block:: bash

   pip install xroms

Optional features come with extras: ``teos10`` (gsw, for the TEOS-10 equation of state), ``geodesic`` (pyproj, for
the ellipsoidal nearest-point search) and ``examples`` (pooch and netCDF4, for the example data), or ``all`` for these
three. ``interpll`` needs xESMF, which is best installed from conda-forge (``conda install -c conda-forge xesmf``).

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Examples and demos

   io
   select_data
   calc
   interpolation
   plotting

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Reference

   api
   migration
   whats_new
   GitHub repository <https://github.com/xoceanmodel/xroms>
