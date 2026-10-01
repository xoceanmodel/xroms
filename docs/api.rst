API
===

.. currentmodule:: xroms

Every function is available as ``xroms.<name>``. Functions take DataArrays, plus the Dataset as ``grid`` where grid
variables are needed, and return canonical dims. The ``ds.xroms`` accessor offers the same calculations by variable
name and returns the Dataset's own naming; ``da.xroms`` has the operations that need no grid
(see :ref:`accessors`).

Opening and conventions
-----------------------

.. autosummary::
   :toctree: generated/

   merge_grid
   decode_time
   add_cf_attrs
   canonicalize
   rename_like
   vertical_params
   VerticalParams
   stretching
   sigma_levels
   rho0
   datasets.fetch_ROMS_example_full_grid

Vertical coordinate
-------------------

.. autosummary::
   :toctree: generated/

   z
   compute_depth
   dz
   surface
   bottom
   depth_band_weights
   depth_average

Grid metrics and masks
----------------------

.. autosummary::
   :toctree: generated/

   dx
   dy
   dA
   dV
   nominal_resolution
   mask_at

Moving between grid positions
-----------------------------

.. autosummary::
   :toctree: generated/

   to_grid
   to_rho
   to_u
   to_v
   to_psi
   to_s_rho
   to_s_w
   order

Derivatives and grid-weighted sums
----------------------------------

.. autosummary::
   :toctree: generated/

   ddxi
   ddeta
   ddz
   hgrad
   gridsum
   gridmean

Physical quantities
-------------------

.. autosummary::
   :toctree: generated/

   speed
   KE
   EKE
   uv_geostrophic
   dudz
   dvdz
   vertical_shear
   relative_vorticity
   convergence
   ertel
   density
   potential_density
   buoyancy
   N2
   M2
   mld

Vectors
-------

.. autosummary::
   :toctree: generated/

   rotate_vectors
   grid_to_earth
   earth_to_grid

Interpolation and slices
------------------------

.. autosummary::
   :toctree: generated/

   zslice
   isoslice
   xisoslice
   interpll
   make_regridder

Selection and subsetting
------------------------

.. autosummary::
   :toctree: generated/

   subset
   trim
   argsel2d
   sel2d

Longitudes
----------

.. autosummary::
   :toctree: generated/

   wrap_longitude
   straddles
   lonlat_at

.. _accessors:

Accessors
---------

``ds.xroms`` and ``da.xroms``.

.. autosummary::
   :toctree: generated/

   accessor.xromsDatasetAccessor
   accessor.xromsDataArrayAccessor
