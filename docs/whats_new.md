# What's New

## v0.4.3 (July 27, 2023)
* zkey is checked for but not required in interpll now

## v0.4.2 (July 27, 2023)
* changes to roms_dataset
    * If "coordinates" are found in attrs for a variable, they are moved to "encoding" now because everything works better then.
    * Recreated the zslice function in the accessor for both Dataset and DataArray (instead of just using the default isoslice).
    * updated docs and tests accordingly.

## v0.4.1 (July 27, 2023)
* can now pass kwargs to xe.Regridder in interpll

## v0.4.0 (July 25, 2023)

* hopefully fixing issue reordering dimensions when extra coords present
* divergence calculation was added to derived.py
* accessor changes:
  * xgrid is run automatically when accessor is used, which could be too slow for some uses
  * div and div_norm properties added to accessor
  * div_norm is the surface divergence normalized by f
* added tests for new functions
* hopefully fixed build issue on several OSes by pinning `h5py < 3.2`, see for reference https://github.com/h5py/h5py/issues/1880, https://github.com/conda-forge/h5py-feedstock/issues/92
* updated docs


## v0.3.3 (July 11, 2023)

* do not use Z coords if 2d


## v0.3.2 (June 23, 2023)

* More fixes to the rotation accessor options


## v0.3.1 (June 22, 2023)

* made east/north variable names have two options


## v0.3.0 (June 12, 2023)

* can rotate along-grid velocities to be eastward and northward
* can also rotate to be along an arbitrary angle (to be along-channel for example)

## v0.2.3 (May 24, 2023)

* updating versioning approach
* the xgcm grid is no longer attached to every variable in a Dataset. Because of this:
  * Several `xroms` accessor functions now require the grid to be input
  * Additionally because of the grid not being available, the Dataset is no longer available within the DataArray accessor, making it so that functions that change the grid size in any dimension no longer know about other coordinates to use. Therefore, these `xroms` accessor functions for DataArrays no longer work (e.g. ddeta, ddxi, etc). All `xroms` Dataset accessor functions still work, and the grid object is still saved to the Dataset `xroms` accessor.
* You can set up the grid object directly in your `xroms` Dataset accessor with `ds.xroms.set_grid(grid)`, otherwise it will be calculated internally.
* `xroms` functions for opening model output files will be deprecated in the future; use `xarray` functions directly instead of opening model output through `xroms`, and then run `xroms.roms_dataset()` to add functionality to your Dataset and to calculate your `xgcm` grid object.
* tests have been updated
* `xroms` works with newest version of `xgcm`
* changed all references to the `xgcm` grid to `xgrid` since there is now a "grid" attribute in some Datasets.
* updated example notebooks to be formal docs
* added a ROMS example dataset, available with `xroms.datasets.fetch_ROMS_example_full_grid()`.
