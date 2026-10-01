# What's New

## v1.0.0 (unreleased)

A rewrite. xroms is now stateless, has no setup step, and works with current xgcm. {doc}`migration` shows how to
update code written for 0.6.

### Design
* No setup step. `roms_dataset` and `set_grid` are gone: open the output with xarray and call xroms. Each call
  computes what it needs from the data in hand, lazily under dask, and nothing is cached. Subsets, time selections and
  edits can therefore no longer leave stale depths, grid metrics or xgcm grids behind. Depths and metrics are computed
  only for the data asked about ({issue}`74`).
* Two layers, one implementation. Functions take DataArrays plus the Dataset as `grid` and return canonical dims. The
  `ds.xroms` accessor offers the same calculations by variable name, and returns results in the Dataset's own naming.
* The ROMS family: Rutgers ROMS, UCLA ROMS, CROCO ({issue}`24`) and REMORA output, detected per call from the file's names and
  attributes. This covers s-coordinate parameters stored as variables or as attributes, CROCO's `VertCoordType`, time
  without a coordinate (UCLA), Cartesian x/y grids and time-varying masks.
* Results keep the right dims. A time dim is never re-broadcast or dropped. A variable is matched to the free surface
  at its own times, with an error naming the options when it can't be (after a time mean, for example).
* Chunked input works, and only the dim being operated on is rechunked. The other dims keep their chunks, and the
  operated dim gets its chunk structure back afterwards.

### New
* Vertical coordinate: `z` at any position, with `reference="mean_sea_level"`, `"surface"` or `"bottom"`, `positive=`
  and `zeta=0`, `"mean"` or a DataArray. Also `compute_depth`, `dz`, `vertical_params`, `stretching` and
  `sigma_levels`. Vertical outputs carry CF `standard_name`, `positive` and `units`.
* Vertical selection and interpolation: `surface`, `bottom`, `depth_band_weights` and `depth_average`; `zslice` onto
  fixed heights or depths, including `method="nearest"`; `isoslice` onto any monotonic field. Positions xroms did not
  label (`z=` in `zslice`, `z_w` in `depth_band_weights`) are described with `positive=`/`reference=`, which win over
  their attrs; reading the sign from attrs xroms did not write warns.
* Grid metrics and masks: `dx`, `dy`, `dA`, `dV`, `nominal_resolution` and `mask_at`, at any position. On the accessor:
  `z_rho`, `z_w`, `z()`, `dz()`, `dx()`, `dy()`, `dA()`, `dV()`, `vertical_params`, `assign_z()`, and `xgcm_grid()`
  for your own xgcm work.
* Opening helpers:
  * `merge_grid` for output with a separate grid file;
  * `decode_time` for UCLA output;
  * `add_cf_attrs`, which adds the attributes cf-xarray and SGRID readers need;
  * `canonicalize` and `rename_like`.
* Longitudes: `wrap_longitude`, `straddles` and `lonlat_at`.
* Vectors: `grid_to_earth` and `earth_to_grid`.
* Selection: `subset(..., halo=)` and `trim`; `argsel2d`/`sel2d` with `method="geodesic"` (pyproj); `make_regridder`, to
  reuse xESMF weights in `interpll` (`interpll(var, regridder=r)`).
* Density and mixed layer: `eos="teos10"` in `density` and `potential_density` (gsw). `mld` takes `threshold`,
  `reference_depth`, `variable="temperature"`, `fill` and `method`.

### Changed results
{doc}`migration` has the details.
* `convergence` is `-(du/dx + dv/dy)`, positive where the flow converges: since 0.5.1 it had returned the divergence.
  `divergence` (and the accessor's `divergence`, `divergence_norm`) is new.
* Land is NaN in `speed`, `KE` and the east/north components (they were 0 there), and in a `gridsum` with nothing to
  sum.
* Horizontal derivatives at constant depth stay on the input's vertical levels, and their slope term uses a
  second-order `ddz`.
* There are no more artificial zeros at the top and bottom of vertical derivatives.
* Layer thicknesses on w levels are correct at the top and bottom.
* `argsel2d`/`sel2d` use the haversine distance.
* Horizontal derivatives of a single selected s-level raise unless `along_s=True`.
* `depth_average` and `gridmean` give NaN where there is no water, and weight only the points with data.
* `interpll` gives NaN outside the model domain (0.6 gave 0), and the `units` of a `gridsum` include the metres summed
  over.

### Removed
Each of these raises an error naming its replacement:
* `roms_dataset`, `open_netcdf`, `open_mfnetcdf`, `open_zarr` and `grid_interp`;
* `ds.xroms.set_grid` and `ds.xroms.xgrid`, and the `include_*` flags;
* the DataArray accessor's `ddxi`, `ddeta`, `ddz`, `gridmean`, `gridsum` and `zslice`;
* the `ds.xroms.w`/`omega` placeholders;
* the `add_verts` and `proj` options, with `lon_vert`/`lat_vert` and the pygridgen dependency.

### Packaging
* `xgcm>=0.10`, no longer pinned to 0.8.1 ({issue}`77`), so xroms installs alongside roms-tools.
* numba is listed explicitly, and Python 3.11 or newer is required.
* No import side effects: no global `keep_attrs`, no warning filters, and no eager imports of cartopy, xesmf or
  cf-xarray. cf-xarray and pygridgen ({issue}`12`) are no longer needed.
* `pyproject.toml`, with optional extras `interp`, `geodesic`, `teos10`, `examples` and `all`. The wheel no longer
  contains the 62 MB example file, which pooch downloads on first use, or the tests.

### Fixes
* Opening output no longer depends on cf-xarray's reading of the dims ({issue}`69`).
* `isoslice` takes a scalar value and an explicit `iso_array`, and depths have their own `zslice` ({issue}`75`).
* `xisoslice` returns the value when the iso value is one of the levels ({issue}`7`).

## v0.6.2 (August 27, 2025)
* catching and ignoring a bunch of warnings from `xgcm`, but still staying with `xgcm` `v0.8.1` until I can update this code to match.
* updating CI test versions to 3.11, 3.12, 3.13


## v0.6.1 (October 28, 2024)
* Correction in a few built-in calculations of u/v grid to rho-grid interpolations of u and v velocities (currently `speed` and `_uv2eastnorth`). In these cases, we need to fill nans with zeros so that the masked locations in the velocity fields are not fully brought forward into the rho mask but are instead interpolated over. By making them 0, they are calculated into the mask\_rho positions by combining them with neighboring cells. If this wasn't done, the fact that they are masked would supersede the neighboring cells and they would be masked in mask\_rho. This needs to be done anytime the velocities are moved from their native grids to the rho or other grids to preserve their locations around masked cells.

## v0.6.0 (February 9, 2024)
* fixed error in `derived.py`'s `uv_geostrophic` function after being pointed out by @ak11283
* updated docs so mostly well-formatted and working

## v0.5.3 (October 11, 2023)
* change to `roms_dataset()` so that input flag `include_3D_metrics` also controls if `ds["3d"] = True`.

## v0.5.2 (October 4, 2023)
* small fix to `roms_dataset()` processing to enable running it twice

## v0.5.1 (September 14, 2023)
* renamed all references to "divergence" to "convergence" instead

## v0.5.0 (September 12, 2023)
* the mixed layer depth function now returns positive values

## v0.4.7 (September 8, 2023)
* Fixed attributes for accessor method `div_norm`

## v0.4.6 (July 31, 2023)
* fixed `ds.xroms.div` and `ds.xroms.div_norm` in the case that `u` and `v` need to be calculated from other velocities like `east` and `north`.

## v0.4.5 (July 27, 2023)
* typo fix

## v0.4.4 (July 27, 2023)
* added accessor function `find_horizontal_velocities()` which returns the names of the horizontal velocities since they sometimes have different names, but still there are only a few options.

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
