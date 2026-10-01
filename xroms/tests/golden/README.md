# Golden outputs from xroms v0.6.2

What the released xroms v0.6.2 (git commit `be8ea34`) computes, recorded so that a
rewritten package can be regression-tested against it.

| file | contents |
| --- | --- |
| `make_golden.py` | generates the three files below |
| `golden_tiny.nc` | 69 outputs on the tiny test input: `xroms/tests/input/ocean_his_0001.nc` merged with `grid.nc`, exactly as the existing tests build it (Vtransform 2, 2 x 3 x 9 x 14) |
| `golden_window.nc` | the same 69 outputs on a 12 x 16 window of the Rutgers-style `xroms/data/ROMS_example_full_grid.nc` (Vtransform 1, 30 levels, sloping bathymetry, only 11-27 m deep) |
| `window_input.nc` | that window before any xroms processing: the input of `golden_window.nc`, for tests of new code |

Both inputs go through
`ds, xgrid = xroms.roms_dataset(ds, include_Z0=True, include_cell_volume=True, include_cell_area=True)`
before anything else. Keys and calls are listed in `build_steps` in `make_golden.py`.

## Regenerate

    PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/ocean-skill/bin/python xroms/tests/golden/make_golden.py

Needs an environment with **xgcm 0.8.1** (the script asserts it), plus `git` and `tar`.
Used here: Python 3.12, xarray 2026.4.0, numpy 2.4.3, cf_xarray 0.11.0, netCDF4 1.7.4.
The v0.6.2 sources are exported with `git archive` into a temporary directory and
imported from there only, never from the working tree or site-packages. The files come
out byte-identical on every run.

## Reading the files

- One data variable per output, named by its key (`z_rho`, `dx`, `to_rho_u`, `ddxi_temp`,
  `speed`, `ertel`, `east`, `rot_x`, `zslice_temp`, `gridmean_temp_YX`, ...). Only values and
  dims are stored: non-dimension coordinates are dropped, dimension coordinates are kept.
- Variable attrs: `xroms_call` is the exact statement that produced it, `v0.6.2_dims` the
  dims as v0.6.2 returned them. Other attrs are v0.6.2's own (units, long_name, ...),
  quirks included; treat them as a record, not a requirement.
- dtypes are whatever v0.6.2 returned. The window's `u, v, temp, salt, zeta` are float32,
  so `to_rho_u`, `to_rho_v`, `to_u_temp`, `to_v_temp`, `to_psi_temp`, `to_s_w_temp`,
  `speed`, `sig0` and `buoy` are float32 there. xroms 1.0 computes them the same way, so
  `test_golden.py` still holds them to rtol 1e-12. Only `KE` and `isoslice_temp_on_salt`,
  which v0.6.2 promoted to float64 and 1.0 keeps in float32, are compared to one float32
  ulp (rtol 1.2e-7).
- Global attrs: `source_commit`, library versions, `created_by`, `input`,
  `roms_dataset_call`, `mask_rho_land_cells`, and `failed_<key>` for any output that could
  not be computed (none at present).

## Things to know

- Neither input has land (`mask_rho` is 1 everywhere), so masking is not exercised.
- The tiny fields are so simple that 8 tiny outputs are identically zero: `ddeta_temp`,
  `ddeta_temp_rho`, `ddeta_zeta`, `ug`, `dudz`, `dvdz`, `vertical_shear`, `vort`. The window
  is the meaningful golden for those.
- Three outputs needed an adapted call. The variable holds the call that worked in
  `xroms_call`, and the plain call and its error in `primary_call` and `primary_call_error`.
  - `rot_x`, `rot_y`: `xroms.rotate_vectors` without `attrs` raises `KeyError: 'name'`,
    because the output of `xroms.to_rho` has no `name` attribute. The recorded call passes
    `attrs=`; values are unaffected, and equal the accessor's `east` and `north`.
  - `isoslice_temp_on_salt`: `xroms.isoslice(ds.temp, ..., iso_array=ds.salt, axis="Z")`
    raises a `ValueError` in `xroms.order`. Because `temp` carries a `z_rho` coordinate,
    `isoslice` re-attaches an interpolated one next to the new `salt` axis; cf_xarray then
    reports `z_rho` instead of `salt` as the Z axis and the transpose leaves `salt` out.
    The recorded call drops the `z_*` coordinates from `temp` first; values are unaffected.
- `zslice_temp` has dimension `z_rho_dim` here, where v0.6.2 names it `z_rho`, because that
  would clash with the stored `z_rho` variable. Its index holds the depths, and attr
  `dims_renamed_for_storage` marks the rename. `isoslice_temp_on_salt` keeps its dimension
  `salt`, indexed by the salinities.
- NaN is meaningful. `N2` and `M2` are NaN on the bottom and top w levels. Slices are NaN
  outside the range of the source levels: in the tiny set the -2 and -0.5 m slices lie
  above the top rho level (about -2.5 m) and are entirely NaN; in the window, 21% of the
  salinity-23 slice is NaN where the surface is saltier than 23.
- Slice values. Depths [m]: tiny -10, -2, -0.5; window -10, -5, -1 (inside the water column
  everywhere; -50, -20, -5 would mostly be below the window's seabed). Salinity: tiny 18, 22;
  window 23, 30. The window's salinity profiles are often not monotonic, so these were
  chosen so that no column crosses them more than once (for a value crossed several times
  xgcm returns the topmost crossing).
