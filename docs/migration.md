# Migrating from xroms 0.6

xroms 1.0 is a clean break. There is no setup step and nothing is stored. Each call computes what it needs from the
data you pass it, lazily under dask, so subsetting, selecting or editing a Dataset can never leave stale depths, grid
metrics or xgcm grids behind. The removed entry points still exist, but they raise an error naming their replacement,
so old scripts fail loudly at the first call that needs changing.

## The short version

```python
# xroms 0.6
ds = xr.open_dataset("ocean_his.nc")
ds, xgrid = xroms.roms_dataset(ds, include_Z0=True, include_cell_volume=True)
ds.xroms.set_grid(xgrid)
dsdxi = xroms.ddxi(ds.salt, xgrid)
z0 = ds.z_rho0

# xroms 1.0
ds = xr.open_dataset("ocean_his.nc")
dsdxi = xroms.ddxi(ds.salt, ds)   # or ds.xroms.ddxi("salt")
z0 = ds.xroms.z(zeta=0)
```

- Delete `roms_dataset` and `set_grid`. If the grid is in a separate file, merge it with
  `ds = xroms.merge_grid(out, grid)`, or pass `grid=grid` to the calls that need it.
- Wherever you passed `xgrid`, pass the Dataset (`grid`) instead, or nothing where no grid is needed.
- Depths and grid metrics are no longer variables added to the Dataset: ask for them with `ds.xroms.z()`,
  `ds.xroms.dz()`, `ds.xroms.dA()` and so on, or attach depths with `ds = ds.xroms.assign_z()`.

## Call by call

| xroms 0.6 | xroms 1.0 |
|---|---|
| `ds, xgrid = xroms.roms_dataset(ds)` | delete it; a separate grid: `ds = xroms.merge_grid(out, grid)`, or `grid=grid` |
| `ds.xroms.set_grid(xgrid)` | delete it |
| `ds.xroms.xgrid` | `ds.xroms.xgcm_grid()`, a fresh xgcm Grid for your own xgcm work |
| `xroms.open_netcdf`, `xroms.open_mfnetcdf`, `xroms.open_zarr` | `xr.open_dataset`, `xr.open_mfdataset`, `xr.open_zarr` |
| `xroms.f(var, xgrid, ...)` | `xroms.f(var, ds, ...)` |
| `xroms.to_rho(var, xgrid)`, and `to_u`, `to_v`, `to_psi`, `to_s_rho`, `to_s_w`, `to_grid` | `xroms.to_rho(var)`: moving between positions needs no grid |
| `xroms.grid_interp(xgrid, da, dim)` | `xroms.to_grid(da, hcoord, scoord)` |
| `xroms.speed(u, v, xgrid)`, `xroms.EKE(ug, vg, xgrid)`, `xroms.vertical_shear(dudz, dvdz, xgrid)` | drop `xgrid`: `xroms.speed(u, v)` |
| `da.xroms.ddxi(xgrid)`, and `ddeta`, `ddz`, `gridmean`, `gridsum`, `zslice` on a DataArray | `ds.xroms.ddxi(da)`, or `xroms.ddxi(da, ds)` |
| `ds.temp.xroms.to_grid(xgrid, hcoord="psi")` | `ds.temp.xroms.to_grid(hcoord="psi")` |
| `ds.z_rho`, `ds.z_w`, `ds.z_rho_u`, `ds.z_w_v`, ... | `ds.xroms.z_rho`, `ds.xroms.z_w`, `ds.xroms.z("u")`, `ds.xroms.z("v", "s_w")`, or `ds = ds.xroms.assign_z()` |
| `ds.z_rho0` (from `include_Z0=True`) | `ds.xroms.z(zeta=0)` |
| `ds.dz`, `ds.dz_w`, `ds.dz_u`, ... | `ds.xroms.dz()`, `ds.xroms.dz(scoord="s_w")`, `ds.xroms.dz("u")` |
| `ds.dx`, `ds.dy`, `ds.dA`, `ds.dV`, `ds.dx_u`, `ds.dA_psi`, ... | `ds.xroms.dx()`, `.dy()`, `.dA()`, `.dV()`, `ds.xroms.dx("u")`, `ds.xroms.dA("psi")` |
| `include_*` flags of `roms_dataset` | gone: everything is computed on demand |
| `xroms.isoslice(var, depths, xgrid)` | `xroms.zslice(var, depths, ds)`, or `ds.xroms.zslice("salt", depths)` |
| `xroms.isoslice(var, values, xgrid, iso_array=arr, axis="Y")` | `xroms.isoslice(var, values, arr, dim="eta_rho")` |
| `xroms.mld(sig0, xgrid, h, mask, thresh=0.03)` | `xroms.mld(sig0, ds, threshold=0.03)`, or `ds.xroms.mld(threshold=0.03)` |
| `xroms.argsel2d(lons, lats, lon0, lat0)` (WGS84 geodesic, cartopy) | the same call, with the haversine distance; `method="geodesic"` uses pyproj |
| `roms_dataset(..., add_verts=True, proj=proj)`, `ds.lon_vert`/`ds.lat_vert` | gone: `pcolormesh(..., shading="auto")` plots from the cell centres |
| `ds.xroms.w`, `ds.xroms.omega` | gone (they were unfinished placeholders); read `ds["w"]` or `ds["omega"]` |

Calls that keep their 0.6 form include the accessor's properties and methods (`ds.xroms.speed`, `ds.xroms.KE`,
`ds.xroms.vort`, `ds.xroms.ertel`, `ds.xroms.N2`, `ds.xroms.ddz("salt")`, `ds.xroms.zslice("temp", depths)`,
`ds.xroms.subset(X=..., Y=...)`, ...), `ds.temp.xroms.order()`, `interpll`, `sel2d`, and `subset`.

## Results that change

- **`convergence` changes sign.** It is now `-(du/dx + dv/dy)`, positive where the flow converges. 0.5.1 renamed
  `divergence` to `convergence` without changing the sign, so 0.5.1 to 0.6.2 returned `du/dx + dv/dy`, the divergence.
  The new `divergence` (and `ds.xroms.divergence`, `ds.xroms.divergence_norm`) gives those values; `convergence_norm`
  changes sign with `convergence`.
- **Land is NaN** in `speed`, `KE` and the earth components (`grid_to_earth`, `ds.xroms.east`/`north` and the rotated
  ones). Masked u and v still count as 0 when averaged to rho points, so the water next to land keeps its values, but
  land points, with no velocity around them, are NaN instead of 0. `gridsum` is NaN where every point summed over is
  missing (land) instead of 0.
- **Horizontal derivatives** (`ddxi`, `ddeta`, `hgrad`, and the calculations built on them) stay on the input's own
  vertical levels; 0.6 moved them to w levels. Pass `scoord="s_w"` for the old placement.
- **No zeros at the top and bottom.** 0.6 padded the vertical edges so that derivatives there came out exactly 0,
  which halved the top and bottom layers of `ddz` and of the slope term of horizontal derivatives. 1.0 takes the edge
  value from the data (a one-sided difference, `sboundary="extend"`). `sboundary="fill"` gives NaN there instead, and
  `sfill_value=0` imposes a zero gradient when you want that physics.
- **`ddz` inside the slope term** uses a second-order three-point stencil on stretched levels, so interior values of
  horizontal derivatives differ slightly from 0.6.
- **Layer thicknesses on w levels** are half-cells at the top and bottom. 0.6's bottom value was wrong (for example
  −97.5 m where +2.5 m is right), and that error reached depth sums and means of w-level variables.
- **A single selected s-level** (for example `ds.salt.isel(s_rho=-1)`) raises in horizontal derivatives unless you
  pass `along_s=True`, which differentiates along that s-surface rather than at constant depth. Differentiate the
  3-D field and then select the level for the constant-depth derivative.
- **A field without time** (a time mean, a groupby result) has no free surface to match. Depth-aware calls then raise
  and ask you to choose: `zeta="mean"`, `zeta=0`, `zeta=<DataArray>` or `z=`.
- **The nearest point** (`argsel2d`, `sel2d`) uses the haversine distance instead of cartopy's WGS84 geodesic. It
  almost always finds the same cell; `method="geodesic"` (with pyproj) is the ellipsoidal distance.
- **Mixed layer depth.** `threshold=` replaces `thresh=` (which still works, with a FutureWarning). The reference value
  is taken at `reference_depth` (default 0: the shallowest level, as before), density uses its increase with depth,
  and a column with no crossing gets the bottom depth (`fill="bottom"`) or NaN (`fill="nan"`).
- **Density** takes `eos="roms"` (the default, ROMS' own equation of state, as before) or `eos="teos10"` (gsw).
- **Interpolation to lon/lat points** (`interpll`) gives NaN at points outside the model domain, where 0.6 gave 0
  (`unmapped_to_nan=False` brings the 0 back).
- **Grid sums** (`gridsum`) multiply the variable's `units` by a metre for each dimension summed over; 0.6 kept the
  units of the variable.

## Naming: canonical dims and the accessor

Rutgers ROMS (and REMORA) files name their u, v and psi dims with aliases (`eta_u`, `xi_v`, `eta_psi`, `xi_psi`).
`roms_dataset` used to rename these to xgcm's names. In 1.0:

- **Pure functions** (`xroms.ddxi(var, ds)`, `xroms.to_u(var)`, ...) accept either naming and always return the
  canonical dims: rho `(eta_rho, xi_rho)`, u `(eta_rho, xi_u)`, v `(eta_v, xi_rho)`, psi `(eta_v, xi_u)`.
- **The accessor** (`ds.xroms.<...>`) returns results in the Dataset's own naming, so they combine with its variables.
- To mix pure-function results with the variables of a Rutgers file, rename the file once with
  `ds = xroms.canonicalize(ds)`, the renaming `roms_dataset` used to do (metadata only), or bring a single result back
  with `xroms.rename_like(da, ds)`. Otherwise `xroms.to_u(ds.temp) * ds.u` would broadcast `eta_rho` against `eta_u`.

## Other changes

- **No global options.** 0.6 set xarray's `keep_attrs=True` for the whole session on `import xroms`. 1.0 leaves xarray's
  options alone, so attributes in your own arithmetic follow your xarray's default: recent versions keep them (2026.4
  does), older ones drop them (2025.7 does). Call `xr.set_options(keep_attrs=True)` yourself if you rely on keeping
  them. xroms results still carry their own attributes.
- **Quiet imports.** `import xroms` no longer filters warnings or imports cartopy, xesmf or cf-xarray.
- **Dependencies.** xgcm is no longer pinned to 0.8.1: 1.0 needs `xgcm>=0.10`, and lists numba, which xgcm's
  `transform` needs. cf-xarray is no longer a dependency, cartopy and pygridgen are no longer used, and pooch and
  netCDF4 moved to an extra. Optional extras: `interp` (xesmf), `geodesic` (pyproj), `teos10` (gsw), `examples`
  (pooch, netCDF4). Python 3.11 or newer.
- **Example data** is downloaded on first use by `xroms.datasets.fetch_ROMS_example_full_grid()` (needs pooch) instead
  of shipping in the package.
- **More of the ROMS family.** UCLA ROMS, CROCO and REMORA output work as well as Rutgers ROMS, detected per call from
  the file's own names and attributes. For UCLA output, `xroms.decode_time(ds)` gives a decoded time coordinate.

{doc}`whats_new` lists everything that is new, and {doc}`api` documents each function.
