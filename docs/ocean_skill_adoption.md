# Adopting xroms 1.0 in ocean-skill

A working note for the ocean-skill proof of concept; the 1.0 migration guide will absorb it. Every row below is pinned
by `xroms/tests/test_parity_oceanskill.py`, which runs wherever ocean-skill is installed (134 pass, plus one strict
xfail for an ocean-skill bug; checked against ocean-skill `c554103`).

## Installing the branch

xroms 1.0 is not released yet. In ocean-skill's `environment.yml`, replace the pip entry `xroms` with:

```yaml
      - "xroms @ git+https://github.com/xoceanmodel/xroms.git@claude/xroms-package-overhaul-0dc46e"
```

xroms 1.0 needs `xgcm>=0.10` (0.6 pinned 0.8.1) and `numba`. `eos="teos10"` uses `gsw`, which ocean-skill already
has, and ocean-skill's own `_z_grid` works with either xgcm. With the branch in a clone of the ocean-skill env,
`pip check` reports only the existing arcosparse/pandas conflict. ocean-skill's suite gives the same results as in its
own env, except for one new failure: a test that calls the removed `xroms.roms_dataset` (see
[First change](#first-change)).
- The 9 failures in `test_cache` and `test_regrid_target` fail in both envs.
- Three kerchunk build tests fail intermittently in both envs; that code imports neither xgcm nor xroms.

## How xroms 1.0 is called

- **No setup step and no state.** Open with xarray, then either merge the grid
  (`xr.merge([out, grid], compat="override")`) or pass `grid=`. Nothing is cached.
- **Canonical dims.** Pure functions take DataArrays plus the Dataset as `grid`, and return canonical dims:
  - rho `(eta_rho, xi_rho)`
  - u `(eta_rho, xi_u)`
  - v `(eta_v, xi_rho)`
  - psi `(eta_v, xi_u)`

  These are the UCLA names ocean-skill already uses.
- **Heights.** `xroms.z` is height above mean sea level, negative down, like ocean-skill's `z_rho`/`z_w`.
  `positive="down"` gives depths, and `reference="surface"` gives depths below the moving free surface.
- **Free surface.** By default xroms uses the Dataset's `zeta`, or the variable `standardize` renames it to
  (`sea_surface_height_above_geoid`).
  - `zeta=0` gives the resting depths.
  - `zeta=<DataArray>` passes it explicitly.
  - A variable that varies in time, with no free surface to match it, raises an error rather than assuming a flat
    surface.
- **Vertical parameters** come from the Dataset (variables or attributes), not from a catalog. Put a catalog's
  `Vtransform`/`hc` on the Dataset, or pass `Vtransform=` to `z`/`vertical_params`.
- **Lazy.** Everything stays lazy. Only the dim being operated on is rechunked, and its chunks are restored afterwards.
- **Coordinates.** Results carry the Dataset's coordinates (lon/lat at the result's position), never its data
  variables.

## Wave 1: the proof of concept

| ocean-skill | xroms | Agreement |
|---|---|---|
| `xroms.roms_dataset(ds)`, then `ds.z_rho`/`ds.z_w` | `xroms.z(ds)`, `xroms.z(ds, scoord="s_w")` | identical |
| `roms.add_depth_coord(std, meta).z_rho` | `xroms.z(ds)` | bit for bit |
| `roms.add_interface_coord(std, meta).z_w` | `xroms.z(ds, scoord="s_w")` | bit for bit |
| the same with `zero_zeta=True` | `xroms.z(ds, zeta=0)` | bit for bit |
| `roms._s_to_z(sigma, Cs, h, zeta, hc, Vtransform)` | `xroms.compute_depth(h, zeta, hc=, Cs=, sigma=, Vtransform=)` | bit for bit |
| `roms._vertical_params(ds, meta)` | `xroms.vertical_params(ds)` (`.hc`, `.Vtransform`) | same values |
| `roms.surface(std, meta)` | `xroms.surface(var)` | bit for bit |
| `roms.depth_band(std, meta, low, high)` | `xroms.depth_band_weights(xroms.z(ds, scoord="s_w", zeta=0), low, high)` | bit for bit (xroms keeps every layer, weighted 0 outside the band) |
| `roms.depth_average(std, meta, low, high)` | `xroms.depth_average(var, ds, shallow=low, deep=high)` | bit for bit over water |
| `roms.to_depth(std, meta, depths)` | `xroms.zslice(var, [-d for d in depths], ds)` | rtol 1e-12 |
| `roms.nearest_depth_levels(std, meta, depths, ref_time=t)` | `xroms.zslice(var, [-d for d in depths], ds, zeta=ds.zeta.sel(time=t), method="nearest")` | bit for bit |
| `roms._decode_time(ds, meta)`, `build._decode_times` (ROMS) | `xroms.decode_time(ds)` | same instants |

## Wave 2

| ocean-skill | xroms | Agreement |
|---|---|---|
| `mld.potential_density(temp, salt, z, lon, lat)` | `xroms.potential_density(temp, salt, eos="teos10", grid=ds) - 1000` | rtol 1e-12 |
| `roms.to_sigma0(std, meta, targets)` | `xroms.isoslice(var, targets, sigma0, dim="s_rho", new_dim="sigma0")` | bit for bit |
| `mld.mld_density_threshold(std, threshold=t, ref_depth=r)` | `xroms.mld(rho, ds, threshold=t, reference_depth=r, fill="nan")` | rtol 1e-9 where density increases with depth |
| `mld.mld_temperature_threshold(std, threshold=t, ref_depth=r)` | `xroms.mld(temp, ds, variable="temperature", threshold=t, reference_depth=r, fill="nan")` | rtol 1e-9 |
| `align.harmonize_longitude(obj, "0-360")` | `xroms.wrap_longitude(obj, "0-360")` | identical |
| `align.harmonize_longitude(obj, align.natural_convention(obj))` | `xroms.wrap_longitude(obj)` | identical except on ties and on the seam |
| `align.natural_convention(obj)` | `xroms.straddles(obj)`, True when the domain crosses the prime meridian | differs on ties |
| `roms._average_to_rho(field, stagger, rho)` | `xroms.to_rho(field)` | bit for bit |
| `roms._add_geographic_velocity(std)` | `xroms.grid_to_earth(u, v, angle)` | bit for bit away from land |
| `roms.add_geographic_velocity_windowed(window, meta)` | `xroms.subset(ds, X=, Y=, halo=1)`, then `grid_to_earth`, then `xroms.trim` | bit for bit |
| `roms._rotate_and_assign` | `xroms.rotate_vectors(x, y, angle)` | bit for bit |
| `align._nearest_indices(lon, lat, lon0, lat0)` | `xroms.argsel2d(lon_rho, lat_rho, lon0, lat0)` | same cell, in either longitude convention |
| `align._point_window(std, "lon", "lat", lon0, lat0, n)` | `xroms.subset(ds, X=, Y=, halo=1)` around `argsel2d`'s cell (`xroms.trim` drops the halo) | bit for bit |
| `standardize`'s `cell_area` coordinate | `xroms.dA(ds)` on the raw grid | within 2.6e-16 |
| `align._cell_km(ds, lon, lat)` | `np.hypot(xroms.dx(ds).median(), xroms.dy(ds).median()) / 1000` | 1% on grids aligned east-north |

`zslice` takes heights, like ocean-skill's `z` coordinate. `zslice(var, depths, ds, positive="down")` gives the same
values, with a positive coordinate.

In the wave 2 rows:
- `sigma0` is `xroms.potential_density(temp, salt, eos="teos10", grid=ds) - 1000`. Instead of `grid=`, you can pass
  `z_points=`, `lon=` and `lat=`.
- `rho` is that same potential density, without subtracting 1000.

## Intended differences

- **Depth average over land.** xroms gives NaN where a column has no water, and averages over the valid levels of a
  column with a missing one. ocean-skill gives 0.0 over masked land.
- **Mixed layer depth.**
  - Defaults: to match ocean-skill, pass `reference_depth=10, fill="nan"`. xroms defaults to 0 and to the bottom
    depth.
  - A reference depth above the shallowest level gives NaN in ocean-skill; xroms uses the shallowest level's value.
  - For density, xroms uses the signed increase, which is what ocean-skill's docstring says. ocean-skill's
    `mld_threshold` uses |Δσ0|, so it also finds a crossing where density decreases with depth.
- **Longitudes.**
  - The seam: ocean-skill's -180..180 frame is [-180, 180), and xroms' is (-180, 180].
  - Which longitudes: `harmonize_longitude` wraps the one longitude it finds. `wrap_longitude` wraps every longitude
    variable (`lon_rho`, `lon_u`, `lon_v`, ...).
  - Ties: on a domain that is contiguous in both frames, `wrap_longitude(obj)` leaves the values as stored, while
    ocean-skill moves them to -180..180.
- **Velocities next to land.** `grid_to_earth` counts masked u/v as 0 when averaging to rho points, as xroms always
  has. Rho points beside land therefore keep a value, and land itself is 0. ocean-skill spreads NaN there instead.
  Wherever ocean-skill has a value, the two agree.
- **Nearest levels over time.** With `ref_time`, ocean-skill keeps the levels it picked at that step for every step.
  `zslice(method="nearest")` follows zeta at each step, unless it is given that step's `zeta=`.
- **Time decoding.** xroms:
  - keeps fractional seconds;
  - decodes day, hour and minute units;
  - raises for units without a fixed length (months), where ocean-skill returns None;
  - refuses a `reference_date` that disagrees with the file;
  - keeps the time dim's own name (`ocean_time` in classic Rutgers output, where ocean-skill moves the data onto
    `time`).
- **Vertical interpolation.** `zslice`/`isoslice` put the new dim where the vertical one was, giving
  `(time, z, eta_rho, xi_rho)`. ocean-skill puts it last.
- **Density.** xroms returns full density; subtract 1000 for sigma0. `roms.to_sigma0` refuses full density, while
  `isoslice` takes either.

## On the standardized Dataset

- `xroms.z(std)` finds the renamed zeta and follows the free surface.
- `standardize` masks `pm` and `pn`, so `xroms.dA(std)` is NaN over land. Take areas from the raw grid, as
  `cell_area` does.
- `_cell_km` is not `xroms.nominal_resolution`, which is the mean spacing and is smaller. On rotated grids, `_cell_km`
  is also short by cos(angle).

## ocean-skill bugs found on the way

- **`depth_band`/`depth_average` with a free surface that varies in time.** They take the first dim of `z_w` that is
  not `s_rho` as the interface dim. That dim is `time`, so the band comes back empty. The compare pipeline avoids
  this by dropping zeta.
- **`mld_threshold` with density.** It tests |Δσ0|, though `mld_density_threshold`'s docstring promises an increase.

## First change

The first change is in `tests/test_roms_classic_e2e.py`, in
`test_classic_depths_match_xroms_on_the_xroms_example_file`. Replace `xroms.roms_dataset` with `xroms.z(ds)` and
`xroms.z(ds, scoord="s_w")` on the example file, opened with `xr.open_dataset`. These match ocean-skill's
`z_rho`/`z_w` exactly.

After that, make the wave-1 swaps one function at a time, using each function's parity test above as the check.
