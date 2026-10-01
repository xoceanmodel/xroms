---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import xarray as xr
import xroms
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import pandas as pd
```

# How to interpolate

There is a different approach for interpolation in:

* time (using `xarray`'s `interp`)
* longitude/latitude (using `xESMF`, through {func}`xroms.interpll`)
* depth (using `xgcm`, through {func}`xroms.zslice` for fixed depths and {func}`xroms.isoslice` for the surfaces of any other quantity, like density)

In this notebook, we will demonstrate each independently as well as considerations for combining the approaches.

The depth functions can be called two ways, both shown below: with the Dataset accessor (`ds.xroms.zslice("v", depths)`), which takes variable names and returns results in the Dataset's own dimension names, or as a function (`xroms.zslice(ds.v, depths, ds)`), which takes DataArrays plus the Dataset that holds the grid and returns results in canonical dimension names.

+++

## Load in data

Load in example dataset. More information in the {doc}`input/output page <io>`. There is no setup step: xarray opens the file and xroms calculates from the Dataset directly.

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds.sizes
```

## Interpolate to...

The following section is examples of different kinds of interpolation.

+++

### Times

Interpolating in time is straight-forward because it is 1D, uncoupled from the other dimensions. So, we can just use the `xarray` `interp` function directly with the desired times. The result is `[ocean_time x s_rho x eta_rho x xi_rho]`, with the new times along `ocean_time`.

Notes:
* The times can be a list, a `pd.date_range` or a DataArray. Times outside of the range of the model output give NaN.
* You can interpolate in time on the whole Dataset (`ds.interp(ocean_time=times)`) or a single DataArray. The example shows interpolation in time on a single DataArray.
* Chunking: in current versions of `xarray`, `interp` works on dask arrays that are chunked in time, but it then splits the other dimensions into smaller chunks. For output that is chunked in time (several files, say), put time in one chunk first, `da.chunk({"ocean_time": -1})`, and chunk it again afterward if you want, `.chunk({"ocean_time": 1})`. Here the example file is already a single time chunk.

Example usage for a DataArray `da`:
> `da.interp(ocean_time=times)`

```{code-cell} ipython3
# times to interpolate to: half an hour after the first output, then every hour
start = pd.Timestamp(ds.ocean_time.values[0])
times = pd.date_range(start + pd.Timedelta("30min"), periods=4, freq="1h")

temp = ds.temp.chunk({"ocean_time": -1})  # time in one chunk
temp_t = temp.interp(ocean_time=times)
temp_t.sizes
```

Results are demonstrated below for a single location.

```{code-cell} ipython3
point = {"s_rho": -1, "eta_rho": 50, "xi_rho": 100}

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ds.temp.isel(**point).plot(ax=ax, marker="o", ms=10, label="model output")
temp_t.isel(**point).plot(ax=ax, marker="s", ls="", label="interpolated")
ax.set_title("surface temperature at one location")
ax.legend()
plt.show()
```

### Longitude/latitude points

Function {func}`xroms.interpll` wraps `xESMF` so that the wrapper can take care of some niceties. It takes in longitude/latitude values and interpolates a variable onto the desired lon/lat positions correctly for a non-flat Earth (bilinearly, by default). It has functionality for returning pairs of points (1D) vs. 2D arrays of points. First we demo the 1D output.

`xESMF` is an optional dependency, installed from conda-forge (`conda install -c conda-forge xesmf`).

The result is dimensions `[ocean_time x s_rho x locations]`.

Notes:
* 1D behavior is the default for `xroms.interpll` but also accessible by inputting `which="pairs"`.
* Input longitude and latitudes (below `lon0` and `lat0`) can be lists, ndarrays or DataArrays.
* The variable is interpolated from its own horizontal points, found from its `lon_rho`/`lat_rho`, `lon_u`/`lat_u` or `lon_v`/`lat_v` coordinates.
* Points outside of the model domain come back as NaN, and so do points whose neighbors include land, where the variable is NaN. Extra keyword arguments go to the `xESMF` regridder, including `method=` (`"bilinear"` by default).

Example usage for a DataArray `da`:
> `xroms.interpll(da, lon0, lat0, which="pairs")`

or with the `xroms` DataArray accessor:
> `da.xroms.interpll(lon0, lat0, which="pairs")`

```{code-cell} ipython3
# use advanced indexing to pull out individual pairs of points to compare with
# rather than 2D array of lon/lat points that would occur otherwise
ie, ix = [24, 100, 121, 30], [31, 198, 239, 142]
indexer = {"eta_rho": xr.DataArray(ie, dims="locations"), "xi_rho": xr.DataArray(ix, dims="locations")}
lat0 = ds.lat_rho.isel(indexer)
lon0 = ds.lon_rho.isel(indexer)
salt_comp = ds.salt.isel(indexer).isel(ocean_time=0, s_rho=-1)
```

Interpolating to the longitudes and latitudes of these grid points should return the model values at them.

```{code-cell} ipython3
salt_pts = xroms.interpll(ds.salt, lon0, lat0, which="pairs")
assert np.allclose(salt_pts.isel(ocean_time=0, s_rho=-1), salt_comp)
salt_pts.sizes
```

Plot the interpolated surface salinity overlaid on the full field to visually check.

```{code-cell} ipython3
surface = {"ocean_time": 0, "s_rho": -1}
salt = ds.salt.isel(**surface)
vmin, vmax = float(salt.min()), float(salt.max())

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
salt.plot(ax=ax, x="lon_rho", y="lat_rho", cmap=cmo.haline, vmin=vmin, vmax=vmax)
ax.scatter(lon0, lat0, c=salt_pts.isel(**surface), s=200, edgecolor="r", vmin=vmin, vmax=vmax, cmap=cmo.haline)
plt.show()
```

We can also use `xroms.interpll` to interpolate to a 2D grid of longitudes and latitudes.

Result is `[ocean_time x s_rho x lat x lon]`.

Notes:
* 2D grids of `glon`, `glat` are found by inputting `which="grid"`.
* Input longitude and latitudes (below `glon` and `glat`) are 1D arrays or lists: the output is on their outer product, with `lat` and `lon` as its coordinates.

Example usage for a DataArray `da`:
> `xroms.interpll(da, glon, glat, which="grid")`

or with `xroms` accessor:
> `da.xroms.interpll(glon, glat, which="grid")`

```{code-cell} ipython3
npts = 5
glon, glat = np.linspace(-92, -91, npts + 1), np.linspace(28, 29, npts)  # still input as 1D arrays
GLON, GLAT = np.meshgrid(glon, glat)  # for plotting
u_grid = xroms.interpll(ds.u, glon, glat, which="grid")
u_grid.sizes
```

Plot to visually inspect results.

```{code-cell} ipython3
u = ds.u.isel(**surface)
vmax = float(abs(u).max())

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
u.plot(ax=ax, x="lon_u", y="lat_u", cmap=cmo.delta, vmin=-vmax, vmax=vmax)
ax.scatter(GLON, GLAT, c=u_grid.isel(**surface), s=200, edgecolor="r", vmin=-vmax, vmax=vmax, cmap=cmo.delta)
plt.show()
```

#### Reusing the regridder

Calculating the interpolation weights is the slow part of `xroms.interpll`, and it is done on every call. To interpolate several variables to the same points, build the regridder once with {func}`xroms.make_regridder` and pass it in with `regridder=` instead of the points. The weights depend on the horizontal grid points of the variable they were made for, so `temp` and `salt` (both on rho points) can share a regridder while `u` and `v` each need their own. (If you give points along with a regridder, they have to be the ones it was made for.)

```{code-cell} ipython3
lons, lats = [-92.5, -91.5, -90.5], [28.0, 28.5, 29.0]
reg = xroms.make_regridder(ds.temp, lons, lats, which="pairs")

temp_ll = xroms.interpll(ds.temp, regridder=reg)
salt_ll = xroms.interpll(ds.salt, regridder=reg)
temp_ll.sizes
```

### Fixed depths

Function {func}`xroms.zslice` interpolates a variable to fixed vertical positions at every horizontal location. It finds the heights of the variable's own points from the Dataset (the bathymetry `h`, the free surface `zeta` and the vertical coordinate parameters), then interpolates linearly along the vertical, using `xgcm`'s `transform`.

The result is dimensions `[ocean_time x z x eta x xi]`, where `z` is the new vertical coordinate holding the depths used to interpolate the variable to; it takes the place of `s_rho`.

Notes:
* `zslice` takes **heights relative to mean sea level, negative down**, like ROMS' `z` ({func}`xroms.z`): `-10` is 10 m below mean sea level. Depths as positive numbers are also possible, with `positive="down"` (see below).
* Input depths can be lists or ndarrays.
* Where a depth is above the top model level or below the bottom one, the result is NaN (the model levels are the centers of the layers, so these limits are a little inside the surface and the seabed). With `mask_edges=False` it holds the value at the edge instead.
* The vertical is rechunked to a single chunk when the interpolation runs (lazily); the other dimensions keep their chunks.
* `xgcm`'s `transform` has more flexibility and functionality than is offered through `xroms.zslice`; this function focuses on just depth interpolation. `xroms.isoslice`, below, is closer to it.

Pick depths that suit the data: how deep does the water get?

```{code-cell} ipython3
print(f"water depth in this domain: {float(ds.h.min()):.0f} to {float(ds.h.max()):.0f} m")
```

We will use 20 depths from 10 to 600 m below mean sea level. They start at 10 m instead of 0 because the top model level lies below the surface (by up to about 1.6 m here), so a height of 0 is above it, and NaN, at most points.

+++

#### with z varying in time

Use the heights of the model's levels as they move with the free surface `zeta`, which varies in time. This is the default.

Example usage with the Dataset accessor, providing the DataArray name:
> `ds.xroms.zslice(varname, depths)`

or as a function, providing a DataArray and the Dataset (`ds` here) that holds the grid:
> `xroms.zslice(da, depths, ds)`

```{code-cell} ipython3
depths = np.linspace(-10, -600, 20)
v_z = ds.xroms.zslice("v", depths)
v_z.sizes
```

The `z` coordinate is labelled with what it is:

```{code-cell} ipython3
v_z.z.attrs
```

The function returns canonical dimension names: `(eta_v, xi_rho)` for v points, where this Rutgers file has `(eta_v, xi_v)`. That would not line up with `ds.v` (the `xi` dimensions would broadcast against each other), so use {func}`xroms.rename_like` to return to the Dataset's names. (Another option is `ds = xroms.canonicalize(ds)` once at the start, to work in canonical names throughout.)

```{code-cell} ipython3
v_fn = xroms.zslice(ds.v, depths, ds)
print(v_fn.dims)

v_fn = xroms.rename_like(v_fn, ds)
print(v_fn.dims)
xr.testing.assert_allclose(v_fn, v_z)
```

Plot to visually inspect results. The model field is plotted in its own terrain-following levels (it needs the heights of those levels as a coordinate: see `ds.xroms.z("v")`) and the interpolated values are the circles at fixed depths, which should match the colors beneath them. Circles below the seabed are empty because they are NaN.

```{code-cell} ipython3
section = {"ocean_time": 0, "xi_v": 100, "eta_v": slice(None, 40)}  # a cross-section down the slope

v_sec = ds.v.isel(**section).assign_coords(z=ds.xroms.z("v").isel(**section))
pts = v_z.isel(**section).isel(eta_v=slice(None, None, 3), z=slice(None, None, 2))
LAT, Z = np.meshgrid(pts.lat_v, pts.z)

vmax = float(abs(v_sec).max())
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
v_sec.plot(ax=ax, x="lat_v", y="z", cmap=cmo.delta, vmin=-vmax, vmax=vmax)
ax.scatter(LAT, Z, c=pts.transpose("z", "eta_v"), s=150, edgecolor="r", vmin=-vmax, vmax=vmax, cmap=cmo.delta, clip_on=False)
plt.show()
```

#### z constant in time

Pass `zeta=0` to use the resting heights, calculated from the bathymetry `h` alone, instead of heights that follow the free surface. Then `z` does not vary in time, which makes the interpolation cheaper, and is a good choice if the free surface moves very little compared to the depths of interest. (This replaces the `include_Z0` and `z_rho_v0` approach of xroms 0.6; see the {doc}`migration guide <migration>`.)

Example usage:
> `ds.xroms.zslice(varname, depths, zeta=0)`

or
> `xroms.zslice(da, depths, ds, zeta=0)`

```{code-cell} ipython3
v_0 = ds.xroms.zslice("v", depths, zeta=0)

# the resting heights have no time dimension
print(ds.xroms.z().dims, ds.xroms.z(zeta=0).dims)
print(f"largest difference between the two: {float(abs(v_z - v_0).max()):.4f} m/s (zeta is at most {float(ds.zeta.max()):.2f} m)")
```

Plot the difference between the two interpolations at a single location (a deep one, where all the depths are in the water) to see the effect of accounting for time-varying depths or not.

```{code-cell} ipython3
loc = {"ocean_time": 0, "eta_v": 10, "xi_v": 250}

fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
v_z.isel(**loc).plot(ax=axs[0], y="z", lw=3, label="z follows zeta")
v_0.isel(**loc).plot(ax=axs[0], y="z", label="zeta=0")
axs[0].set_title("v at one location")
axs[0].legend()
(v_z - v_0).isel(**loc).plot(ax=axs[1], y="z")
axs[1].set_title("difference")
plt.show()
```

#### Depths below the surface, and nearest levels

The depths can be measured from other references with `reference=`: `"mean_sea_level"` (the default), `"surface"` for the moving free surface, or `"bottom"` for heights above the seabed. `positive="down"` gives depths as positive numbers, in the inputs and in the `z` coordinate of the result. For example, temperature at depths below the sea surface:

```{code-cell} ipython3
below = np.arange(2, 101, 2)  # m below the moving sea surface
temp_lin = ds.xroms.zslice("temp", below, reference="surface", positive="down")

# the nearest model level to each depth, instead of interpolating
temp_near = ds.xroms.zslice("temp", below, reference="surface", positive="down", method="nearest")

# 5 m above the seabed (NaN where the lowest model level is higher than that)
temp_bottom = ds.xroms.zslice("temp", [5], reference="bottom")

temp_lin.z.attrs
```

Compare the two methods at one location, along with the model's own levels:

```{code-cell} ipython3
loc = {"ocean_time": 0, "eta_rho": 10, "xi_rho": 250}
levels = ds.temp.isel(**loc).assign_coords(z=ds.xroms.z(reference="surface", positive="down").isel(**loc)).compute()
levels = levels.where(levels.z < 110, drop=True)  # just the upper levels

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
levels.plot(ax=ax, y="z", ls="", marker="o", color="0.6", label="model levels")
temp_lin.isel(**loc).plot(ax=ax, y="z", label="linear")
temp_near.isel(**loc).plot(ax=ax, y="z", drawstyle="steps-mid", label="nearest")
ax.set_ylim(100, 0)
ax.set_title("temperature at one location")
ax.legend()
plt.show()
```

### isoslice

Function {func}`xroms.isoslice` interpolates a variable to where another quantity, the `iso_array`, takes given values, along a dimension `dim`: `xroms.isoslice(var, iso_values, iso_array, dim=..., new_dim=...)`. The new dimension `new_dim` takes the place of `dim` in the result. `xroms.zslice` is `isoslice` with the heights as the `iso_array`; other examples are surfaces of constant density, or cross-sections at a given latitude.

Notes:
* `var` and `iso_array` need to be on the same horizontal points and levels (both on rho points, say), as they are paired by dimension name.
* `dim` is the dimension to interpolate along: an axis, `"Z"` (the default), `"Y"` or `"X"`, or the name of the dimension (`s_rho`, `eta_u`, ...).
* `method="linear"` (default) or `"nearest"`, and `mask_edges` work as in `zslice`.

#### Potential density surfaces

The Dataset accessor property `ds.xroms.sig0` is potential density (the full density, around 1000 to 1030 kg/m^3), so subtract 1000 for the more familiar sigma0. Here we find the depth of three potential density surfaces, and the temperature on them.

Example usage as a function:
> `xroms.isoslice(da, iso_values, iso_array, dim="s_rho", new_dim="sigma0")`

or with the accessor, which also takes variable names:
> `ds.xroms.isoslice(varname, iso_values, iso_array, dim="s_rho", new_dim="sigma0")`

```{code-cell} ipython3
sigma = ds.xroms.sig0 - 1000
sigma_values = [24, 25, 26]

z_iso = xroms.isoslice(ds.xroms.z(), sigma_values, sigma, dim="s_rho", new_dim="sigma0")
temp_iso = xroms.isoslice(ds.temp, sigma_values, sigma, dim="s_rho", new_dim="sigma0")
temp_iso.sizes
```

The result is `[ocean_time x sigma0 x eta_rho x xi_rho]`. It is NaN where the water column does not reach the density, for example in the light water of the shelf.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
z_iso.isel(ocean_time=0).sel(sigma0=25).plot(ax=axs[0], x="lon_rho", y="lat_rho", cmap=cmo.deep_r)
temp_iso.isel(ocean_time=0).sel(sigma0=25).plot(ax=axs[1], x="lon_rho", y="lat_rho", cmap=cmo.thermal)
axs[0].set_title("depth of sigma0 = 25")
axs[1].set_title("temperature on sigma0 = 25")
plt.show()
```

#### Cross-section along a latitude

Use the coordinate as the `iso_array`, and interpolate along the axis it changes along. Here we calculate a cross-section of u-velocity along a latitude of 28.5 degrees, so the `iso_array` is the latitude of the u points and `dim="Y"` (the eta axis). The result has the new dimension `lat`, with the one value 28.5, in place of `eta_u`, and it keeps `xi_u`: it is a section along the xi direction of the grid, at that latitude.

```{code-cell} ipython3
lat0 = 28.5
u_lat = xroms.isoslice(ds.u, [lat0], ds.lat_u, dim="Y", new_dim="lat")
u_lat.sizes
```

To plot the section against longitude and depth, interpolate those in the same way.

```{code-cell} ipython3
lon_lat = xroms.isoslice(ds.lon_u, [lat0], ds.lat_u, dim="Y", new_dim="lat")
z_lat = xroms.isoslice(ds.xroms.z("u"), [lat0], ds.lat_u, dim="Y", new_dim="lat")

u_sec = u_lat.isel(ocean_time=0, lat=0).assign_coords(lon=lon_lat.isel(lat=0), z=z_lat.isel(ocean_time=0, lat=0))
# columns that do not reach 28.5 degrees (or are land) are all NaN, as are their longitudes and heights
u_sec = u_sec.dropna("xi_u", how="all")

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
u_sec.plot(ax=ax, x="lon", y="z", cmap=cmo.delta)
ax.set_title(f"u at {lat0} degrees north")
plt.show()
```

### Multiple locations, depths, and times

A user can simply use multiple of these approaches one after another to interpolate in more dimensions. There are several considerations for the ordering:

* Downsize first

    If you are going to interpolate in time, depth, and lon/lat, consider if one of those interpolation steps will result in much less model output, and if so, do that step first. For example, if you will interpolate to 3 data locations in lon/lat but 30 vertical levels on a 191 x 300 grid, first interpolate in lon/lat before interpolating in z to save time.

* Heights

    After interpolating to lon/lat points the variable is no longer on the model grid: it has no `eta_rho` and `xi_rho` dimensions for `zslice` to look up the bathymetry and free surface with (passing the Dataset as the grid raises an error). Give it the heights of the variable's points with `z=` instead, on the same points, times and levels as the variable. The simplest way to get matching heights is to interpolate them just like the variable: the same `xroms.interpll` (with the same regridder, which also saves time) and the same `interp` in time.

* Chunking

    `xroms.interpll` and `xroms.zslice` take care of chunks. As above, `interp` in time works best with time in one chunk. You can check chunks with `da.chunks`, specify new chunks with `da.chunk({'ocean_time': 1, 's_rho': 5})` and reset any individual dimension chunking by passing in -1, or reset all chunks for a DataArray or Dataset with `ds.chunk(-1)`.

Here we interpolate salinity to the three lon/lat locations from the regridder example, to 13 depths between 2 and 50 m below mean sea level, and to times every half hour.

```{code-cell} ipython3
depths = np.arange(-2, -52, -4)
times = pd.date_range(start, ds.ocean_time.values[-1], freq="30min")
```

Since there are only a few lons/lats, I will start with that, for the salinity and for the heights of its points:

```{code-cell} ipython3
salt_ll = xroms.interpll(ds.salt, regridder=reg)
z_ll = xroms.interpll(ds.xroms.z(), regridder=reg)
print(salt_ll.dims, z_ll.dims)
```

The order of the other two steps probably doesn't matter too much in this case. The heights have a time dimension (the free surface moves), so they get the same time interpolation as the salinity, then everything goes into `zslice`:

```{code-cell} ipython3
salt_t = salt_ll.interp(ocean_time=times)
z_t = z_ll.interp(ocean_time=times)
salt_z = xroms.zslice(salt_t, depths, z=z_t)
salt_z.sizes
```

The result is `[ocean_time x z x locations]`; it is NaN below the seabed, which is shallow at the third location.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
salt_z.isel(ocean_time=0).plot.line(ax=ax, y="z", hue="locations", marker="o")
plt.show()
```

The heights `z_ll` carry xroms' labels (`positive`, `vertical_reference`) through the interpolations, so `zslice` knows what they are. If you ask for another `reference=` or `positive=`, build the heights the same way, for example `ds.xroms.z(reference="surface", positive="down")`.

The heights do not have to be interpolated in time: with `zeta=0` they have no time dimension to interpolate, and `zslice` broadcasts them over time. Doing the steps in another order gives very similar numbers, too: here, depth first on the grid, then the points, then time. Compare both with the result above:

```{code-cell} ipython3
# resting heights at the points (no time dimension)
z_ll0 = xroms.interpll(ds.xroms.z(zeta=0), regridder=reg)
salt_z0 = xroms.zslice(salt_t, depths, z=z_ll0)

# depth first
salt_zfirst = xroms.interpll(ds.xroms.zslice("salt", depths), regridder=reg).interp(ocean_time=times)

for name, other in [("zeta=0", salt_z0), ("depth first", salt_zfirst)]:
    diff = float(abs(other.transpose(*salt_z.dims) - salt_z).max())
    print(f"{name}: largest difference {diff:.3f} (salinity is {float(salt_z.min()):.0f} to {float(salt_z.max()):.0f})")
```
