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

* time (using `xarray interp`)
* longitude/latitude (using `xESMF`)
* depth (using `xgcm`)

In this notebook, we will demonstrate each independently as well considerations for combining the approaches.

+++

## Load in data

Load in example dataset. More information at in {doc}`input/output page <io>`

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True, include_Z0=True)
ds.xroms.set_grid(xgrid)
ds
```

## Interpolate to...

The following section is examples of different kinds of interpolation.

+++

### times

Interpolating in time is straight-forward because it is 1D, uncoupled from the other dimensions. So, we can just use the `xarray interp` function directly with the desired times. The result is `[ocean_time x s_rho x eta x xi]`.

Notes:
* The potentially tricky part is that chunking cannot occur in the direction of interpolation. So, here we reset the chunking and chunk in a different dimension before interpolation, then chunk back to `ocean_time` afterward.
* You can interpolate in time on the whole Dataset or a single DataArray. The example shows interpolation in time on a single DataArray.
* The interpolation times can be sequence-like, but I recommend putting them into a DataArray as follows and demonstrated in the example so that attributes are appropriately added and `cf-xarray` works afterward (also used in the other interpolation routines).
> t0s = xr.DataArray(t0s, dims='ocean_time', attrs={'axis': 'T', 'standard_name': 'time'})

Example usage for a DataArray `da`:
> da.interp(ocean_time=t0s)

```{code-cell} ipython3
# times to interpolate to
startdate = pd.Timestamp(ds.cf["T"][0].values)
t0s = [startdate + pd.Timedelta('30 min') + pd.Timedelta('1 hours')*i for i in range(4)]

# not necessary to change t0s to be a DataArray, but then we can add
# attributes that keep cf-xarray working
t0s = xr.DataArray(t0s, dims=ds.cf["T"].name, attrs={'axis': 'T', 'standard_name': 'time'})

varin = ds.temp

# rechunk from time to vertical dimension
varin = varin.chunk({'ocean_time': -1, 's_rho': 1})

# interpolation
varout = varin.interp(ocean_time=t0s).chunk({'ocean_time':1, 's_rho': -1})
```

Results are demonstrated below for a single location.

```{code-cell} ipython3
varin.cf.isel(Z=-1, Y=50, X=100).plot()
varout.cf.isel(Z=-1, Y=50, X=100).plot(marker='o')
```

### multiple lon, lat locations (1D)

Function `xroms.interpll` wraps `xESMF` so that the wrapper can take care of some niceties. It takes in longitude/latitude values and interpolates a variable onto the desired lon/lat positions correctly for a non-flat Earth. It has functionality for returning pairs of points (1D) vs. 2D arrays of points. First we demo the 1D output.

The result is dimensions `[ocean_time x s_rho x locations]`.

Notes:
* Cannot have chunks in the horizontal dimensions.
* 1D behavior is the default for `xroms.interpll` but also accessible by inputting `which='pairs'`.
* Input longitude and latitudes (below `lon0` and `lat0`) can be lists or ndarrays.

Example usage for a DataArray `da`:
> xroms.interpll(da, lon0, lat0, which='pairs')

or with `xroms` accessor:
> da.xroms.interpll(lon0, lat0, which='pairs')

```{code-cell} ipython3
# use advanced indexing to pull out individual pairs of points to compare with
# rather than 2D array of lon/lat points that would occur otherwise
ie, ix = [24, 100, 121, 30], [31, 198, 239, 142]
varin = ds.salt
indexer = {varin.cf["X"].name: xr.DataArray(ix, dims="locations"), varin.cf["Y"].name: xr.DataArray(ie, dims="locations")}
lat0 = varin.cf["latitude"].isel(indexer)
lon0 = varin.cf["longitude"].isel(indexer)
varcomp = varin.isel(indexer).cf.isel(T=0, Z=-1)
```

```{code-cell} ipython3
varout = xroms.interpll(varin, lon0, lat0, which='pairs')
assert np.allclose(varout.isel(ocean_time=0, s_rho=-1), varcomp)
```

Plot the interpolated surface salinity overlaid on the full field to visually check.

```{code-cell} ipython3
indexer = {'ocean_time': 0, 's_rho': -1}
salt = varin.isel(indexer)
vmin = salt.min().values; vmax = salt.max().values

fig, ax = plt.subplots(1, 1, figsize=(15,10))
salt.cf.plot.pcolormesh(x='longitude', y='latitude', infer_intervals=True, cmap=cmo.haline)
ax.scatter(lon0, lat0, c=varout.isel(indexer), s=200, edgecolor='r', vmin=vmin, vmax=vmax, cmap=cmo.haline)
```

### array of lon, lat locations (2D)

We can also use `xroms.interpll` to interpolate to a 2D grid of longitudes and latitudes.

Result is `[ocean_time x s_rho x lat x lon]`.

Notes:
* Cannot have chunks in the horizontal dimensions.
* 2D grids of lon0, lat0 are found by inputting `which='grid'`.
* Input longitude and latitudes (below `lon0` and `lat0`) can be lists or ndarrays.

Example usage for a DataArray `da`:
> xroms.interpll(da, lon0, lat0, which='grid')

or with `xroms` accessor:
> da.xroms.interpll(lon0, lat0, which='grid')

```{code-cell} ipython3
npts = 5
lon0, lat0 = np.linspace(-92, -91, npts+1), np.linspace(28,29,npts)  # still input as 1D arrays
LON0, LAT0 = np.meshgrid(lon0, lat0)  # for plotting
varin = ds.u

varout = xroms.interpll(varin, lon0, lat0, which='grid')
```

Plot to visually inspect results.

```{code-cell} ipython3
indexer = {'ocean_time': 0, 's_rho': -1}
vmin = abs(varin).min().values; vmax = abs(varin).max().values
vmax = max(vmin, vmax)

fig, ax = plt.subplots(1, 1, figsize=(15,10))
varin.isel(indexer).cf.plot.pcolormesh(x='longitude', y='latitude', infer_intervals=True, cmap=cmo.delta)
ax.scatter(LON0, LAT0, c=varout.isel(indexer), s=200, edgecolor='r', vmin=-vmax, vmax=vmax, cmap=cmo.delta)
```

### variable regridded to fixed depths

Function `xroms.zslice` wraps `xgcm grid.transform` so that the wrapper can take care of some niceties. It  interpolates a variable onto the input depths.

The result is dimensions `[ocean_time x [z coord] x eta x xi]`, where `[z coord]` is the z coordinate used to interpolate the variable to.

Notes:
* Cannot have chunks in the vertical dimension.
* Input depths can be lists or ndarrays.
* `xgcm grid.transform` has more flexibility and functionality than is offered through `xroms.zslice`; this function focuses on just depth interpolation.
* Interpolation to fixed depths can be done using time-varying depths or with constant depths in time; do the latter to save computation time if accuracy isn't very important.

+++

#### with z varying in time

Use the z coordinates associated with the DataArray in the interpolation.

Example usage for a DataArray `da`:
> xroms.isoslice(da, depths, grid, z=z, axis="Z")

More is pre-selected if you used the `xroms` accessor, with a different name of "zslice". With DataArray, need to provide grid:
> da.xroms.zslice(grid, depths)

With Dataset accessor need to provide DataArray name:
> ds.xroms.zslice(varname, depths)


```{code-cell} ipython3
varin = ds.v
varout = xroms.isoslice(varin, np.linspace(0, -600, 20), xgrid)
```

Plot to visually inspect results:

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(14,6))

dss = varin.cf.isel(X=100, ocean_time=0)
dss.where(~dss.isnull().compute(), drop=True).cf.plot(x='latitude', y='vertical', cmap=cmo.delta)

vmin = abs(dss).min().values; vmax = abs(dss).max().values
vmax = max(vmin, vmax)
toplot = varout.cf.isel(T=0, X=100, Y=slice(None,None,10), Z=slice(None,None,3))
X, Z = np.meshgrid(toplot.lat_v, toplot.z_rho_v)
ax.scatter(X, Z, c=toplot, s=200, edgecolor='r', vmin=-vmax, vmax=vmax, cmap=cmo.delta)
```

#### z constant in time

Input separate z coordinates `z0` that don't vary in time for the DataArray to be interpolated to.

Example usage for a DataArray `da`:
> xroms.isoslice(da, depths, grid, z=z0, axis="Z")

More is pre-selected if you used the `xroms` accessor, with a different name of "zslice". With DataArray, need to provide grid:
> da.xroms.zslice(grid, depths, z=z0)

With Dataset accessor need to provide DataArray name:
> ds.xroms.zslice(varname, depths, z=z0)

One complication that is currently necessary is to change the metadata such that `z_rho_v0` is recognized as the vertical coordinate for `ds.v`.

```{code-cell} ipython3
var0 = ds.v

# changes to use z_rho_v0 as vertical coordinate
var0.attrs["coordinates"] = var0.attrs["coordinates"].replace("z_rho_v","z_rho_v0")
var0.z_rho_v0.attrs["positive"] = "up"
var0.z_rho_v0.attrs["standard_name"] = "depth"

varout0 = xroms.isoslice(var0, np.linspace(0, -600, 20), xgrid, iso_array=var0.z_rho_v0)
```

Plot the difference between the two interpolations as a point to see the difference in accounting for time-varying depths and not.

```{code-cell} ipython3
indexer = {'ocean_time': 0, 'Y': 10, 'X': 250}

varout.cf.isel(indexer).cf.plot(y='vertical', figsize=(6,6), lw=3)
varout0.cf.isel(indexer).cf.plot(y='vertical')
```

### multiple locations, depths, and times

A user can simply use multiple of these approaches one after another to interpolate in more dimensions. There are several considerations for the ordering:

* Downsize first

    If you are going to interpolate in time, depth, and lon/lat, consider if one of those interpolation steps will result in much less model output, and if so, do that step first. For example, if you will interpolate to 3 data locations in lon/lat but 50 vertical levels, first interpolate in lon/lat before interpolating in z to save time.

* Chunking

    A DataArray cannot be chunked in the dimension that is being interpolated on. So, in the previous example of interpolating first in lon/lat, the DataArray can have dask chunks in the Z and T directions when calculating the lon/lat interpolation. Then, the DataArray would need to be rechunked so that no chunks are in the Z dimension before interpolating in the Z dimension. Similarly for time. You can check chunks with `da.chunks`, specify new chunks with `da.chunk({'ocean_time': 1, 's_rho': 5})` and reset any individual dimension chunking by passing in -1, or reset all chunks for a DataArray or Dataset with `ds.chunk(-1)`.

```{code-cell} ipython3
varin = ds.salt
lons, lats = [-93, -92, -91], [28, 28.5, 29]
zs = np.linspace(0, -50, 20)
startdate = pd.Timestamp(ds.ocean_time[0].values)
ts = [startdate + pd.Timedelta('30 min')*i for i in range(10)]
ts = xr.DataArray(ts, dims='ocean_time', attrs={'axis': 'T', 'standard_name': 'time'})
```

Since there are only a few lons/lats, I will start with that:

```{code-cell} ipython3
varout = xroms.interpll(varin, lons, lats, which='pairs')
print(varout)
```

The order of the other two steps probably doesn't matter too much in this case:

```{code-cell} ipython3
varout
```

```{code-cell} ipython3
varout2 = varout.interp(ocean_time=ts)
varout3 = xroms.isoslice(varout2, zs, xgrid)
# print(varout3)
```

Note that `cf-xarray` still works on this output:

```{code-cell} ipython3
varout3.cf.describe()
```

### Cross-section or isoslice

A cross-section or isoslice can be calculated using `xroms.isoslice`. A short example is given here, but more examples are given in the `xroms.isoslice` docs. This is the same function used for interpolating variables to fixed depths as demonstrated earlier in this notebook.

Calculate cross-section of u-velocity along latitude of 27 degrees.

```{code-cell} ipython3
grid = ds.xroms.xgrid
lat0 = 27
varin = ds.u
```

```{code-cell} ipython3
xroms.isoslice(varin, np.array([lat0]), xgrid, iso_array=varin.cf['latitude'], axis='Y')
```
