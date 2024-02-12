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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy
```

# How to select data

The {doc}`input/output <io>` notebook demonstrates how to load in data, but now how to select and slice it apart? Much of this is accomplished with the `sel` and `isel` methods in `xarray`, which are demonstrated in detail in this notebook.

Use `sel` to select/slice a Dataset or DataArray by dimension values; the best example of this for ROMS output is selecting certain time using a string representation of a datetime.

    ds.salt.sel(ocean_time='2010-1-1 12:00')
    ds.salt.sel(ocean_time=slice('2010-1-1', '2010-2-1'))

Use `isel` to subdivide a Dataset or DataArray by dimension indices:

    ds.salt.isel(eta_rho=20, xi_rho=100)
    ds.salt.isel(eta_rho=slice(20,100,10), xi_rho=slice(None,None,5))

+++

## `cf-xarray`

`xroms` includes the `cf-xarray` accessor, which allows you to use xarray `sel` and `isel` commands for a DataArray without needing to input the exact grid – just the axes.

With `xarray` alone:

    ds.salt.isel(xi_rho=20, eta_rho=10, s_rho=20, ocean_time=10)

With `cf-xarray` accessor:

    ds.salt.cf.isel(X=20, Y=10, Z=20, T=10)

and get the same thing back. Same for `sel`. The T, Z, Y, X names can be mixed and matched with the actual dimension names. Some of the attribute wrangling in `xroms` is dedicated to making sure that `cf-xarray` can always identify dimensions and coordinates for DataArrays.

You can always check what `cf-xarray` understands about a Dataset or DataArray with

    ds.salt.cf.describe()

+++

### Load in data

More information at in {doc}`input/output page <io>`

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)
ds.xroms.set_grid(xgrid)
ds
```

```{code-cell} ipython3
ds.salt.cf.describe()
```

## Select

+++

### Surface layer slice

The surface in ROMS is given by the last index in the vertical dimension. The easiest way to access this is by indexing into `s_rho`. While normally it is better to access coordinates through keywords to be human-readable, it's not easy to tell what value of `s_rho` gives the surface. In this instance, it's easier to just go by index.

    ds.salt.isel(s_rho=-1)

    ds.salt.cf.isel(Z=-1)  # with cf-xarray

You can also grab the Z level that is "nearest" to 0, the surface, which will give the same vertical level as the other options:

    ds.salt.cf.sel(Z=0, method="nearest")

```{code-cell} ipython3
ds.salt.cf.sel(Z=0, method="nearest")
```

### x/y index slice

For a curvilinear ROMS grid, selecting by the dimensions `xi_rho` or `eta_rho` (or for whichever is the relevant grid) is not very meaningful because they are given by index. Thus the following is possible to get a slice along the index, but it cannot be used to find a slice based on the lon/lat values. For the eta and xi grids, `sel` is equivalent to `isel`.

    ds.temp.sel(xi_rho=20)

    ds.temp.cf.sel(X=20);  # same with cf-xarray accessor

```{code-cell} ipython3
ds.temp.cf.sel(X=20)
```

### Single time

Find the forecast model output available that is closest to now. Note that the `method` keyword argument is not necessary if the desired date/time is exactly a model output time. You can daisy-chain together different `sel` and `isel` calls.

    date = "2009-11-19T13:00"

    ds.salt.isel(s_rho=-1).sel(ocean_time=date, method='nearest')

    ds.salt.cf.isel(Z=-1).cf.sel(T=date, method='nearest')  # with cf-xarray

```{code-cell} ipython3
date = "2009-11-19T13:00"
ds.salt.cf.sel(Z=0, T=date, method='nearest')
```

### Range of time

    time_range = slice(date, pd.Timestamp(date)+pd.Timedelta('3 hours'))

    ds.salt.sel(ocean_time=time_range)

    ds.salt.cf.sel(T=time_range)  # cf-xarray

```{code-cell} ipython3
time_range = slice(date, pd.Timestamp(date)+pd.Timedelta('3 hours'))
ds.salt.cf.sel(T=time_range)  # cf-xarray
```

### Select region

Select a boxed region by min/max lon and lat values.

```{code-cell} ipython3
# want model output only within the box defined by these lat/lon values
lon = np.array([-92, -91])
lat = np.array([28, 29])
```

```{code-cell} ipython3
# this condition defines the region of interest
box = ((lon[0] < ds["salt"].cf["longitude"]) & (ds["salt"].cf["longitude"] < lon[1])
       & (lat[0] < ds["salt"].cf["latitude"]) & (ds["salt"].cf["latitude"] < lat[1])).compute()
```

Plot the model output in the box at the surface

```{code-cell} ipython3
dss = ds.where(box).salt.cf.isel(Z=-1, T=0)
dss.cf.plot(x='longitude', y='latitude')
```

If you don't need the rest of the model output, you can drop it by using `drop=True` in the `where` call.

```{code-cell} ipython3
dss = ds.where(box, drop=True).salt.cf.isel(Z=-1, T=0)
dss.cf.plot(x='longitude', y='latitude')
```

Can calculate a metric within the box:

```{code-cell} ipython3
dss.mean().values
```

### Subset model output

Subset Dataset of model output such that subsetted domain is as if the simulation was run on that size grid. That is, the rho grid is 1 larger than the psi grid in each of xi and eta.

    ds.xroms.subset(X=slice(20,40), Y=slice(50,100))  # with accessor

    xroms.subset(ds, X=slice(20,40), Y=slice(50,100))

```{code-cell} ipython3
ds.xroms.subset(X=slice(20,40), Y=slice(50,100))  # with accessor
```

### Find nearest in lon/lat

This matters for a curvilinear grid.

Can't use `sel` because it will only search in one dimension for the nearest value and the dimensions are indices which are not necessarily geographic distance. Instead need to use a search for distance and use that for the `where` condition from the previous example. This functionality has been wrapped into `xroms.sel2d` (and its partner function `xroms.argsel2d`).

```{code-cell} ipython3
lon0, lat0 = -91, 28
saltsel = ds.salt.xroms.sel2d(lon0, lat0)
```

Or, if you instead want the indices of the nearest grid node returned, you can call `argsel2d`:

```{code-cell} ipython3
ds.salt.xroms.argsel2d(lon0, lat0)
```

Check this function, just to be sure:

```{code-cell} ipython3
dl = 0.05
box = (ds.lon_rho>lon0-dl) & (ds.lon_rho<lon0+dl) & (ds.lat_rho>lat0-dl) & (ds.lat_rho<lat0+dl)
dss = ds.where(box).salt.cf.isel(T=0, Z=-1)

vmin = dss.min().values
vmax = dss.max().values

dss.plot(x='lon_rho', y='lat_rho')
plt.scatter(lon0, lat0, c=saltsel.cf.isel(Z=-1, T=0), s=200, edgecolor='k', vmin=vmin, vmax=vmax)
plt.xlim(lon0-dl,lon0+dl)
plt.ylim(lat0-dl, lat0+dl)
```
