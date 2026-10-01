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
```

# How to select data

The {doc}`input/output <io>` notebook demonstrates how to load in data, but how do you select and slice it apart? Much of this is accomplished with the `sel` and `isel` methods in `xarray`, using the dimension names of your Dataset, which are demonstrated in detail in this notebook. `xroms` adds a few selections that ROMS output needs: the surface and bottom layers, a horizontal subset that keeps the staggered grids consistent, and the grid point nearest to a longitude and latitude.

Use `sel` to select/slice a Dataset or DataArray by dimension values; the best example of this for ROMS output is selecting certain time using a string representation of a datetime.

    ds.salt.sel(ocean_time='2010-1-1 12:00')
    ds.salt.sel(ocean_time=slice('2010-1-1', '2010-2-1'))

Use `isel` to subdivide a Dataset or DataArray by dimension indices:

    ds.salt.isel(eta_rho=20, xi_rho=100)
    ds.salt.isel(eta_rho=slice(20,100,10), xi_rho=slice(None,None,5))

## Load in data

More information in the {doc}`input/output page <io>`.

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
print(dict(ds.sizes))
```

The example is Rutgers ROMS output, which names the dimensions of every position of the grid separately: `(eta_rho, xi_rho)` for rho points, `(eta_u, xi_u)` for u points and `(eta_v, xi_v)` for v points, plus `s_rho` and `s_w` for the vertical levels and `ocean_time`. Use the names that your own Dataset has (UCLA ROMS output, for example, uses `(eta_rho, xi_u)` for u points and `(eta_v, xi_rho)` for v points).

## Select

### Surface layer slice

The surface in ROMS is given by the last index in the vertical dimension. The easiest way to access this is by indexing into `s_rho`. While normally it is better to access coordinates through keywords to be human-readable, it's not easy to tell what value of `s_rho` gives the surface. In this instance, it's easier to just go by index.

```{code-cell} ipython3
ds.salt.isel(s_rho=-1).dims
```

You can also grab the `s_rho` level that is "nearest" to 0, the surface, which will give the same vertical level:

```{code-cell} ipython3
ds.salt.sel(s_rho=0, method="nearest").dims
```

{func}`xroms.surface` and {func}`xroms.bottom` give the top and bottom layers for any output, also those without `s_rho` labels (UCLA ROMS output has none, so the index is the only way there). Both are also available on the Dataset accessor, which takes a variable name. The layer stays marked by a scalar `s_rho` coordinate that holds its label (or its index if there are no labels), so that `xroms` can tell later that this is a single level.

    xroms.surface(ds.salt)
    ds.xroms.surface("salt")  # with accessor
    ds.xroms.bottom("salt")

```{code-cell} ipython3
print(ds.xroms.surface("salt").s_rho.values)
print(ds.xroms.bottom("salt").s_rho.values)
```

### x/y index slice

For a curvilinear ROMS grid, selecting by the dimensions `xi_rho` or `eta_rho` (or for whichever is the relevant grid) is not very meaningful because they are given by index. Thus the following is possible to get a slice along the index, but it cannot be used to find a slice based on the lon/lat values. For the eta and xi dimensions, `sel` is equivalent to `isel`.

```{code-cell} ipython3
print(dict(ds.temp.sel(xi_rho=20).sizes))
```

### Single time

Select the model output that is closest to a date and time. Note that the `method` keyword argument is not necessary if the desired date/time is exactly a model output time. You can daisy-chain together different `sel` and `isel` calls.

```{code-cell} ipython3
date = "2009-11-19T13:00"
ds.salt.isel(s_rho=-1).sel(ocean_time=date, method="nearest").dims
```

### Range of time

```{code-cell} ipython3
time_range = slice(date, pd.Timestamp(date) + pd.Timedelta("3 hours"))
print(dict(ds.salt.sel(ocean_time=time_range).sizes))
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
box = (
    (lon[0] < ds.lon_rho) & (ds.lon_rho < lon[1]) & (lat[0] < ds.lat_rho) & (ds.lat_rho < lat[1])
).compute()
```

Plot the model output in the box at the surface

```{code-cell} ipython3
dss = ds.salt.where(box).isel(ocean_time=0, s_rho=-1)
plt.figure(figsize=(6, 4))
dss.plot(x="lon_rho", y="lat_rho");
```

If you don't need the rest of the model output, you can drop it by using `drop=True` in the `where` call.

```{code-cell} ipython3
dss = ds.salt.where(box, drop=True).isel(ocean_time=0, s_rho=-1)
plt.figure(figsize=(6, 4))
dss.plot(x="lon_rho", y="lat_rho");
```

Can calculate a metric within the box:

```{code-cell} ipython3
dss.mean().values
```

### Subset model output

Subset Dataset of model output such that subsetted domain is as if the simulation was run on that size grid. That is, the rho grid is 1 larger than the psi grid in each of xi and eta. `X` and `Y` are slices of rho indices, and the u, v and psi dimensions keep the points between them, so that every position stays consistent with the others.

    ds.xroms.subset(X=slice(20,40), Y=slice(50,100))  # with accessor

    xroms.subset(ds, X=slice(20,40), Y=slice(50,100))

```{code-cell} ipython3
sub = ds.xroms.subset(X=slice(20, 40), Y=slice(50, 100))  # with accessor
print(dict(sub.sizes))
```

Only whole, contiguous index ranges are allowed. A step other than 1 (`slice(0, 100, 2)`) raises an error, since staggered grids do not survive taking every other point; thin the subset afterwards if you want fewer points.

Calculations that use neighbours, such as derivatives, are one-sided on the points at the edge of a subset. To get the same values as on the full domain, ask for a halo of extra rho points around the subset with `halo=1`, calculate, and take the halo off again with {func}`xroms.trim`:

```{code-cell} ipython3
padded = ds.xroms.subset(X=slice(20, 40), Y=slice(50, 100), halo=1)
conv = xroms.trim(padded.xroms.convergence, 1)  # trim 1 point from each side again
print(dict(conv.sizes))
```

Compare with calculating on the whole domain and selecting afterwards, and with the same calculation on a subset without a halo:

```{code-cell} ipython3
full = ds.xroms.convergence.isel(eta_rho=slice(50, 100), xi_rho=slice(20, 40))
print(float(abs(conv - full).max()))  # with the halo: identical
without_halo = ds.xroms.subset(X=slice(20, 40), Y=slice(50, 100)).xroms.convergence
print(float(abs(without_halo - full).max()))  # without it: different at the edges
```

### Find nearest in lon/lat

This matters for a curvilinear grid.

Can't use `sel` because it will only search in one dimension for the nearest value and the dimensions are indices which are not necessarily geographic distance. Instead need to use a search for distance and use that for the `where` condition from the previous example. This functionality has been wrapped into {func}`xroms.sel2d` (and its partner function {func}`xroms.argsel2d`), which find the nearest grid point by the distance on the sphere (haversine). Use the accessor of the Dataset, which takes the name of a variable and uses the longitudes and latitudes of its horizontal position, or the accessor of a DataArray, which uses its own longitude and latitude coordinates:

```{code-cell} ipython3
lon0, lat0 = -91, 28
saltsel = ds.xroms.sel2d("salt", lon0, lat0)
print(dict(saltsel.sizes))
```

```{code-cell} ipython3
ds.salt.xroms.sel2d(lon0, lat0).identical(saltsel)
```

The result has the vertical and time dimensions and the horizontal ones are gone. Or, if you instead want the indices of the nearest grid node returned, you can call `argsel2d`, as `(eta_rho, xi_rho)` indices:

```{code-cell} ipython3
ds.xroms.argsel2d(lon0, lat0)
```

```{code-cell} ipython3
print(ds.salt.xroms.argsel2d(lon0, lat0))
print(ds.xroms.argsel2d(lon0, lat0, hcoord="u"))  # the nearest u point
print(xroms.argsel2d(ds.lon_rho, ds.lat_rho, lon0, lat0))  # as a function, with the coordinates
```

Several points at once, given as lists or arrays, are collected along a new `points` dimension:

```{code-cell} ipython3
print(dict(ds.xroms.sel2d("salt", [-91, -90.5], [28, 28.5]).sizes))
```

Check this function, just to be sure. The circle is the value that `sel2d` found, drawn at the longitude and latitude we asked for on top of the field around it: its color should match the cell it sits on.

```{code-cell} ipython3
dl = 0.05
box = (
    (ds.lon_rho > lon0 - dl) & (ds.lon_rho < lon0 + dl) & (ds.lat_rho > lat0 - dl) & (ds.lat_rho < lat0 + dl)
)
dss = ds.salt.where(box).isel(ocean_time=0, s_rho=-1)

vmin = float(dss.min())
vmax = float(dss.max())

plt.figure(figsize=(6, 4))
dss.plot(x="lon_rho", y="lat_rho", vmin=vmin, vmax=vmax)
plt.scatter(lon0, lat0, c=saltsel.isel(ocean_time=0, s_rho=-1), s=200, edgecolor="k", vmin=vmin, vmax=vmax)
plt.xlim(lon0 - dl, lon0 + dl)
plt.ylim(lat0 - dl, lat0 + dl);
```

### Longitudes

Grids store longitudes either in -180..180 or in 0..360, and a domain that crosses the prime meridian or the dateline is one contiguous span in only one of them. The box above compares longitudes as they are stored, so it needs longitudes in the convention of the grid. {func}`xroms.wrap_longitude` moves longitudes between the conventions (numbers, arrays and every longitude of a Dataset), and {func}`xroms.straddles` says whether the domain crosses the prime meridian.

```{code-cell} ipython3
print(xroms.straddles(ds))  # False: this domain does not cross the prime meridian
lon360 = xroms.wrap_longitude(ds.lon_rho, "0-360")
print(float(ds.lon_rho.min()), float(lon360.min()))  # the same longitude in the two conventions
print(xroms.wrap_longitude(lon, "0-360"))  # the limits of the box above, for a grid in 0..360
```

The nearest-point search uses the distance on the sphere, so it gives the same point in either convention:

```{code-cell} ipython3
ds360 = xroms.wrap_longitude(ds, "0-360")  # every longitude of the Dataset
print(ds360.xroms.argsel2d(lon0, lat0))
```

## Optional: `cf-xarray`

`xroms` does not use `cf-xarray`, but you can use it with ROMS output, to select by the axes `T`, `Z`, `Y` and `X` instead of by the names of the dimensions, when the Dataset has the attributes `cf-xarray` looks for. {func}`xroms.add_cf_attrs` returns a copy of a Dataset that has them (it only changes metadata, and it makes longitudes and latitudes coordinates). `cf-xarray` has to be installed and imported to get the `cf` accessor.

```{code-cell} ipython3
import cf_xarray

dscf = xroms.add_cf_attrs(ds)
dscf.salt.cf.axes
```

With `xarray` alone:

    ds.salt.isel(s_rho=-1, ocean_time=0)

With the `cf-xarray` accessor:

    dscf.salt.cf.isel(Z=-1, T=0)

and get the same thing back. Same for `sel`. The `T`, `Z`, `Y`, `X` names can be mixed and matched with the actual dimension names.

```{code-cell} ipython3
print(dscf.salt.cf.isel(Z=-1, T=0).dims)
print(dscf.salt.cf.sel(Z=0, method="nearest").dims)  # the level nearest to 0, the surface
print(dscf.salt.cf.sel(Z=0, T=date, method="nearest").dims)
```

The horizontal dimensions of a ROMS file have no values (they are just indices), so `cf-xarray` cannot find the axes `X` and `Y` yet (the output above lists only `Z` and `T`). If you want them, `index_coords=True` adds integer coordinates `0, 1, 2, ...` marked as axes:

```{code-cell} ipython3
dscf = xroms.add_cf_attrs(ds, index_coords=True)
print(dscf.salt.cf.axes)
a = dscf.salt.cf.isel(X=20, Y=10, Z=20, T=1)
b = ds.salt.isel(xi_rho=20, eta_rho=10, s_rho=20, ocean_time=1)
print(float(a), float(b))
```

You can always check what `cf-xarray` understands about a Dataset or DataArray by displaying its `cf` accessor:

    dscf.salt.cf
