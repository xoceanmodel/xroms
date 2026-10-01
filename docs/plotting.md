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
import xroms
import matplotlib.pyplot as plt
import cartopy
import numpy as np
import cmocean.cm as cmo
```

```{code-cell} ipython3
:tags: [remove-cell]

# not shown: cartopy projects the land and state polygons with shapely, which
# raises a harmless RuntimeWarning with some versions of shapely and numpy
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in create_collection", category=RuntimeWarning)
```

# How to plot

This notebook demonstrates how to plot ROMS model output from a planview (x-y) and an x-z cross-section. Static approaches are shown, with `xarray` and with `Matplotlib`; for interactive plots see the end. The `cartopy` package is used for managing projections for mapview plots, which also gives many options for input (some shown below).

+++

## Load in data

Load in example dataset. More information in the {doc}`input/output page <io>`. There is no setup step: xarray opens the file and its variables are ready to plot.

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds.sizes
```

## Setup for plotting

Use `cartopy` when plotting with a projection and/or wanting to add context like coastline. `proj` is the projection of the map, and `pc` (plate carree) is plain longitude/latitude, the coordinates of the model output.

```{code-cell} ipython3
proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
pc = cartopy.crs.PlateCarree()
```

## Static: `xarray`

Can plot directly in `xarray` since it has wrapped many `Matplotlib` plotting routines. This is great for quick plots and continues to improve, but if you need more control then try the next section of plotting directly in `Matplotlib`.

Select with the dimension names of the data, as in `isel(ocean_time=0, s_rho=-1)` for the first time and the surface (the last `s_rho`). The colormaps are from the [`cmocean`](https://matplotlib.org/cmocean/) package, which has colormaps designed for oceanographic variables: for example `cmo.haline` for salinity, `cmo.thermal` for temperature and `cmo.delta` for variables that are positive and negative, like velocity.

+++

### Map view

+++

#### Overview

Plotted with dimension indices instead of coordinates:

```{code-cell} ipython3
ds.v.isel(ocean_time=0, s_rho=-1).plot(cmap=cmo.delta)
plt.show()
```

Plotted with coordinates lon/lat. Each variable is on its own horizontal points, which have their own coordinates: `lon_rho` and `lat_rho` for rho points (`temp`, `salt`, `zeta`, ...), `lon_u` and `lat_u` for u points and `lon_v` and `lat_v` for v points, so here v is plotted against `lon_v` and `lat_v`.

```{code-cell} ipython3
ds.v.isel(ocean_time=0, s_rho=-1).plot(x="lon_v", y="lat_v", cmap=cmo.delta)
plt.show()
```

These are the locations of the cell centers. `xarray` and `Matplotlib` work out the cell corners from them, so the corners (vertices) of the cells are no longer needed as separate coordinates (see `shading="auto"` below).

#### Magnified

A subset of the grid, selected by index:

```{code-cell} ipython3
ds.salt.isel(ocean_time=0, s_rho=-1, xi_rho=slice(100, 300), eta_rho=slice(75, 100)).plot(cmap=cmo.haline)
plt.show()
```

```{code-cell} ipython3
ds.salt.isel(ocean_time=0, s_rho=-1, xi_rho=slice(100, 300), eta_rho=slice(75, 100)).plot(x="lon_rho", y="lat_rho", cmap=cmo.haline)
plt.show()
```

### Cross-section

A cross-section needs a vertical coordinate to plot against. The vertical coordinate of ROMS follows the terrain, so the depth of a model level is different at every location and time. There are two ways to attach the depths to the data as a coordinate.

`ds.xroms.assign_z()` returns a **new** Dataset (`ds` is not changed) with the depths of the rho levels, `z_rho`, and of the w levels, `z_w`, as coordinates. They are heights relative to mean sea level, negative below it. Use `hcoord="u"` or `"v"` for variables on u or v points (the names then end with `_u` or `_v`). The depths are a snapshot, so call `assign_z` again if you change `zeta` or `h`; they are lazy with dask-backed data, as here, but are calculated immediately for data in memory.

Here we plot u-velocity along a line of constant `xi_u`, so against latitude. u is on u points, so ask for their depths:

```{code-cell} ipython3
dsz = ds.xroms.assign_z(hcoord="u")
dsz.z_rho_u.dims
```

The depths come from `zeta`, which is NaN over land, so they are NaN there too, and `pcolormesh` needs finite coordinates. When a cross-section includes land, as this one does, drop the points that are all NaN first:

```{code-cell} ipython3
u = dsz.u.isel(xi_u=200, ocean_time=0).dropna("eta_u", how="all")
u.plot(x="lat_u", y="z_rho_u", cmap=cmo.delta, figsize=(10, 5))
plt.show()
```

The other way is to attach the depths to a single variable, with `ds.xroms.z()` for rho points or `ds.xroms.z("u")` or `ds.xroms.z("v")` for u or v points (see {func}`xroms.z`). For example, temperature along a line of constant `eta_rho`, which does not cross land, against longitude:

```{code-cell} ipython3
temp = ds.temp.assign_coords(z=ds.xroms.z()).isel(eta_rho=20, ocean_time=0)
temp.plot(x="lon_rho", y="z", cmap=cmo.thermal, figsize=(10, 5))
plt.show()
```

## Static: `Matplotlib`

+++

### Map view

+++

#### Overview

Here is a basic plan-view map, using `cartopy` for projection handling. You can add many different types of natural data for context. Shown here are land, coastline and state borders. You can control the resolution of the data by changing the input to `with_scale` (options are `"10m"`, `"50m"`, or `"110m"`, corresponding to 1:10,000,000, 1:50,000,000, and 1:110,000,000 scale). The data come from [Natural Earth](https://www.naturalearthdata.com/), which `cartopy` downloads the first time it needs each file. More `cartopy` feature information available [here](https://scitools.org.uk/cartopy/docs/latest/matplotlib/feature_interface.html).

The model output is added with `ax.pcolormesh`, giving the longitudes and latitudes of the cell centers and `transform=pc` to say that they are longitude and latitude. With `shading="auto"`, `Matplotlib` finds the corners of the cells itself.

```{code-cell} ipython3
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=proj)

# Add natural features
ax.add_feature(cartopy.feature.LAND.with_scale("50m"), facecolor="0.8")
ax.add_feature(cartopy.feature.COASTLINE.with_scale("10m"), edgecolor="0.2")
ax.add_feature(cartopy.feature.STATES.with_scale("10m"), edgecolor="k")

gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, xlocs=np.arange(-104, -80, 2))

# manipulate `gridliner` object to change locations of labels
gl.top_labels = False
gl.right_labels = False

salt = ds.salt.isel(ocean_time=0, s_rho=-1)
mesh = ax.pcolormesh(salt.lon_rho, salt.lat_rho, salt, transform=pc, cmap=cmo.haline, shading="auto")
fig.colorbar(mesh, ax=ax, label="salinity")
plt.show()
```

#### Magnified

Use `set_extent` to narrow the view to magnify a subregion. A finer resolution of the features is worthwhile when magnified. The `xarray` plot also draws on a `Matplotlib` axes, so it can add the output here too.

```{code-cell} ipython3
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=proj)

ax.set_extent([-94, -90, 27.5, 30], crs=pc)
ax.add_feature(cartopy.feature.LAND.with_scale("10m"), facecolor="0.8")
ax.add_feature(cartopy.feature.COASTLINE.with_scale("10m"), edgecolor="0.2")
ax.add_feature(cartopy.feature.STATES.with_scale("10m"), edgecolor="k")
gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, xlocs=np.arange(-104, -80, 2))

# manipulate `gridliner` object to change locations of labels
gl.top_labels = False
gl.right_labels = False

ds.salt.isel(ocean_time=0, s_rho=-1).plot(ax=ax, x="lon_rho", y="lat_rho", transform=pc, cmap=cmo.haline)
plt.show()
```

### Cross-section

A cross-section has no projection to manage, so it is plotted as in the `xarray` section above, with the depths attached as a coordinate. It is on a `Matplotlib` axes, so it is easy to add to. Here, contours of potential density (`ds.xroms.sig0`, minus 1000 for the more familiar sigma0) are drawn on the temperature section.

```{code-cell} ipython3
z = ds.xroms.z()
temp = ds.temp.assign_coords(z=z).isel(eta_rho=20, ocean_time=0)
sigma0 = (ds.xroms.sig0 - 1000).assign_coords(z=z).isel(eta_rho=20, ocean_time=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
temp.plot(ax=ax, x="lon_rho", y="z", cmap=cmo.thermal)
contours = sigma0.plot.contour(ax=ax, x="lon_rho", y="z", levels=[22, 24, 25, 26, 27], colors="k", linewidths=0.8)
ax.clabel(contours, fmt="%g")
plt.show()
```

## Interactive

For interactive plots, with zoom, pan and widgets to vary over time or depth, send the same `xarray` objects to [`hvplot`](https://hvplot.holoviz.org/).
