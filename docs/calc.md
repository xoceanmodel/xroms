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
import xgcm
import numpy as np
import xroms
import matplotlib.pyplot as plt
```

# How to calculate with `xarray` and `xroms`

Here we demonstrate a number of calculations built into `xroms`, through accessors to `DataArrays` and `Datasets`.

## `xarray` Datasets

Use an `xarray` accessor in `xroms` to easily perform calculations with syntax

    ds.xroms.[method]

Importantly, the `xroms` accessor to a `Dataset` is initialized with an `xgcm` grid object (or you can input a previously-calculated grid object), stored at `ds.xroms.xgrid`, which is used to perform the basic grid calculations. More on this under "How to set up grid" below.

The built-in native calculations are properties of the `xroms` accessor and are not functions.

The accessor functions can take in the horizontal then vertical grid label you want the calculation to be on as options:

    ds.xroms.ddz('u', hcoord='rho', scoord='s_rho')  # to make sure result is on rho horizontal grid and s_rho vertical grid, a function

or

    ds.xroms.dudz  # return on native grid it is calculated on, a property

Other inputs are available for functions when the calculation involves a derivative and there is a choice for how to treat the boundary (`hboundary` and `hfill_value` for horizontal calculations and `sboundary` and `sfill_value` for vertical calculations). More information on those inputs can be found in the docs for `xgcm` such as under found under:

    ds.xroms.xgrid.interp?

## `xarray` DataArrays

A few of the more basic methods in `xroms` are available to `DataArrays` too. `xroms` methods for `DataArrays` require the grid object to be input:

    ds.temp.xroms.to_grid(xgrid, hcoord='psi', scoord='s_w')

+++

## Attributes

`xroms` provides attributes as metadata to track calculations, provide context, and to be used as indicators for plots.

The option to always keep attributes in `xarray` is turned on in the call to `xroms`.

## `cf-xarray`

Some functionality is added by using the package `cf-xarray`. Necessary attributes are added to datasets when the following call is run:

    ds, xgrid = xroms.roms_dataset(ds)

For example, when all CF Convention attributes are available in the Dataset, you can refer to dimensions and coordinates generically, regardless of the actual variable names.

* For dimensions:
  * ds.cf["T"], ds.cf["Z"], ds.cf["Y"], ds.cf["X"]
* For coordinates:
  * ds.cf["time], ds.cf["vertical"], ds.cf["latitude"], ds.cf["longitude"]

+++

### Load in data

More information on input/output in {doc}`input/output page <io>`. For model output available at <url>, you can find your dataset with and chunks according to the dataset itself (though if none are known by the dataset this will use no chunks):

    ds = xr.open_dataset(url, chunks={})

Also, an example ROMS dataset is available with `xroms` that we will read in for this tutorial.

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds
```

```{code-cell} ipython3
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)
```

```{code-cell} ipython3
# add grid to xrom accessor explicitly
ds.xroms.set_grid(xgrid)
```

```{code-cell} ipython3
ds.xroms.xgrid
```

## `xgcm` grid and extra ROMS coordinates

+++

### How to set up grid

The package `xcgm` has many nice grid functions for ROMS users, however, a bit of set up is required to connect from ROMS to the `xgcm` standardds. This grid set up does that.

The `grid` object contains metrics (X, Y, Z) with distances for each grid ('dx', 'dx_u', 'dx_v', 'dx_psi', and 'dz', 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v', 'dz_psi', 'dz_w_psi'), and all of these as grid coordinates too.

After setting up your Dataset, you should add coordinates and other information to the dataset and set up an `xgcm` grid object with:

    ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)

If you want to use the `xroms` accessor, add the grid object explicitly with:

    ds.xroms.set_grid(xgrid)

If you don't do this step, the first time the grid object is required it will be set up, though you can't choose which input flags to use in that case.

The xgcm grid object is then available at

    ds.xroms.xgrid


### Grid lengths

Distances between grid nodes on every ROMS grid can be calculated and set up in the `xgcm` grid object — some by default and some have to be requested by the user with optional flags.

* Horizontal grids:
 * inverse distances between nodes are given in an analogous way to distance (*i.e.*, `ds.pm` and `ds.pn_psi`)
 * distances between nodes are given in meters by dx's and dy's stored in ds, such as: `ds.dx` for the `rho` grid and `ds.dy_psi` for the `psi` grid, calculated from inverse distances
* Vertical grids:
 * There are lazily-evaluated z-coordinates for both `rho` and `w` vertical grids for each horizontal grid.
 * There are also arrays of z distances between nodes, called dz's, available for each combination of grids. For example, there is `ds.dz_u` for z distances on the `u` horizontal and `rho` vertical grid, and there is `ds.dz_w_v` for z distances on the `v` horizontal and `w` vertical grid. These are `[ocean_time x s_* x eta_* x xi_*]` arrays.
 * Arrays of z distances relative to a sea level of 0 are also available. They have analogous names to the previous entries but with "0" on the end. They are computationally faster to use because they do not vary in time. They are also less accurate for this reason but it depends on your use as to how much that matters.

+++

### Grid areas

* Horizontal
  * rho grid `ds.dA`, psi grid `ds.dA_psi`, u grid `ds.dA_u`, v grid `ds.dA_v`
* Vertical
  * These aren't built in but can easily be calculated. For example, for cell areas in the x direction on the rho horizontal and rho vertical grids: `ds.dx * ds.dz`.

+++

### Grid volumes

Time varying: All 8 combinations of 4 horizontal grids and 2 vertical grids are available if `include_cell_volume==True` in `roms_dataset()`, such as: `ds.dV` (rho horizontal, rho vertical), and `ds.dV_w_v` (w vertical, v horizontal).

A user can easily calculate the same but for time-constant dz's, for example as:

    ds['dV_w'] = ds.dx * ds.dy * ds.dz_w0  # w vertical, rho horizontal, constant in time

You can calculate the full domain volume in time with:

    ds.dV.sum(('s_rho', 'eta_rho', 'xi_rho'))

Or, using `cf-xarray` with:

    ds.dV.cf.sum(('Z', 'Y', 'X'))

```{code-cell} ipython3
ds.dV.cf.sum(('Z', 'Y', 'X'))  # with cf-xarray accessor
```

## Change grids

A ROMS user frequently needs to move between horizontal and vertical grids, so it is built into many of the function wrappers, but you can also do it as a separate function. It can also be done directly to `Datasets` with the `xroms` accessor. Here we change salinity from its default grids to be on the psi grid horizontally and the s_w grid vertically:

    ds.xroms.to_grid('salt', 'psi', 's_w')

You can also use the `xroms` function directly instead of using the `xarray` accessor if you prefer to have more options. Here is the equivalent call to the accessor, using the same defaults:

    xroms.to_grid(ds["salt"], xgrid,
                  hcoord="psi", scoord="s_w",
                  hboundary="extend", hfill_value=None,
                  sboundary="extend", sfill_value=None)

```{code-cell} ipython3
ds.xroms.to_grid('salt', 'psi', 's_w')
```

## Dimension ordering convention

By convention, ROMS DataArrays should be in the order ['T', 'Z', 'Y', 'X'], for however many of these dimensions they contain. The following function does this for you:

    xroms.order(ds.temp);  # function call

    ds.temp.xroms.order();  # accessor

```{code-cell} ipython3
ds.temp.xroms.order()  # accessor
```

## Basic computations

These are all functions, not properties.

+++

### `xarray`

Many [computations](http://xarray.pydata.org/en/stable/computation.html) are built into `xarray` itself. Often it is possible to input the dimension over which to perform a computation by name, such as:

    arr.sum(dim="xi_rho")

or

    arr.sum(dim=("xi_rho","eta_rho"))

Note that many basic `xarray` calculations should be used with caution when using with ROMS output, since a ROMS grid can be stretched both horizontally and vertically. When using these functions, consider if your calculation should account for variable grid cell distances, areas, or volumes. Additionally, it is straight-forward to use basic grid functions from `xarray` on a ROMS time dimension (resampling, differentiation, interpolation, etc), however, be careful before using these functions on spatial dimensions for the same reasons as before.

    ds.salt.mean(dim=("xi_rho","eta_rho"))

    # same call but using cf-xarray
    ds.salt.cf.mean(("Y","X"))

```{code-cell} ipython3
ds.salt.cf.mean(("Y","X"))
```

#### `xroms` grid-based metrics

Spatial metrics that account for the variable grid cell sizing in ROMS (both curvilinear horizontal and s vertical) are available by wrapping `xgcm` functions. These also have the additional benefit that the user can change grids and attributes are tracked. The available functions are:

* gridsum
* gridmean

Example usage:

    xroms.gridsum(ds.temp, xgrid, dim)  # function call

    ds['temp'].xroms.gridsum(xgrid, dim)  # accessor

where dimension names in the `xgcm` convention are 'Z', 'Y', or 'X'. `dim` can be a string, list, or tuple of combinations of these names for dimensions to average over.

+++

##### sum

```{code-cell} ipython3
uint = ds.u.xroms.gridsum(xgrid, "Z")
```

```{code-cell} ipython3
uint.xroms.gridsum(xgrid, 'Y')
```

##### mean

```{code-cell} ipython3
vint = ds.v.xroms.gridmean(xgrid, "Z")
vint.xroms.gridmean(xgrid, "Y")
```

## Derivatives

+++

### Vertical

Syntax is:

    ds.xroms.ddz("salt")  # accessor to dataset

    xroms.ddz(ds.salt, xgrid)  # No accessor

Other options:

    ds.xroms.ddz('salt', hcoord='psi', scoord='s_rho', sboundary='extend', sfill_value=np.nan);  # Dataset

    xroms.ddz(ds.salt, xgrid, hcoord='psi', scoord='s_rho', sboundary='extend', sfill_value=np.nan);  # No accessor

```{code-cell} ipython3
ds.xroms.ddz('salt')  # Dataset
```

### Horizontal

Syntax:

    ds.xroms.ddxi('u');  # horizontal xi-direction gradient (accessor)

    ds.xroms.ddeta('u');  #  horizontal eta-direction gradient (accessor)

    dtempdxi, dtempdeta = xroms.hgrad(ds.temp, xgrid)  # both gradients simultaneously, as function

    xroms.ddxi(ds.temp, xgrid)  # individual derivative, as function

    xroms.ddeta(ds.temp, xgrid)  # individual derivative, as function

```{code-cell} ipython3
ds.xroms.ddxi('u')  # horizontal xi-direction gradient
```

### Time

Use `xarray` directly for this.

```{code-cell} ipython3
ddt = ds.differentiate('ocean_time', datetime_unit='s')
ddt
```

## Built-in Physical Calculations

These are all properties of the accessor, so should be called without (). Demonstrated below are the calculations using the accessor and not using the accessor.

+++

### Horizontal speed

    ds.xroms.speed  # accessor

    xroms.speed(ds.u, ds.v, xgrid)  # function

```{code-cell} ipython3
ds.xroms.speed
```

### Kinetic energy

    ds.xroms.KE  # accessor

    # without the accessor you need to manage this yourself — first calculate speed to then calculate KE
    speed = xroms.speed(ds.u, ds.v, xgrid)
    xroms.KE(ds.rho0, speed)

```{code-cell} ipython3
ds.xroms.KE
```

### Geostrophic velocities

    ds.xroms.ug  # accessor, u component
    ds.xroms.vg  # accessor, v component

    ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)  # function

```{code-cell} ipython3
ds.xroms.ug
```

### Eddy kinetic energy (EKE)

    ds.xroms.EKE  # accessor

    ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)
    xroms.EKE(ug, vg, xgrid)

```{code-cell} ipython3
ds.xroms.EKE
```

### Vertical shear

Since it is a common use case, there are specific methods to return the u and v components of vertical shear on their own grids. These are just available for Datasets.

    ds.xroms.dudz
    ds.xroms.dvdz

    xroms.dudz(ds.u, xgrid)
    xroms.dvdz(ds.v, xgrid)

    # already on same grid:
    ds.xroms.vertical_shear

    dudz = xroms.dudz(ds.u, xgrid)
    dvdz = xroms.dvdz(ds.v, xgrid)
    xroms.vertical_shear(dudz, dvdz, xgrid)

```{code-cell} ipython3
ds.xroms.dudz
```

The magnitude of the vertical shear is also a built-in derived variable for the `xroms` accessor:

```{code-cell} ipython3
ds.xroms.vertical_shear
```

### Vertical vorticity

    ds.xroms.vort

    xroms.relative_vorticity(ds.u, ds.v, xgrid)

```{code-cell} ipython3
ds.xroms.vort
```

### Horizontal divergence

Horizontal component of the currents divergence.

    ds.xroms.div

    xroms.divergence(ds.u, ds.v, xgrid)

```{code-cell} ipython3
ds.xroms.div
```

### Normalized surface divergence

Horizontal component of the currents divergence at the surface, normalized by $f$. This is only available through the accessor.

    ds.xroms.div_norm

```{code-cell} ipython3
ds.xroms.div_norm
```

### Ertel potential vorticity

The accessor assumes you want the Ertel potential vorticity of the buoyancy:

    ds.xroms.ertel

    sig0 = xroms.potential_density(ds.temp, ds.salt)
    buoyancy = xroms.buoyancy(sig0, rho0=ds.rho0)
    xroms.ertel(buoyancy, ds.u, ds.v, ds.f, xgrid, scoord='s_w')

Alternatively, the user can access the original function and use a different tracer for this calculation (in this example, "dye_01"), and can return the result on a different vertical grid, for example:

    xroms.ertel(ds.dye_01, ds.u, ds.v, ds.f, xgrid, scoord='s_w')

```{code-cell} ipython3
ds.xroms.ertel
```

### Density

    ds.xroms.rho

    xroms.density(ds.temp, ds.salt)

```{code-cell} ipython3
ds.xroms.rho
```

### Potential density

    ds.xroms.sig0

    xroms.potential_density(ds.temp, ds.salt)

```{code-cell} ipython3
ds.xroms.sig0
```

### Buoyancy

    ds.xroms.buoyancy

    sig0 = xroms.potential_density(ds.temp, ds.salt);
    xroms.buoyancy(sig0)

```{code-cell} ipython3
ds.xroms.buoyancy
```

### Buoyancy frequency

Also called vertical buoyancy gradient.

    ds.xroms.N2

    rho = xroms.density(ds.temp, ds.salt)  # calculate rho if not in output
    xroms.N2(rho, xgrid)

```{code-cell} ipython3
ds.xroms.N2
```

### Horizontal buoyancy gradient

    ds.xroms.M2

    rho = xroms.density(ds.temp, ds.salt)  # calculate rho if not in output
    xroms.M2(rho, xgrid)

```{code-cell} ipython3
ds.xroms.M2
```

### Mixed layer depth

This is not a property since the threshold is a parameter and needs to be input.

    ds.xroms.mld(thresh=0.03)

    sig0 = xroms.potential_density(ds.temp, ds.salt);
    xroms.mld(sig0, xgrid, ds.h, ds.mask_rho, thresh=0.03)

```{code-cell} ipython3
ds.xroms.mld(thresh=0.03)
```

## Other calculations

### Rotations

If your ROMS grid is curvilinear, you'll need to rotate your u and v velocities from along the grid axes to being eastward and northward. You can do this with


    ds.xroms.east

    ds.xroms.north

Additionally, if you want to rotate your velocity to be a different orientation, for example to be along-channel, you can do that with

    ds.xroms.east_rotated(angle)

    ds.xroms.north_rotated(angle)


## Time-based calculations including climatologies

+++

### Rolling averages in time

Here is an example of computing a rolling average in time. Nothing happens in this example because we only have two time steps to use, however, it does demonstrate the syntax. If more time steps were available we would update `ds.salt.rolling(ocean_time=1)` to include more time steps to average over in a rolling sense.

More information about rolling operations [is available](http://xarray.pydata.org/en/stable/computation.html#rolling-window-operations).

```{code-cell} ipython3
roll = ds.salt.rolling(ocean_time=1, center=True, min_periods=1).mean()
roll.isel(s_rho=-1, eta_rho=10, xi_rho=20).plot(alpha=0.5, lw=2)
ds.salt.isel(s_rho=-1, eta_rho=10, xi_rho=20).plot(ls=":", lw=2)
```

### Resampling in time

Can't have any chunks in the time dimension to do this. More info: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html

+++

#### Upsample

Upsample to a higher resolution in time. Makes sense to interpolate to fill in data when upsampling, but can also forward or backfill, or just add nan's.

```{code-cell} ipython3
dstest = ds.resample(ocean_time='30min', restore_coord_dims=True).interpolate()
```

Plot to visually inspect results

```{code-cell} ipython3
ds.salt.cf.isel(Y=30, X=20, Z=-1).plot(marker='o')
dstest.salt.cf.isel(Y=30, X=20, Z=-1).plot(marker='x')
```

#### Downsample

Resample down to lower resolution in time. This requires appending a method to aggregate the extra data, such as a `mean`. Note that other options can be used to shift the result within the interval of aggregation in various ways. Just the syntax is shown here since we only have two time steps to work with.

    dstest = ds.resample(ocean_time='6H').mean()
    ds.salt.cf.isel(Y=30, X=20, Z=-1).plot(marker='o')
    dstest.salt.cf.isel(Y=30, X=20, Z=-1).plot(marker='x')

+++

#### Seasonal average, over time

This is an example of [resampling](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html).

    da.cf.resample({'T': [time frequency string]}).reduce([aggregation function])

For example, calculate the mean temperature every quarter in time with the following:

    ds.temp.cf.resample({'T': 'QS'}).reduce(np.mean)

or the aggregation function can be appended on the end directly with:

    ds.temp.cf.resample({'T': 'QS'}).mean()

The result of this calculation is a time series of downsampled chunks of output in time, the frequency of which is selected by input "time frequency string", and aggregated by input "aggregation function".

Examples of the time frequency string are:
* "QS": quarters, starting in January of each year and averaging three months.
  * Also available are selections like "QS-DEC", quarters but starting with December to better align with seasons. Other months are input options as well.
* "MS": monthly
* "D": daily
* "H": hourly
* Many more options are given [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

Examples of aggregation functions are:
* np.mean
* np.max
* np.min
* np.sum
* np.std

Result of downsampling a 4D salt array from hourly to 6-hourly, for example, gives: `[ocean_time x s_rho x eta_rho x xi_rho]`, where `ocean_time` has about 1/6 of the number of entries reflecting the aggregation in time.

```{code-cell} ipython3
ds.temp.cf.resample(indexer={'T': '6H'}).reduce(np.mean)
```

### Seasonal mean over all available time

This is how to average over the full dataset period by certain time groupings using xarray `groupby` which is like pandas version. In this case we show the seasonal mean averaged across the full model time period. The syntax for this is:

    da.salt.cf.groupby('T.[time string]').reduce([aggregation function])

For example, to average salt by season:

    da.salt.cf.groupby('T.season').reduce(np.mean)

or

    da.salt.cf.groupby('T.season').mean()

Options for the time string include:
* 'season'
* 'year'
* 'month'
* 'day'
* 'hour'
* 'minute'
* 'second'
* 'dayofyear'
* 'week'
* 'dayofweek'
* 'weekday'
* 'quarter'

More information about options for time (including "derived" datetime coordinates) is [here](https://xarray.pydata.org/en/v0.16.0/time-series.html#datetime-components).

Examples of aggregation functions are:
* np.mean
* np.max
* np.min
* np.sum
* np.std

Result of averaging over seasons for a 4D salt array returns, for example: `[season x s_rho x eta_rho x xi_rho]`, where `season` has 4 entries, each covering 3 months of the year.

```{code-cell} ipython3
# this example has only 1 season because it is a short example file
ds.temp.cf.groupby('T.season').mean()
```
