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
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xroms
```

# How to calculate with `xarray` and `xroms`

Here we demonstrate a number of calculations built into `xroms`, through the accessor to `Datasets` and as functions. Everything is calculated from the variables of your Dataset when you ask for it: there is nothing to set up and nothing is stored (see the {doc}`input/output page <io>`).

## Load in data

More information on input/output in the {doc}`input/output page <io>`. Your own output opens with `xarray`, for example with the chunks stored in the file:

    ds = xr.open_dataset("ocean_his.nc", chunks={})

Also, an example ROMS dataset is available with `xroms` that we will read in for this tutorial. It is Rutgers ROMS output, so it has the dimensions `eta_u`, `xi_v` and so on for the other positions of the grid, and it has the grid in the same file. (If your grid is kept in a file apart from the output, merge it into the Dataset or pass it as `grid=`, see the {doc}`input/output page <io>`.)

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
print(dict(ds.sizes))
print(list(ds.data_vars))
```

## `xarray` Datasets

Use an `xarray` accessor in `xroms` to easily perform calculations with syntax

    ds.xroms.[method]

The accessor holds nothing but a reference to your Dataset: no grid object and no cache. Each call looks up what it needs in the Dataset (`pm`, `pn`, `h`, `zeta`, the s-coordinate parameters, ...) and calculates from that, so what you get always matches the Dataset as it is now. The methods take variable names (or DataArrays) and return results in the naming of your Dataset's own dimensions, with the coordinates of your Dataset.

The built-in native calculations are properties of the `xroms` accessor and are not functions (`ds.xroms.speed`, `ds.xroms.dudz`), and the methods that need input are functions (`ds.xroms.ddz("u")`).

The accessor functions can take in the horizontal then vertical grid label you want the calculation to be on as options (`hcoord`: `rho`, `u`, `v` or `psi`; `scoord`: `s_rho` or `s_w`):

```{code-cell} ipython3
# result on the rho horizontal grid and the s_rho vertical grid
ds.xroms.ddz("u", hcoord="rho", scoord="s_rho").dims
```

Other inputs are available for functions when the calculation involves a derivative and there is a choice for how to treat the boundary (`hboundary` and `hfill_value` for horizontal calculations and `sboundary` and `sfill_value` for vertical calculations), and for the s-coordinates, how to choose the free surface that the depths follow (`zeta`). See "Derivatives" below.

## `xarray` DataArrays

A few of the more basic methods in `xroms` are available to `DataArrays` too: those that need nothing but the DataArray itself, such as moving between grids (`to_grid`, `to_rho`, `to_u`, `to_v`, `to_psi`, `to_s_rho`, `to_s_w`), `order`, `sel2d`, `argsel2d`, `isoslice` and `interpll`.

```{code-cell} ipython3
ds.temp.xroms.to_grid(hcoord="psi", scoord="s_w").dims
```

Calculations that need grid variables (derivatives, grid sums and means, interpolation to depths) are only available through the Dataset (`ds.xroms.ddz("temp")`) or as a function that gets the Dataset (`xroms.ddz(ds.temp, ds)`). The DataArray methods of older versions of `xroms` for those raise an error that says what to use instead.

+++

## Attributes

`xroms` provides attributes as metadata to track calculations, provide context, and to be used as indicators for plots. The attributes of a result are its `name`, `long_name` and `units` and, for vertical positions, the CF `standard_name` and `positive`.

```{code-cell} ipython3
print(ds.xroms.speed.attrs)
print(ds.xroms.z_rho.attrs)
```

What `xroms` does not do any more is to change how `xarray` treats attributes. In `xroms` 0.6, importing the package switched on the `xarray` option `keep_attrs` for the whole session. Now your own arithmetic follows the defaults of `xarray`, which depend on your version of `xarray`: recent versions keep attributes through almost every operation (2026.4 does), older versions drop them in arithmetic and reductions (2025.7 does). Keeping attributes is not always right, either: the units of `speed**2` are not `m/s`. You can choose for a bit of code, or for your whole session:

```{code-cell} ipython3
speed = ds.xroms.speed
print(xr.get_options()["keep_attrs"])  # xroms leaves it at "default"
print((0.5 * speed**2).attrs)  # what your version of xarray does by default
with xr.set_options(keep_attrs=False):
    print((0.5 * speed**2).attrs)  # without attributes
with xr.set_options(keep_attrs=True):
    print((0.5 * speed**2).attrs)  # with the attributes of speed, units and all
```

    xr.set_options(keep_attrs=True)  # for the rest of the session

## Grid metrics on demand

There is no grid object to set up. The lengths, areas, volumes and depths of the grid are calculated from the variables of the Dataset when you ask for them (lazily, if the Dataset has `dask` arrays), at any of the positions of the grid.

### Grid lengths and areas

* Horizontal grids:
  * the spacing between nodes in meters, `ds.xroms.dx(hcoord)` along xi and `ds.xroms.dy(hcoord)` along eta, is `1/pm` and `1/pn` (the inverse distances stored in the model output), averaged to the position `hcoord` (`rho` by default).
  * cell areas in square meters, `ds.xroms.dA(hcoord)`, are `dx * dy`.
* Vertical grids:
  * layer thicknesses (`ds.xroms.dz(hcoord, scoord)`, positive and in meters) at the rho levels `s_rho` and at the w levels `s_w`.
  * heights (`ds.xroms.z_rho`, `ds.xroms.z_w`, and `ds.xroms.z(hcoord, scoord)`), described below.

```{code-cell} ipython3
print(ds.xroms.dx().dims, ds.xroms.dx().attrs["units"])
print(ds.xroms.dy("v").dims)  # spacing along eta at v points
print(ds.xroms.dA("psi").dims)  # cell areas at psi points
```

The thicknesses follow the free surface `zeta`, so they have a time dimension. With `zeta=0` they are those of the ocean at rest, which do not vary in time:

```{code-cell} ipython3
print(ds.xroms.dz().dims)
print(ds.xroms.dz(zeta=0).dims)
print(ds.xroms.dz(scoord="s_w").dims)  # at the w levels
```

### Depths

The vertical position `z` is a height relative to mean sea level, so it is negative in the water. `ds.xroms.z_rho` and `ds.xroms.z_w` are the heights at the rho points on the rho levels and on the w levels (the layer interfaces). `ds.xroms.z(hcoord, scoord)` gives them at any horizontal position and level, and takes options for `zeta` (as above), `positive="down"` to get depths, and `reference="surface"` or `"bottom"` to measure from the moving free surface or from the sea floor (`mean_sea_level` is the default).

```{code-cell} ipython3
print(ds.xroms.z_rho.dims)
print(ds.xroms.z("u", "s_w").dims)  # at u points and w levels
depth = ds.xroms.z(positive="down", reference="surface")  # depth below the moving free surface
print(depth.attrs["positive"], depth.attrs["standard_name"])
```

To attach the depths to a Dataset as coordinates, see `ds.xroms.assign_z()` on the {doc}`input/output page <io>`.

The mean spacing of the grid is in {func}`xroms.nominal_resolution` (in meters; with `units="degrees"` it is in degrees of longitude at the latitude `lat`, by default the middle of the domain):

```{code-cell} ipython3
print(xroms.nominal_resolution(ds))
print(xroms.nominal_resolution(ds, units="degrees"))
```

### Grid volumes

Time varying: cell volumes are `dz * dA`, `ds.xroms.dV(hcoord, scoord)`, for any of the 4 horizontal positions and 2 vertical positions. You can calculate the full domain volume in time with:

    ds.xroms.dV().sum(("s_rho", "eta_rho", "xi_rho"))

The free surface of this example is NaN over land, so that is also where the cell volumes are NaN, and the sum is the volume of water. If your `zeta` has values on land, mask them first, for example with `ds.mask_rho`.

```{code-cell} ipython3
ds.xroms.dV().sum(("s_rho", "eta_rho", "xi_rho")).values
```

A volume that does not vary in time, for the ocean at rest, is `ds.xroms.dV(zeta=0)`.

### `xgcm` grid for your own calculations

`ds.xroms.xgcm_grid()` returns a new `xgcm` grid object, for when you want to do your own calculations with `xgcm`. `xroms` itself does not store or use one. It works with the canonical dimension names (see "Change grids" below), so use it with `xroms.canonicalize(ds)`. It has the axes `X`, `Y` and `Z` and the horizontal metrics at rho, u, v and psi points: `dx` (`("X",)`), `dy` (`("Y",)`) and the cell area `dA` (`("X", "Y")`). That is enough for `g.average(da, ["X", "Y"])`, an average weighted by cell area, or `g.derivative(da, "X")`. The option `vertical_metrics=True` adds the layer thicknesses for `g.integrate(da, "Z")`.

```{code-cell} ipython3
g = ds.xroms.xgcm_grid()
dsc = xroms.canonicalize(ds)  # the same Dataset, with the canonical dimension names
g.average(dsc.temp.isel(s_rho=-1), ["X", "Y"]).values  # mean surface temperature, weighted by cell area (same as gridmean below)
```

## Change grids

A ROMS user frequently needs to move between horizontal and vertical grids, so it is built into many of the function wrappers, but you can also do it as a separate function. It can also be done directly to `Datasets` with the `xroms` accessor. Here we change salinity from its default grids to be on the psi grid horizontally and the s_w grid vertically:

    ds.xroms.to_grid('salt', 'psi', 's_w')

You can also use the `xroms` function directly instead of using the `xarray` accessor if you prefer to have more options. No grid is needed to move between grids, since it only averages neighboring points. Here is the equivalent call to the accessor, using the same defaults:

    xroms.to_grid(ds["salt"], hcoord="psi", scoord="s_w",
                  hboundary="extend", hfill_value=np.nan,
                  sboundary="extend", sfill_value=np.nan)

The functions `xroms.to_rho`, `xroms.to_u`, `xroms.to_v`, `xroms.to_psi`, `xroms.to_s_rho` and `xroms.to_s_w` move to one position, and `hboundary` and `sboundary` say what to do at the edges ("extend" copies the nearest value; "fill" uses `hfill_value` or `sfill_value`).

```{code-cell} ipython3
ds.xroms.to_grid("salt", "psi", "s_w").dims
```

The example file has no psi dimensions of its own, so the result has those that psi points share with v points along eta (`eta_v`) and with u points along xi (`xi_u`).

### Names of dimensions

The functions of `xroms` always return the **canonical** dimensions: `(eta_rho, xi_rho)` for rho points, `(eta_rho, xi_u)` for u points, `(eta_v, xi_rho)` for v points and `(eta_v, xi_u)` for psi points, `s_rho` and `s_w` for the vertical. The accessor returns results in the dimensions of your Dataset instead. For this Rutgers ROMS file, u points have the dimensions `(eta_u, xi_u)`, so the same calculation has different dimension names when it is done with a function and with the accessor:

```{code-cell} ipython3
print(ds.u.dims)  # as in the file
print(xroms.to_u(ds.temp).dims)  # function: canonical dimensions
print(ds.xroms.to_grid("temp", "u").dims)  # accessor: the dimensions of the Dataset
```

If you combine the result of a function with variables of your Dataset that use other names, `xarray` does not know that `eta_u` and `eta_rho` are the same positions, and it broadcasts them against each other to a new, extra dimension:

```{code-cell} ipython3
print((ds.u - xroms.to_u(ds.temp)).dims)  # eta_u and eta_rho are both there
```

There are three ways to avoid it: use the accessor; use `xroms.canonicalize(ds)` once, which renames the dimensions of a Dataset to the canonical ones (only the names change); or, with a single result, `xroms.rename_like(result, ds)`, which gives it the dimensions of `ds`.

```{code-cell} ipython3
dsc = xroms.canonicalize(ds)
print(dsc.u.dims)
print((dsc.u - xroms.to_u(dsc.temp)).dims)  # as it should be
print((ds.u - xroms.rename_like(xroms.to_u(ds.temp), ds)).dims)  # also
```

## Dimension ordering convention

By convention, ROMS DataArrays should be in the order ['T', 'Z', 'Y', 'X'], for however many of these dimensions they contain, and the results of `xroms` are. The following function does this for you, for any other DataArray (other dimensions go last):

    xroms.order(ds.temp);  # function call

    ds.temp.xroms.order();  # accessor

```{code-cell} ipython3
scrambled = ds.temp.transpose("xi_rho", "eta_rho", "s_rho", "ocean_time")
print(scrambled.dims)
print(scrambled.xroms.order().dims)  # accessor
```

## Basic computations

### `xarray`

Many [computations](https://docs.xarray.dev/en/stable/computation.html) are built into `xarray` itself. Often it is possible to input the dimension over which to perform a computation by name, such as:

    arr.sum(dim="xi_rho")

or

    arr.sum(dim=("xi_rho","eta_rho"))

Note that many basic `xarray` calculations should be used with caution when using with ROMS output, since a ROMS grid can be stretched both horizontally and vertically. When using these functions, consider if your calculation should account for variable grid cell distances, areas, or volumes. Additionally, it is straight-forward to use basic grid functions from `xarray` on a ROMS time dimension (resampling, differentiation, interpolation, etc), however, be careful before using these functions on spatial dimensions for the same reasons as before.

```{code-cell} ipython3
ds.salt.mean(dim=("xi_rho", "eta_rho")).dims
```

### `xroms` grid-based metrics

Spatial metrics that account for the variable grid cell sizing in ROMS (both curvilinear horizontal and s vertical) are available as `gridsum` and `gridmean`. They multiply by the grid spacing at the position of the variable (`dx`, `dy` and/or `dz`) before summing, and for the mean, points without data (NaN, as on land) carry no weight. The result keeps the attributes of the variable, with a `long_name` that says what was summed or averaged; for a sum, the `units` gain a metre for each dimension summed over. The available functions are:

* gridsum
* gridmean

Example usage:

    xroms.gridsum(ds.temp, ds, dims)  # function call

    ds.xroms.gridsum("temp", dims)  # accessor

where `dims` is a string, list or tuple of the axes 'Z', 'Y' and 'X', or of the names of dimensions (such as 's_rho', 'eta_u' and 'xi_rho') to sum or average over.

Here, the mean temperature at the surface across the domain, from the plain `xarray` mean (each cell counts the same) and weighted by cell area, with the axes or with the names of the dimensions:

```{code-cell} ipython3
surface_temp = ds.temp.isel(s_rho=-1)
print(surface_temp.mean(("eta_rho", "xi_rho")).values)
print(ds.xroms.gridmean(surface_temp, ("X", "Y")).values)
print(ds.xroms.gridmean(surface_temp, ("eta_rho", "xi_rho")).values)
```

+++

#### sum

For example, u integrated over the water column (`Z`) is the transport per width, and integrating that along eta (`Y`) as well gives a transport:

```{code-cell} ipython3
uint = ds.xroms.gridsum("u", "Z")
print(uint.dims, uint.attrs["units"])
print(ds.xroms.gridsum(uint, "Y").dims)  # the result of one calculation can go into the next
print(ds.xroms.gridsum("u", ("Z", "Y")).dims)  # or do both at once
```

#### mean

The mean of v over the water column, weighted by the thickness of the layers, and then over `Y` as well. Land carries no weight.

```{code-cell} ipython3
vint = ds.xroms.gridmean("v", "Z")
print(vint.dims)
print(ds.xroms.gridmean(vint, "Y").dims)
print(ds.xroms.gridmean("v", ("Z", "Y")).dims)  # all at once
```

### Depth average

`ds.xroms.depth_average("temp")` is the thickness-weighted mean over the water column, for a variable on the rho levels. With `shallow` and `deep`, it is the mean over a band of depths: both are depths in meters, positive down, measured from `reference`, which is the mean sea level unless it is `"surface"` (the moving free surface). Where there is no water in the band (land), the result is NaN.

    ds.xroms.depth_average("temp")  # the whole water column

    xroms.depth_average(ds.temp, ds, shallow=0, deep=10, reference="surface")  # function

```{code-cell} ipython3
# the mean of the upper 10 m below the free surface
upper10 = ds.xroms.depth_average("temp", shallow=0, deep=10, reference="surface")
print(ds.xroms.depth_average("temp").dims)
print(upper10.dims)
```

## Derivatives

### Vertical

Syntax is:

    ds.xroms.ddz("salt")  # accessor to dataset

    xroms.ddz(ds.salt, ds)  # No accessor, with the Dataset that has the grid variables

Other options:

    ds.xroms.ddz('salt', hcoord='psi', scoord='s_rho', sboundary='extend', sfill_value=np.nan);  # Dataset

    xroms.ddz(ds.salt, ds, hcoord='psi', scoord='s_rho', sboundary='extend', sfill_value=np.nan);  # No accessor

The result lands where the difference of neighboring levels does: on the other vertical grid, from the rho levels `s_rho` to the w levels `s_w` or the other way around. With `scoord`, you can have it on the levels of the input (a second order calculation on those levels) or anywhere else, and with `hcoord` on another horizontal grid.

```{code-cell} ipython3
print(ds.xroms.ddz("salt").dims)
print(ds.xroms.ddz("salt", scoord="s_rho").dims)
```

### Horizontal

Syntax:

    ds.xroms.ddxi('u');  # horizontal xi-direction gradient (accessor)

    ds.xroms.ddeta('u');  #  horizontal eta-direction gradient (accessor)

    dtempdxi, dtempdeta = ds.xroms.hgrad("temp")  # both gradients simultaneously, as accessor

    dtempdxi, dtempdeta = xroms.hgrad(ds.temp, ds)  # both gradients simultaneously, as function

    xroms.ddxi(ds.temp, ds)  # individual derivative, as function

    xroms.ddeta(ds.temp, ds)  # individual derivative, as function

These are derivatives at constant depth, as you probably want, and not along the terrain-following levels: where the levels are sloped, they are corrected with the vertical derivative. The result keeps the vertical levels of the input, there are no zeros at the top and bottom. In the horizontal, it lands one position over, where the difference of neighbors is: a derivative along xi of a field on rho points is on u points, and one of a field on u points is on rho points.

```{code-cell} ipython3
for name in ["temp", "u", "v"]:
    print(
        name, ds[name].dims[-2:],
        "d/dxi:", ds.xroms.ddxi(name).dims[-2:],
        "d/deta:", ds.xroms.ddeta(name).dims[-2:],
    )
```

Both at once:

```{code-cell} ipython3
dtempdxi, dtempdeta = ds.xroms.hgrad("temp")
print(dtempdxi.dims)
print(dtempdeta.dims)
```

#### Boundaries

At the edges of the domain, there is no neighbor for the difference of the points at the edge. `hboundary` and `sboundary` say what to do about it, for horizontal and vertical derivatives (and for moving between grids, above):

* `"extend"` is the default, and uses one-sided values from the data at the edge: the nearest derivative that could be calculated.
* `"fill"` puts the value of `hfill_value` or `sfill_value` there, which is NaN by default. A value of 0 imposes a zero gradient.

`xroms` never pads the field with its own edge value before differencing, which gave zeros in the first and last values in older versions. Here, the derivative of salinity over the bottom two and top two w levels in a column:

```{code-cell} ipython3
column = dict(ocean_time=0, eta_rho=100, xi_rho=150)
extended = ds.xroms.ddz("salt").isel(**column)
filled = ds.xroms.ddz("salt", sboundary="fill").isel(**column)
print(extended.values[[0, 1, -2, -1]])
print(filled.values[[0, 1, -2, -1]])
```

#### Derivatives of one level

A derivative at constant depth needs the vertical levels around each point. If you select a single level of a field before you differentiate it horizontally (`ds.salt.isel(s_rho=-1)`), `xroms` raises an error rather than give you the derivative along that level of the terrain-following coordinate:

```{code-cell} ipython3
try:
    ds.xroms.ddxi(ds.salt.isel(s_rho=-1))
except ValueError as err:
    print(err)
```

For the gradient of the surface field, differentiate the 3D field and then select the surface, which is the gradient at constant depth. Or, if the derivative along the surface level is what you want, say so with `along_s=True` (it is also the way to differentiate a field that is not on levels of s, such as a depth average that kept the name of its variable):

```{code-cell} ipython3
dsaltdxi = ds.xroms.ddxi("salt").isel(s_rho=-1)  # at constant depth
dsaltdxi_along_s = ds.xroms.ddxi(ds.salt.isel(s_rho=-1), along_s=True)  # along the surface level
print(dsaltdxi.dims, dsaltdxi_along_s.dims)
```

### Time

Use `xarray` directly for this.

```{code-cell} ipython3
ddt = ds.salt.differentiate("ocean_time", datetime_unit="s")
ddt.dims
```

## Built-in Physical Calculations

These are all properties of the accessor, so should be called without (), except for the mixed layer depth, which needs a threshold. Demonstrated below are the calculations using the accessor and using the function, which takes the arrays and, if it needs depths or grid metrics, the Dataset as `grid`. Each shows where the result lands, by its dimensions: the accessor in the dimensions of the Dataset and the function in the canonical ones.

+++

### Horizontal speed

The magnitude of the velocity on rho points [m/s]. Velocities that are masked (NaN) count as 0 when they are moved to the rho points, so that land does not spread into the neighboring water.

    ds.xroms.speed  # accessor

    xroms.speed(ds.u, ds.v)  # function

```{code-cell} ipython3
print("accessor:", ds.xroms.speed.dims)
print("function:", xroms.speed(ds.u, ds.v).dims)
```

### Kinetic energy

On rho points [kg/(m s²)], from the speed and the reference density `rho0`: the one of the Dataset (a variable or an attribute of the file, else 1025).

    ds.xroms.KE  # accessor

    # without the accessor you need to manage this yourself — first calculate speed to then calculate KE
    speed = xroms.speed(ds.u, ds.v)
    xroms.KE(xroms.rho0(ds), speed)

```{code-cell} ipython3
speed = xroms.speed(ds.u, ds.v)
print("accessor:", ds.xroms.KE.dims)
print("function:", xroms.KE(xroms.rho0(ds), speed).dims)
```

### Geostrophic velocities

From the slope of the free surface `zeta` and the Coriolis parameter `f` [m/s]. `ug` is on u points and `vg` on v points, along the xi and eta directions of the grid (not east and north).

    ds.xroms.ug  # accessor, u component
    ds.xroms.vg  # accessor, v component

    ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, ds)  # function

```{code-cell} ipython3
ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, ds)
print("accessor:", ds.xroms.ug.dims, ds.xroms.vg.dims)
print("function:", ug.dims, vg.dims)
```

### Eddy kinetic energy (EKE)

Of the geostrophic velocities, on rho points [m²/s²].

    ds.xroms.EKE  # accessor

    ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, ds)
    xroms.EKE(ug, vg)

```{code-cell} ipython3
print("accessor:", ds.xroms.EKE.dims)
print("function:", xroms.EKE(ug, vg).dims)
```

### Vertical shear

Since it is a common use case, there are specific methods to return the u and v components of vertical shear on their own grids [1/s]: `dudz` on u points and `dvdz` on v points, both on the w levels. These are just available for Datasets.

    ds.xroms.dudz
    ds.xroms.dvdz

    xroms.dudz(ds.u, ds)
    xroms.dvdz(ds.v, ds)

    # already on same grid:
    ds.xroms.vertical_shear

    dudz = xroms.dudz(ds.u, ds)
    dvdz = xroms.dvdz(ds.v, ds)
    xroms.vertical_shear(dudz, dvdz)

```{code-cell} ipython3
print("accessor:", ds.xroms.dudz.dims, ds.xroms.dvdz.dims)
print("function:", xroms.dudz(ds.u, ds).dims, xroms.dvdz(ds.v, ds).dims)
```

The magnitude of the vertical shear is also a built-in derived variable for the `xroms` accessor, on rho points and w levels:

```{code-cell} ipython3
dudz = xroms.dudz(ds.u, ds)
dvdz = xroms.dvdz(ds.v, ds)
print("accessor:", ds.xroms.vertical_shear.dims)
print("function:", xroms.vertical_shear(dudz, dvdz).dims)
```

### Vertical vorticity

The vertical component of the relative vorticity, `dv/dxi - du/deta` at constant depth, on psi points [1/s].

    ds.xroms.vort

    xroms.relative_vorticity(ds.u, ds.v, ds)

```{code-cell} ipython3
print("accessor:", ds.xroms.vort.dims)
print("function:", xroms.relative_vorticity(ds.u, ds.v, ds).dims)
```

### Horizontal convergence

Horizontal component of the currents convergence, `du/dx + dv/dy` at constant depth, on rho points [1/s].

    ds.xroms.convergence

    xroms.convergence(ds.u, ds.v, ds)

```{code-cell} ipython3
print("accessor:", ds.xroms.convergence.dims)
print("function:", xroms.convergence(ds.u, ds.v, ds).dims)
```

### Normalized surface convergence

Horizontal component of the currents convergence at the surface, normalized by `f`. It does not have a vertical dimension, and it is dimensionless. This is only available through the accessor.

    ds.xroms.convergence_norm

```{code-cell} ipython3
ds.xroms.convergence_norm.dims
```

### Ertel potential vorticity

The accessor assumes you want the Ertel potential vorticity of the buoyancy, on rho points and levels:

    ds.xroms.ertel

    sig0 = xroms.potential_density(ds.temp, ds.salt)
    buoyancy = xroms.buoyancy(sig0, rho0=xroms.rho0(ds))
    xroms.ertel(buoyancy, ds.u, ds.v, ds.f, ds, scoord='s_w')

Alternatively, the user can access the original function and use a different tracer for this calculation (in this example, "dye_01"), and can return the result on a different vertical grid, for example:

    xroms.ertel(ds.dye_01, ds.u, ds.v, ds.f, ds, scoord='s_w')

```{code-cell} ipython3
sig0 = xroms.potential_density(ds.temp, ds.salt)
buoyancy = xroms.buoyancy(sig0, rho0=xroms.rho0(ds))
print("accessor:", ds.xroms.ertel.dims)
print("function:", xroms.ertel(buoyancy, ds.u, ds.v, ds.f, ds, scoord="s_w").dims)  # on w levels
```

### Density

The in-situ density [kg/m³], calculated with the equation of state of ROMS. The pressure comes from the depth of each point, so the function needs the Dataset as `grid` (or the heights as `z`). The accessor gives the variable `rho` of the Dataset if it has one.

    ds.xroms.rho

    xroms.density(ds.temp, ds.salt, grid=ds)

```{code-cell} ipython3
print("accessor:", ds.xroms.rho.dims)
print("function:", xroms.density(ds.temp, ds.salt, grid=ds).dims)
```

### Potential density

Referenced to the surface [kg/m³]. Here and above, this is the density itself, not the anomaly: subtract 1000 for sigma.

    ds.xroms.sig0

    xroms.potential_density(ds.temp, ds.salt)

```{code-cell} ipython3
print("accessor:", ds.xroms.sig0.dims)
print("function:", xroms.potential_density(ds.temp, ds.salt).dims)
```

### Density with TEOS-10

Both of the density functions use the equation of state of ROMS by default, `eos="roms"`. With `eos="teos10"` they use TEOS-10 instead, through the package `gsw`. TEOS-10 converts the practical salinity to absolute salinity at the pressure and the location of each point, and the potential temperature to conservative temperature, so it needs the heights of the points (`grid=ds`, or `z_points=`) and the longitudes and latitudes (those of the DataArrays, or `lon=` and `lat=`).

```{code-cell} ipython3
sig0_teos = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
print(sig0_teos.dims, sig0_teos.attrs["long_name"])
# the difference from the equation of state of ROMS at the surface, kg/m3
print(float((sig0_teos - ds.xroms.sig0).isel(ocean_time=0, s_rho=-1).mean()))
```

### Buoyancy

[m/s²], from the potential density and `rho0`.

    ds.xroms.buoyancy

    sig0 = xroms.potential_density(ds.temp, ds.salt);
    xroms.buoyancy(sig0, rho0=xroms.rho0(ds))

```{code-cell} ipython3
print("accessor:", ds.xroms.buoyancy.dims)
print("function:", xroms.buoyancy(sig0, rho0=xroms.rho0(ds)).dims)
```

### Buoyancy frequency

Also called vertical buoyancy gradient, `N²` [1/s²]. It is on the w levels, between the rho levels of the density that it is calculated from, and it is NaN at the top and bottom w levels, unless you ask for `sboundary="extend"`.

    ds.xroms.N2

    rho = xroms.density(ds.temp, ds.salt, grid=ds)  # calculate rho if not in output
    xroms.N2(rho, ds)

```{code-cell} ipython3
rho = xroms.density(ds.temp, ds.salt, grid=ds)
print("accessor:", ds.xroms.N2.dims)
print("function:", xroms.N2(rho, ds).dims)
```

### Horizontal buoyancy gradient

`M²` [1/s²], on rho points and the levels of the density, with the horizontal derivatives at constant depth.

    ds.xroms.M2

    rho = xroms.density(ds.temp, ds.salt, grid=ds)  # calculate rho if not in output
    xroms.M2(rho, ds)

```{code-cell} ipython3
print("accessor:", ds.xroms.M2.dims)
print("function:", xroms.M2(rho, ds).dims)
```

### Mixed layer depth

This is not a property since the threshold is a parameter and needs to be input. The mixed layer depth [m, positive] is the shallowest depth below `reference_depth` at which the potential density has increased by more than `threshold` from its value at `reference_depth`, interpolated between levels. It has no vertical dimension. The defaults are `reference_depth=0` (the shallowest level) and `fill="bottom"` (a column that never gets denser by the threshold has the depth of the bottom as mixed layer depth; `fill="nan"` gives NaN there).

    ds.xroms.mld(threshold=0.03)

    sig0 = xroms.potential_density(ds.temp, ds.salt);
    xroms.mld(sig0, ds, threshold=0.03)

```{code-cell} ipython3
print("accessor:", ds.xroms.mld(threshold=0.03).dims)
print("function:", xroms.mld(sig0, ds, threshold=0.03).dims)
```

The calculation is lazy like all the others, and land (where there are no profiles) is NaN:

```{code-cell} ipython3
mld = ds.xroms.mld(threshold=0.03).isel(ocean_time=0).compute()
print(float(mld.min()), float(mld.max()), int(mld.isnull().sum()), "land points")
```

## Other calculations

### Rotations

If your ROMS grid is curvilinear, you'll need to rotate your u and v velocities from along the grid axes to being eastward and northward. You can do this with

    ds.xroms.east

    ds.xroms.north

The velocities are on rho points. If the Dataset has `u_eastward` and `v_northward`, those are used. Without the accessor, `grid_to_earth` rotates `u` and `v` on their own grids with the grid `angle` (in radians), and gives `east` and `north`:

    east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)

Additionally, if you want to rotate your velocity to be a different orientation, for example to be along-channel, you can do that with

    ds.xroms.east_rotated(angle)

    ds.xroms.north_rotated(angle)

These rotate the vector (east, north) by `angle` (in radians unless you give `isradians=False`; positive counterclockwise, or clockwise with `reference="compass"`) and give the x and y components of the result. To get the component along a direction `theta`, counterclockwise from east, rotate by `-theta`. The function that does it for any pair of components is `xroms.rotate_vectors(x, y, angle)`, and `xroms.earth_to_grid` goes the other way.

```{code-cell} ipython3
east, north = ds.xroms.east, ds.xroms.north
print(east.dims, east.attrs["units"])

theta = np.deg2rad(30)  # an along-channel direction, 30 degrees counterclockwise from east
along = ds.xroms.east_rotated(-theta)
by_hand = east * np.cos(theta) + north * np.sin(theta)
print(bool(np.allclose(along, by_hand, equal_nan=True)))

east2, north2 = xroms.grid_to_earth(ds.u, ds.v, ds.angle)  # function
print(bool(np.allclose(east2, east, equal_nan=True)))

x, y = xroms.rotate_vectors(1, 0, 90, isradians=False)  # any pair of components, here just numbers
print(round(x, 12), round(y, 12))
```

## Functions instead of the accessor

The accessor is for convenience: it gets the variables and the grid from your Dataset, and returns results as the dimensions of your Dataset. The functions underneath can be used on their own. They take DataArrays and, to get the grid variables they need (`pm` and `pn`, `h`, `zeta`, the parameters of the s-coordinates), the Dataset that has them as `grid` (with a grid file that is kept apart from the output, this is the merged Dataset, see the {doc}`input/output page <io>`). They return canonical dimensions, so canonicalize your Dataset first, once, and everything lines up:

```{code-cell} ipython3
dsc = xroms.canonicalize(ds)  # only the dimension names change
print(dsc.u.dims)
print(xroms.ddz(dsc.salt, dsc).dims)
print(xroms.speed(dsc.u, dsc.v).dims)
print(xroms.gridmean(dsc.v, dsc, ("Z", "Y")).dims)
print((dsc.u - xroms.to_u(dsc.temp)).dims)  # no extra dimension
```

## Time-based calculations including climatologies

+++

### Rolling averages in time

Here is an example of computing a rolling average in time. Nothing happens in this example because we only have two time steps to use, however, it does demonstrate the syntax. If more time steps were available we would update `ds.salt.rolling(ocean_time=1)` to include more time steps to average over in a rolling sense.

More information about rolling operations [is available](https://docs.xarray.dev/en/stable/computation.html#rolling-window-operations).

```{code-cell} ipython3
roll = ds.salt.rolling(ocean_time=1, center=True, min_periods=1).mean()
plt.figure(figsize=(6, 3))
roll.isel(s_rho=-1, eta_rho=10, xi_rho=20).plot(alpha=0.5, lw=2)
ds.salt.isel(s_rho=-1, eta_rho=10, xi_rho=20).plot(ls=":", lw=2);
```

### Resampling in time

More info: [`resample` in the xarray docs](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html).

+++

#### Upsample

Upsample to a higher resolution in time. Makes sense to interpolate to fill in data when upsampling, but can also forward or backfill, or just add nan's.

```{code-cell} ipython3
saltup = ds.salt.resample(ocean_time="30min").interpolate()
print(dict(saltup.sizes))
```

Plot to visually inspect results

```{code-cell} ipython3
point = dict(eta_rho=30, xi_rho=20, s_rho=-1)
plt.figure(figsize=(6, 3))
ds.salt.isel(**point).plot(marker="o")
saltup.isel(**point).plot(marker="x");
```

#### Downsample

Resample down to lower resolution in time. This requires appending a method to aggregate the extra data, such as a `mean`. Note that other options can be used to shift the result within the interval of aggregation in various ways. Just the syntax is shown here since we only have two time steps to work with.

    saltdown = ds.salt.resample(ocean_time='6h').mean()
    ds.salt.isel(eta_rho=30, xi_rho=20, s_rho=-1).plot(marker='o')
    saltdown.isel(eta_rho=30, xi_rho=20, s_rho=-1).plot(marker='x')

+++

#### Seasonal average, over time

This is an example of [resampling](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html).

    da.resample(ocean_time=[time frequency string]).reduce([aggregation function])

For example, calculate the mean temperature every quarter in time with the following:

    ds.temp.resample(ocean_time='QS').reduce(np.mean)

or the aggregation function can be appended on the end directly with:

    ds.temp.resample(ocean_time='QS').mean()

The result of this calculation is a time series of downsampled chunks of output in time, the frequency of which is selected by input "time frequency string", and aggregated by input "aggregation function".

Examples of the time frequency string are:
* "QS": quarters, starting in January of each year and averaging three months.
  * Also available are selections like "QS-DEC", quarters but starting with December to better align with seasons. Other months are input options as well.
* "MS": monthly
* "D": daily
* "h": hourly
* Many more options are given [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

Examples of aggregation functions are:
* np.mean
* np.max
* np.min
* np.sum
* np.std

Result of downsampling a 4D salt array from hourly to 6-hourly, for example, gives: `[ocean_time x s_rho x eta_rho x xi_rho]`, where `ocean_time` has about 1/6 of the number of entries reflecting the aggregation in time.

```{code-cell} ipython3
print(dict(ds.temp.resample(ocean_time="6h").reduce(np.mean).sizes))
```

### Seasonal mean over all available time

This is how to average over the full dataset period by certain time groupings using xarray `groupby` which is like pandas version. In this case we show the seasonal mean averaged across the full model time period. The syntax for this is:

    da.groupby('ocean_time.[time string]').reduce([aggregation function])

For example, to average salt by season:

    ds.salt.groupby('ocean_time.season').reduce(np.mean)

or

    ds.salt.groupby('ocean_time.season').mean()

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

More information about options for time (including "derived" datetime coordinates) is [here](https://docs.xarray.dev/en/stable/user-guide/time-series.html#datetime-components).

Examples of aggregation functions are:
* np.mean
* np.max
* np.min
* np.sum
* np.std

Result of averaging over seasons for a 4D salt array returns, for example: `[season x s_rho x eta_rho x xi_rho]`, where `season` has 4 entries, each covering 3 months of the year.

```{code-cell} ipython3
# this example has only 1 season because it is a short example file
print(dict(ds.temp.groupby("ocean_time.season").mean().sizes))
```

### Calculations on a time mean that need depths

Many calculations of `xroms` need the depths of the points: derivatives, grid sums and means over `Z`, the depth average, the buoyancy frequency and more. The depths follow the free surface `zeta`, which varies in time. After a time mean (or a resample or a groupby), the field has no time left to match with `zeta`, and `xroms` does not guess which free surface you want. It raises an error that tells you what to do:

```{code-cell} ipython3
temp_mean = ds.temp.mean("ocean_time")
print(temp_mean.dims)
try:
    ds.xroms.depth_average(temp_mean)
except ValueError as err:
    print(err)
```

For a time mean, the free surface to use is the time mean of `zeta`: `zeta="mean"`. The other choices are `zeta=0` (the depths of the ocean at rest) and `zeta=` a DataArray without time, or depths `z=`.

```{code-cell} ipython3
# the mean over the water column of the time mean temperature
temp_mean_avg = ds.xroms.depth_average(temp_mean, zeta="mean")
print(temp_mean_avg.dims)

# volume-weighted mean of the time mean temperature over the whole domain
print(float(ds.xroms.gridmean(temp_mean, ("X", "Y", "Z"), zeta="mean")))
```

Interpolation to depths and to isosurfaces is shown in {doc}`interpolation`.
