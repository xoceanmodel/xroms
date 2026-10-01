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

# How to load data

Read your model output with `xarray`, then call `xroms` on it: there is no setup step. More about input/output with `xarray` can be found [here](https://docs.xarray.dev/en/stable/user-guide/io.html).

```{code-cell} ipython3
import os
import tempfile

import xarray as xr
import xroms
```

## Open model output with `xarray`

Which `xarray` function to use depends on where your output is:

* `xr.open_dataset(path, chunks={})` for a netCDF file;
* `xr.open_mfdataset(paths, ...)` for output split over many files, such as one file per day (more below);
* `xr.open_zarr(path)` for a Zarr store;
* `xr.open_dataset(url, chunks={})` for a remote dataset served with OPeNDAP, for example by a THREDDS server;
* kerchunk or icechunk references, for output that stays where it is but is read as one Zarr-like dataset.

`xroms` only sees the Dataset you give it, so how the Dataset was opened does not matter to it. The calls below are shown but not run here, since they need your files, a server or optional packages (`zarr`, `kerchunk`, `icechunk`):

```python
ds = xr.open_dataset("ocean_his.nc", chunks={})          # a netCDF file
ds = xr.open_zarr("ocean_his.zarr")                      # a Zarr store

url = "https://my.thredds.server/thredds/dodsC/roms/ocean_his.nc"
ds = xr.open_dataset(url, chunks={})                     # OPeNDAP / THREDDS
```

A [kerchunk](https://fsspec.github.io/kerchunk/) reference file describes where the data of many netCDF files sit, so they can be read as one dataset without copying them:

```python
ds = xr.open_dataset(
    "reference://",
    engine="zarr",
    backend_kwargs={"consolidated": False, "storage_options": {"fo": "ocean_his_refs.json"}},
    chunks={},
)
```

An [icechunk](https://icechunk.io) repository is opened from its storage and read through a session:

```python
import icechunk

storage = icechunk.s3_storage(bucket="my-bucket", prefix="roms/ocean_his", from_env=True)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
```

## Demo using the example dataset

`xroms` comes with a small example, a Rutgers ROMS history file from the Texas and Louisiana shelf. It is downloaded the first time it is used (this needs the optional package `pooch`) and kept in a cache. It is opened with `xr.open_dataset(..., chunks={})` and nothing else is done to it: its grid is in the same file, so `xroms` can be called on it right away.

```{code-cell} ipython3
ds = xroms.datasets.fetch_ROMS_example_full_grid()
print(dict(ds.sizes))
```

```{code-cell} ipython3
ds.xroms.speed  # horizontal speed on rho points
```

## Some specific notes

### Chunks

Chunks break model output into smaller units for use with `dask`. Passing `chunks` when opening a dataset requires `dask`. With `chunks={}` the chunks stored in the file are used (one chunk for the whole array if the file is not chunked); you can also choose them, for example `chunks={"ocean_time": 1}`, or change them later with `ds.chunk(...)`. This can be formalized more by setting up a `dask` cluster. The best sizing of chunks is not clear *a priori* and requires some testing; the [`dask` documentation](https://docs.dask.org/en/stable/array-chunks.html) has guidance.

With chunked data every `xroms` result is lazy, a `dask` array that is only computed when you ask for values (to plot, `.values`, `.compute()`), and it is chunked like its inputs. (A vertical derivative needs whole columns, so while it works it rechunks the vertical dimension, and it chunks the result like the input again.)

The example file is not chunked, so it has one chunk. Here is how to change that:

```{code-cell} ipython3
print(ds.salt.chunks)
print(ds.salt.chunk({"ocean_time": 1}).chunks)  # one chunk per time
```

### `open_mfdataset()`

When output is spread over many files, `xr.open_mfdataset()` opens them as one Dataset, combined along time. `xroms` works with `xarray`'s plain defaults and gives the same results with the keywords below. But history files repeat the grid and the s-coordinate parameters (`h`, `pm`, `pn`, `mask_rho`, `Cs_r`, `hc`, ...) in every file, and by default `xarray` then repeats them along time too. Asking for the following keeps just one copy of everything that does not depend on time (these were the defaults of `xroms.open_mfnetcdf` in 0.6):

    xr.open_mfdataset(paths, data_vars="minimal", coords="minimal", compat="override")

Here it is with two small history files from the tests of `xroms` (a glob such as `"ocean_his_*.nc"` works for any number of files; add `parallel=True` to open many files in parallel). `plain` is what `xarray` does by default, written out as `data_vars="all"` (leaving the keywords out gives a `FutureWarning`, see below):

```{code-cell} ipython3
paths = "../xroms/tests/input/ocean_his_000?.nc"
plain = xr.open_mfdataset(paths, data_vars="all")
ds_mf = xr.open_mfdataset(paths, data_vars="minimal", coords="minimal", compat="override")
print(ds_mf.sizes["ocean_time"], "times")
print("plain defaults:", plain.h.dims, plain.hc.dims)
print("minimal/override:", ds_mf.h.dims, ds_mf.hc.dims)
```

The repeated copies are not just wasteful: results that only depend on the grid, like `dx`, also get a time dimension.

```{code-cell} ipython3
print(plain.xroms.dx().dims)
print(ds_mf.xroms.dx().dims)
```

`xarray` is moving to defaults like these, and recent versions warn about it (a `FutureWarning`) if you do not set the keywords yourself. Giving all three, as above, avoids the warning. They go together: `compat="override"` can only be used with `coords="minimal"`, and with only `data_vars="minimal"` the warning is about `compat` instead. In these versions you can also opt in to the new defaults everywhere with `xr.set_options(use_new_combine_kwarg_defaults=True)`.

### `open_zarr()`

Some useful keyword argument selections for reading in files with `xr.open_zarr()` are:

    {'consolidated': True, 'drop_variables': 'dstart'}

and for concatenating several stores together with `xr.concat()`:

    {'dim': 'ocean_time', 'data_vars': 'minimal', 'coords': 'minimal'}

For example (not run here):

```python
datasets = [xr.open_zarr(path, consolidated=True, drop_variables="dstart") for path in paths]
ds = xr.concat(datasets, dim="ocean_time", data_vars="minimal", coords="minimal")
```

## There is no setup step

Before 1.0, you called `ds, xgrid = xroms.roms_dataset(ds)` once to add `z` coordinates and metrics to the Dataset and to build an `xgcm` grid, which was stored for `xroms` to use (`ds.xroms.xgrid`, `ds.xroms.set_grid(xgrid)`). All of that is gone: open the Dataset with `xarray` and call `xroms`, which finds what it needs in the Dataset when it needs it.

| In 0.6 | In 1.0 |
| --- | --- |
| `ds, xgrid = xroms.roms_dataset(ds)` and `ds.xroms.set_grid(xgrid)` | nothing |
| `ds.z_rho`, `ds.dz`, `ds.dA`, `ds.dV`, ... added to the Dataset | computed when asked for: `ds.xroms.z_rho`, `ds.xroms.dz()`, `ds.xroms.dA()`, `ds.xroms.dV()`; `ds.xroms.assign_z()` adds the depths as coordinates |
| `xroms.open_netcdf`, `xroms.open_mfnetcdf`, `xroms.open_zarr` | `xr.open_dataset`, `xr.open_mfdataset`, `xr.open_zarr` |
| the `xgcm` grid in `ds.xroms.xgrid` | `ds.xroms.xgcm_grid()` gives you a new one for your own `xgcm` work |

Calling a removed function raises an error that names its replacement. The full table of changes is in the {doc}`migration guide <migration>`.

## Nothing is stored, and everything is lazy

Every `xroms` call computes from the data in hand, and nothing is stored in or next to your Dataset: no grid object, no cache. So a result can never be out of date. Subset the Dataset or edit it, and the next call uses what it now holds. Here the free surface is raised by 1 m, and the depth of the top level follows:

```{code-cell} ipython3
point = dict(ocean_time=0, s_rho=-1, eta_rho=100, xi_rho=150)  # a point at the top level
print(float(ds.xroms.z_rho.isel(**point)))
higher = ds.assign(zeta=ds.zeta + 1)
print(float(higher.xroms.z_rho.isel(**point)))
```

If the data are `dask` arrays, as when you open them with `chunks`, calls return immediately with a lazy result and compute only when you ask for values:

```{code-cell} ipython3
speed = ds.xroms.speed
type(speed.data)
```

Because nothing is cached, something that is expensive and that you use again and again belongs in a variable. `.persist()` computes it once and keeps the result in memory (still as a `dask` array), and `.load()` computes it into a numpy array:

```{code-cell} ipython3
speed = ds.xroms.speed.persist()
print(float(speed.isel(s_rho=-1).mean()))
print(type(ds.xroms.speed.load().data))
```

## A separate grid file

UCLA ROMS, `roms-tools` and often CROCO write the grid (`h`, `pm`, `pn`, the masks, `angle`, `f`, longitudes and latitudes) to a file of its own, apart from the model output. You can merge the two once:

    ds = xroms.merge_grid(out, grid)

{func}`xroms.merge_grid` is like `xr.merge([out, grid], compat="override")` with some extras: where both have a variable or an attribute the one in the output is kept; it checks that both cover the same points (so nothing is padded with NaN); and the longitudes and latitudes at every position become coordinates, so that every variable and every `xroms` result carries them. Only metadata changes: the data stay lazy.

Or leave them apart and pass the grid to the methods of the accessor as `grid=`:

    out.xroms.z(grid=grid)
    out.xroms.ddz("salt", grid=grid)

The accessor completes the grid with what only the output has (its free surface `zeta` and, for UCLA ROMS, the s-coordinate parameters that are in its attributes), so the results are the same as with the merged Dataset. The functions of `xroms` take the Dataset that has all the grid variables they need as their `grid` argument, as in `xroms.ddz(out.salt, ds)`, which for depths and everything that uses them means the merged Dataset.

### Example: UCLA ROMS output

The tests of `xroms` include small pieces of real UCLA ROMS files: a restart file and its grid file, cut down to 10 by 8 rho points. The restart file has no coordinates at all: `time` is a dimension without values, and `ocean_time` is a variable in seconds with the start date only in its `long_name`.

```{code-cell} ipython3
d = "../xroms/tests/input/"
out = xr.open_dataset(d + "ucla_rst.nc")
grid = xr.open_dataset(d + "ucla_grd.nc")
print(dict(out.sizes))
```

Merge the grid in and use {func}`xroms.decode_time` to get a `time` coordinate with dates:

```{code-cell} ipython3
ds_ucla = xroms.merge_grid(out, grid)
ds_ucla = xroms.decode_time(ds_ucla)
ds_ucla.time.values
```

That is all the preparation. The s-coordinate parameters (`theta_s`, `theta_b`, `hc`, `Cs_r`, `Cs_w`) are global attributes of the file, which `xroms` reads on its own, so depths and all the calculations work:

```{code-cell} ipython3
speed = ds_ucla.xroms.speed.sel(time="1998-01-06")
print(speed.dims)
print(float(speed.isel(s_rho=-1).max()), "m/s at the surface at most")
```

## Optional helpers

These are not needed to use `xroms`, but can be handy.

### `xroms.canonicalize()`

Rutgers ROMS and REMORA give every position its own dimensions (`eta_u`, `xi_v`, `eta_psi`, `xi_psi`), while UCLA ROMS and CROCO use shared ones. The functions of `xroms` return results with the canonical dimension names: `(eta_rho, xi_rho)` for rho points, `(eta_rho, xi_u)` for u points, `(eta_v, xi_rho)` for v points and `(eta_v, xi_u)` for psi points. {func}`xroms.canonicalize` renames a Dataset or DataArray to them. It only changes names; no data are touched. See {doc}`calc` for when it matters.

```{code-cell} ipython3
print(ds.u.dims)
print(xroms.canonicalize(ds).u.dims)
```

### `xroms.add_cf_attrs()`

{func}`xroms.add_cf_attrs` returns a copy of a Dataset with the attributes that `cf-xarray` looks for and with the longitudes and latitudes as coordinates. `xroms` does not use `cf-xarray` itself. See {doc}`select_data` for an example.

### `ds.xroms.assign_z()`

`ds.xroms.assign_z()` returns a **new** Dataset with the depths of the rho points on the rho levels and on the w levels attached as the coordinates `z_rho` and `z_w`, for example to plot against depth. It is a snapshot of the `zeta` and `h` the Dataset has now, so call it again after you change them. For `dask` arrays the depths are lazy, but for data that are in memory (numpy arrays) they are computed right away.

```{code-cell} ipython3
dsz = ds.xroms.assign_z()
print(dsz.z_rho.dims)
print("z_rho" in ds.coords)  # ds itself is unchanged
print(type(dsz.z_rho.data))  # lazy
in_memory = ds.isel(ocean_time=0).load()
print(type(in_memory.xroms.assign_z().z_rho.data))  # computed right away
```

## Save output

After model output has been read in with `xarray`, it can be used for calculations and/or subsetted, then easily saved back out to a file (in this case saving out only the first time step):

    ds.isel(ocean_time=0).to_netcdf('filename.nc')

`xroms` results are DataArrays with the coordinates of the Dataset and attributes of their own (`name`, `long_name`, `units`), so a result can be put back into the Dataset with `assign` and saved along with the rest:

```{code-cell} ipython3
ds_out = ds.assign(speed=ds.xroms.speed)
print(ds_out.speed.dims)
print(ds_out.sizes == ds.sizes)  # no new dimensions
```

The next cell saves only the speed at the first time. It writes to a temporary directory so that building this page leaves no file behind; use a file name of your own.

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "speed.nc")
    ds_out[["speed"]].isel(ocean_time=0).to_netcdf(path)
    with xr.open_dataset(path) as saved:
        print(saved.speed.dims, saved.speed.attrs["units"])
```
