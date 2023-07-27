
# How to load data

You should read in your model output of choice using `xarray`; more information on input/output with `xarray` can be found [here](https://docs.xarray.dev/en/stable/user-guide/io.html).

**Note:** There are a few functions to read in model output with `xroms` but they are scheduled to be removed in future versions of `xroms`.

## Some specific notes

### Chunks

Chunks are used to break up model output into smaller units for use with `dask`. Inputting chunks into a call to open a dataset requires the use of `dask`. This can be formalized more by setting up a `dask` cluster. The best sizing of chunks is not clear *a priori* and requires some testing.

### `open_mfdataset()`

Some useful keyword argument selections for when using `xr.open_mfdataset()` are suggested here:

    {'compat': 'override', 'combine': 'by_coords',
         'data_vars': 'minimal', 'coords': 'minimal', 'parallel': True}

For example,

    xr.open_mfdataset(url, compat='override', combine='by_coords',
         data_vars='minimal', coords='minimal', parallel=True}

### `open_zarr()`

Some useful keyword argument selections are for reading in files with `xr.open_zarr()` are:

    {'consolidated': True, 'drop_variables': 'dstart'}

and for concatenating the files together:

    {'dim': 'ocean_time', 'data_vars': 'minimal', 'coords': 'minimal'}


## Suggested Workflow for `xroms:

1. Read in model output using the appropriate `xarray` function.
2. Supplement your Dataset and calculate an `xgcm` grid object with:

    ds, xgrid = xroms.roms_dataset(ds)

The function adds `z` coordinates and other useful metrics to the `Dataset`, including the z coordinates on each horizontal grid (e.g., z_rho_u), and the z coordinates relative to mean sea level (e.g., z_rho0). It also sets up an xgcm grid object for the Dataset, which is stored necessary for many `xroms` functions, and can be stored and accessible in the accessor (`ds.xroms.xgrid`).

There are optional flags for `xroms.roms_dataset()` for what all metrics to lazily calculate since it can be time-consuming despite being lazily loaded and calculated; see the {doc}`API docs <api>` for details.

Alternatively, `roms_dataset()` will be run automatically the first time you use the Dataset accessor and `xgrid` will be stored in the object. Note that the default input flags to `roms_dataset()` are used in the case and if you want to have more control over that, you can use the following to override the xgrid stored

    ds, xgrid = xroms.roms_dataset(ds, [other flags you want to use])
    ds.xroms.set_grid(xgrid)

+++

## Demo workflow using example dataset

```
import xarray as xr
import xroms

ds = xroms.datasets.fetch_ROMS_example_full_grid()
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)
ds.xroms.set_grid(xgrid)
```

## Save output

After model output has been read in with `xarray`, it can be used for calculations and/or subsetted, then easily saved back out to a file (in this case saving out only the first time step):

    ds.isel(ocean_time=0).to_netcdf('filename.nc')
