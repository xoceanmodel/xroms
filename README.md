## XROMS

`xroms` contains functions for commonly used scripts for working with ROMS output in xarray. The first two read in zarr and netCDF output:

    ds = open_roms_netcdf_dataset(files, chunks)
    ds = open_roms_zarr_dataset(files, chunks)
    
`chunks` defaults to `chunks = {'ocean_time':1}`, which is only intended as a passable first guess. 

Vertical coordinates are appended to the dataset, and an xgcm grid object is created, using

    ds, grid = roms_dataset(ds, Vtransform=None)

By default, `Vtransform` will be read from the dataset, and the appropriate transformation for the vertical coordinate `z` will be applied.
