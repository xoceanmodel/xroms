## XROMS

`xroms` contains functions for commonly used scripts for working with ROMS output in xarray. The first two read in netCDF and zarr output:

    ds = open_netcdf(files, chunks)
    ds = open_zarr(files, chunks)
    
`chunks` defaults to `chunks = {'ocean_time':1}`, which is only intended as a passable first guess. 

Vertical coordinates are appended to the dataset, and an xgcm grid object is created, using

    ds, grid = roms_dataset(ds, Vtransform=None)

By default, `Vtransform` will be read from the dataset, and the appropriate transformation for the vertical coordinate `z` will be applied.

The `xroms.roms_seawater` subpackage contains an equation of state based on the one in ROMS, `density`, the related `buoyancy`, and a function that calculates N<sup>2</sup>, `stratification_frequency`. 

The `xroms.utilities` subpackage contains a function to calculate horizontal gradients on an s-grid, and some helper functions to interpolate to either rho- or psi-points.


### Installation

You can install this locally and so that it is editable (`-e`), and with the required packages:

    git clone git@github.com:hetland/xroms.git
    cd xroms
    pip install -r requirements.txt -e .

Or:

    pip install git+git://github.com/hetland/xroms