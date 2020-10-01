"""Test interpolation functions with known coordinates to make sure results are correct"""

import cartopy
import numpy as np
import xarray as xr

import xroms


grid = xr.open_dataset("xroms/tests/input/grid.nc")
ds = xroms.open_netcdf("xroms/tests/input/ocean_his_0001.nc", chunks={})
# ds = xr.open_dataset('xroms/tests/input/ocean_his_0001.nc')
# combine the two:
ds = ds.merge(grid, overwrite_vars=True, compat="override")
# ds, grid = xroms.roms_dataset(ds, Vt)
ie, ix = 2, 3


def test_interpll():

    indexer = {"eta_rho": [ie], "xi_rho": [ix]}
    out = xroms.interpll(ds.temp, ds.lon_rho.isel(indexer), ds.lat_rho.isel(indexer))
    np.allclose(out.squeeze(), ds.temp.isel(indexer).squeeze())
