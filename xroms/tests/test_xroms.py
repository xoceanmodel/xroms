'''Test xroms functions.'''

import xroms
import xarray as xr
import numpy as np
import cartopy


grid = xr.open_dataset('xroms/tests/input/grid.nc')
ds = xr.open_dataset('xroms/tests/input/ocean_his_0001.nc')
# combine the two:
ds = ds.merge(grid, overwrite_vars=True, compat='override')
ds, grid = xroms.roms_dataset(ds, Vtransform=2)


# def test_relative_vorticity():
    
#     x.relative_vorticity(ds, grid, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None)
#     assert np.allclose(xroms.relative_vorticity(ds, grid),0)

# def test_mld():
#     sig0 = np.linspace(0,1,50)
#     xroms.mld(sig0, h, mask, z=None, thresh=0.03)

def test_dataset():
    
    pass
    # change to `dataset1` from `1roms_dataset`
    

def test_hgrad():
    
    pass