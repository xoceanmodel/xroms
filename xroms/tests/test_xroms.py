'''Test functions in xroms that are not loading.'''

import xroms
import xarray as xr
import numpy as np
import cartopy


# grid = xr.open_dataset('tests/input/grid.nc')
# ds = xr.open_dataset('tests/input/ocean_his_0001.nc')
# # combine the two:
# ds = ds.merge(grid, overwrite_vars=True, compat='override')
# ds, grid = xroms.roms_dataset(ds, Vtransform=2)
# tris = xroms.interp.setup(ds, whichgrids=['u'])
# ie, ix = 2, 3


# def test_relative_vorticity():
#     assert np.allclose(xroms.relative_vorticity(ds, grid),0)


def test_dataset():
    
    pass
    # change to `dataset1` from `1roms_dataset`
    

def test_hgrad():
    
    pass