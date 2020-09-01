import xroms
import xarray as xr
import numpy as np
import cartopy
# import accessor


grid = xr.open_dataset('xroms/tests/input/grid.nc')
ds = xr.open_dataset('xroms/tests/input/ocean_his_0001.nc')
# combine the two:
ds = ds.merge(grid, overwrite_vars=True, compat='override')
ds['Vtransform'] = 2
ds.xroms
# ds, grid = xroms.roms_dataset(ds, Vtransform=2)
# tris = xroms.interp.setup(ds, whichgrids=['u'])
# ie, ix = 2, 3


def test_relative_vorticity():
    assert np.allclose(ds.xroms.vort('rho', 's_rho'),0)

def test_dudz():
    assert np.allclose(ds.xroms.dudz('rho', 's_rho'),0)

def test_dvdz():
    assert np.allclose(ds.xroms.dvdz('rho', 's_rho'),0)
