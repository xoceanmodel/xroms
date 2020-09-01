'''Test accessor functions.'''

import xroms
import xarray as xr
import numpy as np
import cartopy


grid = xr.open_dataset('xroms/tests/input/grid.nc')
ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
# ds = xr.open_dataset('xroms/tests/input/ocean_his_0001.nc')
# combine the two:
ds = ds.merge(grid, overwrite_vars=True, compat='override')
# ds['Vtransform'] = 2
ds.xroms

# functions in test files:
xl, yl, N = 14, 9, 3
z_rho = np.array([-97.5025, -50.05  ,  -2.5975])
u = np.linspace(0,1.2, xl-1)
v = np.linspace(-0.1,0.1,yl-1)[:,np.newaxis]
temp = np.linspace(20,15,N)
salt = np.linspace(15,25,N)

def test_z_rho():
    assert np.allclose(ds.z_rho[0,:,0,0], z_rho)

def test_rho():
    assert np.allclose(ds.xroms.rho()[0,:,0,0], xroms.density(temp, salt, z_rho))

def test_speed():
    assert np.allclose(ds.xroms.speed().mean(),np.sqrt(u**2 + v**2).mean())

def test_KE():
    rho = xroms.density(temp, salt, z_rho)[:,np.newaxis, np.newaxis]
    s = (u**2 + v**2)[np.newaxis,:,:]
    KE = 0.5*rho*s
    assert np.allclose(ds.xroms.KE().mean(),KE.mean())

def test_relative_vorticity():
    assert np.allclose(ds.xroms.vort('rho', 's_rho'),0)

def test_dudz():
    assert np.allclose(ds.xroms.dudz('rho', 's_rho'),0)

def test_dvdz():
    assert np.allclose(ds.xroms.dvdz('rho', 's_rho'),0)
