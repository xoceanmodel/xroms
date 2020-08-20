'''Test utilities in xroms'''

import xroms
import xarray as xr
import numpy as np
import cartopy

    
def test_argsel2d():
    '''Check that function returns correct indices.'''
    
    lon0, lat0 = -95.8, 27.1
    ds = xr.open_dataset('tests/input/grid.nc')
    assert xroms.utilities.argsel2d(ds, lon0, lat0, whichgrid='rho') == (1,0)
    assert xroms.utilities.argsel2d(ds, lon0, lat0, whichgrid='psi') == (0,0)
    assert xroms.utilities.argsel2d(ds, lon0, lat0, whichgrid='u') == (0,0)
    assert xroms.utilities.argsel2d(ds, lon0, lat0, whichgrid='v') == (1,0)
    assert xroms.utilities.argsel2d(ds, lon0, lat0, whichgrid='vert') == (1,1)


def test_sel2d():
    '''Check that function returns correct value.'''
    
    lon0, lat0 = -95.8, 27.1
    ds = xr.open_dataset('tests/input/ocean_his_0001.nc')
    
    # all of the u velocities are equal to 0.1, and v to 0.0,
    # but this at least does check that the mechanics are working
    assert xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='u').u.isel(s_rho=0, ocean_time=0).values == 0.1
    assert xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='v').v.isel(s_rho=0, ocean_time=0).values == 0.0
    

def test_xisoslice():
    '''Test xisoslice function.'''

    ds = xr.open_dataset('tests/input/ocean_his_0001.nc')

    # test longitude slice
    assert xroms.utilities.xisoslice(ds.lon_u, -95, ds.u, 'xi_u').max().values == 0.1
    
    # test when slice isn't along a value that is equal along that slice
    # convert to projected space
    proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    pc = cartopy.crs.PlateCarree()
    x_u, y_u = proj.transform_points(pc, ds.lon_u.values, ds.lat_u.values)[...,:2].T
    ds['x_u'] = (('eta_u','xi_u'), x_u.T)
    res = xroms.utilities.xisoslice(ds.x_u, 380000, ds.lon_u, 'xi_u').mean().values
    assert np.allclose(res, -94.19483976)
    
    # test getting latitude for longitude slice
    res = xroms.utilities.xisoslice(ds.lon_rho, -94.5, ds.lat_rho, 'xi_rho').mean().values
    assert np.allclose(res, 27.75001934)
