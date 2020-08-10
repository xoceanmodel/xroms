'''Test utilities in xroms'''

import xroms
import xarray as xr
import numpy as np
import cartopy


grid = xr.open_dataset('tests/input/grid.nc')
ds = xr.open_dataset('tests/input/ocean_his_0001.nc')
ll2xe_dict = xroms.utilities.ll2xe_setup(ds, whichgrids=['u'])
ie, ix = 2, 3


def test_interp_1coord():
    '''Test interp with 1 input coord in several forms.'''

    varin = ds.u.isel(ocean_time=0, s_rho=-1)
    varoutcomp = varin.isel(eta_u=ie, xi_u=ix).persist()

    # lon0/lat0 as 0d DataArrays
    lon0 = ds.lon_u.isel(eta_u=ie, xi_u=ix)
    lat0 = ds.lat_u.isel(eta_u=ie, xi_u=ix)
    varout = xroms.utilities.interp(varin, lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    # lon0/lat0 as 0d numpy arrays
    lon0 = ds.lon_u.isel(eta_u=ie, xi_u=ix).values
    lat0 = ds.lat_u.isel(eta_u=ie, xi_u=ix).values
    varout = xroms.utilities.interp(varin, lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = float(ds.lon_u.isel(eta_u=ie, xi_u=ix))
    lat0 = float(ds.lat_u.isel(eta_u=ie, xi_u=ix))
    varout = xroms.utilities.interp(varin, lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = [float(ds.lon_u.isel(eta_u=ie, xi_u=ix))]
    lat0 = [float(ds.lat_u.isel(eta_u=ie, xi_u=ix))]
    varout = xroms.utilities.interp(varin, lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = np.array([float(ds.lon_u.isel(eta_u=ie, xi_u=ix))])
    lat0 = np.array([float(ds.lat_u.isel(eta_u=ie, xi_u=ix))])
    varout = xroms.utilities.interp(varin, lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    
def test_ll2xe_interp_col():
    '''Interpolate to pts with full water column.'''
    
    lon0, lat0 = [-94.5, -95.0], [27.8, 28.1]

    # find locations of lon0/lat0 in grid space
    xi0, eta0 = xroms.utilities.ll2xe(ll2xe_dict['u'], lon0, lat0)

    # perform interpolation
    var0 = ds.u.isel(ocean_time=0).interp(xi_u=xi0, eta_u=eta0)
    
    res = np.ones(var0.shape)
    res[:,0] *= 0.95
    res[:,1] *= 0.6
    
    assert np.allclose(var0, res)


def test_ll2xe_interp_pts():
    '''Test interpolation to several 2d points at surface.'''

    # want u at the following coords
    lon0, lat0 = [-94.5, -95.0], [27.8, 28.1]

    # find locations of lon0/lat0 in grid space
    xi0, eta0 = xroms.utilities.ll2xe(ll2xe_dict['u'], lon0, lat0)

    # perform interpolation
    u0 = ds.interp(xi_u=xi0, eta_rho=eta0).u.isel(ocean_time=0, s_rho=-1)
    
    assert np.allclose(u0, [0.95, 0.6])


def test_ll2xe_u_lat():
    '''Test interpolation coords and results for u grid.'''
    
    
    lon0 = ds.lon_u
    lat0 = ds.lat_u
    xi0, eta0 = xroms.utilities.ll2xe(ll2xe_dict['u'], lon0, lat0)

    # This tests the interpolation process for finding grid coords described here:
    J, I = ds.lon_u.shape
    X, Y = np.meshgrid(np.arange(I), np.arange(J))

    assert np.allclose(xi0,X)
    assert np.allclose(eta0,Y)
   

    # test that resulting subsequent xarray interpolation is correct
    # here we compare interpolated latitudes with original latitudes
    lat_u0 = ds.interp(xi_u=xi0, eta_u=eta0).lat_u
    inan = np.isnan(lat_u0)
    assert np.allclose(lat_u0.values[~inan.values], ds.lat_u.values[~inan.values])


def test_argsel2d():
    '''Check that function returns correct indices.
    
    This compares with previous calculation.'''
    
    lon0, lat0 = -95.8, 27.1
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='rho') == (0,1)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='psi') == (0,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='u') == (0,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='v') == (0,1)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='vert') == (1,1)


def test_argsel2d_exact():
    '''test for exact index.'''
    
    lon0, lat0 = -95.928571, 27.166685  # one corner of grid
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='rho') == (1,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='psi') == (0,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='u') == (1,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='v') == (0,0)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='vert') == (1,1)

    lon0, lat0 = -94.071429, 28.333351  # other corner of grid
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='rho') == (8,13)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='psi') == (7,12)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='u') == (8,12)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='v') == (7,13)
    assert xroms.utilities.argsel2d(grid, lon0, lat0, whichgrid='vert') == (8,13)

    
def test_sel2d():
    '''Check that function returns correct value for scalar inputs.'''
    
    lon0, lat0 = -94.8, 28.0
    
    assert xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='u').u.isel(s_rho=0, ocean_time=0) == 0.7
    assert xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='v').v.isel(s_rho=0, ocean_time=0) == 0.0
    
    
def test_sel2d_list():
    '''Test sel2d for lon0/lat0 as list.'''

    lon0, lat0 = [-95.7,-94.8], [27.4,28.0]
    
    assert (xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='u').u.isel(s_rho=0, ocean_time=0) == [0.1, 0.7]).all()
    assert (xroms.utilities.sel2d(ds, lon0, lat0, whichgrid='v').v.isel(s_rho=0, ocean_time=0) == [0.0, 0.0]).all()


def test_xisoslice():
    '''Test xisoslice function.'''

    # test longitude slice of u
    res = xroms.utilities.xisoslice(ds.lon_u, -95, ds.u, 'xi_u')
    assert np.allclose(res, 0.6)
    
    # test latitude slice of u
    res = xroms.utilities.xisoslice(ds.lat_u, 28, ds.u, 'eta_u').std()
    assert np.allclose(res, 0.37416574)
    
    # test when slice isn't along a value that is equal along that slice
    # convert to projected space
    proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    pc = cartopy.crs.PlateCarree()
    x_u, y_u = proj.transform_points(pc, ds.lon_u.values, ds.lat_u.values)[...,:2].T
    ds['x_u'] = (('eta_u','xi_u'), x_u.T)
    res = xroms.utilities.xisoslice(ds.x_u, 380000, ds.lon_u, 'xi_u').mean()
    assert np.allclose(res, -94.19483976)
    
    # test getting latitude for longitude slice
    res = xroms.utilities.xisoslice(ds.lon_rho, -94.5, ds.lat_rho, 'xi_rho').mean().values
    assert np.allclose(res, 27.75001934)
