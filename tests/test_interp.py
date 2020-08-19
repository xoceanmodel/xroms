'''Test interpolation functions with known coordinates to make sure results are correct'''

import xroms
import xarray as xr
import numpy as np
import cartopy


grid = xr.open_dataset('tests/input/grid.nc')
ds = xr.open_dataset('tests/input/ocean_his_0001.nc')
# combine the two:
ds = ds.merge(grid, overwrite_vars=True, compat='override')
ds, grid = xroms.roms_dataset(ds, Vtransform=2)
tris = xroms.interp.setup(ds, whichgrids=['u'])
ie, ix = 2, 3

def test_llzslice_1coord():
    '''Test llzslice with 1 input coord in several forms.'''

    varin = ds.u.isel(ocean_time=0, s_rho=-1)
    varoutcomp = varin.isel(eta_rho=ie, xi_u=ix).persist()

    # lon0/lat0 as 0d DataArrays
    lon0 = ds.lon_u.isel(eta_rho=ie, xi_u=ix)
    lat0 = ds.lat_u.isel(eta_rho=ie, xi_u=ix)
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    # lon0/lat0 as 0d numpy arrays
    lon0 = ds.lon_u.isel(eta_rho=ie, xi_u=ix).values
    lat0 = ds.lat_u.isel(eta_rho=ie, xi_u=ix).values
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))
    lat0 = float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = [float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))]
    lat0 = [float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))]
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = np.array([float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))])
    lat0 = np.array([float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))])
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    
def test_llzt_1coord():
    '''Test llzt with 1 input coord in several forms.'''

    varin = ds.u.isel(ocean_time=0, s_rho=-1)
    varoutcomp = varin.isel(eta_rho=ie, xi_u=ix).persist()

    # lon0/lat0 as 0d DataArrays
    lon0 = ds.lon_u.isel(eta_rho=ie, xi_u=ix)
    lat0 = ds.lat_u.isel(eta_rho=ie, xi_u=ix)
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    # lon0/lat0 as 0d numpy arrays
    lon0 = ds.lon_u.isel(eta_rho=ie, xi_u=ix).values
    lat0 = ds.lat_u.isel(eta_rho=ie, xi_u=ix).values
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))
    lat0 = float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = [float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))]
    lat0 = [float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))]
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)
    
    lon0 = np.array([float(ds.lon_u.isel(eta_rho=ie, xi_u=ix))])
    lat0 = np.array([float(ds.lat_u.isel(eta_rho=ie, xi_u=ix))])
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varoutcomp)

    
def test_llzslice_2coord():
    '''Test llzslice with multiple xy locations and depths.'''
    
    # test subset of grid nodes
    indexer = {'eta_rho': slice(ie, ie+2), 'xi_u': slice(ix, ix+3)}
    lon0 = ds.lon_u.isel(indexer)
    lat0 = ds.lat_u.isel(indexer)
    varin = ds.u.isel(ocean_time=0, s_rho=-1)
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varin.isel(indexer))
    
    # test subset of gridnodes at 1 known s_rho values
    # take mean of resultant z0s for s_rho levels to check 1 depth instead of full array
    indexerll = {'eta_rho': slice(ie, ie+2), 'xi_u': slice(ix, ix+3)}
    indexer = {'eta_rho': slice(ie, ie+2), 'xi_u': slice(ix, ix+3), 's_rho': 1}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u.isel(ocean_time=0)
    # take mean across neighboring nodes for depths to calculate at for testing purposes
    z0s = varin.isel(indexer).z_rho_u.mean(('eta_rho','xi_u')).values
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))    
    
    # test subset of gridnodes at multiple known s_rho values
    # take mean of resultant z0s for s_rho levels to check 3 depths instead of full array
    indexerll = {'eta_rho': slice(ie, ie+2), 'xi_u': slice(ix, ix+3)}
    indexer = {'eta_rho': slice(ie, ie+2), 'xi_u': slice(ix, ix+3), 's_rho': slice(1,3)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u.isel(ocean_time=0)
    # take mean across neighboring nodes for depths to calculate at for testing purposes
    z0s = varin.isel(indexer).z_rho_u.mean(('eta_rho','xi_u')).values
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))

    # test 1 gridnode at known s_rho values
    indexerll = {'eta_rho': ie, 'xi_u': ix}
    indexer = {'eta_rho': ie, 'xi_u': ix, 's_rho': slice(0,2)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u.isel(ocean_time=0)
    z0s = varin.isel(indexer).z_rho_u.values
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))
    
    # test 2d array of lon/lat locations (not consecutive grid nodes) at 1 depth
    # z0s as np.array
    indexerll = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix], 's_rho': 1}
    lon0 = ds.lon_u.isel(indexerll).values
    lat0 = ds.lat_u.isel(indexerll).values
    varin = ds.u.isel(ocean_time=0)
    z0s = varin.isel(indexer).z_rho_u.mean().values
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))

    # test 2d array of lon/lat locations (not consecutive grid nodes) at 1 depth
    # z0s as DataArray output
    indexerll = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix], 's_rho': 0}
    lon0 = ds.lon_u.isel(indexerll).values
    lat0 = ds.lat_u.isel(indexerll).values
    varin = ds.u.isel(ocean_time=0)
    z0s = varin.isel(indexer).z_rho_u.mean()
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))
    
    # test 2d array of lon/lat locations (not consecutive grid nodes) at 1 depth
    # z0s as float
    indexerll = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix], 's_rho': 2}
    lon0 = ds.lon_u.isel(indexerll).values
    lat0 = ds.lat_u.isel(indexerll).values
    varin = ds.u.isel(ocean_time=0)
    z0s = float(varin.isel(indexer).z_rho_u.mean())
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s)
    assert np.allclose(varout, varin.isel(indexer))

    # test 2d array of lon/lat locations (not consecutive grid nodes)
    indexerll = {'eta_rho': [ie,ie*ie], 'xi_u': [ix,ix*ix]}
    lon0 = ds.lon_u.isel(indexerll).values
    lat0 = ds.lat_u.isel(indexerll).values
    varin = ds.u.isel(ocean_time=0, s_rho=2)
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0)
    assert np.allclose(varout, varin.isel(indexerll))

    
def test_llzslice_triplets():
    '''Test llzslice with triplets.'''

    # test z,y,x triplets
    indexerll = {'eta_rho': ie, 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': ie, 'xi_u': [ix,ix*ix], 's_rho': slice(0,2)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u.isel(ocean_time=0)
    z0s = varin.isel(indexer).z_rho_u.mean(('xi_u')).squeeze()
    ie2 = xr.DataArray([ie,ie], dims="pts")
    ix2 = xr.DataArray([ix,ix*ix], dims="pts")
    is2 = xr.DataArray([0,1], dims="pts")
    indexer2 = {'eta_rho': ie2, 'xi_u': ix2, 's_rho': is2}
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s, triplets=True)
    assert np.allclose(varout, varin.isel(indexer2))

    # test t,z,y,x triplets
    indexerll = {'eta_rho': ie, 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': ie, 'xi_u': [ix,ix*ix], 's_rho': slice(0,2)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u
    t0s = varin.ocean_time[0:2]
    z0s = np.diag(varin.isel(indexer).z_rho_u.sel(ocean_time=t0s).mean(('xi_u')).values)
    ie2 = xr.DataArray([ie,ie], dims="pts")
    ix2 = xr.DataArray([ix,ix*ix], dims="pts")
    is2 = xr.DataArray([0,1], dims="pts")
    indexer2 = {'s_rho': is2, 'eta_rho': ie2, 'xi_u': ix2}
    varout = xroms.interp.llzslice(varin, tris['u'], lon0, lat0, z0s, triplets=True)
    varout = varout.transpose('ocean_time','pts')
    assert np.allclose(varout, varin.isel(indexer2).sel(ocean_time=t0s))    

    
def test_llzt_triplets():
    '''Test llzt with triplets.'''

    # test z,y,x triplets
    indexerll = {'eta_rho': ie, 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': ie, 'xi_u': [ix,ix*ix], 's_rho': slice(0,2)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u.isel(ocean_time=0)
    z0s = varin.isel(indexer).z_rho_u.mean(('xi_u')).squeeze()
    ie2 = xr.DataArray([ie,ie], dims="pts")
    ix2 = xr.DataArray([ix,ix*ix], dims="pts")
    is2 = xr.DataArray([0,1], dims="pts")
    indexer2 = {'eta_rho': ie2, 'xi_u': ix2, 's_rho': is2}
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0, z0s, zetaconstant=True)
    assert np.allclose(varout, varin.isel(indexer2))

    # test t,z,y,x triplets
    indexerll = {'eta_rho': ie, 'xi_u': [ix,ix*ix]}
    indexer = {'eta_rho': ie, 'xi_u': [ix,ix*ix], 's_rho': slice(0,2)}
    lon0 = ds.lon_u.isel(indexerll)
    lat0 = ds.lat_u.isel(indexerll)
    varin = ds.u
    t0s = varin.ocean_time[0:2]
    z0s = np.diag(varin.isel(indexer).z_rho_u.sel(ocean_time=t0s).mean(('xi_u')).values)
    ie2 = xr.DataArray([ie,ie], dims="pts")
    ix2 = xr.DataArray([ix,ix*ix], dims="pts")
    is2 = xr.DataArray([0,1], dims="pts")
    indexer2 = {'s_rho': is2, 'eta_rho': ie2, 'xi_u': ix2}
    varout = xroms.interp.llzt(varin, tris['u'], lon0, lat0, z0s, t0s=t0s)
    assert np.allclose(varout, varin.isel(indexer2).sel(ocean_time=t0s))    


def test_ll2xe_u_lat():
    '''Test interpolation coords for u grid.'''
    
    lon0 = ds.lon_u
    lat0 = ds.lat_u
    xi0, eta0 = xroms.interp.ll2xe(tris['u'], lon0, lat0)

    # This tests the interpolation process for finding grid coords described here:
    J, I = ds.lon_u.shape
    X, Y = np.meshgrid(np.arange(I), np.arange(J))

    assert np.allclose(xi0,X)
    assert np.allclose(eta0,Y)