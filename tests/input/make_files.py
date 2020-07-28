'''Make example netCDF and zarr files for xroms testing.'''


import numpy as np
import xarray as xr
import pandas as pd
import pygridgen
from pyproj import Proj
import octant


def grid():
    '''Make test grid.
    
    It is spherical.'''
    
    # Define dimensions and parameters
    xl = 15  # horizontal dimensions
    yl = 10
    h = 100
    
    # set up projection
    inputs = {'proj': 'lcc', 'ellps': 'clrk66', 'datum': 'NAD27',
          'lat_1': 22.5, 'lat_2': 31.0, 'lat_0': 30, 'lon_0': -94,
          'x_0': 0, 'y_0': 0}
    proj = Proj(**inputs)

    # set up horizontal grid
    lonv = np.linspace(-96, -94, xl)
    latv = np.linspace(27, 28.5, yl)
    lon_vert, lat_vert = np.meshgrid(lonv, latv)
    grd = pygridgen.grid.CGrid_geo(lon_vert, lat_vert, proj)  
    
    # add-on
    grd.h = h
    
    # save grid
    octant.roms.write_grd(grd, filename='grid.nc', full_output=True, verbose=True)
    
    
def output(to='netcdf'):
    '''Make test model output to go with grid.'''
    
    # Read in grid for sizing info
    grid = xr.open_dataset('grid.nc')
    xl = grid.lon_rho.shape[1]
    yl = grid.x_rho.shape[0]

    # Add in vertical grid info
    N = 3  # vertical
    s_rho = np.linspace(-0.975,-0.025,N)
    s_w = np.linspace(-1,0,N+1)
    hc = 0.
    Cs_r = np.linspace(-0.975,-0.025,N)
    Cs_w = np.linspace(-1,0,N+1)
    theta_b = 0.
    hmin = 100.
    theta_s = 1e-4
    tcline = 0.

    # Time
    tl = 4  # three time outputs
    dt = '1 hour'  # 1 hour between outputs
    startdate = pd.Timestamp('2013-12-17')
    ts = [startdate + it*pd.Timedelta(dt) for it in range(tl)]
    
    # Make velocity fields
    u = 0.1*np.ones((tl,N,yl,xl-1)) 
    v = np.zeros((tl,N,yl-1,xl))

    # Save file, starting from grid file
    # 1st file
    out1 = xr.open_dataset('grid.nc')
    out1['ocean_time'] = ts[:2]
    out1['u'] = (('ocean_time','s_rho','eta_u','xi_u'), u[:2])
    out1['v'] = (('ocean_time','s_rho','eta_v','xi_v'), v[:2])
    out1 = out1.assign_coords(s_rho=s_rho, lat_u=grid.lat_u, lon_u=grid.lon_u, lat_v=grid.lat_v, lon_v=grid.lon_v, lat_rho=grid.lat_rho, lon_rho=grid.lon_rho)
    # 2nd file
    out2 = xr.open_dataset('grid.nc')
    out2['ocean_time'] = ts[2:]
    out2['u'] = (('ocean_time','s_rho','eta_u','xi_u'), u[2:])
    out2['v'] = (('ocean_time','s_rho','eta_v','xi_v'), v[2:])
    out2 = out2.assign_coords(s_rho=s_rho, lat_u=grid.lat_u, lon_u=grid.lon_u, lat_v=grid.lat_v, lon_v=grid.lon_v, lat_rho=grid.lat_rho, lon_rho=grid.lon_rho)
    
    fname1 = 'ocean_his_0001'
    fname2 = 'ocean_his_0002'

    if to=='netcdf':
        out1.to_netcdf(fname1 + '.nc', mode='w')
        out2.to_netcdf(fname2 + '.nc', mode='w')

    elif to=='zarr':
        out1.to_zarr(fname1, mode='w', consolidated=True)
        out2.to_zarr(fname2, mode='w', consolidated=True)
    