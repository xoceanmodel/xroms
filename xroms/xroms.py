import xarray as xr
import xgcm
from warnings import warn
import cartopy


from .utilities import to_rho, to_psi
from .roms_seawater import buoyancy


def roms_dataset(ds, Vtransform=None, add_verts=True, proj=None):
    '''Return a dataset that is aware of ROMS coordinatates and an associated xgcm grid object with metrics
    
    Note that this could be very slow if dask is not on.

    Input
    -----

    ds :     xarray dataset

    Output
    ------

    ds :     xarray dataset
             dimensions are renamed to be consistent with xgcm
             vertical coordinates are added

    grid:    xgcm grid object
             includes ROMS metrics
    '''
    
    ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho', 'xi_psi': 'xi_u', 'eta_psi': 'eta_v'})

    coords={'X':{'center':'xi_rho', 'inner':'xi_u'},
        'Y':{'center':'eta_rho', 'inner':'eta_v'},
        'Z':{'center':'s_rho', 'outer':'s_w'}}

    grid = xgcm.Grid(ds, coords=coords, periodic=[])

    if 'Vtransform' in ds.variables.keys():
        Vtransform = ds.Vtransform

    if Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho/ds.h)
        Zo_w = ds.hc * (ds.s_w - ds.Cs_w) + ds.Cs_w * ds.h
        z_w = Zo_w + ds.zeta * (1 + Zo_w/ds.h)
        # also include z coordinates with mean sea level (constant over time)
        z_rho0 = Zo_rho
        z_w0 = Zo_w
    elif Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho
        Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
        z_w = ds.zeta + (ds.zeta + ds.h) * Zo_w
        # also include z coordinates with mean sea level (constant over time)
        z_rho0 = ds.h * Zo_rho
        z_w0 = ds.h * Zo_w
        
    else:
        print('Need a Vtransform of 1 or 2.')

    ds.coords['z_w'] = z_w.transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho',
                                     transpose_coords=False)
    ds.coords['z_w_u'] = grid.interp(ds.z_w, 'X')
    ds.coords['z_w_v'] = grid.interp(ds.z_w, 'Y')
    ds.coords['z_w_psi'] = grid.interp(ds.z_w_u, 'Y')

    ds.coords['z_rho'] = z_rho.transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho',
                                     transpose_coords=False)
    ds.coords['z_rho_u'] = grid.interp(ds.z_rho, 'X')
    ds.coords['z_rho_v'] = grid.interp(ds.z_rho, 'Y')
    ds.coords['z_rho_psi'] = grid.interp(ds.z_rho_u, 'Y')
    # also include z coordinates with mean sea level (constant over time)
    ds.coords['z_rho0'] = z_rho0.transpose('s_rho', 'eta_rho', 'xi_rho',
                                     transpose_coords=False)
    ds.coords['z_rho_u0'] = grid.interp(ds.z_rho0, 'X')
    ds.coords['z_rho_v0'] = grid.interp(ds.z_rho0, 'Y')
    ds.coords['z_rho_psi0'] = grid.interp(ds.z_rho_u0, 'Y')
    ds.coords['z_w0'] = z_w0.transpose('s_w', 'eta_rho', 'xi_rho',
                                     transpose_coords=False) 
    ds.coords['z_w_u0'] = grid.interp(ds.z_w0, 'X')
    ds.coords['z_w_v0'] = grid.interp(ds.z_w0, 'Y')
    ds.coords['z_w_psi0'] = grid.interp(ds.z_w_u0, 'Y')
    
    # add vert grid, esp for plotting pcolormesh
    if add_verts:
        import pygridgen
        if proj is None:
            proj = cartopy.crs.LambertConformal(central_longitude=-98,    central_latitude=30)
        pc = cartopy.crs.PlateCarree()
        # project points for this calculation
        xr, yr = proj.transform_points(pc, ds.lon_rho.values, ds.lat_rho.values)[...,:2].T
        xr = xr.T; yr = yr.T
        # calculate vert locations
        xv, yv = pygridgen.grid.rho_to_vert(xr, yr, ds.pm, ds.pn, ds.angle)
        # project back
        lon_vert, lat_vert = pc.transform_points(proj, xv, yv)[...,:2].T
        lon_vert = lon_vert.T; lat_vert = lat_vert.T
        # add new coords to ds
        ds.coords['lon_vert'] = (('eta_vert', 'xi_vert'), lon_vert)
        ds.coords['lat_vert'] = (('eta_vert', 'xi_vert'), lat_vert)

    ds['pm_v'] = grid.interp(ds.pm, 'Y')
    ds['pn_u'] = grid.interp(ds.pn, 'X')
    ds['pm_u'] = grid.interp(ds.pm, 'X')
    ds['pn_v'] = grid.interp(ds.pn, 'Y')
    ds['pm_psi'] = grid.interp(grid.interp(ds.pm, 'Y'),  'X') # at psi points (eta_v, xi_u)
    ds['pn_psi'] = grid.interp(grid.interp(ds.pn, 'X'),  'Y') # at psi points (eta_v, xi_u)

    ds['dx'] = 1/ds.pm
    ds['dx_u'] = 1/ds.pm_u
    ds['dx_v'] = 1/ds.pm_v
    ds['dx_psi'] = 1/ds.pm_psi

    ds['dy'] = 1/ds.pn
    ds['dy_u'] = 1/ds.pn_u
    ds['dy_v'] = 1/ds.pn_v
    ds['dy_psi'] = 1/ds.pn_psi

    ds['dz'] = grid.diff(ds.z_w, 'Z')
    ds['dz_w'] = grid.diff(ds.z_rho, 'Z', boundary='fill')
    ds['dz_u'] = grid.interp(ds.dz, 'X')
    ds['dz_w_u'] = grid.interp(ds.dz_w, 'X')
    ds['dz_v'] = grid.interp(ds.dz, 'Y')
    ds['dz_w_v'] = grid.interp(ds.dz_w, 'Y')
    ds['dz_psi'] = grid.interp(ds.dz_v, 'X')
    ds['dz_w_psi'] = grid.interp(ds.dz_w_v, 'X')

    # also include z coordinates with mean sea level (constant over time)
    ds['dz0'] = grid.diff(ds.z_w0, 'Z')
    ds['dz_w0'] = grid.diff(ds.z_rho0, 'Z', boundary='fill')
    ds['dz_u0'] = grid.interp(ds.dz0, 'X')
    ds['dz_w_u0'] = grid.interp(ds.dz_w0, 'X')
    ds['dz_v0'] = grid.interp(ds.dz0, 'Y')
    ds['dz_w_v0'] = grid.interp(ds.dz_w0, 'Y')
    ds['dz_psi0'] = grid.interp(ds.dz_v0, 'X')
    ds['dz_w_psi0'] = grid.interp(ds.dz_w_v0, 'X')

    ds['dA'] = ds.dx * ds.dy

    metrics = {
        ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'], # X distances
        ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
        ('Z',): ['dz', 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v', 'dz_psi', 'dz_w_psi'], # Z distances
        ('X', 'Y'): ['dA'] # Areas
    }
    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

    return ds, grid


def open_netcdf(files, chunks=None):
    '''Return an xarray.Dataset based on a list of netCDF files

    Inputs:
    files       A list of netCDF files

    Output:
    ds          An xarray.Dataset

    Options:
    chunks      The specified chunks for the DataSet.
                Default: chunks = {'ocean_time':1}
    '''

    if chunks is None:
        chunks = {'ocean_time':1}   # A basic chunking option

    return xr.open_mfdataset(files, compat='override', combine='by_coords',
                                 data_vars='minimal', coords='minimal', chunks=chunks)


def open_zarr(files, chunks=None):
    '''Return an xarray.Dataset based on a list of zarr files

    Inputs:
    files       A list of zarr files

    Output:
    ds          An xarray.Dataset

    Options:
    chunks      The specified chunks for the DataSet.
                Default: chunks = {'ocean_time':1}
    '''
    if chunks is None:
        chunks = {'ocean_time':1}   # A basic chunking option

    opts = {'consolidated': True,
            'chunks': chunks, 'drop_variables': 'dstart'}
    return xr.concat(
        [xr.open_zarr(file, **opts) for file in files],
        dim='ocean_time', data_vars='minimal', coords='minimal')
    

def hgrad(q, grid, which='both', z=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Return gradients of property q in the ROMS curvilinear grid native xi- and eta- directions
    
    The main purpose of this it to account for the fact that ROMS vertical coordinates are 
    sigma coordinates.

    Inputs:
    ------

    q               DataArray, Property to take gradients of

    grid            xgcm object, Grid object associated with DataArray q

    Outputs:
    -------

    dqdxi, dqdeta   Gradients of q in the xi- and eta-directions


    Options:
    -------
    
    which           string ('both'). 'both': return both components of hgrad. 'xi': return only 
                     xi-direction. 'eta': return only eta-direction.

    z               DataArray. The vertical depths associated with q. Default is to find the
                    coordinate of q that starts with 'z_', and use that.

    hboundary        Passed to `grid` method calls for horizontal calculations. Default is `extend`
    sboundary        Passed to `grid` method calls for vertical calculations. Default is `extend`
    '''

    if z is None:
        coords = list(q.coords)
        z_coord_name = coords[[coord[:2] == 'z_' for coord in coords].index(True)]
        z = q[z_coord_name]
        
    if which in ['both','xi']:

        dqdx = grid.interp(grid.derivative(q, 'X', boundary=hboundary, fill_value=hfill_value), 'Z', boundary=sboundary, fill_value=sfill_value)
        dqdz = grid.interp(grid.derivative(q, 'Z', boundary=sboundary, fill_value=sfill_value), 'X', boundary=hboundary, fill_value=hfill_value)
        dzdx = grid.interp(grid.derivative(z, 'X', boundary=hboundary, fill_value=hfill_value), 'Z', boundary=sboundary, fill_value=sfill_value)
        dzdz = grid.interp(grid.derivative(z, 'Z', boundary=sboundary, fill_value=sfill_value), 'X', boundary=hboundary, fill_value=hfill_value)

        dqdxi = dqdx*dzdz - dqdz*dzdx

    if which in ['both','eta']:
        
        dqdy = grid.interp(grid.derivative(q, 'Y', boundary=hboundary, fill_value=hfill_value), 'Z', boundary=sboundary, fill_value=sfill_value)
        dqdz = grid.interp(grid.derivative(q, 'Z', boundary=sboundary, fill_value=sfill_value), 'Y', boundary=hboundary, fill_value=hfill_value)
        dzdy = grid.interp(grid.derivative(z, 'Y', boundary=hboundary, fill_value=hfill_value), 'Z', boundary=sboundary, fill_value=sfill_value)
        dzdz = grid.interp(grid.derivative(z, 'Z', boundary=sboundary, fill_value=sfill_value), 'Y', boundary=hboundary, fill_value=hfill_value)

        dqdeta = dqdy*dzdz - dqdz*dzdy
    
    if which == 'both':
        return dqdxi, dqdeta
    elif which == 'xi':
        return dqdxi
    elif which == 'eta':
        return dqdeta
    else:
        print('nothing being returned from hgrad')
    

def relative_vorticity(ds, grid, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Return the vertical component of the relative vorticity on rho-points
    
    
    Inputs:
    ------
    ds              ROMS dataset, needs to include grid metrics: dz_rho_u, dz_rho_v
    
    grid            xgcm object, Grid object associated with DataArray phi
    
    
    Outputs:
    -------
    rel_vort        The relative vorticity, v_x - u_y, on rho-points.


    Options:
    -------
    hboundary        Passed to `grid` method calls. Default is `extend`
    
    '''
    
    dvdxi = hgrad(ds.v, grid, which='xi', hboundary=hboundary, hfill_value=hfill_value, 
                                          sboundary=sboundary, sfill_value=sfill_value)
    dudeta = hgrad(ds.u, grid, which='eta', hboundary=hboundary, hfill_value=hfill_value, 
                                            sboundary=sboundary, sfill_value=sfill_value)

    return dvdxi - dudeta
