import xarray as xr
import xgcm
from warnings import warn
import cartopy
import xroms
import numpy as np


from .utilities import xisoslice, to_grid
from .roms_seawater import buoyancy

g = 9.81  # m/s^2

def roms_dataset(ds, Vtransform=None, add_verts=False, proj=None):
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

    rename = {}
    if 'eta_u' in ds.dims:
        rename['eta_u'] = 'eta_rho'
    if 'xi_v' in ds.dims:
        rename['xi_v'] = 'xi_rho'
    if 'xi_psi' in ds.dims:
        rename['xi_psi'] = 'xi_u'
    if 'eta_psi' in ds.dims:
        rename['eta_psi'] = 'eta_v'
    ds = ds.rename(rename)

#     ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho', 'xi_psi': 'xi_u', 'eta_psi': 'eta_v'})

    coords={'X':{'center':'xi_rho', 'inner':'xi_u'},
        'Y':{'center':'eta_rho', 'inner':'eta_v'},
        'Z':{'center':'s_rho', 'outer':'s_w'}}

    grid = xgcm.Grid(ds, coords=coords, periodic=[])

    if "Vtransform" in ds.variables.keys():
        Vtransform = ds.Vtransform

    assert Vtransform is not None, 'Need a Vtransform of 1 or 2, either in the Dataset or input to the function.'


    if Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / ds.h)
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
        
    # the dims present in this process could be different depending on the output sent in with the Dataset
    dims_rho = list(ds.salt.dims)
    dims_rho0 = dims_rho.copy()
    if np.sum(['time' in dim for dim in dims_rho0]):
        dims_rho0.pop(['time' in dim for dim in dims_rho0].index(True))
    dims_w = dims_rho.copy()
    # if s_rho is included in dims for this Dataset, rename it to s_w for dims_w
    if dims_w.count('s_rho'):
        dims_w[dims_w.index('s_rho')] = 's_w'
    dims_w0 = dims_w.copy()
    if np.sum(['time' in dim for dim in dims_w0]):
        dims_w0.pop(['time' in dim for dim in dims_w0].index(True))

    ds.coords['z_w'] = z_w.transpose(*dims_w,
                                     transpose_coords=False)
#     ds.coords['z_w'] = z_w.transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho',
#                                      transpose_coords=False)
    ds.coords['z_w_u'] = grid.interp(ds.z_w, 'X')
    ds.coords['z_w_v'] = grid.interp(ds.z_w, 'Y')
    ds.coords['z_w_psi'] = grid.interp(ds.z_w_u, 'Y')

    ds.coords['z_rho'] = z_rho.transpose(*dims_rho,
                                     transpose_coords=False)
#     ds.coords['z_rho'] = z_rho.transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho',
#                                      transpose_coords=False)
    ds.coords['z_rho_u'] = grid.interp(ds.z_rho, 'X')
    ds.coords['z_rho_v'] = grid.interp(ds.z_rho, 'Y')
    ds.coords['z_rho_psi'] = grid.interp(ds.z_rho_u, 'Y')
    # also include z coordinates with mean sea level (constant over time)
    ds.coords['z_rho0'] = z_rho0.transpose(*dims_rho0,
                                     transpose_coords=False)
#     ds.coords['z_rho0'] = z_rho0.transpose('s_rho', 'eta_rho', 'xi_rho',
#                                      transpose_coords=False)
    ds.coords['z_rho_u0'] = grid.interp(ds.z_rho0, 'X')
    ds.coords['z_rho_v0'] = grid.interp(ds.z_rho0, 'Y')
    ds.coords['z_rho_psi0'] = grid.interp(ds.z_rho_u0, 'Y')
    ds.coords['z_w0'] = z_w0.transpose(*dims_w0,
                                     transpose_coords=False)
#     ds.coords['z_w0'] = z_w0.transpose('s_w', 'eta_rho', 'xi_rho',
#                                      transpose_coords=False)
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

    # grid areas
    ds['dA'] = ds.dx * ds.dy
    ds['dA_u'] = ds.dx_u * ds.dy_u
    ds['dA_v'] = ds.dx_v * ds.dy_v
    ds['dA_psi'] = ds.dx_psi * ds.dy_psi
    
    # volume
    ds['dV'] = ds.dz * ds.dx * ds.dy   # rho vertical, rho horizontal
    ds['dV_w'] = ds.dz_w * ds.dx * ds.dy  # w vertical, rho horizontal
    ds['dV_u'] = ds.dz_u * ds.dx_u * ds.dy_u  # rho vertical, u horizontal
    ds['dV_w_u'] = ds.dz_w_u * ds.dx_u * ds.dy_u  # w vertical, u horizontal
    ds['dV_v'] = ds.dz_v * ds.dx_v * ds.dy_v  # rho vertical, v horizontal
    ds['dV_w_v'] = ds.dz_w_v * ds.dx_v * ds.dy_v  # w vertical, v horizontal
    ds['dV_psi'] = ds.dz_psi * ds.dx_psi * ds.dy_psi  # rho vertical, psi horizontal
    ds['dV_w_psi'] = ds.dz_w_psi * ds.dx_psi * ds.dy_psi  # w vertical, psi horizontal

    metrics = {
        ("X",): ["dx", "dx_u", "dx_v", "dx_psi"],  # X distances
        ("Y",): ["dy", "dy_u", "dy_v", "dy_psi"],  # Y distances
        ("Z",): [
            "dz",
            "dz_u",
            "dz_v",
            "dz_w",
            "dz_w_u",
            "dz_w_v",
            "dz_psi",
            "dz_w_psi",
        ],  # Z distances
        ("X", "Y"): ["dA"],  # Areas
    }
    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

    return ds, grid


def open_netcdf(files, chunks=None, Vtransform=None, add_verts=False, proj=None, parallel=True):
    '''Return an xarray.Dataset based on a list of netCDF files

    Inputs:
    files       Where to find the model output. `files` could be: A list of netCDF file
                names, a string of netCDF file name, or a string of a thredds server 
                containing model output.

    Output:
    ds          An xarray.Dataset

    Options:
    chunks      The specified chunks for the DataSet.
                Default: chunks = {'ocean_time':1}
    parallel    (True) To be passed to `xarray open_mfdataset`.
    '''

    if chunks is None:
        chunks = {"ocean_time": 1}  # A basic chunking, ok, but maybe not the best

    if isinstance(files, list):
        ds = xr.open_mfdataset(files, compat='override', combine='by_coords',
                               data_vars='minimal', coords='minimal', 
                               chunks=chunks, parallel=parallel)
    elif isinstance(files, str):
        ds = xr.open_dataset(files, chunks=chunks)

    ds, grid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)
    ds.attrs['grid'] = grid
    # also put grid into every variable with at least 2D
    for var in ds.variables:
        if ds[var].ndim > 1:
            ds[var].attrs['grid'] = ds.attrs['grid']

    return ds


def open_zarr(files, chunks=None, Vtransform=None, add_verts=False, proj=None):
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
    ds = xr.concat(
        [xr.open_zarr(file, **opts) for file in files],
        dim='ocean_time', data_vars='minimal', coords='minimal')

    ds, grid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)
    ds.attrs['grid'] = grid
    # also put grid into every variable with at least 2D
    for var in ds.variables:
        if ds[var].ndim > 1:
            ds[var].attrs['grid'] = ds.attrs['grid']
    
    return ds


def hgrad(q, grid, which='both', z=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None, attrs=None):
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
        try:
            coords = list(q.coords)
            z_coord_name = coords[[coord[:2] == "z_" for coord in coords].index(True)]
            z = q[z_coord_name]
            is3D = True
        except:
            # if we get here that means that q doesn't have z coords (like zeta)
            is3D = False
    else:
        is3D = True
            

    if which in ['both','xi']:

        if is3D:
            dqdx = grid.interp(grid.derivative(q, 'X', boundary=hboundary, fill_value=hfill_value), 
                               'Z', boundary=sboundary, fill_value=sfill_value)
            dqdz = grid.interp(grid.derivative(q, 'Z', boundary=sboundary, fill_value=sfill_value), 
                               'X', boundary=hboundary, fill_value=hfill_value)
            dzdx = grid.interp(grid.derivative(z, 'X', boundary=hboundary, fill_value=hfill_value), 
                               'Z', boundary=sboundary, fill_value=sfill_value)
            dzdz = grid.interp(grid.derivative(z, 'Z', boundary=sboundary, fill_value=sfill_value), 
                               'X', boundary=hboundary, fill_value=hfill_value)

            dqdxi = dqdx*dzdz - dqdz*dzdx
        
        else:  # 2D variables
            dqdxi = grid.derivative(q, 'X', boundary=hboundary, fill_value=hfill_value)

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs['name'] = 'd' + q.name  + 'dxi'
            attrs['units'] = '1/m * ' + attrs.setdefault('units', 'units')
            attrs['long_name']  = 'horizontal xi derivative of ' + attrs.setdefault('long_name', 'var')
            attrs['grid'] = grid
        dqdxi = xroms.to_grid(dqdxi, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                             hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)  

    if which in ['both','eta']:
        
        if is3D:
            dqdy = grid.interp(grid.derivative(q, 'Y', boundary=hboundary, fill_value=hfill_value), 
                               'Z', boundary=sboundary, fill_value=sfill_value)
            dqdz = grid.interp(grid.derivative(q, 'Z', boundary=sboundary, fill_value=sfill_value), 
                               'Y', boundary=hboundary, fill_value=hfill_value)
            dzdy = grid.interp(grid.derivative(z, 'Y', boundary=hboundary, fill_value=hfill_value), 
                               'Z', boundary=sboundary, fill_value=sfill_value)
            dzdz = grid.interp(grid.derivative(z, 'Z', boundary=sboundary, fill_value=sfill_value), 
                               'Y', boundary=hboundary, fill_value=hfill_value)

            dqdeta = dqdy*dzdz - dqdz*dzdy
        
        else:  # 2D variables
            dqdeta = grid.derivative(q, 'Y', boundary=hboundary, fill_value=hfill_value)

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs['name'] = 'd' + q.name  + 'deta'
            attrs['units'] = '1/m * ' + attrs.setdefault('units', 'units')
            attrs['long_name']  = 'horizontal eta derivative of ' + attrs.setdefault('long_name', 'var')
            attrs['grid'] = grid
        dqdeta = xroms.to_grid(dqdeta, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                              hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)  
        

    if which == 'both':
        return dqdxi, dqdeta
    elif which == 'xi':
        return dqdxi
    elif which == 'eta':
        return dqdeta
    else:
        print('nothing being returned from hgrad')


def relative_vorticity(u, v, grid, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Return the vertical component of the relative vorticity on rho-points


    Inputs:
    ------
    u, v            (DataArray) xi, eta components of velocity

    grid            xgcm object, Grid object associated with DataArray phi


    Outputs:
    -------
    rel_vort        The relative vorticity, v_x - u_y, on rho-points.


    Options:
    -------
    hboundary        Passed to `grid` method calls. Default is `extend`

    '''

    dvdxi = hgrad(v, grid, which='xi', hboundary=hboundary, hfill_value=hfill_value,
                                          sboundary=sboundary, sfill_value=sfill_value)
    dudeta = hgrad(u, grid, which='eta', hboundary=hboundary, hfill_value=hfill_value,
                                            sboundary=sboundary, sfill_value=sfill_value)

    var = dvdxi - dudeta
    attrs = {'name': 'vort', 'long_name': 'vertical component of vorticity', 
             'units': '1/s', 'grid': grid}
    var = to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                 hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)  

    return var


def KE(rho, speed, grid, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan):
    '''Calculate kinetic energy [kg/(m*s^2)].
    
    Inputs:
    hcoord     string (None). Name of horizontal grid to interpolate variable
               to. Options are 'rho' and 'psi'.
    scoord     string (None). Name of vertical grid to interpolate variable
               to. Options are 's_rho' and 's_w'.
    '''
    
    attrs = {'name': 'KE', 'long_name': 'kinetic energy', 
             'units': 'kg/(m*s^2)', 'grid': grid}
    var = 0.5*rho*speed**2
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    return var


def speed(u, v, grid, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan):
    '''Calculate horizontal speed [m/s].
    
    Inputs:
    hcoord     string (None). Name of horizontal grid to interpolate variable
               to. Options are 'rho' and 'psi'.
    scoord     string (None). Name of vertical grid to interpolate variable
               to. Options are 's_rho' and 's_w'.
    '''
    
    attrs = {'name': 's', 'long_name': 'horizontal speed', 
             'units': 'm/s', 'grid': grid}
    u = xroms.to_rho(u, grid, boundary=hboundary, fill_value=hfill_value)
    v = xroms.to_rho(v, grid, boundary=hboundary, fill_value=hfill_value)
    var = np.sqrt(u**2 + v**2)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs, 
                        hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    return var


def ertel(phi, u, v, f, grid, hcoord='rho', scoord='s_rho',
          hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Return Ertel potential vorticity of phi.

    Inputs:
    ------
    phi             (DataArray) Conservative tracer. Usually this would be 
                    the buoyancy but could be another approximately 
                    conservative tracer. The buoyancy can be calculated as:
                    > xroms.buoyancy(temp, salt, 0)
                    and then input as `phi`. 
                    
    u, v            (DataArray) xi, eta components of velocity
    
    f               (DataArray) Coriolis array

    grid            xgcm object, Grid object associated with Dataset.


    Outputs:
    -------
    epv             The ertel potential vorticity
                    epv = -v_z * phi_x + u_z * phi_y + (f + v_x - u_y) * phi_z

    Options:
    -------
    hcoord     string (None). Name of horizontal grid to interpolate variable
               to. Options are 'rho' and 'psi'.
    scoord     string (None). Name of vertical grid to interpolate variable
               to. Options are 's_rho' and 's_w'.
    boundary        Passed to `grid` method calls. Default is `extend`
    '''

    # get the components of the grad(phi)
    phi_xi, phi_eta = hgrad(phi, grid, hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    phi_xi = xroms.to_grid(phi_xi, grid, hcoord=hcoord, scoord=scoord)
    phi_eta = xroms.to_grid(phi_eta, grid, hcoord=hcoord, scoord=scoord)
    phi_z = xroms.ddz(phi, grid, hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=np.nan)

    # vertical shear (horizontal components of vorticity)
    u_z = xroms.dudz(u, grid, hcoord=hcoord, scoord=scoord)
    v_z = xroms.dvdz(v, grid, hcoord=hcoord, scoord=scoord)

    # vertical component of vorticity on rho grid
    vort = relative_vorticity(u, v, grid, hcoord=hcoord, scoord=scoord)

    # combine terms to get the ertel potential vorticity
    epv = -v_z * phi_xi + u_z * phi_eta + (f + vort) * phi_z

    attrs = {'name': 'ertel', 'long_name': 'ertel potential vorticity', 
             'units': 'tracer/(m*s)', 'grid': grid}
    epv = xroms.to_grid(epv, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)

    return epv


def uv_geostrophic(zeta, f, grid, hcoord=None, scoord=None,
          hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Calculate geostrophic velocities from zeta.
    
    Copy of copy of surf_geostr_vel of IRD Roms_Tools.
    
    v = g * zeta_eta / (d eta * f)
    u = -g * zeta_xi / (d xi * f)
    '''
    
    attrsu = {'name': 'u_geo', 'long_name': 'geostrophic u velocity', 
            'units': 'm/s', 'grid': grid}
    attrsv = {'name': 'v_geo', 'long_name': 'geostrophic v velocity', 
            'units': 'm/s', 'grid': grid}
    
    # calculate derivatives of zeta
    dzetadxi, dzetadeta = hgrad(zeta, grid)#, hcoord='rho', hboundary='extend')
    
    # calculate geostrophic velocities
    vbar = g*dzetadeta/xroms.to_v(f, grid, boundary='extend')
    ubar = -g*dzetadxi/xroms.to_u(f, grid, boundary='extend')
    
    ubar = xroms.to_grid(ubar, grid, hcoord=hcoord, scoord=scoord, attrs=attrsu,
                         hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    vbar = xroms.to_grid(vbar, grid, hcoord=hcoord, scoord=scoord, attrs=attrsv,
                         hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    return ubar, vbar


def EKE(ug, vg, grid, hcoord='rho', scoord='s_rho',
          hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
    '''Calculate EKE.'''
    
    attrs = {'name': 'EKE', 'long_name': 'eddy kinetic energy', 
            'units': 'm^2/s^2', 'grid': grid}
    
    # make sure geostrophic velocities are on rho grid
    ug = xroms.to_rho(ug, grid, boundary='extend')
    vg = xroms.to_rho(vg, grid, boundary='extend')
    
    var = 0.5*(ug**2 + vg**2)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                     hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    return var
