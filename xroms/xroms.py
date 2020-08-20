import xarray as xr
import xgcm
from warnings import warn

from .utilities import to_rho, to_psi
from .roms_seawater import buoyancy

def roms_dataset(ds, Vtransform=None):
    """Return a dataset that is aware of ROMS coordinatates and an associated xgcm grid object with metrics

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
    """
    ds = ds.rename(
        {"eta_u": "eta_rho", "xi_v": "xi_rho", "xi_psi": "xi_u", "eta_psi": "eta_v"}
    )

    coords = {
        "X": {"center": "xi_rho", "inner": "xi_u"},
        "Y": {"center": "eta_rho", "inner": "eta_v"},
        "Z": {"center": "s_rho", "outer": "s_w"},
    }

    grid = xgcm.Grid(ds, coords=coords, periodic=[])

    if "Vtransform" in ds.variables.keys():
        Vtransform = ds.Vtransform

    if Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / ds.h)
        Zo_w = ds.hc * (ds.s_w - ds.Cs_w) + ds.Cs_w * ds.h
        z_w = Zo_w + ds.zeta * (1 + Zo_w / ds.h)
    elif Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho
        Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
        z_w = ds.zeta + (ds.zeta + ds.h) * Zo_w

    ds.coords["z_w"] = z_w.transpose(
        "ocean_time", "s_w", "eta_rho", "xi_rho", transpose_coords=False
    )
    ds.coords["z_w_u"] = grid.interp(ds.z_w, "X")
    ds.coords["z_w_v"] = grid.interp(ds.z_w, "Y")
    ds.coords["z_w_psi"] = grid.interp(ds.z_w_u, "Y")

    ds.coords["z_rho"] = z_rho.transpose(
        "ocean_time", "s_rho", "eta_rho", "xi_rho", transpose_coords=False
    )
    ds.coords["z_rho_u"] = grid.interp(ds.z_rho, "X")
    ds.coords["z_rho_v"] = grid.interp(ds.z_rho, "Y")
    ds.coords["z_rho_psi"] = grid.interp(ds.z_rho_u, "Y")

    ds["pm_v"] = grid.interp(ds.pm, "Y")
    ds["pn_u"] = grid.interp(ds.pn, "X")
    ds["pm_u"] = grid.interp(ds.pm, "X")
    ds["pn_v"] = grid.interp(ds.pn, "Y")
    ds["pm_psi"] = grid.interp(
        grid.interp(ds.pm, "Y"), "X"
    )  # at psi points (eta_v, xi_u)
    ds["pn_psi"] = grid.interp(
        grid.interp(ds.pn, "X"), "Y"
    )  # at psi points (eta_v, xi_u)

    ds["dx"] = 1 / ds.pm
    ds["dx_u"] = 1 / ds.pm_u
    ds["dx_v"] = 1 / ds.pm_v
    ds["dx_psi"] = 1 / ds.pm_psi

    ds["dy"] = 1 / ds.pn
    ds["dy_u"] = 1 / ds.pn_u
    ds["dy_v"] = 1 / ds.pn_v
    ds["dy_psi"] = 1 / ds.pn_psi

    ds["dz"] = grid.diff(ds.z_w, "Z")
    ds["dz_w"] = grid.diff(ds.z_rho, "Z", boundary="fill")
    ds["dz_u"] = grid.interp(ds.dz, "X")
    ds["dz_w_u"] = grid.interp(ds.dz_w, "X")
    ds["dz_v"] = grid.interp(ds.dz, "Y")
    ds["dz_w_v"] = grid.interp(ds.dz_w, "Y")
    ds["dz_psi"] = grid.interp(ds.dz_v, "X")
    ds["dz_w_psi"] = grid.interp(ds.dz_w_v, "X")

    ds["dA"] = ds.dx * ds.dy

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

def open_netcdf(files, chunks=None):
    '''Return an xarray.Dataset based on a list of netCDF files

    Inputs:
    files       A list of netCDF files

    Output:
    ds          An xarray.Dataset

    Options:
    chunks      The specified chunks for the DataSet.
                Default: chunks = {'ocean_time':1}
    """

    if chunks is None:
        chunks = {"ocean_time": 1}  # A basic chunking, ok, but maybe not the best

    return xr.open_mfdataset(
        files,
        compat="override",
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        chunks=chunks,
    )


def open_zarr(files, chunks=None):
    '''Return an xarray.Dataset based on a list of zarr files

    Inputs:
    files       A list of zarr files

    Output:
    ds          An xarray.Dataset

    Options:
    chunks      The specified chunks for the DataSet.
                Default: chunks = {'ocean_time':1}
    """
    if chunks is None:

        chunks = {'ocean_time':1}   # A basic chunking option

    opts = {'consolidated': True,
            'chunks': chunks, 'drop_variables': 'dstart'}
    return xr.concat(
        [xr.open_zarr(file, **opts) for file in files],
        dim='ocean_time', data_vars='minimal', coords='minimal')
    

def hgrad(q, grid, z=None, boundary="extend", to=None):
    """Return gradients of property q in the ROMS curvilinear grid native xi- and eta- directions

    Inputs:
    ------

    q               DataArray, Property to take gradients of

    grid            xgcm object, Grid object associated with DataArray q

    Outputs:
    -------

    dqdxi, dqdeta   Gradients of q in the xi- and eta-directions


    Options:
    -------

    z               DataArray. The vertical depths associated with q. Default is to find the
                    coordinate of q that starts with 'z_', and use that.

    to              By default, gradient values are located at the midpoints between q points
                    in each direction of the gradient. Optionally, these can be interpolated
                    to either rho- or psi-points passing `rho` or `psi`

    boundary        Passed to `grid` method calls. Default is `extend`
    '''

    if z is None:
        coords = list(q.coords)
        z_coord_name = coords[[coord[:2] == "z_" for coord in coords].index(True)]
        z = q[z_coord_name]

    dqdx = grid.interp(
        grid.derivative(q, "X", boundary=boundary), "Z", boundary=boundary
    )
    dqdz = grid.interp(
        grid.derivative(q, "Z", boundary=boundary), "X", boundary=boundary
    )
    dzdx = grid.interp(
        grid.derivative(z, "X", boundary=boundary), "Z", boundary=boundary
    )
    dzdz = grid.interp(
        grid.derivative(z, "Z", boundary=boundary), "X", boundary=boundary
    )

    dqdxi = dqdx * dzdz - dqdz * dzdx

    dqdy = grid.interp(
        grid.derivative(q, "Y", boundary=boundary), "Z", boundary=boundary
    )
    dqdz = grid.interp(
        grid.derivative(q, "Z", boundary=boundary), "Y", boundary=boundary
    )
    dzdy = grid.interp(
        grid.derivative(z, "Y", boundary=boundary), "Z", boundary=boundary
    )
    dzdz = grid.interp(
        grid.derivative(z, "Z", boundary=boundary), "Y", boundary=boundary
    )

    dqdeta = dqdy * dzdz - dqdz * dzdy

    if to == "rho":
        return to_rho(dqdxi, grid), to_rho(dqdeta, grid)
    if to == "psi":
        return to_psi(dqdxi), to_psi(dqdeta)
    else:
        return dqdxi, dqdeta


def relative_vorticity(ds, grid, boundary='extend', to='rho'):
    '''Return the relative vorticity on rho-points
    
    
    Inputs:
    ------
    ds              ROMS dataset, needs to include grid metrics: dz_rho_u, dz_rho_v
    
    grid            xgcm object, Grid object associated with DataArray phi
    
    
    Outputs:
    -------
    rel_vort        The relative vorticity, v_x - u_y, on rho-points.


    Options:
    -------
    boundary        Passed to `grid` method calls. Default is `extend`
    
    to              Grid to interpolate to, default is horizontal and vertical rho-points, 
                    otherwise on horizontal psi-points, vertical w-points.
    '''

    dvdx = grid.interp(grid.derivative(ds.v, 'X', boundary=boundary), 'Z', boundary=boundary)
    dvdz = grid.interp(grid.derivative(ds.v, 'Z', boundary=boundary), 'X', boundary=boundary)
    dzdx = grid.interp(grid.derivative(ds.z_rho_v, 'X', boundary=boundary), 'Z', boundary=boundary)
    dzdz = grid.interp(grid.derivative(ds.z_rho_v, 'Z', boundary=boundary), 'X', boundary=boundary)

    dvdxi = dvdx*dzdz - dvdz*dzdx

    dudy = grid.interp(grid.derivative(ds.u, 'Y', boundary=boundary), 'Z', boundary=boundary)
    dudz = grid.interp(grid.derivative(ds.u, 'Z', boundary=boundary), 'Y', boundary=boundary)
    dzdy = grid.interp(grid.derivative(ds.z_rho_u, 'Y', boundary=boundary), 'Z', boundary=boundary)
    dzdz = grid.interp(grid.derivative(ds.z_rho_u, 'Z', boundary=boundary), 'Y', boundary=boundary)

    dudeta = dudy*dzdz - dudz*dzdy

    if to == 'rho':
        return ( to_rho(grid.interp(dvdxi, 'Z', boundary=boundary), grid) - 
                 to_rho(grid.interp(dudeta, 'Z', boundary=boundary), grid) )
    else:
        return dvdxi - dudeta


def ertel(ds, grid, tracer=None, boundary='extend'):
    '''Return gradients of property q in the ROMS curvilinear grid native xi- and eta- directions

    Inputs:
    ------
    ds              ROMS dataset
    
    grid            xgcm object, Grid object associated with DataArray phi
    
    
    Outputs:
    -------
    epv             The ertel potential vorticity
                    epv = -v_z * phi_x + u_z * phi_y + (f + v_x - u_y) * phi_z

    Options:
    -------
    tracer          The tracer to use in calculating EPV. Default is buoyancy. 
                    Buoyancy calculated from salt and temp if rho is not present.

    boundary        Passed to `grid` method calls. Default is `extend`
    '''
    
    # load appropriate tracer into 'phi', defalut rho. Use EOS if necessary
    if tracer is None:
        if 'rho' in ds.variables:
            phi =  -9.8 * ds.rho/1000.0 
        else:
            phi = buoyancy(ds.temp, ds.salt, 0)
    else:
        phi = ds[tracer]
    
    # get the components of the grad(phi)
    phi_xi, phi_eta = hgrad(phi, grid, to='rho', boundary=boundary)
    phi_xi = grid.interp(phi_xi, 'Z', boundary=boundary)
    phi_eta = grid.interp(phi_eta, 'Z', boundary=boundary)
    
    phi_z = grid.derivative(phi, 'Z', boundary=boundary)
    phi_z = grid.interp(phi_z, 'Z', boundary=boundary)
    
    # vertical shear (horizontal components of vorticity)
    v_z = grid.derivative(ds.v, 'Z', boundary=boundary)
    v_z = grid.interp(grid.interp(v_z, 'Y', boundary=boundary), 'Z', boundary=boundary)
    
    u_z = grid.derivative(ds.u, 'Z', boundary=boundary)
    u_z = grid.interp(grid.interp(u_z, 'X', boundary=boundary), 'Z', boundary=boundary)

    # vertical component of vorticity
    rel_vort = relative_vorticity(ds, grid)
    
    # combine terms to get the ertel potential vorticity
    epv = -v_z * phi_xi + u_z * phi_eta + (ds.f + rel_vort) * phi_z
    
    # add coordinates
    try:
        epv.coords['lon_rho'] = ds.coords['lon_rho']
        epv.coords['lat_rho'] = ds.coords['lat_rho']
        epv.coords['z_rho'] = ds.coords['z_rho']
        epv.coords['ocean_time'] = ds.coords['ocean_time']
    except:
        warn('Could not append coordinates')

    return epv
