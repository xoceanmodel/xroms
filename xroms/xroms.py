import xarray as xr
import xgcm
from warnings import warn
import cartopy
import xroms
import numpy as np
import cf_xarray


xr.set_options(keep_attrs=True)

# from .utilities import xisoslice, to_grid
# from .roms_seawater import buoyancy

g = 9.81  # m/s^2

def roms_dataset(ds, Vtransform=None, add_verts=False, proj=None):
    '''Modify Dataset to be aware of ROMS coordinates, with matching xgcm grid object.

    Inputs
    ------
    ds: Dataset
        xarray Dataset with model output
    Vtransform: int, None
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection 
        to be input too.
    proj: cartopy crs projection, None
        Should match geographic area of model domain. Required if `add_verts=True`, 
        otherwise not used. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Same dataset as input, but with dimensions renamed to be consistent with `xgcm` and
        with vertical coordinates and metrics added.
    grid: xgcm grid object
        Includes ROMS metrics so can be used for xgcm grid operations, which mostly have 
        been wrapped into xroms.
             
    Notes
    -----
    Note that this could be very slow if dask is not on.
    
    This does not need to be run by the user if `xroms` functions `open_netcdf` or 
    `open_zarr` are used for reading in model output, since run in those functions.
    
    This also uses `cf-xarray` to manage dimensions of variables.
    
    Example usage
    -------------
    >>> ds, grid = xroms.roms_dataset(ds)
    '''
    
    if add_verts:
        assert proj is not None, 'To add "verts" grid, input projection "proj".'

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

        
    # modify attributes for using cf-xarray
    tdims = [dim for dim in ds.dims if dim[:3] == 'xi_']
    for dim in tdims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {'axis': 'X'})
    tdims = [dim for dim in ds.dims if dim[:4] == 'eta_']
    for dim in tdims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {'axis': 'Y'})
    ds.ocean_time.attrs['axis'] = 'T'
    ds.ocean_time.attrs['standard_name'] = 'time'
    tcoords = [coord for coord in ds.coords if coord[:2] == 's_']
    for coord in tcoords:
        ds[coord].attrs['axis'] = 'Z'

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

    ds.coords['z_w'] = z_w.cf.transpose(*[dim for dim in ["T", "Z", "Y", "X"] if dim in z_w.cf.get_valid_keys()])
#     ds.coords['z_w'] = z_w.transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho', transpose_coords=False)
    ds.coords['z_w_u'] = grid.interp(ds.z_w, 'X')
    ds.coords['z_w_v'] = grid.interp(ds.z_w, 'Y')
    ds.coords['z_w_psi'] = grid.interp(ds.z_w_u, 'Y')

    ds.coords['z_rho'] = z_rho.cf.transpose(*[dim for dim in ["T", "Z", "Y", "X"] if dim in z_rho.cf.get_valid_keys()])
#     ds.coords['z_rho'] = z_rho.transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho', transpose_coords=False)
    ds.coords['z_rho_u'] = grid.interp(ds.z_rho, 'X')
    ds.coords['z_rho_v'] = grid.interp(ds.z_rho, 'Y')
    ds.coords['z_rho_psi'] = grid.interp(ds.z_rho_u, 'Y')
    # also include z coordinates with mean sea level (constant over time)
    ds.coords['z_rho0'] = z_rho0.cf.transpose(*[dim for dim in ["T", "Z", "Y", "X"] if dim in z_rho0.cf.get_valid_keys()])
#     ds.coords['z_rho0'] = z_rho0.transpose('s_rho', 'eta_rho', 'xi_rho', transpose_coords=False)
    ds.coords['z_rho_u0'] = grid.interp(ds.z_rho0, 'X')
    ds.coords['z_rho_v0'] = grid.interp(ds.z_rho0, 'Y')
    ds.coords['z_rho_psi0'] = grid.interp(ds.z_rho_u0, 'Y')
    ds.coords['z_w0'] = z_w0.cf.transpose(*[dim for dim in ["T", "Z", "Y", "X"] if dim in z_w0.cf.get_valid_keys()])
#     ds.coords['z_w0'] = z_w0.transpose('s_w', 'eta_rho', 'xi_rho', transpose_coords=False)
    ds.coords['z_w_u0'] = grid.interp(ds.z_w0, 'X')
    ds.coords['z_w_v0'] = grid.interp(ds.z_w0, 'Y')
    ds.coords['z_w_psi0'] = grid.interp(ds.z_w_u0, 'Y')

    # add vert grid, esp for plotting pcolormesh
    if add_verts:
        import pygridgen
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
    
    if 'rho0' not in ds:
        ds['rho0'] = 1025  # kg/m^3

    # cf-xarray
    # areas
#     ds.coords["cell_area"] = ds['dA']
#     ds.coords["cell_area_u"] = ds['dA_u']
#     ds.coords["cell_area_v"] = ds['dA_v']
#     ds.coords["cell_area_psi"] = ds['dA_psi']
#     # and set proper attributes
#     ds.temp.attrs["cell_measures"] = "area: cell_area, volume: cell_volume"
#     ds.salt.attrs["cell_measures"] = "area: cell_area"
#     ds.u.attrs["cell_measures"] = "area: cell_area_u"
#     ds.v.attrs["cell_measures"] = "area: cell_area_v"
#     # volumes
#     ds.coords["cell_volume"] = ds['dV']
# #     ds.temp.attrs["cell_measures"] = "volume: cell_volume"
    
#     ds['temp'].attrs['cell_measures'] = 'area: cell_area'
#     tcoords = [coord for coord in ds.variables if coord[:2] == 'dA']
#     for coord in tcoords:
#         ds[coord].attrs['cell_measures'] = 'area: cell_area'
#     # add coordinates attributes for variables
    if 'positive' in ds.s_rho.attrs:
        ds.s_rho.attrs.pop('positive')    
    if 'positive' in ds.s_w.attrs:
        ds.s_w.attrs.pop('positive')    
#     ds['z_rho'].attrs['positive'] = 'up'
    tcoords = [coord for coord in ds.coords if coord[:2] == 'z_' and '0' not in coord]
    for coord in tcoords:
        ds[coord].attrs['positive'] = 'up'
#         ds[dim] = (dim, np.arange(ds.sizes[dim]), {'axis': 'Y'})
#     ds['z_rho'].attrs['vertical'] = 'depth'
#     ds['temp'].attrs['coordinates'] = 'lon_rho lat_rho z_rho ocean_time'
#     [del ds[var].encoding['coordinates'] for var in ds.variables if 'coordinates' in ds[var].encoding]
    for var in ds.variables:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']

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

    ds.attrs['grid'] = grid
    # also put grid into every variable with at least 2D
    for var in ds.variables:
        if ds[var].ndim > 1:
            ds[var].attrs['grid'] = grid

    return ds, grid


def open_netcdf(file, chunks={"ocean_time": 1}, xrargs={},
                Vtransform=None, add_verts=False, proj=None):
    '''Return Dataset based on a single thredds or physical location.

    Inputs
    ------
    file: str
        Where to find the model output. `file` could be: 
        * a string of a single netCDF file name, or 
        * a string of a thredds server address containing model output.
    chunks: dict
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict
        Keyword arguments to be passed to `xarray.open_dataset`. See `xarray` docs 
        for options.
    Vtransform: int, None
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection 
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, None
        Should match geographic area of model domain. Required if `add_verts=True`, 
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and 
        output is read in lazily instead of eagerly.

    Example usage
    -------------
    >>> ds = xroms.open_netcdf(file)
    '''
    
    words = 'Model location should be given as string. If have list of multiple locations, use `open_mfdataset`.'
    assert isinstance(file, str), words

    ds = xr.open_dataset(file, chunks=chunks, **xrargs)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, grid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)

    return ds


def open_mfnetcdf(files, chunks={"ocean_time": 1}, xrargs={},  
                Vtransform=None, add_verts=False, proj=None):
    '''Return Dataset based on a list of netCDF files.

    Inputs
    ------
    files: list of strings
        Where to find the model output. `files` can be a list of netCDF file names.
    chunks: dict
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict
        Keyword arguments to be passed to `xarray.open_mfdataset`.
        Anything input by the user overwrites the default selections saved in this 
        function. Defaults are:
            {'compat': 'override', 'combine': 'by_coords',
             'data_vars': 'minimal', 'coords': 'minimal', 'parallel': True}
        Many other options are available; see xarray docs.
    Vtransform: int, None
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection 
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, None
        Should match geographic area of model domain. Required if `add_verts=True`, 
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and 
        output is read in lazily instead of eagerly.

    Example usage
    -------------
    >>> ds = xroms.open_mfnetcdf(files)
    '''
    
    words = 'Model location should be given as list of strings. If have single location, use `open_dataset`.'
    assert isinstance(files, list), words
        
    xrargsin = {'compat': 'override', 'combine': 'by_coords',
                'data_vars': 'minimal', 'coords': 'minimal', 
                'parallel': True}
    
    # use input xarray arguments and prioritize them over xroms defaults.
    xrargsin = {**xrargsin, **xrargs}

    ds = xr.open_mfdataset(files, chunks=chunks, **xrargsin)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, grid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)

    return ds


def open_zarr(files, chunks={"ocean_time": 1}, xrargs={}, xrconcatargs={},
              Vtransform=None, add_verts=False, proj=None):
    '''Return a Dataset based on a list of zarr files

    Inputs
    ------
    files: list of strings
        A list of zarr file directories.
    chunks: dict
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict
        Keyword arguments to be passed to `xarray.open_zarr`.
        Anything input by the user overwrites the default selections saved in this 
        function. Defaults are:
            {'consolidated': True, 'drop_variables': 'dstart'}
        Many other options are available; see xarray docs.
    xrconcatargs: dict
        Keyword arguments to be passed to `xarray.concat` for combining zarr files 
        together. Anything input by the user overwrites the default selections saved in this 
        function. Defaults are:
            {'dim': 'ocean_time', 'data_vars': 'minimal', 'coords': 'minimal'}
        Many other options are available; see xarray docs.
    Vtransform: int, None
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection 
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, None
        Should match geographic area of model domain. Required if `add_verts=True`, 
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and 
        output is read in lazily instead of eagerly.

    Example usage
    -------------
    >>> ds = xroms.open_zarr(files)
    '''
        
    # keyword arguments to go to `open_zarr`:
    xrargsin = {'consolidated': True, 'drop_variables': 'dstart'}  
    # use input xarray arguments and prioritize them over xroms defaults.
    xrargsin = {**xrargsin, **xrargs}
    
    # keyword arguments to go to `concat`:
    xrconcatargsin = {'dim': 'ocean_time', 'data_vars': 'minimal', 'coords': 'minimal'}
    # use input xarray arguments and prioritize them over xroms defaults.
    xrconcatargsin = {**xrconcatargsin, **xrconcatargs}

    # open files
    ds = xr.concat( [xr.open_zarr(file, **xrargsin) for file in files], **xrconcatargsin)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, grid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)
    
    return ds
