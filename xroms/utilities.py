import xarray as xr
import numpy as np


def argsel2d(ds, lon0, lat0, whichgrid='rho', proj=None):
    '''Return the indices that select nearest grid node.
    
    The order of the indices is first the y or latitude axes, 
    then x or longitude.'''

    return sel2d(ds, lon0, lat0, proj=proj, whichgrid=whichgrid, argsel=True)


def sel2d(ds, lon0, lat0, proj=None, whichgrid='rho', argsel=False):
    '''`sel` in lon and lat simultaneously.
    
    Inputs:
    ds: xarray Dataset with model output
    lon0, lat0: Point of interest in longitude/latitude
    proj (optional): cartopy projection for converting from geographic to 
      projected coordinates. If not input, a Lambert Conformal Conic 
      projection centered at lon0, lat0 will be used for the conversion.
    whichgrid (optional): Which ROMS grid to find node for. Default is 'rho' grid.
      Options are 'rho', 'psi', 'u', 'v', 'vert'.
    argsel (optional): This option is available so that function `argsel2d` 
      is possible as a wrapper to this function.
    
    Return ds subsetted to grid node nearest lon0, lat0 calculated in 2D 
    for grid `whichgrid`.
    '''
    
    import cartopy
    
    if proj is None:
        proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    pc = cartopy.crs.PlateCarree()
    
    # which grid? rho, u, v, etc
    grids = ['rho', 'u', 'v', 'psi', 'vert']
    err = 'whichgrid must be a str of one of: ' + ', '.join(grids)
    assert whichgrid in grids, err
    lon = ds['lon_' + whichgrid].values; lat = ds['lat_' + whichgrid].values
    
    # convert grid points from lon/lat to a reasonable projection for calculating distances
    x, y = proj.transform_points(pc, lon, lat)[...,:2].T

    # convert point of interest
    x0, y0 = proj.transform_point( lon0, lat0, pc )

    # calculate distance from point of interest
    dist = np.sqrt( (x - x0)**2 + (y - y0)**2 )

    iy, ix = np.where(dist==dist.min())
    
    # if being called from argsel2d, return indices instead
    if argsel:
        return iy, ix
        
    # normal scenario
    else:
        # xidim, etadim are like xi_rho, eta_rho, for whichever grid
        xidim = 'xi_' + whichgrid
        etadim = 'eta_' + whichgrid
        indexer = {xidim: ix, etadim: iy}
        return ds.isel(indexer) 


def to_rho(var, grid, boundary='extend'):
    if var.dims[-1] != 'xi_rho':
        var = grid.interp(var, 'X', to='center', boundary=boundary)
    if var.dims[-2] != 'eta_rho':
        var = grid.interp(var, 'Y', to='center', boundary=boundary)
    return var


def to_psi(var, grid, boundary='extend'):
    if var.dims[-1] != 'xi_u':
        var = grid.interp(var, 'X', to='inner', boundary=boundary)
    if var.dims[-2] != 'eta_v':
        var = grid.interp(var, 'Y', to='inner', boundary=boundary)
    return var


def xisoslice(iso_array, iso_value, projected_array, coord):
    '''Calculate an isosurface

    This function calculates the value of projected_array on
    an isosurface in the array iso_array defined by iso_value. 
    
    Note that `xisoslice` 
    requires that iso_array be monotonic. If iso_value is not monotonic it will  
    still run but values may be incorrect where not monotonic.
    
    Performs lazy evaluation.
        
    
    Inputs:
    iso_array:       xarray.DataArray in which the isosurface is defined
    iso_value:       float: value of the isosurface in iso_array
    projected_array: xarray.DataArray in which to project values on the isosurface.
                     This can have multiple time outputs.
                     Needs to be broadcastable from iso_array?
    coord:           string: coordinate associated with the dimension along which to project

    Output:
    iso_values:      xarray.DataArray: values of projected_array on the isosurface
    
    
    Examples:
    
    Calculate lat-z slice of salinity along a constant longitude value (-91.5):
        sl = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.salt, 'xi_rho')
    
    Calculate a lon-lat slice at a constant z value (-10):
        sl = xroms.utilities.xisoslice(ds.z_rho, -10, ds.temp, 's_rho')
    
    Calculate the depth of a specific isohaline (33):
        sl = xroms.utilities.xisoslice(ds.salt, 33, ds.z_rho, 's_rho')
    
    In addition to calculating the slices themselves, you may need to calculate 
    related coordinates for plotting. For example, to accompany the lat-z slice, 
    you may want the following:

        # calculate z values (s_rho)
        slz = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.z_rho, 'xi_rho')

        # calculate latitude values (eta_rho)
        sllat = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.lat_rho, 'xi_rho')

        # assign these as coords to be used in plot
        sl = sl.assign_coords(z=slz, lat=sllat)

        # points that should be masked
        slmask = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.mask_rho, 'xi_rho')

        # drop masked values
        sl = sl.where(slmask==1, drop=True)

    '''
    
    Nm = len(iso_array[coord]) - 1

    lslice = {coord: slice(None, -1)}
    uslice = {coord: slice(1, None)}

    prop = iso_array - iso_value

    propl = prop.isel(**lslice)
    propl.coords[coord] = np.arange(Nm)
    propu = prop.isel(**uslice)
    propu.coords[coord] = np.arange(Nm)

    zc = xr.where((propu*propl)<0.0, 1.0, 0.0)

    varl = projected_array.isel(**lslice)
    varl.coords[coord] = np.arange(Nm)
    varu = projected_array.isel(**uslice)
    varu.coords[coord] = np.arange(Nm)

    propl = (propl*zc).sum(coord)
    propu = (propu*zc).sum(coord)
    varl = (varl*zc).sum(coord)
    varu = (varu*zc).sum(coord)

    return varl - propl*(varu-varl)/(propu-propl)
