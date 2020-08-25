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
    lon0, lat0: Point(s) of interest in longitude/latitude. lon0, lat0 
        can be scalars or they can be lists or arrays, but can only have one dimension if they are arrays.
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
        proj = cartopy.crs.LambertConformal(central_longitude=-98,    central_latitude=30)
    pc = cartopy.crs.PlateCarree()
    
    # which grid? rho, u, v, etc
    grids = ['rho', 'u', 'v', 'psi', 'vert']
    err = 'whichgrid must be a str of one of: ' + ', '.join(grids)
    assert whichgrid in grids, err
    lon = ds['lon_' + whichgrid].values; lat = ds['lat_' + whichgrid].values
    
    # convert grid points from lon/lat to a reasonable projection for calculating distances
    x, y = proj.transform_points(pc, lon, lat)[...,:2].T

    # difference for whether single or multiple points
    if isinstance(lon0, int) or isinstance(lon0, float):
        # convert point of interest
        x0, y0 = proj.transform_point( lon0, lat0, pc )
        # calculate distance from point of interest
        dist = np.sqrt( (x - x0)**2 + (y - y0)**2 ).T
        iy, ix = np.where(dist==dist.min())
    else:
        if isinstance(lon0,list):
            lon0 = np.array(lon0)
            lat0 = np.array(lat0)
        x0, y0 = proj.transform_points(pc, lon0, lat0)[...,:2].T

        # calculate distance from point of interest
        # dimensions are those of x/y followed by those of x0/y0
        dist = np.sqrt( (x.T[...,np.newaxis] - x0)**2 + (y.T[...,np.newaxis] - y0)**2 )

        # the first `axes` axes are for dist, last one for lon0/lat0
        axes = tuple(np.arange(dist.ndim))[:-1]
        iy, ix = np.where(dist==dist.min(axis=axes))[:2]
    
    # if being called from argsel2d, return indices instead
    if argsel:
        return iy, ix
        
    # normal scenario
    else:
        # xidim, etadim are like xi_rho, eta_rho, for whichever grid
        xidim = 'xi_' + whichgrid
        etadim = 'eta_' + whichgrid
        # http://xarray.pydata.org/en/stable/interpolation.html#advanced-interpolation
        # use advanced indexing to pull out the list of points instead of slices
        ix = xr.DataArray(ix, dims="pts")
        iy = xr.DataArray(iy, dims="pts")
        indexer = {xidim: ix, etadim: iy}
        return ds.isel(indexer) 


def to_rho(var, grid, boundary='extend'):
    if 'xi_rho' not in var.dims:
        var = grid.interp(var, 'X', to='center', boundary=boundary)
    if 'eta_rho' not in var.dims:
        var = grid.interp(var, 'Y', to='center', boundary=boundary)
    return var


def to_psi(var, grid, boundary='extend'):

    if 'xi_u' not in var.dims:
        var = grid.interp(var, 'X', to='inner', boundary=boundary)
    if 'eta_v' not in var.dims:
        var = grid.interp(var, 'Y', to='inner', boundary=boundary)
    return var


def to_s_rho(var, grid, boundary='extend'):
    '''Convert from s_w to s_rho vertical grid.'''
    
    # only change if not already on s_rho
    if 's_rho' not in var.dims:
        var = grid.interp(var, 'Z', to='center', boundary=boundary)
    return var
    
    
def to_s_w(var, grid, boundary='extend'):
    '''Convert from s_rho to s_w vertical grid.'''
    
    # only change if not already on s_w
    if 's_w' not in var.dims:
        var = grid.interp(var, 'Z', to='outer', boundary=boundary)
    return var


def to_grid(var, grid, hcoord=None, scoord=None):
    '''Implement grid changes to variable var using input strings.

    Inputs:
    var        DataArray
    hcoord     string (None). Name of horizontal grid to interpolate variable
               to. Options are 'rho' and 'psi'.
    scoord     string (None). Name of vertical grid to interpolate variable
               to. Options are 's_rho' and 's_w'.

    Example usage:
    Change 'salt' variable in Dataset ds to be on psi horizontal and s_w vertical grids
    > xroms.to_grid(ds.salt, grid, 'psi', 's_w')
    '''
    
    name = var.name

    if hcoord is not None:
        if hcoord == 'rho':
            var = to_rho(var, grid)
        elif hcoord == 'psi':
            var = to_psi(var, grid)
        else:
            print('no change to horizontal grid')

    if scoord is not None:
        if scoord == 's_rho':
            var = to_s_rho(var, grid)
        elif scoord == 's_w':
            var = to_s_w(var, grid)
        else:
            print('no change to vertical grid')

    var.name = name
    
    return var


def ddz(var, grid, sboundary='extend', sfill_value=np.nan):
    '''Calculate d/dz for a variable.
    
    Inputs:
    var        DataArray
    grid       xgcm grid object
        
    Example usage:
    > xroms.ddz(ds.salt, grid)
    '''

    return grid.derivative(var, 'Z', boundary=sboundary, fill_value=sfill_value)


def calc_ddz(var, grid, outname=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
    '''Wrap ddz and to_grid and name.
    '''

    var = ddz(var, grid, sboundary=sboundary, sfill_value=sfill_value)
    var = to_grid(var, grid, hcoord, scoord)
    if outname is not None:
        var.name = outname
    return var

    
def xisoslice(iso_array, iso_value, projected_array, coord, printwarning=False):
    '''Calculate an isosurface

    This function calculates the value of projected_array on
    an isosurface in the array iso_array defined by iso_value. 
    
    Note that `xisoslice` 
    requires that iso_array be monotonic. If iso_value is not monotonic it will  
    still run but values may be incorrect where not monotonic.
    If iso_value is exactly in iso_array or the value is passed twice in iso_array, 
    a message will be printed. iso_value is changed a tiny amount in this case to
    account for it being in iso_array exactly. The latter case is not deal with.
    
    Performs lazy evaluation.
        
    
    Inputs:
    iso_array:       xarray.DataArray in which the isosurface is defined
    iso_value:       float: value of the isosurface in iso_array
    projected_array: xarray.DataArray in which to project values on the isosurface.
                     This can have multiple time outputs.
                     Needs to be broadcastable from iso_array?
    coord:           string: coordinate associated with the dimension along which to project
    printwarning:    boolean (False): set to True to have warning returned if iso_value is
                     exactly equal to a value in iso_array, in which case an extra 
                     step is taken.

    Output:
    iso_values:      xarray.DataArray: values of projected_array on the isosurface
    
    
    Examples:
    
    Calculate lat-z slice of salinity along a constant longitude value (-91.5):
        sl = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.salt, 'xi_rho')
    
    Calculate a lon-lat slice at a constant z value (-10):
        sl = xroms.utilities.xisoslice(ds.z_rho, -10, ds.temp, 's_rho')
    
    Calculate a lon-lat slice at a constant z value (-10) but without zeta changing in time:
    (use ds.z_rho0 which is relative to mean sea level and does not vary in time)
        sl = xroms.utilities.xisoslice(ds.z_rho0, -10, ds.temp, 's_rho')
    
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
    
    # length of the projected coordinate, minus one
    Nm = len(iso_array[coord]) - 1

    # A 'lower' slice including all but the last value, and an
    # 'upper' slice including all but the first value
    lslice = {coord: slice(None, -1)}
    uslice = {coord: slice(1, None)}

    # prop is now the array on which to calculate the isosurface, with
    # the iso_value subtracted so that the isosurface is defined by 
    # prop == 0
    prop = iso_array - iso_value

    # propl are the prop values in the lower slice
    propl = prop.isel(**lslice)
    propl.coords[coord] = np.arange(Nm)
    # propu in the upper slice
    propu = prop.isel(**uslice)
    propu.coords[coord] = np.arange(Nm)

    # Find the location where prop changes sign, meaning it bounds the
    # desired isosurface. zc has a length of Nm in the projected dimension
    # and may be considered to be an array in between the values in the
    # projected dimension. zc==1 means the prop changed signs crossing this
    # value, so that the isovalue occurs between those two values.
    zc = xr.where((propu*propl)<=0.0, 1.0, 0.0)
    
    # saving these comments for now in case want to switch back, but this approach is 
    # more accurate when it works but doesn't always work 
#     # if condition is True, either iso_value exactly matches at least one entry in iso_array
#     # or iso_value is passed more than once (iso_array is not monotonic)
#     if (zc.sum(coord) == 2).sum() > 0:
#         if printwarning:
#             words = '''either iso_value exactly matches at least one entry in iso_array or 
#                         iso_value is passed more than once (iso_array is not monotonic. 
#                         iso_value is being adjusted slightly to account for the former case 
#                         with an approximation.'''
#             print(words)
#         if iso_value == 0:
#             iso_value = 0.00001
#         else:
#             iso_value *= 1.00001
#         # redo these calculations
#         prop = iso_array - iso_value
#         # propl are the prop values in the lower slice
#         propl = prop.isel(**lslice)
#         propl.coords[coord] = np.arange(Nm)
#         # propu in the upper slice
#         propu = prop.isel(**uslice)
#         propu.coords[coord] = np.arange(Nm)
# #         zc = xr.where((propu*propl)<=0.0, 1.0, 0.0)
#         test = (propu*propl)
#         cond = (test<=0.0) + np.isclose(test,np.zeros_like(test))
#         zc = xr.where(cond, 1.0, 0.0)


    # Get the upper and lower slices of the array that will be projected
    # on the isosurface
    varl = projected_array.isel(**lslice)
    varl.coords[coord] = np.arange(Nm)
    varu = projected_array.isel(**uslice)
    varu.coords[coord] = np.arange(Nm)

    # propl*zc extracts the value of prop below the iso_surface.
    # propu*zc above. Extract similar values for the projected array. 
    propl = (propl*zc).sum(coord)
    propu = (propu*zc).sum(coord)
    varl = (varl*zc).sum(coord)
    varu = (varu*zc).sum(coord)

    # A linear fit to of the projected array to the isosurface.
    out =  varl - propl*(varu-varl)/(propu-propl)
    
    check = (zc.sum(coord) == 2)
    if check.sum() > 0:
        if printwarning:
            words = '''either iso_value exactly matches at least one entry in iso_array or 
                        iso_value is passed more than once (iso_array is not monotonic. 
                        iso_value is being adjusted slightly to account for the former case 
                        with an approximation.'''
            print(words)
        # where iso_value is located in iso_array, divide result by 2
        out = xr.where(check, out/2, out)

    return out
