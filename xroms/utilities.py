import xarray as xr
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.tri as mtri
import xroms

vargrid = {}
vargrid['u'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_u', 'grid': 'u', 'z': 'z_rho_u', 'z0': 'z_rho_u0'}
vargrid['v'] = {'s': 's_rho', 'eta': 'eta_v', 'xi': 'xi_rho', 'grid': 'v', 'z': 'z_rho_v', 'z0': 'z_rho_v0'}
vargrid['temp'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_rho', 'grid': 'rho', 'z': 'z_rho', 'z0': 'z_rho0'}
vargrid['salt'] = vargrid['temp']



def ll2xe_setup(ds, whichgrids=None):
    '''Set up for using ll2xe().
    
    Set up Delaunay triangulation by calculating triangulation and functions for 
    calculating grid coords from lon/lat pairs and save into and return ds object.
    
    Create a separate triangulation setup for each grid since otherwise it impacts 
    the performance, especially at edges. Can input keyword whichgrids to only 
    calculate for particular grids — this is intended for testing purposes to save time.
    '''    
    ll2xe_dict = {}
    # Set up Delaunay triangulation of grid space in lon/lat coordinates
    
    if whichgrids is None:
        whichgrids = ['rho', 'u', 'v', 'psi']
        
    for whichgrid in whichgrids:
        lonkey = 'lon_' + whichgrid
        latkey = 'lat_' + whichgrid

        # Triangulation for curvilinear space to grid space
        # Have to use SciPy's Triangulation to be more robust.
        # http://matveichev.blogspot.com/2014/02/matplotlibs-tricontour-interesting.html
        lon = ds[lonkey].values.flatten()
        lat = ds[latkey].values.flatten()
        pts = np.column_stack((lon, lat))
        tess = Delaunay(pts)
        tri = mtri.Triangulation(lon, lat, tess.simplices.copy())
        # For the triangulation, need to preprocess the mask to get rid of potential 
        # flat triangles at the boundaries.
        # http://matplotlib.org/1.3.1/api/tri_api.html#matplotlib.tri.TriAnalyzer
        mask = mtri.TriAnalyzer(tri).get_flat_tri_mask(0.01, rescale=False)
        tri.set_mask(mask)

        # Next set up the grid (or index) space grids: integer arrays that are the
        # shape of the horizontal model grid.
        J, I = ds[lonkey].shape
        X, Y = np.meshgrid(np.arange(I), np.arange(J))

        # these are the functions for interpolating X and Y (the grid arrays) onto
        # lon/lat coordinates. That is, the functions for calculating grid space coords
        # corresponding to input lon/lat coordinates.
        fx = mtri.LinearTriInterpolator(tri, X.flatten())
        fy = mtri.LinearTriInterpolator(tri, Y.flatten())

        ll2xe_dict[whichgrid] = {'name': whichgrid, 'tri': tri, 'fx': fx, 'fy': fy}
    
    return ll2xe_dict


def setup_indexer(vardims, xi0, eta0):
    indexer = {}
    indexer[vardims['xi']] = xi0
    indexer[vardims['eta']] = eta0        
    return indexer


def calc_zslices(varin, z0s, zetaMSL):
    """Can't have chunking in z direction for this function."""
    
    ## fix up shapes, etc
    # if 0d DataArray, convert to float first
    if isinstance(z0s, xr.DataArray) and z0s.ndim==0:
        z0s = float(z0s)
    # if scalar, change to array
    if isinstance(z0s, (int,float)):
        z0s = np.array(z0s)
    # if 0D, change to 1d
    if z0s.ndim == 0:
        z0s = z0s.reshape(1)    
        
    if zetaMSL:
        zarray = varin[vargrid[varin.name]['z0']]
    else:
        zarray = varin[vargrid[varin.name]['z0']]

    # Projecting 3rd input onto constant value z0 in iso_array (1st input)
    if z0s is not None:
        sls = []
        for z0 in z0s:
            sl = xroms.utilities.xisoslice(zarray, z0, varin, 's_rho')
            sl = sl.expand_dims({'z': [z0]})
            sls.append(sl)
        sl = xr.concat(sls, dim='z')
    else:
        sl = varin

    return sl


def interp(varin, lon0, lat0, z0s=None, zetaMSL=False, triplets=False):
    '''
    
    Inputs:
    varin: xarray DataArray containing variable to be interpolated
    ll2xe_dict: previously-calculated output from ll2xe_setup function, already subsetted to whichgrid
    
    zetaMSL (False): Input as True to not consider time-varying zeta for depth-interpolation.
    triplets (False): If True, input arrays must be the same shapes (CAN THEY BE 2D?)
      and assumed that the inputs are triplets for interpolation.
      Otherwise, only lon0/lat0 need to match shape and they will be found for all 
      input z0s values.
    
    Note: May need to transpose dimensions after interpolation, e.g.,:
    var.transpose('ocean_time','z','pts')
    '''
    varname = varin.name; whichgrid = vargrid[varname]['grid']
    
    if triplets:
        dims = 'pts2'  # intermediate name so that 2nd interpolation is to 'pts'
#         assert lon0.shape == lat0.shape == z0s.shape
    else:
        dims = None
    
    # find all necessary z slices, so setting up z interpolation first, here:
    if z0s is not None:
        sl = calc_zslices(varin, z0s, zetaMSL)
    else:
        sl = varin
#     print(lon0, lat0, dims)
    # for lon/lat interpolation, find locations of lon0/lat0 in grid space
    xi0, eta0 = xroms.utilities.ll2xe(ll2xe_dict[whichgrid], lon0, lat0, dims=dims)
    
    # Set up indexer for desired interpolation directions
    indexer = setup_indexer(vargrid[varname], xi0, eta0)

    # set up lazy interpolation
    var = sl.interp(indexer)
    
    # if want triplets back, need to select the points down the diagonal of the
    # array of pts2 returned
    if triplets:
        pts = xr.DataArray(np.arange(len(lon0)), dims="pts")
        z0s = xr.DataArray(np.arange(len(lon0)), dims="pts")
        var = var.isel(pts2=pts, z=z0s)  # isel bc selecting from interpolated object
    
    return var
    
    
def ll2xe(ll2xe_dict, lon0, lat0, dims=None):
    '''Find equivalent xi/eta coords for lon/lat.
    
    Example usage to find coords on rho grid:
    xi0, eta0 = xroms.utilities.ll2xe(ll2xe_dict['rho'], lon0, lat0)
    '''
    
    # use these dimensions if not otherwise assigned
    if dims is None:
        # if shape of lon0/lat0 is more than 1d, leave dims empty. NOT TRUE NOW?
        # otherwise if 1d, add dims="pts" to interpolate to just those points and not array
#         if lon0.ndim > 1:
        if np.asarray(lon0).ndim > 1:
            dims = ("eta_pts","xi_pts")
        elif np.asarray(lon0).ndim == 1:
            dims = "pts"
        else:
            # Interpolate to xi0,eta0 and add a new dimension to select these points
            # alone using advanced indexing and advanced interpolation in xarray.
            # http://xarray.pydata.org/en/stable/interpolation.html
            dims = ()
#             dims = "pts"
        
    # calculate grid coords
    xi0 = ll2xe_dict['fx'](lon0, lat0)
    eta0 = ll2xe_dict['fy'](lon0, lat0)
    
    # assign dimensions for future interpolation
    xi0 = xr.DataArray(xi0, dims=dims)
    eta0 = xr.DataArray(eta0, dims=dims)


    return xi0, eta0


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
