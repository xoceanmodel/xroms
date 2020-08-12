from scipy.spatial import Delaunay
import matplotlib.tri as mtri
import numpy as np
import xroms
import xarray as xr


def setup(ds, whichgrids=None):
    '''Set up for using ll2xe().
    
    Set up Delaunay triangulation by calculating triangulation and functions for 
    calculating grid coords from lon/lat pairs and save into and return ds object.
    
    Create a separate triangulation setup for each grid since otherwise it impacts 
    the performance, especially at edges. Can input keyword whichgrids to only 
    calculate for particular grids — this is intended for testing purposes to save time.
    
    Usage is demonstrated in ll2xe().
    '''    
    tris = {}
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

        tris[whichgrid] = {'name': whichgrid, 'tri': tri, 'fx': fx, 'fy': fy}
    
    return tris
    
    
def ll2xe(trisgrid, lon0, lat0, dims=None):
    '''Find equivalent xi/eta coords for lon/lat.
    
    trisgrid: dictionary tris (from setup()), selected down to grid to use for this function.
    lon0, lat0: lon/lat coordinate pairs at which to find equivalent grid space coordinates.
    dims (None): if None, function will use ("eta_pts", "xi_pts") and "pts" for dimensions 
        associated with output xi0/eta0. Otherwise, can be input but needs to match the 
        dimentionality of lon0/lat0.
    
    Example usage to find coords on rho grid:
    xi0, eta0 = xroms.ll2xe(tris['rho'], lon0, lat0)
    '''
    
    # use these dimensions if not otherwise assigned
    if dims is None:
        # if shape of lon0/lat0 is more than 1d, leave dims empty. NOT TRUE NOW?
        # otherwise if 1d, add dims="pts" to interpolate to just those points and not array
        if np.asarray(lon0).ndim > 2:
            print('lon0/lat0 should have ndim of 2 or less.')    
        elif np.asarray(lon0).ndim == 2:
            dims = ("eta_pts","xi_pts")
        elif np.asarray(lon0).ndim == 1:
            dims = "pts"
        else:
            # Interpolate to xi0,eta0 and add a new dimension to select these points
            # alone using advanced indexing and advanced interpolation in xarray.
            # http://xarray.pydata.org/en/stable/interpolation.html
            dims = ()
    else:
        assert len(dims) == np.asarray(lon0).ndim, 'there needs to be a dimension for each dimension of lon0'
        
    # calculate grid coords
    xi0 = trisgrid['fx'](lon0, lat0)
    eta0 = trisgrid['fy'](lon0, lat0)
    
    # assign dimensions for future interpolation
    xi0 = xr.DataArray(xi0, dims=dims)
    eta0 = xr.DataArray(eta0, dims=dims)

    return xi0, eta0


def calc_zslices(varin, z0s, zetaconstant=False):
    """Wrapper for `xisoslice` for multiple constant depths.
    
    varin: DataArray containing variable to be interpolated.
    z0s: vertical depths to interpolate to.    
    zetaconstant (False): Input as True to not consider time-varying zeta for depth-interpolation.
    
    Note: need to have run roms_dataset to get z info into dataset.
    Note: Can't have chunking in z direction for this function."""
    
    ## fix up shapes, etc
    # if 0d DataArray, convert to float first
    if isinstance(z0s, xr.DataArray) and z0s.ndim==0:
        z0s = float(z0s)
    # if scalar, change to array
    if isinstance(z0s, (int,float,list)):
        z0s = np.array(z0s)
    # if 0D, change to 1d
    if z0s.ndim == 0:
        z0s = z0s.reshape(1)    
        
    if zetaconstant:
        zarray = varin[[dim for dim in varin.coords if 'z' in dim and '0' in dim][0]]
    else:
        zarray = varin[[dim for dim in varin.coords if 'z' in dim and not '0' in dim][0]]

    # Projecting 3rd input onto constant value z0 in iso_array (1st input)
    if z0s is not None:
        sls = []
        for z0 in z0s:
            sl = xroms.xisoslice(zarray, z0, varin, 's_rho')
            sl = sl.expand_dims({'z': [z0]})
            sls.append(sl)
        sl = xr.concat(sls, dim='z')
    else:
        sl = varin

    return sl


def interp(varin, trisgrid, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
    '''
    
    Inputs:
    varin: xarray DataArray containing variable to be interpolated
    trisgrid: previously-calculated output from setup function, already subsetted to whichgrid
    lon0, lat0: lon/lat coordinate pairs to interpolate to.
    z0s (None): vertical depths to interpolate to.    
    zetaconstant (False): Input as True to not consider time-varying zeta for depth-interpolation.
    triplets (False): If True, input arrays must be the same shapes and 1D
      and assumed that the inputs are triplets for interpolation.
      Otherwise, only lon0/lat0 need to match shape and they will be found for all 
      input z0s values.
    
    Note: Can't have chunking in eta or xi direction for this function, or 
        for z if z0s is not None. Can reset chunks before calling this function with:
        > varin.chunk(-1) 
        or input a dictionary to the call with the new chunk choices.
    Note: May need to transpose dimensions after interpolation, e.g.,:
    var.transpose('ocean_time','z','pts')
    '''
    
    if triplets:
        dims = ['pts2']  # intermediate name so that 2nd interpolation is to 'pts'
        assert np.asarray(lon0).squeeze().shape == np.asarray(lat0).squeeze().shape == np.asarray(z0s).squeeze().shape
    else:
        dims = None
    
    # find all necessary z slices, so setting up z interpolation first, here:
    if z0s is not None:
        sl = calc_zslices(varin, z0s, zetaconstant)
    else:
        sl = varin

    # for lon/lat interpolation, find locations of lon0/lat0 in grid space
    xi0, eta0 = ll2xe(trisgrid, lon0, lat0, dims=dims)
    
    # Set up indexer for desired interpolation directions
    xidim = [dim for dim in varin.dims if 'xi' in dim][0]
    etadim = [dim for dim in varin.dims if 'eta' in dim][0]
    indexer = {xidim: xi0, etadim: eta0}

    # set up lazy interpolation
    var = sl.interp(indexer)
    
    # if want triplets back, need to select the points down the diagonal of the
    # array of pts2 returned
    if triplets:
        pts = xr.DataArray(np.arange(len(lon0)), dims=["pts"])
        z0s = xr.DataArray(np.arange(len(lon0)), dims=["pts"])
        var = var.isel(pts2=pts, z=z0s)  # isel bc selecting from interpolated object
    
    return var
