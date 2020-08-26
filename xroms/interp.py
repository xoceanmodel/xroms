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

    if isinstance(whichgrids, str):
        whichgrids = [whichgrids]
    
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
        
    # figure out s coord, whether 's_rho' or 's_w' for varin
    scoord = [dim for dim in varin.coords if 's' in dim][0]

    # Projecting 3rd input onto constant value z0 in iso_array (1st input)
    if z0s is not None:
        sls = []
        for z0 in z0s:
            sl = xroms.xisoslice(zarray, z0, varin, scoord)
            sl = sl.expand_dims({'z': [z0]})
            sls.append(sl)
        sl = xr.concat(sls, dim='z')
    else:
        sl = varin

    return sl


def llzslice(varin, trisgrid, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
    '''Interpolation, best for depth slices.
    
    This function uses `xroms.utilities.xisoslice` to calculate slices of depth. 
    Delaunay triangulation is used to find horizontal locations in grid space from 
    input lon/lat coordinates. The interpolation in eta/xi space is done through 
    xarray's `interp`. Time interpolation can easily be done after running this 
    function. Some of this process is not stream-lined since a DataArray cannot be 
    chunked in the interpolation dimension, complicated the process.
    
    Inputs:
    varin: xarray DataArray containing variable to be interpolated
    trisgrid: previously-calculated output from setup function, already subsetted to whichgrid
    lon0, lat0: lon/lat coordinate pairs to interpolate to.
    z0s (None): vertical depths to interpolate to. Shape does not need to match lon0/lat0 since
      unless `triplets=True` is also input, values will be calculated for lon0/lat0 for every 
      value in z0s.
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


def llzt(varin, trisgrid, lon0, lat0, z0s=None, t0s=None, zetaconstant=False):
    '''Interpolation, best for triplets or quadruplets (t,z,y,x).
    
    This function uses Delaunay triangulation to find eta/xi grid coords of 
    lon0/lat0 coordinates. Then it loops over:
     * z (zetaconstant=True) to calculate the z,y,x interpolation. Time 
       interpolation can be done subsequently. `varin` cannot be chunked in 
       the z/y/x dimensions. 
     * or z and t (zetaconstant=False) to calculate the t, z, y, x 
       interpolation. In this time, all 4 dimensions are interpolated to 
       simultaneously. No dimensions can be chunked over in this case.

    Inputs:
    varin: xarray DataArray containing variable to be interpolated
    trisgrid: previously-calculated output from setup function, already subsetted to whichgrid
    lon0/lat0/z0s: inputs to interpolate varin to. Must be lists.
    z0s (None): vertical depths to interpolate to.    
    t0s (None): times to interpolate to.
    zetaconstant (False): Input as True to not consider time-varying zeta for depth-interpolation.

    If zetaconstant is False, varin can't be read in with dask/can't have 
        chunks in any dimension since all will be interpolated in. Chunks 
        can be reset beforehand with: `ds.chunk(-1)`
    If zetaconstant is True, time interpolation does not occur in this function 
        because dask and chunks couldn't be used in that case at all. But, it 
        is very easy to interpolate afterward:
    > varout2 = varout.chunk(-1).interp(ocean_time=t0s)  # reset time chunks
    > it0s = xr.DataArray(np.arange(len(lon0)), dims=["pts"])  # advanced indexing to pull triplets out
    > varout.isel(ocean_time=it0s)  # pull triplets out
    '''

    assert np.asarray(lon0).squeeze().shape == np.asarray(lat0).squeeze().shape, 'lon0 and lat0 need the same shape'
    
    if z0s is not None and not zetaconstant:  # zeta/depth varies in time
        assert t0s is not None, 'need t0s if considering zeta in time for depths'
        assert np.asarray(t0s).squeeze().shape == np.asarray(lat0).squeeze().shape, 't0s shape needs to match lon0 and others'
        if varin.chunks is not None:
            chunktest = np.sum([len(chunk) for chunk in varin.chunks]) == len(varin.chunks)
            assert chunktest, 'varin cannot be read in with dask (have chunks) if zetaconstant is False. See example for work around.'
    elif z0s is not None and zetaconstant:  # zeta/depth is constant
        assert t0s is None, 'will not interpolate in time if zetaconstant is True, do after function call. See header for example.'
    
    # find locations of lon0/lat0 in grid space
    xi0, eta0 = xroms.interp.ll2xe(trisgrid, lon0, lat0)

    # generic dimensions
    xidim = [dim for dim in varin.dims if 'xi' in dim][0]
    etadim = [dim for dim in varin.dims if 'eta' in dim][0]

    if z0s is not None:
        assert np.asarray(lat0).squeeze().shape == np.asarray(z0s).squeeze().shape, 'z0s needs the same shape as lon0/lat0'

        # figure out s coord, whether 's_rho' or 's_w' for varin
        scoord = [dim for dim in varin.coords if 's' in dim][0]
        
        if zetaconstant:
            zcoord = [dim for dim in varin.coords if 'z' in dim and '0' in dim][0]
            # find s0 without needing to interpolate in time
            s0 = []
            for z0, ie, ix in zip(z0s, eta0, xi0):
                s0.append(varin.interp({etadim: ie, xidim: ix}).swap_dims({scoord: zcoord}).interp({zcoord:z0})[scoord])#.values
            s0 = xr.DataArray(s0, dims=["pts"])  # eta0/xi0 already set up this way
            indexer = {etadim: eta0, xidim: xi0, scoord: s0}
            varout = varin.interp(indexer)

        else:
            zcoord = [dim for dim in varin.coords if 'z' in dim and not '0' in dim][0]
            varout = []
            for t0, z0, ie, ix in zip(t0s, z0s, eta0, xi0):
                varout.append(varin.interp({etadim: ie, xidim: ix, 'ocean_time':t0}).swap_dims({scoord: zcoord}).interp({zcoord:z0}))
            varout = xr.concat(varout, dim='pts')
    else:
        varout = varin.interp({etadim: eta0, xidim: xi0})

    return varout