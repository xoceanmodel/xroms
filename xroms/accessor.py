import xarray as xr
import cartopy
import xroms
import numpy as np
from xgcm import grid as xgrid

    
g = 9.81  # m^2/s
    
@xr.register_dataset_accessor("xroms")
class xromsDatasetAccessor:
    def __init__(self, ds, add_verts=False, proj=None):

        self.ds = ds
        if proj is None:
            self.proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
        else:
            self.proj = proj
            
        # if ds wasn't read in with an xroms load function, it probably doesn't have a grid object
        if 'grid' in ds.attrs:
            self.ds, grid = xroms.roms_dataset(self.ds, add_verts=add_verts, proj=self.proj)
            self.ds.attrs['grid'] = grid
            
        self._tris = None
    
    
    def to_grid(self, varname, hcoord=None, scoord=None):
        '''Implement grid changes to variable in Dataset using input strings.
        
        Inputs:
        varname    string. Name of variable to change that is available in self.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   varname to. Options are 'rho', 'psi', 'u', 'v'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   varname to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        Change 'salt' variable in Dataset ds to be on psi horizontal and s_w vertical grids
        > ds.xroms.to_grid('salt', 'psi', 's_w')  
        '''
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        return xroms.to_grid(self.ds[varname], self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord)
    
    
    def ddz(self, varname, attrs=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        Example usage:
        > ds.xroms.ddz('salt', hcoord='psi', scoord='w')
        '''
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        return xroms.ddz(self.ds[varname], self.ds.attrs['grid'], attrs=attrs, hcoord=hcoord, 
                         scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def ddxi(self, varname, attrs=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/dxi for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray (None). The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddxi('salt', hcoord='psi', scoord='w')
        '''
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        return xroms.ddxi(self.ds[varname], self.ds.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)


    def ddeta(self, varname, attrs=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/deta for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddeta('salt', hcoord='psi', scoord='w')
        '''
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        return xroms.ddeta(self.ds[varname], self.ds.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)
        
    
#     def resample(self, **kwargs):
#         '''Wrapper for xarray `resample`.
        
#         Example usage:
#         > ds.xroms.resample(ocean_time='3H')
#         '''
#         attrs = self.ds.attrs.copy()
#         self.ds = self.ds.resample(**kwargs)
#         self.ds.attrs = attrs
#         return self.ds
        
    
#     def isel(self, **kwargs):
#         '''Wrapper for xarray `isel`.
        
#         Example usage:
#         > ds.xroms.isel(xi_rho=slice(20,25), eta_rho=slice(30,40), s_rho=10, ocean_time=1)
#         '''
        
#         self.ds = self.ds.isel(**kwargs)
#         return self.ds
    
    
#     def sel(self, **kwargs):
#         '''Wrapper for xarray `sel`.
        
#         Example usage:
#         > ds.xroms.sel(xi_rho=slice(20,25), eta_rho=35, ocean_time=slice('2020-1-1','2020-1-2'))
#         '''
        
#         return self.sel(**kwargs)


    def dudz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dudz.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''
        return xroms.dudz(self.ds.u, self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)

    
    def dvdz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dvdz.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''
        return xroms.dvdz(self.ds.v, self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def KE(self, hcoord=None, scoord=None):
        '''Calculate kinetic energy.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        > ds.xroms.KE()
        '''

        return xroms.KE(self.rho(), self.speed(), self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord)
    
    
    def speed(self, hcoord=None, scoord=None):
        '''Calculate horizontal speed from u and v components.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''

        return xroms.speed(self.ds.u, self.ds.v, self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord)
    
    
    def vort(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
        '''Calculate vertical relative vorticity from ds.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho', 'psi', 'u', 'v'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''

        return xroms.relative_vorticity(self.ds.u, self.ds.v, self.ds.attrs['grid'], hcoord=hcoord, scoord=scoord, 
                                       hboundary=hboundary, hfill_value=hfill_value,
                                       sboundary=sboundary, sfill_value=sfill_value)

    
    def ertel(self, tracer='buoyancy', hcoord='rho', scoord='s_rho', 
              hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
        '''MATCH xroms docs once finished there.
        
        tracer      (string, 'buoyancy') tracer can be "buoyancy" or the key name of the 
                    tracer you want to use instead of buoyancy.
        '''
        
        if tracer == 'buoyancy':
            phi = self.buoyancy(hcoord=hcoord, scoord=scoord)
        else:
            phi = self.ds[tracer]

        return xroms.ertel(phi, self.ds.u, self.ds.v, self.ds.f, self.ds.attrs['grid'], 
                           hcoord=hcoord, scoord=scoord, hboundary='extend', 
                           hfill_value=None, sboundary='extend', sfill_value=None)
    
    
    def rho(self, hcoord=None, scoord=None, z=None):
        '''Return existing rho or calculate from salt/temp.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray or int. The vertical depths associated with density. Default is to 
                   use the z_rho coordinates, but could instead input 0 to get
                   potential density.
        
        '''
        
        if z is None:
            z = self.ds.z_rho
        
        if 'rho' in self.ds.variables:
            var = self.ds.rho
        else:
            var = xroms.density(self.ds.temp, self.ds.salt, z, self.ds.attrs['grid'], hcoord=None, scoord=None)

        return var

    
    def buoyancy(self, hcoord=None, scoord=None):
        '''Calculate buoyancy.'''
        try:
            rho0 = self.ds.rho0
        except:
            rho0 = 1025  # kg/m^3
        return xroms.buoyancy(self.ds.temp, self.ds.salt, 0, self.ds.attrs['grid'], 
                              rho0, hcoord=hcoord, scoord=scoord)
        
    
    def sig0(self, hcoord=None, scoord=None):
        '''Calculate potential density from salt/temp.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.        
        '''
        
        return xroms.sig0(self.ds.temp, self.ds.salt, self.ds.attrs['grid'], hcoord=None, scoord=None)

    
    def N2(self, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan):
        '''Calculate buoyancy frequency squared, or vertical buoyancy gradient.'''
        
        try:
            rho0 = self.ds.rho0
        except:
            rho0 = 1025  # kg/m^3

        return xroms.N2(self.rho(), self.ds.attrs['grid'], rho0, hcoord=hcoord, scoord=scoord, 
                        hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def M2(self, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan, z=None):
        '''Calculate the horizontal buoyancy gradient.
        
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        '''
        try:
            rho0 = self.ds.rho0
        except:
            rho0 = 1025  # kg/m^3
        
        return xroms.N2(self.rho(), self.ds.attrs['grid'], rho0, hcoord=hcoord, scoord=scoord, 
                        hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def mld(self, hcoord=None, scoord=None, thresh=0.03):
        '''Calculate mixed layer depth.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        > ds.xroms.mld().isel(ocean_time=0).plot(vmin=-20, vmax=0)
        '''

        return xroms.mld(self.sig0(), self.ds.h, self.ds.mask_rho, self.ds.attrs['grid'], thresh=thresh, hcoord=hcoord, scoord=scoord)
    
    
#     @property
#     def idgrid(self):
#         '''Return string name of grid DataArray is on.
    
#         Examples usage:
#         > xroms.id_grid(ds.salt)
#         returns 
#         'rho'
#         '''
#         if self._idgrid is None:
#             self._idgrid = xroms.id_grid(self.da)
#         return self._idgrid

    @property
    def tris(self):
        
        # triangulation calculations
        if self._tris is None:
            self._tris = xroms.interp.setup(self.ds)  # setup for all grids
        return self._tris
        
    def llzslice(self, varname, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
        
        self.tris
        da = self.ds[varname]
        idgrid = xroms.id_grid(da)
        tri = self.tris[idgrid]
        return xroms.interp.llzslice(da, tri, lon0, lat0, z0s=z0s, zetaconstant=zetaconstant, triplets=triplets)
        
        
    def llzt(self, varname, lon0, lat0, z0s=None, t0s=None, zetaconstant=False):
        
        self.tris
        da = self.ds[varname]
        idgrid = xroms.id_grid(da)
        tri = self.tris[idgrid]
        return xroms.interp.llzt(da, tri, lon0, lat0, z0s=z0s, t0s=t0s, zetaconstant=zetaconstant)
    
    
    def calc_zslices(self, varname, z0s, zetaconstant=False):
        
        da = self.ds[varname]
        return xroms.interp.calc_zslices(da, z0s, zetaconstant=False)    
    
    
    def ll2xe(self, whichgrid, lon0, lat0, dims=None):
        
        tri = self.tris[whichgrid]
        return xroms.interp.ll2xe(tri, lon0, lat0, dims=None)
    
    
@xr.register_dataarray_accessor("xroms")
class xromsDataArrayAccessor:
    def __init__(self, da):

        self.da = da
        self._idgrid = None
        self._tri = None
        
        # make sure grid is present
        words = 'An `xgcm` grid must be an attribute of the incoming DataArray. Use `xroms.open_*` function to read in dataset.'
        assert isinstance(da.attrs['grid'], xgrid.Grid), words
    
    
    def to_grid(self, hcoord=None, scoord=None):
        '''Implement grid changes to DataArray using input strings.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   varname to. Options are 'rho', 'psi', 'u', 'v'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   varname to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        Change 'salt' variable in Dataset ds to be on psi horizontal and s_w vertical grids
        > ds.salt.xroms.to_grid('psi', 's_w')  
        '''

        return xroms.to_grid(self.da, self.da.attrs['grid'], hcoord=hcoord, scoord=scoord)
        

    def ddz(self, attrs=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.

        Inputs:
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   varname to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   varname to. Options are 's_rho' and 's_w'.
        
        Example usage:
        > ds.salt.xroms.ddz(hcoord='rho', scoord='rho')
        '''
        
        return xroms.ddz(self.da, self.da.attrs['grid'], attrs=attrs, hcoord=hcoord, 
                              scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def ddxi(self, attrs=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/dxi for a variable.

        Inputs:
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.salt.xroms.ddxi(hcoord='psi', scoord='w')
        '''
        
        return xroms.ddxi(self.da, self.da.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)
    
    
    def ddeta(self, attrs=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/deta for a variable.

        Inputs:
        attrs      dictionary (None). Metadata to replace what is assumed for resultant DataArray.
                   For example, `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.salt.xroms.ddeta(hcoord='psi', scoord='w')
        '''
        
        return xroms.ddeta(self.da, self.da.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)
    

    def groupbytime(self, timeperiod, attrs=None, hcoord=None, scoord=None):
        '''DOCS'''
        
        return xroms.groupbytime(self.da, self.da.attrs['grid'], timeperiod, attrs=attrs, hcoord=hcoord, scoord=scoord)
    

    def downsampletime(self, timefrequency, aggfunction=np.mean, attrs=None, hcoord=None, scoord=None):
        '''DOCS'''

        return xroms.downsampletime(self.da, self.da.attrs['grid'], timefrequency, aggfunction=aggfunction, 
                                 attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def mean(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take mean over all or selected dimensions. Provide attributes.

        NOTE: You may want to be using `gridmean` instead of `mean` to account 
        for the variable spatial grid.
        
        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.mean(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def sum(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take sum over all or selected dimensions. Provide attributes.

        NOTE: You may want to be using `gridsum` instead of `sum` to account 
        for the variable spatial grid.
        
        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.sum(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)


    def max(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take max over all or selected dimensions. Provide attributes.

        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.max(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def min(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take min over all or selected dimensions. Provide attributes.

        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.min(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
       

    def std(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take std over all or selected dimensions. Provide attributes.

        NOTE: You may want to be using `gridstd` instead of `std` to account 
        for the variable spatial grid. (not currently available)
        
        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.std(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
         

    def var(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take variance over all or selected dimensions. Provide attributes.

        NOTE: You may want to be using `gridvar` instead of `var` to account 
        for the variable spatial grid.
        
        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.var(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def median(self, dim=None, attrs=None, hcoord=None, scoord=None):
        '''Take median over all or selected dimensions. Provide attributes.

        dim        (None) Can be None to average over all dimensions, a string of
                    one dimension to average over, or a list or tuple of dimension
                    names to average over.
        '''

        return xroms.median(self.da, self.da.attrs['grid'], dim=dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def gridsum(self, dim, attrs=None, hcoord=None, scoord=None):
        '''Take sum with respect to spatial grid over dim. Provide attributes.
        
        dim         dimension names in the `xgcm` convention are 'Z', 'Y', 
                    or 'X'. dim can be a string, list, or tuple of combinations  
                    of these names for dimensions to average over.
                                
        Note that the following two approaches are equivalent:
        > app1 = ds.u.xroms.gridsum(('Z','X'))
        > app2 = (ds.u*ds.dz_u * ds.dx_u).sum(('s_rho','xi_u'))
        > np.allclose(app1, app2)    
        '''

        return xroms.gridsum(self.da, self.da.attrs['grid'], dim, attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def gridmean(self, dim, attrs=None, hcoord=None, scoord=None):
        '''Take mean with respect to spatial grid over dim. Provide attributes.
        
        dim         dimension names in the `xgcm` convention are 'Z', 'Y', 
                    or 'X'. dim can be a string, list, or tuple of combinations  
                    of these names for dimensions to average over.

        Note that the following two approaches are equivalent:
        > app1 = ds.u.xroms.gridmean(('Y','X'))
        > app2 = (ds.u*ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))/(ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))
        > np.allclose(app1, app2)
        '''

        return xroms.gridmean(self.da, self.da.attrs['grid'], dim, attrs=attrs, hcoord=hcoord, scoord=scoord)

    
    def isel(self, xi=None, eta=None, s=None, t=None, **kwargs):
        '''Wrapper for xarray `isel` without needing to specify grid.
        
        Example usage:
        > ds.salt.xroms.isel(xi=slice(20,25), eta=slice(30,40), s=10, t=1)
        '''
                
        indexer = xroms.build_indexer(self.da, xi, eta, s, t)
        
        return self.da.isel(indexer, **kwargs)
    
    
    def sel(self, xi=None, eta=None, s=None, t=None, **kwargs):
        '''Wrapper for xarray `sel` without needing to specify grid.
        
        Example usage:
        > ds.salt.xroms.sel(xi=slice(20,25), eta=35, t=slice('2020-1-1','2020-1-2'))
        '''
        
        indexer = xroms.build_indexer(self.da, xi, eta, s, t)
        
        return self.da.sel(indexer, **kwargs)
    
    
#     def interp(self, xi=None, eta=None, s=None, t=None):
#         '''Wrapper for xarray `interp` without needing to specify grid.
        
#         Example usage:
#         > ds.salt.xroms.interp(xi=20.5, eta=35.5, t='2020-1-1')
#         '''
        
#         indexer = xroms.build_indexer(self.da, xi, eta, s, t)
        
#         return self.da.interp(indexer)

    @property
    def idgrid(self):
        '''Return string name of grid DataArray is on.
    
        Examples usage:
        > xroms.id_grid(ds.salt)
        returns 
        'rho'
        '''
        if self._idgrid is None:
            self._idgrid = xroms.id_grid(self.da)
        return self._idgrid

    @property
    def tri(self):
        
        # triangulation calculations
        if self._tri is None:
            self._tri = xroms.interp.setup(self.da, self.idgrid)[self.idgrid]  # setup for this variable da
        return self._tri
        
    def llzslice(self, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
        
        self.tri
        return xroms.interp.llzslice(self.da, self.tri, lon0, lat0, z0s=z0s, zetaconstant=zetaconstant, triplets=triplets)
        
        
    def llzt(self, lon0, lat0, z0s=None, t0s=None, zetaconstant=False):
        
        self.tri
        return xroms.interp.llzt(self.da, self.tri, lon0, lat0, z0s=z0s, t0s=t0s, zetaconstant=zetaconstant)
    
    
    def calc_zslices(self, z0s, zetaconstant=False):
        
        return xroms.interp.calc_zslices(self.da, z0s, zetaconstant=False)
    
    
    def ll2xe(self, lon0, lat0, dims=None):
        
        return xroms.interp.ll2xe(self.tri, lon0, lat0, dims=None)