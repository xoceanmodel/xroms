import xarray as xr
import cartopy
import xroms
import numpy as np

# vargrid = {}
# vargrid['u'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_u', 
#                 'grid': 'u', 'z': 'z_rho_u', 'z0': 'z_rho_u0'}
# vargrid['v'] = {'s': 's_rho', 'eta': 'eta_v', 'xi': 'xi_rho', 
#                 'grid': 'v', 'z': 'z_rho_v', 'z0': 'z_rho_v0'}
# vargrid['temp'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_rho', 
#                    'grid': 'rho', 'z': 'z_rho', 'z0': 'z_rho0'}
# vargrid['salt'] = vargrid['temp']
    
g = 9.81  # m^2/s
    
@xr.register_dataset_accessor("xroms")
class xromsDatasetAccessor:
    def __init__(self, ds, proj=None):

        self.ds = ds
        if proj is None:
            self.proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
        else:
            self.proj = proj
        
        self.ds, grid = xroms.roms_dataset(self.ds, add_verts=True, proj=self.proj)
        self.grid = grid
        self._tris = None
        
        self._rho = None
        self.rho
    
    
    def to_grid(self, varname, hcoord=None, scoord=None):
        '''Implement grid changes to variable in Dataset using input strings.
        
        Inputs:
        varname    string. Name of variable to change that is available in self.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   varname to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   varname to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        Change 'salt' variable in Dataset ds to be on psi horizontal and s_w vertical grids
        > ds.xroms.to_grid('salt', 'psi', 's_w')  
        '''

        return xroms.to_grid(self.ds[varname], self.grid, hcoord=hcoord, scoord=scoord)
    
    
    def ddz(self, varname, outname=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        Example usage:
        > ds.xroms.ddz('salt', outname='dsaltdz', hcoord='psi', scoord='w')
        '''
        
        return xroms.ddz(self.ds[varname], self.grid, outname=outname, hcoord=hcoord, 
                         scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def ddxi(self, varname, outname=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/dxi for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray (None). The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddxi('salt', outname='dsaltdxi', hcoord='psi', scoord='w')
        '''
        
        return xroms.ddxi(self.ds[varname], self.grid, outname=outname, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)


    def ddeta(self, varname, outname=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/deta for a variable.

        Inputs:
        varname    string. Name of variable to change that is available in self.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddeta('salt', outname='dsaltdeta', hcoord='psi', scoord='w')
        '''
        
        return xroms.ddeta(self.ds[varname], self.grid, outname=outname, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)


    def dudz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dudz from ds.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''

        return self.ddz('u', 'dudz', hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)

    
    def dvdz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dvdz from ds.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''

        return self.ddz('v', 'dvdz', hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def vort(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
        '''Calculate vertical relative vorticity from ds.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''

        var = xroms.relative_vorticity(self.ds, self.grid, hboundary=hboundary, hfill_value=hfill_value,
                                                           sboundary=sboundary, sfill_value=sfill_value)
        var.name = 'vort'
        var = var.xroms.to_grid(self.grid, hcoord, scoord)  # now DataArray
        return var
    
    
    def ertel(self, hcoord='rho', scoord='s_rho', tracer=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
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
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        tracer          The tracer to use in calculating EPV. Default is buoyancy. 
                        Buoyancy calculated from salt and temp if rho is not present.

        boundary        Passed to `grid` method calls. Default is `extend`
        '''

        # load appropriate tracer into 'phi', defalut rho. Use EOS if necessary
        if tracer is None:
            if 'rho' in self.ds.variables:
                phi =  -9.8 * self.ds.rho/1000.0 
            else:
                phi = xroms.buoyancy(self.ds.temp, self.ds.salt, 0)
        else:
            phi = self.ds[tracer]

        # get the components of the grad(phi)
        phi_xi, phi_eta = xroms.hgrad(phi, self.grid, hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
        phi_xi = phi_xi.xroms.to_grid(self.grid, hcoord, scoord)
        phi_eta = phi_eta.xroms.to_grid(self.grid, hcoord, scoord)
        phi_z = phi.xroms.ddz(self.grid, 'dphidz', hcoord, scoord, sboundary=sboundary, sfill_value=np.nan)

        # vertical shear (horizontal components of vorticity)
        u_z = self.dudz(hcoord, scoord)
        v_z = self.dvdz(hcoord, scoord)

        # vertical component of vorticity on rho grid
        vort = self.vort(hcoord, scoord)

        # combine terms to get the ertel potential vorticity
        epv = -v_z * phi_xi + u_z * phi_eta + (self.ds.f + vort) * phi_z

        return epv

    
    @property
    def rho(self, hcoord=None, scoord=None):
        '''Return existing rho or calculate from salt/temp.
        
        Inputs:
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        
        '''
        
        if self._rho is None:
            if 'rho' in self.ds.variables:
                self._rho = self.ds.rho
            else:
                self._rho = xroms.density(self.ds.temp, self.ds.salt, self.ds.z_rho)
            self._rho = self._rho.xroms.to_grid(self.grid, hcoord, scoord)  # var is now DataArray
        return self._rho
    
    
    def N2(self, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan):
        '''Calculate buoyancy frequency squared, or vertical buoyancy gradient.'''
        
        drhodz = self.rho.xroms.ddz(self.grid, 'drhodz', hcoord, scoord, sboundary=sboundary, sfill_value=sfill_value)
        return -g*drhodz/self.ds.rho0
    
    
    def M2(self, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan, z=None):
        '''Calculate the horizontal buoyancy gradient.
        
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        '''
        
        drhodxi = self.rho.xroms.ddxi(self.grid, outname=None, hcoord=hcoord, scoord=scoord, 
                                      hboundary=hboundary, hfill_value=hfill_value, 
                                      sboundary=sboundary, sfill_value=sfill_value, z=None)
        drhodeta = self.rho.xroms.ddeta(self.grid, outname=None, hcoord=hcoord, scoord=scoord, 
                                      hboundary=hboundary, hfill_value=hfill_value, 
                                      sboundary=sboundary, sfill_value=sfill_value, z=None)
        
        return np.sqrt(drhodxi**2 + drhodeta**2) * g/self.ds.rho0
        

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
    
    
    def to_grid(self, grid, hcoord=None, scoord=None):
        '''Implement grid changes to DataArray using input strings.
        
        Inputs:
        grid       xgcm grid object with metrics that apply to this DataArray.
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   varname to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   varname to. Options are 's_rho' and 's_w'.
                   
        Example usage:
        Change 'salt' variable in Dataset ds to be on psi horizontal and s_w vertical grids
        > ds.salt.xroms.to_grid(ds.xroms.grid, 'psi', 's_w')  
        '''
        
        return xroms.to_grid(self.da, grid, hcoord=hcoord, scoord=scoord)
        

    def ddz(self, grid, outname=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.

        Inputs:
        grid       xgcm grid object
        
        Example usage:
        > ds.salt.xroms.ddz(ds.xroms.grid, outname='dsaltdz', hcoord='rho', scoord='rho')
        '''
        
        return xroms.ddz(self.da, grid, outname=outname, hcoord=hcoord, 
                              scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def ddxi(self, grid, outname=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/dxi for a variable.

        Inputs:
        grid       xgcm grid object
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddxi('salt', outname='dsaltdxi', hcoord='psi', scoord='w')
        '''
        
        return xroms.ddxi(self.da, grid, outname=outname, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)
    
    
    def ddeta(self, grid, outname=None, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan, z=None):
        '''Calculate d/deta for a variable.

        Inputs:
        grid       xgcm grid object
        hcoord     string (None). Name of horizontal grid to interpolate variable
                   to. Options are 'rho' and 'psi'.
        scoord     string (None). Name of vertical grid to interpolate variable
                   to. Options are 's_rho' and 's_w'.
        z          DataArray. The vertical depths associated with q. Default is to find the
                   coordinate of var that starts with 'z_', and use that.
        
        Example usage:
        > ds.xroms.ddeta('salt', outname='dsaltdeta', hcoord='psi', scoord='w')
        '''
        
        return xroms.ddxi(self.da, grid, outname=outname, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, z=z)
        
    
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