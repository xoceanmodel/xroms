import xarray as xr
import cartopy
import xroms
import numpy as np
from xgcm import grid as xgrid

xr.set_options(keep_attrs=True)
    
g = 9.81  # m/s^2
    
@xr.register_dataset_accessor("xroms")
class xromsDatasetAccessor:
    def __init__(self, ds):

        self.ds = ds
            
        # if ds wasn't read in with an xroms load function, it probably doesn't have a grid object
        if 'grid' not in ds.attrs:
            self.ds, grid = xroms.roms_dataset(self.ds)
            self.grid = grid    
    
    @property
    def speed(self):
        '''Calculate horizontal speed [m/s] from u and v components, on rho/rho grids.
        
        Notes
        -----
        speed = np.sqrt(u^2 + v^2)
        
        Uses 'extend' for horizontal boundary.  
        
        See `xroms.speed` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.speed
        '''

        if 'speed' not in self.ds:
            var = xroms.speed(self.ds.u, self.ds.v, self.grid, hboundary='extend')
            self.ds['speed'] = var
        return self.ds.speed
    
    
    @property
    def KE(self):
        '''Calculate kinetic energy [kg/(m*s^2)], on rho/rho grids.
                   
        Notes
        -----
        Uses speed that has been extended out to the rho grid and rho0.
        
        See `xroms.KE` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.KE
        '''
        
        if 'KE' not in self.ds:
            var = xroms.KE(self.ds.rho0, self.speed)
            self.ds['KE'] = var
        return self.ds.KE    
    
    
    @property
    def ug(self):
        '''Calculate geostrophic u velocity from zeta, on u grid.

        Notes
        -----
        ug = -g * zeta_xi / (d xi * f)  # on u grid
        
        See `xroms.uv_geostrophic` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.ug
        '''
        
        if 'ug' not in self.ds:
            ug = xroms.uv_geostrophic(self.ds.zeta, self.ds.f, self.grid, hboundary='extend', hfill_value=None, which='xi')
            self.ds['ug'] = ug
        return self.ds['ug']
    
    
    @property
    def vg(self):
        '''Calculate geostrophic v velocity from zeta, on v grid.

        Notes
        -----
        vg = g * zeta_eta / (d eta * f)  # on v grid
        
        See `xroms.uv_geostrophic` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.vg
        '''
        
        if 'vg' not in self.ds:
            vg = xroms.uv_geostrophic(self.ds.zeta, self.ds.f, self.grid, hboundary='extend', hfill_value=None, which='eta')
            self.ds['vg'] = vg
        return self.ds['vg']
        

    @property
    def EKE(self):
        '''Calculate EKE [m^2/s^2], on rho grid.
        
        Notes
        -----
        EKE = 0.5*(ug^2 + vg^2)
        Puts geostrophic speed on rho grid.
        
        See `xroms.EKE` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.EKE
        '''
        
        if 'EKE' not in self.ds:
            var = xroms.EKE(self.ug, self.vg, self.grid, hboundary='extend')    
            self.ds['EKE'] = var
        return self.ds['EKE']

    
    @property
    def dudz(self):
        '''Calculate dudz [1/s] on u/w grids.
        
        Notes
        -----
        See `xroms.dudz` for full docstring.
        
        `sboundary` is set to 'extend'.
        
        
        Example usage
        -------------
        >>> ds.xroms.dudz
        '''

        if 'dudz' not in self.ds:
            var = xroms.dudz(self.ds.u, self.grid, sboundary='extend')
            self.ds['dudz'] = var
        return self.ds['dudz']

    
    @property
    def dvdz(self):
        '''Calculate dvdz [1/s] on v/w grids.
        
        Notes
        -----
        See `xroms.dvdz` for full docstring.
        
        `sboundary` is set to 'extend'.
        
        
        Example usage
        -------------
        >>> ds.xroms.dvdz
        '''

        if 'dvdz' not in self.ds:
            var = xroms.dvdz(self.ds.v, self.grid, sboundary='extend')
            self.ds['dvdz'] = var
        return self.ds['dvdz']
        
    
    @property
    def vertical_shear(self):
        '''Calculate vertical shear [1/s], rho/w grids.
        
        Notes
        -----
        See `xroms.vertical_shear` for full docstring.
        
        `hboundary` is set to 'extend'.
        
        Example usage
        -------------
        >>> ds.xroms.vertical_shear
        '''
        
        if 'shear' not in self.ds:
            var = xroms.vertical_shear(self.dudz, self.dvdz, self.grid, hboundary='extend')    
            self.ds['shear'] = var
        return self.ds['shear']
    
    
    @property
    def vort(self):
        '''Calculate vertical relative vorticity, psi/w grids.
        
        Notes
        -----
        See `xroms.relative_vorticity` for full docstring.
        
        `hboundary` and `sboundary` both set to 'extend'.
        
        Example usage
        -------------
        >>> ds.xroms.vort
        '''

        if 'vort' not in self.ds:
            var = xroms.relative_vorticity(self.ds.u, self.ds.v, self.grid, 
                                           hboundary='extend', sboundary='extend')
            self.ds['vort'] = var
        return self.ds.vort

    
    @property
    def ertel(self):
        '''Calculate Ertel potential vorticity of buoyancy on rho/rho grids.
        
        Notes
        -----
        See `xroms.ertel` for full docstring.
        
        `hboundary` and `sboundary` both set to 'extend'.
        
        Example usage
        -------------
        >>> ds.xroms.ertel
        '''

        return xroms.ertel(self.buoyancy, self.ds.u, self.ds.v, self.ds.f, self.grid, 
                           hcoord='rho', scoord='s_rho', hboundary='extend', 
                           hfill_value=None, sboundary='extend', sfill_value=None)

    
    @property
    def rho(self):
        '''Return existing rho or calculate, on rho/rho grids.
        
        Notes
        -----
        See `xroms.density` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.rho
        '''
        
        if 'rho' not in self.ds:
            var = xroms.density(self.ds.temp, self.ds.salt, self.ds.z_rho)
            self.ds['rho'] = var

        return self.ds.rho
       
        
    @property
    def sig0(self):
        '''Calculate potential density referenced to z=0, on rho/rho grids.
        
        Notes
        -----
        See `xroms.potential_density` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.sig0
        '''

        if 'sig0' not in self.ds:
            var = xroms.potential_density(self.ds.temp, self.ds.salt, 0)
            self.ds['sig0'] = var
        return self.ds.sig0

    
    @property
    def buoyancy(self):
        '''Calculate buoyancy on rho/rho grids.
        
        Notes
        -----
        See `xroms.buoyancy` for full docstring.
        
        Example usage
        -------------
        >>> ds.xroms.buoyancy
        '''

        if 'buoyancy' not in self.ds:
            var = xroms.buoyancy(self.sig0, self.ds.rho0)
            self.ds['buoyancy'] = var
        return self.ds.buoyancy

    @property
    def N2(self):
        '''Calculate buoyancy frequency squared on rho/w grids.
        
        Notes
        -----
        See `xroms.N2` for full docstring.
        
        `sboundary` set to 'fill' with `sfill_value=np.nan`.
        
        Example usage
        -------------
        >>> ds.xroms.N2
        '''
        
        if 'N2' not in self.ds:
            var = xroms.N2(self.rho, self.grid, self.ds.rho0, sboundary='fill', sfill_value=np.nan)
            self.ds['N2'] = var
        return self.ds.N2

    
    @property
    def M2(self):
        '''Calculate the horizontal buoyancy gradient on rho/w grids.
        
        Notes
        -----
        See `xroms.M2` for full docstring.
        
        `hboundary` set to 'extend' and `sboundary='fill'` with `sfill_value=np.nan`.
        
        Example usage
        -------------
        >>> ds.xroms.M2
        '''
        
        if 'M2' not in self.ds:
            var = xroms.M2(self.rho, self.grid, self.ds.rho0,  
                            hboundary='extend', sboundary='fill', sfill_value=np.nan)
            self.ds['M2'] = var
        return self.ds.M2
    
    
    def mld(self, thresh=0.03):
        '''Calculate mixed layer depth [m].
        
        Inputs
        ------
        thresh: float
            Threshold for detection of mixed layer [kg/m^3]
            
        Notes
        -----
        See `xroms.mld` for full docstring.
                   
        Example usage
        -------------
        >>> ds.xroms.mld(thresh=0.03).isel(ocean_time=0).plot(vmin=-20, vmax=0)
        '''

        return xroms.mld(self.sig0, self.ds.h, self.ds.mask_rho, thresh=thresh)
    
    
    def ddxi(self, varname, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
             sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/dxi for a variable.

        Inputs
        ------
        varname: str
            Name of variable in Dataset to operate on.
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for calculating horizontal derivative of var. This same value 
            will be used for all horizontal grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for calculating horizontal derivative of var. This same value will 
            be used for all vertical grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        
        Returns
        -------
        DataArray of dqdxi, the gradient of q in the xi-direction with 
        attributes altered to reflect calculation.

        Notes
        -----
        dqdxi = dqdx*dzdz - dqdz*dzdx

        Derivatives are taken in the ROMS curvilinear grid native xi-direction.

        These derivatives properly account for the fact that ROMS vertical coordinates are
        s coordinates and therefore can vary in time and space.
    
        This will alter the number of points in the xi and s dimensions. 

        Example usage
        -------------
        >>> ds.xroms.ddxi('salt')
        '''
        
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        assert varname in self.ds, 'variable called "varname" must be in Dataset'
        return xroms.ddxi(self.ds[varname], self.grid, attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value)


    def ddeta(self, varname, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
              sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/deta for a variable.

        Inputs
        ------
        varname: str
            Name of variable in Dataset to operate on.
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for calculating horizontal derivative of var. This same value 
            will be used for grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for calculating horizontal derivative of var. This same value will 
            be used for grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

        Returns
        -------
        DataArray of dqdeta, the gradient of q in the eta-direction with 
        attributes altered to reflect calculation.

        Notes
        -----
        dqdeta = dqdy*dzdz - dqdz*dzdy

        Derivatives are taken in the ROMS curvilinear grid native eta-direction.

        These derivatives properly account for the fact that ROMS vertical coordinates are
        s coordinates and therefore can vary in time and space.
    
        This will alter the number of points in the eta and s dimensions. 

        Example usage
        -------------
        >>> ds.xroms.ddeta('salt')
        '''
        
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        assert varname in self.ds, 'variable called "varname" must be in Dataset'
        return xroms.ddeta(self.ds[varname], self.grid, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value, attrs=attrs)
    
    
    def ddz(self, varname, hcoord=None, scoord=None, hboundary='extend', hfill_value=None,
            sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/dz for a variable.
    
        Inputs
        ------
        varname: str
            Name of variable in Dataset to operate on.
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection for 
            calculating z derivative. This same value will be used for grid 
            changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary fill value 
            associated with sboundary input.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

        Returns
        -------
        DataArray of vertical derivative of variable with 
        attributes altered to reflect calculation.

        Notes
        -----
        This will alter the number of points in the s dimension. 

        Example usage
        -------------
        >>> ds.xroms.ddz('salt')
        '''
        
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        assert varname in self.ds, 'variable called "varname" must be in Dataset'
        return xroms.ddz(self.ds[varname], self.grid, hcoord=hcoord, scoord=scoord, 
                         hboundary=hboundary, hfill_value=hfill_value,
                         sboundary=sboundary, sfill_value=sfill_value, attrs=attrs)

    
    def to_grid(self, varname, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
                sboundary='extend', sfill_value=None):
        '''Implement grid changes.

        Inputs
        ------
        varname: str
            Name of variable in Dataset to operate on.
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.

        Returns
        -------
        DataArray interpolated onto hcoord horizontal and scoord
        vertical grids.

        Notes
        -----
        If var is already on selected grid, nothing happens.

        Example usage
        -------------
        >>> ds.xroms.to_grid('salt', hcoord='rho', scoord='w')
        '''
        
        assert isinstance(varname, str), 'varname should be a string of the name of a variable stored in the Dataset'
        assert varname in self.ds, 'variable called "varname" must be in Dataset'
        return xroms.to_grid(self.ds[varname], self.grid, hcoord=hcoord, scoord=scoord, 
                             hboundary=hboundary, hfill_value=hfill_value,
                             sboundary=sboundary, sfill_value=sfill_value)
  

#     @property
#     def tris(self):
        
#         # triangulation calculations
#         if self._tris is None:
#             self._tris = xroms.interp.setup(self.ds)  # setup for all grids
#         return self._tris
        
#     def llzslice(self, varname, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
        
#         self.tris
#         da = self.ds[varname]
#         idgrid = xroms.id_grid(da)
#         tri = self.tris[idgrid]
#         return xroms.interp.llzslice(da, tri, lon0, lat0, z0s=z0s, zetaconstant=zetaconstant, triplets=triplets)
        
        
#     def llzt(self, varname, lon0, lat0, z0s=None, t0s=None, zetaconstant=False):
        
#         self.tris
#         da = self.ds[varname]
#         idgrid = xroms.id_grid(da)
#         tri = self.tris[idgrid]
#         return xroms.interp.llzt(da, tri, lon0, lat0, z0s=z0s, t0s=t0s, zetaconstant=zetaconstant)
    
    
#     def calc_zslices(self, varname, z0s, zetaconstant=False):
        
#         da = self.ds[varname]
#         return xroms.interp.calc_zslices(da, z0s, zetaconstant=False)    
    
    
#     def ll2xe(self, whichgrid, lon0, lat0, dims=None):
        
#         tri = self.tris[whichgrid]
#         return xroms.interp.ll2xe(tri, lon0, lat0, dims=None)
    
    
    
    
@xr.register_dataarray_accessor("xroms")
class xromsDataArrayAccessor:
    def __init__(self, da):

        self.da = da
   
    
    def to_grid(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
                sboundary='extend', sfill_value=None):
        '''Implement grid changes.

        Inputs
        ------
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.

        Returns
        -------
        DataArray interpolated onto hcoord horizontal and scoord
        vertical grids.

        Notes
        -----
        If var is already on selected grid, nothing happens.

        Example usage
        -------------
        >>> ds.salt.xroms.to_grid(hcoord='rho', scoord='w')
        '''

        return xroms.to_grid(self.da, self.da.attrs['grid'], hcoord=hcoord, scoord=scoord, 
                             hboundary=hboundary, hfill_value=hfill_value,
                             sboundary=sboundary, sfill_value=sfill_value)
        

    def ddz(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None,
            sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/dz for a variable.

        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for grid changes.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection for 
            calculating z derivative. This same value will be used for grid 
            changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary fill value 
            associated with sboundary input.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict, None
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

        Returns
        -------
        DataArray of vertical derivative of variable with 
        attributes altered to reflect calculation.

        Notes
        -----
        This will alter the number of points in the s dimension. 

        Example usage
        -------------
        >>> ds.salt.xroms.ddz()
        '''
        
        return xroms.ddz(self.da, self.da.attrs['grid'], hcoord=hcoord, scoord=scoord, 
                         hboundary=hboundary, hfill_value=hfill_value,
                         sboundary=sboundary, sfill_value=sfill_value, attrs=attrs)
    
    
    def ddxi(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
             sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/dxi for variable.

        Inputs
        ------
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for calculating horizontal derivative of var. This same value 
            will be used for all horizontal grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for calculating horizontal derivative of var. This same value will 
            be used for all vertical grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`
        
        Returns
        -------
        DataArray of dqdxi, the gradient of q in the xi-direction with 
        attributes altered to reflect calculation.

        Notes
        -----
        dqdxi = dqdx*dzdz - dqdz*dzdx

        Derivatives are taken in the ROMS curvilinear grid native xi-direction.

        These derivatives properly account for the fact that ROMS vertical coordinates are
        s coordinates and therefore can vary in time and space.
    
        This will alter the number of points in the xi and s dimensions. 

        Example usage
        -------------
        >>> ds.salt.xroms.ddxi()
        '''
        
        return xroms.ddxi(self.da, self.da.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value)
    
    
    def ddeta(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, 
              sboundary='extend', sfill_value=None, attrs=None):
        '''Calculate d/deta for a variable.

        Inputs
        ------
        hcoord: string, None.
            Name of horizontal grid to interpolate output to. 
            Options are 'rho', 'psi', 'u', 'v'.
        scoord: string, None. 
            Name of vertical grid to interpolate output to. 
            Options are 's_rho', 's_w', 'rho', 'w'.
        hboundary: string, None
            Passed to `grid` method calls; horizontal boundary selection 
            for calculating horizontal derivative of var. This same value 
            will be used for grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        hfill_value: float, None
            Passed to `grid` method calls; horizontal boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        sboundary: string, None
            Passed to `grid` method calls; vertical boundary selection 
            for calculating horizontal derivative of var. This same value will 
            be used for grid changes too.
            From xgcm documentation:
            A flag indicating how to handle boundaries:
            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Neumann boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Dirichlet boundary condition.
        sfill_value: float, None
            Passed to `grid` method calls; vertical boundary selection 
            fill value.
            From xgcm documentation:
            The value to use in the boundary condition with `boundary='fill'`.
        attrs: dict
            Dictionary of attributes to add to resultant arrays. Requires that 
            q is DataArray. For example:
            `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

        Returns
        -------
        DataArray of dqdeta, the gradient of q in the eta-direction with 
        attributes altered to reflect calculation.

        Notes
        -----
        dqdeta = dqdy*dzdz - dqdz*dzdy

        Derivatives are taken in the ROMS curvilinear grid native eta-direction.

        These derivatives properly account for the fact that ROMS vertical coordinates are
        s coordinates and therefore can vary in time and space.
    
        This will alter the number of points in the eta and s dimensions. 

        Example usage
        -------------
        >>> ds.salt.xroms.ddeta()
        '''
        
        return xroms.ddeta(self.da, self.da.attrs['grid'], attrs=attrs, hcoord=hcoord, scoord=scoord, 
                          hboundary=hboundary, hfill_value=hfill_value, 
                          sboundary=sboundary, sfill_value=sfill_value)
    
    
    def argsel2d(self, lon0, lat0):
        """Find the indices of coordinate pair closest to another point.

        Inputs
        ------
        lon0: float, int
            Longitude of comparison point.
        lat0: float, int
            Latitude of comparison point.

        Returns
        -------
        Indices in eta, xi of closest location to lon0, lat0.

        Notes
        -----
        This function uses Great Circle distance to calculate distances assuming 
        longitudes and latitudes as point coordinates. Uses cartopy function 
        `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

        Example usage
        -------------
        >>> ds.temp.xroms.argsel2d(-96, 27)
        """
        
        return xroms.argsel2d(self.da.cf["longitude"], self.da.cf["latitude"], lon0, lat0)
    
    
    def sel2d(self, lon0, lat0):
        """Find the value of the var at closest location to lon0,lat0.

        Inputs
        ------
        lon0: float, int
            Longitude of comparison point.
        lat0: float, int
            Latitude of comparison point.

        Returns
        -------
        DataArray value(s) of closest location to lon0/lat0.

        Notes
        -----
        This function uses Great Circle distance to calculate distances assuming 
        longitudes and latitudes as point coordinates. Uses cartopy function 
        `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

        This wraps `argsel2d`.

        Example usage
        -------------
        >>> ds.temp.xroms.sel2d(-96, 27)
        """
        
        return xroms.sel2d(self.da, self.da.cf["longitude"], self.da.cf["latitude"], lon0, lat0)
    

#     def groupbytime(self, timeperiod, attrs=None, hcoord=None, scoord=None):
#         '''DOCS'''
        
#         return xroms.groupbytime(self.da, self.da.attrs['grid'], timeperiod, attrs=attrs, hcoord=hcoord, scoord=scoord)
    

#     def downsampletime(self, timefrequency, aggfunction=np.mean, attrs=None, hcoord=None, scoord=None):
#         '''DOCS'''

#         return xroms.downsampletime(self.da, self.da.attrs['grid'], timefrequency, aggfunction=aggfunction, 
#                                  attrs=attrs, hcoord=hcoord, scoord=scoord)
        

    def gridmean(self, dim):
        '''Calculate mean accounting for variable spatial grid.

        Inputs
        ------
        dim: str, list, tuple
            Spatial dimension names to average over. In the `xgcm` 
            convention, the allowable names are 'Z', 'Y', or 'X'.

        Returns
        -------
        DataArray or ndarray of average calculated over dim accounting 
        for variable spatial grid.
    
        Notes
        -----
        If result is DataArray, long name attribute is modified to describe
        calculation.

        Example usage
        -------------
        Note that the following two approaches are equivalent:
        >>> app1 = ds.u.xroms.gridmean(('Y','X'))
        >>> app2 = (ds.u*ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))/(ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))
        >>> np.allclose(app1, app2)
        '''

        return xroms.gridmean(self.da, self.da.attrs['grid'], dim)
        

    def gridsum(self, dim):
        '''Calculate sum accounting for variable spatial grid.

        Inputs
        ------
        dim: str, list, tuple
            Spatial dimension names to sum over. In the `xgcm` 
            convention, the allowable names are 'Z', 'Y', or 'X'.

        Returns
        -------
        DataArray or ndarray of sum calculated over dim accounting 
        for variable spatial grid.

        Notes
        -----
        If result is DataArray, long name attribute is modified to describe
        calculation.

        Example usage
        -------------
        Note that the following two approaches are equivalent:
        >>> app1 = ds.u.xroms.gridsum(('Z','X'))
        >>> app2 = (ds.u*ds.dz_u * ds.dx_u).sum(('s_rho','xi_u'))
        >>> np.allclose(app1, app2)    
        '''

        return xroms.gridsum(self.da, self.da.attrs['grid'], dim)
    
    
#     def interp(self, xi=None, eta=None, s=None, t=None):
#         '''Wrapper for xarray `interp` without needing to specify grid.
        
#         Example usage:
#         > ds.salt.xroms.interp(xi=20.5, eta=35.5, t='2020-1-1')
#         '''
        
#         indexer = xroms.build_indexer(self.da, xi, eta, s, t)
        
#         return self.da.interp(indexer)

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

#     @property
#     def tri(self):
        
#         # triangulation calculations
#         if self._tri is None:
#             self._tri = xroms.interp.setup(self.da, self.idgrid)[self.idgrid]  # setup for this variable da
#         return self._tri
        
#     def llzslice(self, lon0, lat0, z0s=None, zetaconstant=False, triplets=False):
        
#         self.tri
#         return xroms.interp.llzslice(self.da, self.tri, lon0, lat0, z0s=z0s, zetaconstant=zetaconstant, triplets=triplets)
        
        
#     def llzt(self, lon0, lat0, z0s=None, t0s=None, zetaconstant=False):
        
#         self.tri
#         return xroms.interp.llzt(self.da, self.tri, lon0, lat0, z0s=z0s, t0s=t0s, zetaconstant=zetaconstant)
    
    
#     def calc_zslices(self, z0s, zetaconstant=False):
        
#         return xroms.interp.calc_zslices(self.da, z0s, zetaconstant=False)
    
    
#     def ll2xe(self, lon0, lat0, dims=None):
        
#         return xroms.interp.ll2xe(self.tri, lon0, lat0, dims=None)