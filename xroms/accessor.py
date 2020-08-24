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
    
    
    def ddz(self, varname, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.'''
        
        res = self.grid.derivative(self.ds[varname], 'Z', boundary=sboundary, fill_value=sfill_value)
        return res
    
    
    def to_grid(self, varname, hcoord=None, scoord=None):
        '''Implement grid changes using input strings.'''
        
        var = self.ds[varname]
        
        if hcoord is not None:
            if hcoord == 'rho':
                var = xroms.to_rho(var, self.grid)
            elif hcoord == 'psi':
                var = xroms.to_psi(var, self.grid)
            else:
                print('no change to horizontal grid')
                
        if scoord is not None:
            if scoord == 's_rho':
                var = xroms.to_s_rho(var, self.grid)
            elif scoord == 's_w':
                var = xroms.to_s_w(var, self.grid)
            else:
                print('no change to vertical grid')
            
        return var
        
        
    def calc_ddz(self, varname, outname=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Wrap ddz and to_grid and name.'''

        var = self.ddz(varname, sboundary=sboundary, sfill_value=sfill_value)
        var = self.to_grid(varname, hcoord, scoord)
        if outname is not None:
            var.name = outname
        return var

    
    def dudz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dudz from ds.'''

        return self.calc_ddz('u', 'dudz', hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)

    
    def dvdz(self, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Calculate dvdz from ds.'''

        return self.calc_ddz('v', 'dvdz', hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    
    
    def vort(self, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='extend', sfill_value=None):
        '''Calculate vertical relative vorticity from ds.'''

        var = xroms.relative_vorticity(self.ds, self.grid, hboundary=hboundary, hfill_value=hfill_value,
                                                           sboundary=sboundary, sfill_value=sfill_value)
        var = var.xroms.to_grid(self.grid, hcoord, scoord)  # now DataArray
        var.name = 'vort'
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
        phi_z = phi.xroms.calc_ddz(self.grid, 'dphidz', hcoord, scoord, sboundary=sboundary, sfill_value=np.nan)

        # vertical shear (horizontal components of vorticity)
        u_z = self.dudz(hcoord, scoord)
        v_z = self.dvdz(hcoord, scoord)

        # vertical component of vorticity on rho grid
        vort = self.vort(hcoord, scoord)

        # combine terms to get the ertel potential vorticity
        epv = -v_z * phi_xi + u_z * phi_eta + (self.ds.f + vort) * phi_z

        return epv

    
    def get_rho(self, hcoord=None, scoord=None):
        '''Return existing rho or calculate from salt/temp.'''
        
        if 'rho' in self.ds.variables:
            var = self.ds.rho
        else:
            var = xroms.density(self.ds.temp, self.ds.salt, self.ds.z_rho)
        var = self.to_grid(var, hcoord, scoord)
        return var
    
    
    def N2(self, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan):
        '''Calculate buoyancy frequency squared.'''
        
        rho = self.get_rho(hcoord, scoord)
        drhodz = self.calc_ddz(rho, 'drhodz', hcoord, scoord, hboundary=hboundary, hfill_value=hfill_value,
                                                              sboundary=sboundary, sfill_value=sfill_value)
        g = 9.81
        return -g/self.ds.rho0*drhodz
    
    
    
@xr.register_dataarray_accessor("xroms")
class xromsDataArrayAccessor:
    def __init__(self, da):

        self.da = da
        

    def ddz(self, grid, sboundary='extend', sfill_value=np.nan):
        '''Calculate d/dz for a variable.'''
        
        res = grid.derivative(self.da, 'Z', boundary=sboundary, fill_value=sfill_value)
        return res
    
    
    def to_grid(self, grid, hcoord=None, scoord=None):
        '''Implement grid changes using input strings.'''
        
        var = self.da
        
        if hcoord is not None:
            if hcoord == 'rho':
                var = xroms.to_rho(var, grid)
            elif hcoord == 'psi':
                var = xroms.to_psi(var, grid)
            else:
                print('no change to horizontal grid')
                
        if scoord is not None:
            if scoord == 's_rho':
                var = xroms.to_s_rho(var, grid)
            elif scoord == 's_w':
                var = xroms.to_s_w(var, grid)
            else:
                print('no change to vertical grid')
            
        return var
        
        
    def calc_ddz(self, grid, outname=None, hcoord=None, scoord=None, sboundary='extend', sfill_value=np.nan):
        '''Wrap ddz and to_grid and name.'''

        var = self.ddz(grid, sboundary=sboundary, sfill_value=sfill_value)
        var = self.to_grid(grid, hcoord, scoord)
        if outname is not None:
            var.name = outname
        return var
