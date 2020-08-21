import xarray as xr
import cartopy
import xroms

# vargrid = {}
# vargrid['u'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_u', 
#                 'grid': 'u', 'z': 'z_rho_u', 'z0': 'z_rho_u0'}
# vargrid['v'] = {'s': 's_rho', 'eta': 'eta_v', 'xi': 'xi_rho', 
#                 'grid': 'v', 'z': 'z_rho_v', 'z0': 'z_rho_v0'}
# vargrid['temp'] = {'s': 's_rho', 'eta': 'eta_rho', 'xi': 'xi_rho', 
#                    'grid': 'rho', 'z': 'z_rho', 'z0': 'z_rho0'}
# vargrid['salt'] = vargrid['temp']

# THESE ARE BY DEFINITION THINGS THAT WANT TO KNOW ABOUT GRID, SO IT IS REQUIRED

@xr.register_dataset_accessor("xroms")
# @xr.register_dataarray_accessor("xroms")
class xromsAccessor:
    def __init__(self, xarray_obj, proj=None):

        self.ds = xarray_obj
        if proj is None:
            self.proj = cartopy.crs.LambertConformal(central_longitude=-98,    central_latitude=30)
        else:
            self.proj = proj
        self.ds, grid = xroms.roms_dataset(self.ds, add_verts=True, proj=self.proj)
        self.grid = grid
    
    
    def ddz(self, var, boundary='extend'):
        '''Calculate d/dz for a variable.'''
        
        res = self.grid.derivative(var, 'Z', boundary=boundary)
        return res
    
    def grids(self, var, hcoord, scoord):
        '''Implement grid changes using input strings.'''
        
        if hcoord == 'rho':
            var = xroms.to_rho(var, self.grid)
        elif hcoord == 'psi':
            var = xroms.to_psi(var, self.grid)
        else:
            print('no change to horizontal grid')

        if scoord == 's_rho':
            var = xroms.to_s_rho(var, self.grid)
        elif scoord == 's_w':
            var = xroms.to_s_w(var, self.grid)
        else:
            print('no change to vertical grid')
            
        return var
        
    def calc_ddz(self, var, outname, hcoord, scoord, boundary='extend'):
        '''Wrap ddz and grids and name.'''

        var = self.ddz(var, boundary=boundary)
        var = self.grids(var, hcoord, scoord)
        var.name = outname
        return var

    
    def dudz(self, hcoord, scoord, boundary='extend'):
        '''Calculate dudz from ds.'''

        return self.calc_ddz(self.ds.u, 'dudz', hcoord, scoord, boundary=boundary)

    def dvdz(self, hcoord, scoord, boundary='extend'):
        '''Calculate dvdz from ds.'''

        return self.calc_ddz(self.ds.v, 'dvdz', hcoord, scoord, boundary=boundary)
    
    
    def vort(self, hcoord, scoord, boundary='extend'):
        '''Calculate vertical relative vorticity from ds.'''

        var = xroms.relative_vorticity(self.ds, self.grid, boundary=boundary)
        var = self.grids(var, hcoord, scoord)
        var.name = 'vort'
        return var
    
    def ertel(self, hcoord, scoord, tracer=None, boundary='extend'):
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
        phi_xi, phi_eta = xroms.hgrad(phi, self.grid, boundary=boundary)
        phi_xi = self.grids(phi_xi, hcoord, scoord)
        
        phi_z = self.calc_ddz(phi, 'dphidz', hcoord, scoord, boundary=boundary)

        # vertical shear (horizontal components of vorticity)
        u_z = self.dudz('rho', 's_rho')
        v_z = self.dvdz('rho', 's_rho')

        # vertical component of vorticity on rho grid
        vort = self.vort('rho', 's_rho')

        # combine terms to get the ertel potential vorticity
        epv = -v_z * phi_xi + u_z * phi_eta + (self.ds.f + vort) * phi_z

        # add coordinates
        try:
            epv.coords['lon_rho'] = self.ds.coords['lon_rho']
            epv.coords['lat_rho'] = self.ds.coords['lat_rho']
            epv.coords['z_rho'] = self.ds.coords['z_rho']
            epv.coords['ocean_time'] = self.ds.coords['ocean_time']
        except:
            warn('Could not append coordinates')

        return epv

    