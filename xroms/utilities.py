import xarray as xr


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
    an isosurface in the array iso_array defined by iso_val.

    Inputs:
    iso_array:       xarray.DataArray in which the isosurface is defined
    iso_value:       float: value of the isosurface in iso_array
    projected_array: xarray.DataArray in which to project values on the isosurface
                     Needs to have the same dimensions and shape as iso_array
    coord:           string: coordinate associated with the dimension along which to project

    Output:
    iso_values:      xarray.DataArray: values of projected_array on the isosurface
    '''
    Nm = len(iso_array.coord) - 1

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


def salt_variance(ds, salt):
    '''Returns the three volume-averaged terms for the decomposition of salinity variance: 
    total, vertical, and horizontal variance. See Li et al. (2018) JPO for details.
    Inputs:
    ----
    ds: DataArray
    salt: DataArray
    Outputs:
    ----
    svar: total salinity variance
    svert: vertical salinity variance
    shorz: horizontal salinity variance
    '''
    
    dV = ds.dx*ds.dy*ds.dz
    V = dV.sum(dim = ['eta_rho','xi_rho','s_rho'])
    z = ds.dz.sum(dim = 's_rho')

    sbar = ((salt*dV).sum(dim = ['eta_rho','xi_rho','s_rho']))/V

    svar = (((salt-sbar)**2*dV).sum(dim = ['eta_rho','xi_rho','s_rho']))/V

    svbar = ((salt*ds.dz).sum(dim = ['s_rho']))/z
    svert = (((salt-svbar)**2*dV).sum(dim = ['eta_rho','xi_rho','s_rho']))/V
    shorz = (svar-svert)
    return svar, svert, shorz
