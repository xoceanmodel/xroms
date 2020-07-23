import xarray as xr
import numpy as np


def sel2d(ds, lon0, lat0):
    '''`sel` in lon and lat simultaneously.
    
    Return ds subsetted to grid node nearest lon0, lat0 calculating in 2D.
    '''
    
    import cartopy
    proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    pc = cartopy.crs.PlateCarree()

    # convert grid points from lon/lat to a reasonable projection for calculating distances
    x, y = proj.transform_points( pc, ds.lon_rho.values, ds.lat_rho.values )[...,:2].T

    # convert point of interest
    x0, y0 = proj.transform_point( lon0, lat0, pc )

    # calculate distance from point of interest
    dist = np.sqrt( (x - x0)**2 + (y - y0)**2 )

    ix, iy = np.where(dist==dist.min())

    return ds.isel(xi_rho=ix, eta_rho=iy)


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
