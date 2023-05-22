"""Test roms_seawater functions."""

import numpy as np
import xarray as xr

from xgcm import grid as xgrid

import xroms


grid1 = xr.open_dataset("xroms/tests/input/grid.nc")
# ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
ds = xr.open_dataset("xroms/tests/input/ocean_his_0001.nc")
# combine the two:
ds = ds.merge(grid1, overwrite_vars=True, compat="override")
ds, grid = xroms.roms_dataset(ds, include_3D_metrics=True)
# missing psi grid in variables
ds = ds.assign_coords({"lon_psi": ds.lon_psi, "lat_psi": ds.lat_psi})

# functions in test files:
xl, yl, N = 14, 9, 3
z_rho = np.array([-97.5025, -50.05, -2.5975])  # eta_rho=0, xi_rho=0
dx = 14168.78734979  # when eta_rho=0
dy = 18467.47219268  # when eta_rho=0
u = np.linspace(0, 1.2, xl - 1)
v = np.linspace(-0.1, 0.1, yl - 1)[:, np.newaxis]
temp = np.linspace(15, 20, N)
salt = np.linspace(25, 15, N)
zeta = np.linspace(-0.1, 0.1, xl)
g = 9.81
rho0 = 1025


def test_rho():
    rho = xroms.density(temp, salt, z_rho)
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    assert np.allclose(xrho[0, :, 0, 0], rho)


def test_potential_density():
    sig0 = xroms.density(temp, salt, 0)
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    assert np.allclose(xsig0[0, :, 0, 0], sig0)


def test_buoyancy():
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    xbuoy = xroms.buoyancy(xsig0)
    sig0 = xroms.density(temp, salt, 0)
    buoy = -g * sig0 / rho0
    assert np.allclose(xbuoy[0, :, 0, 0], buoy)


def test_N2():
    rho = xroms.density(temp, salt, z_rho)
    drhodz = (rho[2] - rho[0]) / (z_rho[2] - z_rho[0])
    var = -g * drhodz / rho0
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    # compare above estimate with two averaged since
    # there is a grid change
    assert np.allclose(xroms.N2(xrho, grid)[0, 1:3, 0, 0].mean(), var)


# def test_M2():
#     z_rho_xi2 = np.array([-97.50173077, -50.03461538,  -2.5675])
# #     z_rho_xi1 = np.array([-97.50211538, -50.04230769, -2.5825])
#     rho_xi0 = xroms.density(temp, salt, z_rho)
# #     rho_xi1 = xroms.density(temp, salt, z_rho_xi1)
#     rho_xi2 = xroms.density(temp, salt, z_rho_xi2)
#     drhodxi = (rho_xi2[1] - rho_xi0[1]) / (2 * dx)
#     drhodeta = 0
#     var = np.sqrt(drhodxi ** 2 + drhodeta ** 2) * g / rho0
#     xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
#     assert np.allclose(xroms.M2(xrho, grid), var)


def test_mld():
    # choose threshold so that z_rho[-2] is the mld
    xsig0 = xroms.density(ds.temp, ds.salt, 0)
    thresh = xsig0[0, -2, 0, 0] - xsig0[0, -1, 0, 0]
    assert np.allclose(
        xroms.mld(xsig0, grid, ds.h, ds.mask_rho, thresh=thresh)[0, 0, 0], z_rho[-2]
    )
