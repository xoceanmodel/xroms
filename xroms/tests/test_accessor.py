"""Test accessor functions."""

import cartopy
import numpy as np
import xarray as xr

from xgcm import grid as xgrid

import xroms


grid1 = xr.open_dataset("xroms/tests/input/grid.nc")
# ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
ds = xr.open_dataset("xroms/tests/input/ocean_his_0001.nc")
# combine the two:
ds = ds.merge(grid1, overwrite_vars=True, compat="override")
ds, grid = xroms.roms_dataset(ds)

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


def test_grid():
    assert isinstance(ds.xroms.grid, xgrid.Grid)


def test_ddz():
    ddz = (salt[2] - salt[0]) / (z_rho[2] - z_rho[0])
    assert np.allclose(ds.xroms.ddz("salt")[0, 1, 0, 0], ddz)


def test_ddeta():
    ddeta = (v[2] - v[0]) / (2 * dy)
    assert np.allclose(ds.xroms.ddeta("v")[0, 1, 1, 0], ddeta)


def test_ddxi():
    # compare away from boundaries and for
    # correct dx value (eta=0)
    ddxi = (u[2] - u[0]) / (2 * dx)
    assert np.allclose(ds.xroms.ddxi("u")[0, 1, 0, 1], ddxi)


def test_z_rho():
    assert np.allclose(ds.z_rho[0, :, 0, 0], z_rho)


def test_rho():
    assert np.allclose(ds.xroms.rho[0, :, 0, 0], xroms.density(temp, salt, z_rho))


def test_sig0():
    assert np.allclose(ds.xroms.sig0[0, :, 0, 0], xroms.density(temp, salt, 0))


def test_N2():
    rho = xroms.density(temp, salt, z_rho)
    drhodz = (rho[2] - rho[0]) / (z_rho[2] - z_rho[0])
    var = -g * drhodz / rho0
    # compare above estimate with two averaged since
    # there is a grid change
    assert np.allclose(ds.xroms.N2[0, 1:3, 0, 0].mean(), var)


def test_M2():
    z_rho_xi1 = np.array([-97.50211538, -50.04230769, -2.5825])
    rho_xi0 = xroms.density(temp, salt, z_rho)
    rho_xi1 = xroms.density(temp, salt, z_rho_xi1)
    drhodxi = (rho_xi1[1] - rho_xi0[1]) / (2 * dx)
    drhodeta = 0
    var = np.sqrt(drhodxi ** 2 + drhodeta ** 2) * g / rho0


def test_mld():
    # choose threshold so that z_rho[-2] is the mld
    sig0 = xroms.density(temp, salt, 0)
    thresh = sig0[-2] - sig0[-1]
    assert np.allclose(ds.xroms.mld(thresh=thresh)[0, 0, 0], z_rho[-2])


def test_speed():
    assert np.allclose(
        ds.xroms.speed.mean(), np.sqrt(u ** 2 + v ** 2).mean(), rtol=1e-2
    )


def test_KE():
    rho = xroms.density(temp, salt, z_rho)[:, np.newaxis, np.newaxis]
    s2 = (u ** 2 + v ** 2)[np.newaxis, :, :]
    KE = 0.5 * rho * s2
    assert np.allclose(
        ds.xroms.KE.xroms.to_grid(hcoord="psi").mean(), KE.mean(), rtol=1e-2
    )


def test_relative_vorticity():
    assert np.allclose(ds.xroms.vort, 0)


def test_dudz():
    assert np.allclose(ds.xroms.dudz, 0)


def test_dvdz():
    assert np.allclose(ds.xroms.dvdz, 0)
