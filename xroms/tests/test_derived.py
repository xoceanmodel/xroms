"""Test derived functions."""

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


def test_speed():

    # run speed in xroms
    xvar = xroms.speed(ds.u, ds.v, grid)

    # calculate own version of speed
    var = np.sqrt(u**2 + v**2)

    # these aren't identical because of grid shifting
    assert np.allclose(xvar.mean(), var.mean(), rtol=1e-2)


def test_KE():
    rho = xroms.density(temp, salt, z_rho)[:, np.newaxis, np.newaxis]
    s2 = (u**2 + v**2)[np.newaxis, :, :]
    KE = 0.5 * rho * s2

    # xroms
    s = xroms.speed(ds.u, ds.v, grid)
    assert np.allclose(
        xroms.to_grid(xroms.KE(ds.rho0, s), grid, hcoord="psi").mean(),
        KE.mean(),
        rtol=1e-2,
    )


def test_uv_geostrophic():
    vg = 0  # zeta only varies in xi direction
    # test one corner of domain
    f = ds.f[0, 0].values
    # correct dx value (eta=0)
    dzetadxi = (zeta[2] - zeta[0]) / (2 * dx)
    ug = -g * dzetadxi / f

    assert np.allclose(
        xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="xi")[0, 0, 0], ug
    )
    assert np.allclose(xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="eta"), vg)


def test_EKE():
    vg = 0  # zeta only varies in xi direction
    # test one corner of domain
    f = ds.f[0, 0].values
    # correct dx value (eta=0)
    dzetadxi = (zeta[2] - zeta[0]) / (2 * dx)
    ug = -g * dzetadxi / f
    EKE = 0.5 * (ug**2 + vg**2)

    xug, xvg = xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="both")

    assert np.allclose(xroms.EKE(xug, xvg, grid)[0, 0, 0], EKE)


def test_dudz():
    assert np.allclose(xroms.dudz(ds.u, grid), 0)


def test_dvdz():
    assert np.allclose(xroms.dvdz(ds.v, grid), 0)


def test_vertical_shear():
    xdudz = xroms.dudz(ds.u, grid)
    xdvdz = xroms.dvdz(ds.v, grid)
    assert np.allclose(xroms.vertical_shear(xdudz, xdvdz, grid), 0)


def test_relative_vorticity():
    assert np.allclose(xroms.relative_vorticity(ds.u, ds.v, grid), 0)


def test_divergence():
    dudxi = (ds.u[0, -1, 0, 2] - ds.u[0, -1, 0, 0]) / (ds.dx[0, 1] + ds.dx[0, 0])
    dvdeta = (ds.v[0, -1, 2, 0] - ds.v[0, -1, 0, 0]) / (ds.dy[1, 0] + ds.dy[0, 0])
    calc = dudxi + dvdeta
    # choose middle divergence value in depth because of boundary effects?
    assert np.allclose(xroms.divergence(ds.u, ds.v, grid)[0, 1, 1, 1], calc, atol=2e8)


def test_ertel():
    v_z = 0
    u_z = 0
    # test one corner of domain
    f = ds.f[0, 0].values
    vort = 0
    sig0 = xroms.density(temp, salt, 0)
    buoy = -g * sig0 / rho0
    phi_z = (buoy[2] - buoy[0]) / (z_rho[2] - z_rho[0])
    ertel = -v_z + u_z + (f + vort) * phi_z

    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    xbuoy = xroms.buoyancy(xsig0)

    assert np.allclose(xroms.ertel(xbuoy, ds.u, ds.v, ds.f, grid)[0, 1, 0, 0], ertel)


def test_w():
    # VRX
    pass


def test_omega():
    # VRX
    pass
