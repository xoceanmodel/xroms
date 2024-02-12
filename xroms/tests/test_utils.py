"""Test utilities in xroms"""

import cartopy
import numpy as np
import xarray as xr

import xroms


grid1 = xr.open_dataset("xroms/tests/input/grid.nc")
# ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
ds = xr.open_dataset("xroms/tests/input/ocean_his_0001.nc")
# combine the two:
ds = ds.merge(grid1, overwrite_vars=True, compat="override")
ds, grid = xroms.roms_dataset(ds, include_3D_metrics=True)
# # missing psi grid in variables
# ds = ds.assign_coords({'lon_psi': ds.lon_psi, 'lat_psi': ds.lat_psi})

xl, yl, N = 14, 9, 3
z_rho = np.array([-97.5025, -50.05, -2.5975])  # eta_rho=0, xi_rho=0
dz = np.array([33.3, 33.3, 33.3])  # eta_rho=0, xi_rho=0
dx = 14168.78734979  # when eta_rho=0
dy = 18467.47219268  # when eta_rho=0
u = np.linspace(0, 1.2, xl - 1)
v = np.linspace(-0.1, 0.1, yl - 1)[:, np.newaxis]
temp = np.linspace(15, 20, N)
salt = np.linspace(25, 15, N)
zeta = np.linspace(-0.1, 0.1, xl)
g = 9.81
rho0 = 1025


def test_ddz():
    ddz = (salt[2] - salt[0]) / (z_rho[2] - z_rho[0])
    assert np.allclose(xroms.ddz(ds.salt, grid)[0, 1, 0, 0], ddz)


def test_ddeta():
    ddeta = (v[2] - v[0]) / (2 * dy)
    assert np.allclose(xroms.ddeta(ds.v, grid)[0, 1, 1, 0], ddeta)


def test_ddxi():
    # compare away from boundaries and for
    # correct dx value (eta=0)
    ddxi = (u[2] - u[0]) / (2 * dx)
    assert np.allclose(xroms.ddxi(ds.u, grid)[0, 1, 0, 1], ddxi)


def test_to_rho():

    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        var = xroms.to_rho(ds[testvar], grid)
        # check for correct dims
        assert "xi_rho" in var.dims
        assert "eta_rho" in var.dims


def test_to_u():

    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        var = xroms.to_u(ds[testvar], grid)
        # check for correct dims
        assert "xi_u" in var.dims
        assert "eta_rho" in var.dims


def test_to_v():

    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        var = xroms.to_v(ds[testvar], grid)
        # check for correct dims
        assert "xi_rho" in var.dims
        assert "eta_v" in var.dims


def test_to_psi():

    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        var = xroms.to_psi(ds[testvar], grid)
        # check for correct dims
        assert "xi_u" in var.dims
        assert "eta_v" in var.dims


def test_to_s_rho():

    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        var = xroms.to_s_rho(ds[testvar], grid)
        # check for correct dims
        assert "s_rho" in var.dims


def test_to_s_w():

    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        var = xroms.to_s_w(ds[testvar], grid)
        # check for correct dims
        assert "s_w" in var.dims


def test_to_grid():

    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        for scoord in ["s_rho", "s_w"]:
            var = xroms.to_grid(ds[testvar], grid, hcoord="rho", scoord=scoord)
            # check for correct dims
            assert scoord in var.dims
            assert "eta_rho" in var.dims
            assert "xi_rho" in var.dims

            var = xroms.to_grid(ds[testvar], grid, hcoord="u", scoord=scoord)
            assert scoord in var.dims
            assert "eta_rho" in var.dims
            assert "xi_u" in var.dims

            var = xroms.to_grid(ds[testvar], grid, hcoord="v", scoord=scoord)
            assert scoord in var.dims
            assert "eta_v" in var.dims
            assert "xi_rho" in var.dims

            var = xroms.to_grid(ds[testvar], grid, hcoord="psi", scoord=scoord)
            # import pdb; pdb.set_trace()
            assert scoord in var.dims
            assert "eta_v" in var.dims
            assert "xi_u" in var.dims


def test_argsel2d():
    """Check that function returns correct indices.

    This compares with previous calculation."""

    lon0, lat0 = -95.8, 27.1
    assert xroms.argsel2d(ds.lon_rho, ds.lat_rho, lon0, lat0) == (0, 1)
    assert xroms.argsel2d(ds.lon_psi, ds.lat_psi, lon0, lat0) == (0, 0)
    assert xroms.argsel2d(ds.lon_u, ds.lat_u, lon0, lat0) == (0, 0)
    assert xroms.argsel2d(ds.lon_v, ds.lat_v, lon0, lat0) == (0, 1)
    assert xroms.argsel2d(ds.lon_vert, ds.lat_vert, lon0, lat0) == (1, 1)


def test_argsel2d_exact():
    """test for exact index."""

    lon0, lat0 = -95.928571, 27.166685  # one corner of grid
    assert xroms.argsel2d(ds.lon_rho, ds.lat_rho, lon0, lat0) == (0, 0)
    assert xroms.argsel2d(ds.lon_psi, ds.lat_psi, lon0, lat0) == (0, 0)
    assert xroms.argsel2d(ds.lon_u, ds.lat_u, lon0, lat0) == (1, 0)
    assert xroms.argsel2d(ds.lon_v, ds.lat_v, lon0, lat0) == (0, 0)
    assert xroms.argsel2d(ds.lon_vert, ds.lat_vert, lon0, lat0) == (1, 1)

    lon0, lat0 = -94.071429, 28.333351  # other corner of grid
    assert xroms.argsel2d(ds.lon_rho, ds.lat_rho, lon0, lat0) == (7, 13)
    assert xroms.argsel2d(ds.lon_psi, ds.lat_psi, lon0, lat0) == (7, 12)
    assert xroms.argsel2d(ds.lon_u, ds.lat_u, lon0, lat0) == (8, 12)
    assert xroms.argsel2d(ds.lon_v, ds.lat_v, lon0, lat0) == (7, 13)
    assert xroms.argsel2d(ds.lon_vert, ds.lat_vert, lon0, lat0) == (8, 13)


def test_sel2d():
    """Check that function returns correct value for scalar inputs."""

    lon0, lat0 = -94.8, 28.0

    assert (
        xroms.sel2d(ds.u, ds.lon_u, ds.lat_u, lon0, lat0).isel(s_rho=0, ocean_time=0)
        == 0.7
    )
    assert np.allclose(
        xroms.sel2d(ds.v, ds.lon_v, ds.lat_v, lon0, lat0).isel(s_rho=0, ocean_time=0),
        0.042857,
    )


def test_xisoslice():
    """Test xisoslice function."""

    # test longitude slice of u
    res = xroms.xisoslice(ds.lon_u, -95, ds.u, "xi_u")
    assert np.allclose(res, 0.6)

    # test latitude slice of u
    res = xroms.xisoslice(ds.lat_u, 28, ds.u, "eta_rho").std()
    assert np.allclose(res, 0.37416574)

    # test when slice isn't along a value that is equal along that slice
    # convert to projected space
    proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    pc = cartopy.crs.PlateCarree()
    x_u, y_u = proj.transform_points(pc, ds.lon_u.values, ds.lat_u.values)[..., :2].T
    ds["x_u"] = (("eta_u", "xi_u"), x_u.T)
    res = xroms.xisoslice(ds.x_u, 380000, ds.lon_u, "xi_u").mean()
    assert np.allclose(res, -94.19483976)

    # test getting latitude for longitude slice
    res = xroms.xisoslice(ds.lon_rho, -94.5, ds.lat_rho, "xi_rho").mean().values
    assert np.allclose(res, 27.75001934)

    # test requesting an exact iso_value that is in the iso_array, in xi_rho
    lon0 = ds.lon_rho[4, 7]
    res = xroms.xisoslice(ds.lon_rho, lon0, ds.lat_rho, "xi_rho")
    assert np.allclose(res, ds.lat_rho[:, 7])

    # test requesting an exact iso_value that is in the iso_array, in eta_rho
    lat0 = ds.lat_rho[3, 4]
    res = xroms.xisoslice(ds.lat_rho, lat0, ds.lon_rho, "eta_rho")
    assert np.allclose(res, ds.lon_rho[3, :])


def test_gridmean():
    var1 = (ds.salt * ds.dx).cf.sum("X") / ds.dx.cf.sum("X")
    var2 = xroms.gridmean(ds.salt, grid, "X")
    assert np.allclose(var1, var2)

    var1 = (ds.salt * ds.dy).cf.sum("Y") / ds.dy.cf.sum("Y")
    var2 = xroms.gridmean(ds.salt, grid, "Y")
    assert np.allclose(var1, var2)

    var1 = (ds.u * ds.dx_u).cf.sum("X") / ds.dx_u.cf.sum("X")
    var2 = xroms.gridmean(ds.u, grid, "X")
    assert np.allclose(var1, var2)

    var1 = (ds.u * ds.dy_u).cf.sum("Y") / ds.dy_u.cf.sum("Y")
    var2 = xroms.gridmean(ds.u, grid, "Y")
    assert np.allclose(var1, var2)


def test_gridsum():
    var1 = (ds.salt * ds.dx).cf.sum("X")
    var2 = xroms.gridsum(ds.salt, grid, "X")
    assert np.allclose(var1, var2)

    var1 = (ds.salt * ds.dy).cf.sum("Y")
    var2 = xroms.gridsum(ds.salt, grid, "Y")
    assert np.allclose(var1, var2)

    var1 = (ds.u * ds.dx_u).cf.sum("X")
    var2 = xroms.gridsum(ds.u, grid, "X")
    assert np.allclose(var1, var2)

    var1 = (ds.u * ds.dy_u).cf.sum("Y")
    var2 = xroms.gridsum(ds.u, grid, "Y")
    assert np.allclose(var1, var2)
