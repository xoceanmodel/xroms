"""Test accessor functions by ensuring accessor and xroms
functions return same values."""

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


def test_grid():
    assert isinstance(ds.xroms.grid, xgrid.Grid)


def test_speed():

    acc = ds.xroms.speed

    assert np.allclose(acc, xroms.speed(ds.u, ds.v, grid))

    # also check attributes
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    # cf-xarray: make sure all Axes and Coordinates available in output
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_KE():

    s = xroms.speed(ds.u, ds.v, grid)
    acc = ds.xroms.KE

    assert np.allclose(acc, xroms.KE(ds.rho0, s))

    # also check attributes
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    # cf-xarray: make sure all Axes and Coordinates available in output
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_uv_geostrophic():

    acc = ds.xroms.ug
    assert np.allclose(acc, xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="xi"))
    # also check attributes
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    # cf-xarray: make sure all Axes and Coordinates available in output
    items = ["T", "X", "Y", "longitude", "latitude", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())

    acc = ds.xroms.vg
    assert np.allclose(acc, xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="eta"))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "longitude", "latitude", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_EKE():

    acc = ds.xroms.EKE
    xug, xvg = xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="both")
    assert np.allclose(acc, xroms.EKE(xug, xvg, grid))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "longitude", "latitude", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_dudz():
    acc = ds.xroms.dudz
    assert np.allclose(acc, xroms.dudz(ds.u, grid))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_dvdz():
    acc = ds.xroms.dvdz
    assert np.allclose(acc, xroms.dvdz(ds.v, grid))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_vertical_shear():
    xdudz = ds.xroms.dudz
    xdvdz = ds.xroms.dvdz
    acc = ds.xroms.vertical_shear
    assert np.allclose(acc, xroms.vertical_shear(xdudz, xdvdz, grid))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_relative_vorticity():
    acc = ds.xroms.vort
    assert np.allclose(acc, 0)
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_ertel():
    acc = ds.xroms.ertel
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    xbuoy = xroms.buoyancy(xsig0)
    assert np.allclose(acc, xroms.ertel(xbuoy, ds.u, ds.v, ds.f, grid))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_w():
    # VRX
    pass


#     acc = ds.xroms.w
#     assert np.allclose(acc, xroms.w(ds.u, ds.v, grid))
#     acc.name == acc.attrs['name']
#     acc.attrs['grid'] == ds.xroms.grid
#     items = ['T','X','Y','Z','longitude','latitude','vertical','time']
#     assert set(items).issubset(acc.cf.get_valid_keys())


def test_omega():
    # VRX
    pass


#     acc = ds.xroms.omega
#     assert np.allclose(acc, xroms.omega(ds.u, ds.v, grid))
#     acc.name == acc.attrs['name']
#     acc.attrs['grid'] == ds.xroms.grid
#     items = ['T','X','Y','Z','longitude','latitude','vertical','time']
#     assert set(items).issubset(acc.cf.get_valid_keys())


def test_rho():
    acc = ds.xroms.rho
    assert np.allclose(acc, xroms.density(ds.temp, ds.salt, ds.z_rho))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_sig0():
    acc = ds.xroms.sig0
    assert np.allclose(acc, xroms.potential_density(ds.temp, ds.salt, 0))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_buoyancy():
    acc = ds.xroms.buoyancy
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    assert np.allclose(acc, xroms.buoyancy(xsig0))
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_N2():
    acc = ds.xroms.N2
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    assert np.allclose(acc, xroms.N2(xrho, grid), equal_nan=True)
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_M2():
    acc = ds.xroms.M2
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    assert np.allclose(acc, xroms.M2(xrho, grid), equal_nan=True)
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_mld():
    acc = ds.xroms.mld(thresh=0.03)
    sig0 = xroms.potential_density(ds.temp, ds.salt, 0)
    assert np.allclose(acc, xroms.mld(sig0, ds.h, ds.mask_rho), equal_nan=True)
    acc.name == acc.attrs["name"]
    acc.attrs["grid"] == ds.xroms.grid
    items = ["T", "X", "Y", "longitude", "latitude", "time"]
    assert set(items).issubset(acc.cf.get_valid_keys())


def test_ddxi():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        acc = ds[testvar].xroms.ddxi()
        assert np.allclose(acc, xroms.ddxi(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())

        acc = ds.xroms.ddxi(testvar)
        assert np.allclose(acc, xroms.ddxi(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())


def test_ddeta():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        acc = ds[testvar].xroms.ddeta()
        assert np.allclose(acc, xroms.ddeta(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())

        acc = ds.xroms.ddeta(testvar)
        assert np.allclose(acc, xroms.ddeta(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())


def test_ddz():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        acc = ds[testvar].xroms.ddz()
        assert np.allclose(acc, xroms.ddz(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())

        acc = ds.xroms.ddz(testvar)
        assert np.allclose(acc, xroms.ddz(ds[testvar], grid))
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())


def test_to_grid():

    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        for scoord in ["s_w", "s_rho"]:
            for hcoord in ["rho", "u", "v", "psi"]:
                acc = ds[testvar].xroms.to_grid(hcoord=hcoord, scoord=scoord)
                assert np.allclose(
                    acc, xroms.to_grid(ds[testvar], grid, hcoord=hcoord, scoord=scoord)
                )
                acc.name == acc.attrs["name"]
                acc.attrs["grid"] == ds.xroms.grid
                items = [
                    "T",
                    "X",
                    "Y",
                    "Z",
                    "longitude",
                    "latitude",
                    "vertical",
                    "time",
                ]
                assert set(items).issubset(acc.cf.get_valid_keys())

                acc = ds.xroms.to_grid(testvar, hcoord=hcoord, scoord=scoord)
                assert np.allclose(
                    acc, xroms.to_grid(ds[testvar], grid, hcoord=hcoord, scoord=scoord)
                )
                acc.name == acc.attrs["name"]
                acc.attrs["grid"] == ds.xroms.grid
                items = [
                    "T",
                    "X",
                    "Y",
                    "Z",
                    "longitude",
                    "latitude",
                    "vertical",
                    "time",
                ]
                assert set(items).issubset(acc.cf.get_valid_keys())


def test_sel2d():
    lon0, lat0 = -94.8, 28.0
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        acc = ds[testvar].xroms.sel2d(lon0, lat0)
        out = xroms.sel2d(
            ds[testvar],
            ds[testvar].cf["longitude"],
            ds[testvar].cf["latitude"],
            lon0,
            lat0,
        )
        assert np.allclose(acc, out)
        acc.name == acc.attrs["name"]
        acc.attrs["grid"] == ds.xroms.grid
        items = ["T", "X", "Y", "Z", "longitude", "latitude", "vertical", "time"]
        assert set(items).issubset(acc.cf.get_valid_keys())


def test_argsel2d():
    lon0, lat0 = -94.8, 28.0
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        inds = ds[testvar].xroms.argsel2d(lon0, lat0)
        outinds = xroms.argsel2d(
            ds[testvar].cf["longitude"], ds[testvar].cf["latitude"], lon0, lat0
        )
        assert np.allclose(inds, outinds)


def test_gridmean():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        axis = "X"
        var1 = ds[testvar].xroms.gridmean(axis)
        var2 = xroms.gridmean(ds[testvar], grid, axis)
        assert np.allclose(var1, var2)


def test_gridsum():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        axis = "X"
        var1 = ds[testvar].xroms.gridsum(axis)
        var2 = xroms.gridsum(ds[testvar], grid, axis)
        assert np.allclose(var1, var2)


def test_interpll():
    ie, ix = 2, 3
    indexer = {"eta_rho": [ie], "xi_rho": [ix]}
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        var1 = xroms.interpll(
            ds[testvar], ds.lon_rho.isel(indexer), ds.lat_rho.isel(indexer)
        )
        var2 = ds[testvar].xroms.interpll(
            ds.lon_rho.isel(indexer), ds.lat_rho.isel(indexer)
        )
        assert np.allclose(var1, var2)


def test_zslice():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        varin = ds[testvar]
        depths = np.asarray(ds[testvar].cf["vertical"][0, :, 0, 0].values)
        varout = xroms.isoslice(varin, depths, grid, axis="Z")
        varcomp = ds[testvar].xroms.isoslice(depths, axis="Z")
        assert np.allclose(
            varout.cf.isel(T=0, Y=0, X=0), varcomp.cf.isel(T=0, Y=0, X=0)
        )
