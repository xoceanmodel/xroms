"""Test accessor functions by ensuring accessor and xroms
functions return same values."""

import numpy as np
import pytest
import xarray as xr

# from xgcm import grid as xgrid
import xgcm.grid

import xroms


grid1 = xr.open_dataset("xroms/tests/input/grid.nc")
# ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
ds = xr.open_dataset("xroms/tests/input/ocean_his_0001.nc")
# combine the two:
ds = ds.merge(grid1, overwrite_vars=True, compat="override")
ds, grid = xroms.roms_dataset(ds)

axesTZYX = ["T", "Z", "Y", "X"]
axesTYX = ["T", "Y", "X"]
coordnamesTZYX = ["time", "vertical", "latitude", "longitude"]
coordnamesTYX = ["time", "latitude", "longitude"]

dim_dict = {
    "rho": {
        "s_rho": ["ocean_time", "s_rho", "eta_rho", "xi_rho"],
        "s_w": ["ocean_time", "s_w", "eta_rho", "xi_rho"],
        None: ["ocean_time", "eta_rho", "xi_rho"],
    },
    "u": {
        "s_rho": ["ocean_time", "s_rho", "eta_rho", "xi_u"],
        "s_w": ["ocean_time", "s_w", "eta_rho", "xi_u"],
        None: ["ocean_time", "eta_rho", "xi_u"],
    },
    "v": {
        "s_rho": ["ocean_time", "s_rho", "eta_v", "xi_rho"],
        "s_w": ["ocean_time", "s_w", "eta_v", "xi_rho"],
        None: ["ocean_time", "eta_v", "xi_rho"],
    },
    "psi": {
        "s_rho": ["ocean_time", "s_rho", "eta_v", "xi_u"],
        "s_w": ["ocean_time", "s_w", "eta_v", "xi_u"],
        None: ["ocean_time", "eta_v", "xi_u"],
    },
}

coord_dict = {
    "rho": {
        "s_rho": ["ocean_time", "z_rho", "lat_rho", "lon_rho"],
        "s_w": ["ocean_time", "z_w", "lat_rho", "lon_rho"],
        None: ["ocean_time", "lat_rho", "lon_rho"],
    },
    "u": {
        "s_rho": ["ocean_time", "z_rho_u", "lat_u", "lon_u"],
        "s_w": ["ocean_time", "z_w_u", "lat_u", "lon_u"],
        None: ["ocean_time", "lat_u", "lon_u"],
    },
    "v": {
        "s_rho": ["ocean_time", "z_rho_v", "lat_v", "lon_v"],
        "s_w": ["ocean_time", "z_w_v", "lat_v", "lon_v"],
        None: ["ocean_time", "lat_v", "lon_v"],
    },
    "psi": {
        "s_rho": ["ocean_time", "z_rho_psi", "lat_psi", "lon_psi"],
        "s_w": ["ocean_time", "z_w_psi", "lat_psi", "lon_psi"],
        None: ["ocean_time", "lat_psi", "lon_psi"],
    },
}


def test_grid():
    assert isinstance(ds.xroms.xgrid, xgcm.grid.Grid)


def test_speed():

    acc = ds.xroms.speed

    assert np.allclose(acc, xroms.speed(ds.u, ds.v, grid))

    # also check attributes
    assert acc.name == acc.attrs["name"]
    # cf-xarray: make sure all Axes and Coordinates available in output
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_KE():

    s = xroms.speed(ds.u, ds.v, grid)
    acc = ds.xroms.KE

    assert np.allclose(acc, xroms.KE(ds.rho0, s))

    # also check attributes
    assert acc.name == acc.attrs["name"]
    # cf-xarray: make sure all Axes and Coordinates available in output
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_uv_geostrophic():

    acc = ds.xroms.ug
    assert np.allclose(acc, xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="xi"))
    # also check attributes
    assert (
        acc.name == acc.attrs["name"]
    )  # cf-xarray: make sure all Axes and Coordinates available in output
    hcoord = "u"
    scoord = None
    dims = dim_dict[hcoord][scoord]
    axes = axesTYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord

    acc = ds.xroms.vg
    assert np.allclose(acc, xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="eta"))
    assert acc.name == acc.attrs["name"]
    hcoord = "v"
    scoord = None
    dims = dim_dict[hcoord][scoord]
    axes = axesTYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_EKE():

    acc = ds.xroms.EKE
    xug, xvg = xroms.uv_geostrophic(ds.zeta, ds.f, grid, which="both")
    assert np.allclose(acc, xroms.EKE(xug, xvg, grid))
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = None
    dims = dim_dict[hcoord][scoord]
    axes = axesTYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_dudz():
    acc = ds.xroms.dudz
    assert np.allclose(acc, xroms.dudz(ds.u, grid))
    assert acc.name == acc.attrs["name"]
    hcoord = "u"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_dvdz():
    acc = ds.xroms.dvdz
    assert np.allclose(acc, xroms.dvdz(ds.v, grid))
    assert acc.name == acc.attrs["name"]
    hcoord = "v"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_vertical_shear():
    xdudz = ds.xroms.dudz
    xdvdz = ds.xroms.dvdz
    acc = ds.xroms.vertical_shear
    assert np.allclose(acc, xroms.vertical_shear(xdudz, xdvdz, grid))
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_relative_vorticity():
    acc = ds.xroms.vort
    assert np.allclose(acc, 0)
    assert acc.name == acc.attrs["name"]
    hcoord = "psi"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_div():
    acc = ds.xroms.div
    assert np.allclose(acc, xroms.divergence(ds["u"], ds["v"], grid))


def test_div_norm():
    acc = ds.xroms.div_norm
    assert np.allclose(
        acc, xroms.divergence(ds["u"], ds["v"], grid).cf.isel(Z=-1) / ds["f"]
    )


def test_ertel():
    acc = ds.xroms.ertel
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    xbuoy = xroms.buoyancy(xsig0)
    assert np.allclose(acc, xroms.ertel(xbuoy, ds.u, ds.v, ds.f, grid))
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


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
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_sig0():
    acc = ds.xroms.sig0
    assert np.allclose(acc, xroms.potential_density(ds.temp, ds.salt, 0))
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_buoyancy():
    acc = ds.xroms.buoyancy
    xsig0 = xroms.potential_density(ds.temp, ds.salt)
    assert np.allclose(acc, xroms.buoyancy(xsig0))
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_rho"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_N2():
    acc = ds.xroms.N2
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    assert np.allclose(acc, xroms.N2(xrho, grid), equal_nan=True)
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_M2():
    acc = ds.xroms.M2
    xrho = xroms.density(ds.temp, ds.salt, ds.z_rho)
    assert np.allclose(acc, xroms.M2(xrho, grid), equal_nan=True)
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = "s_w"
    dims = dim_dict[hcoord][scoord]
    axes = axesTZYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTZYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_mld():
    acc = ds.xroms.mld(thresh=0.03)
    sig0 = xroms.potential_density(ds.temp, ds.salt, 0)
    assert np.allclose(acc, xroms.mld(sig0, grid, ds.h, ds.mask_rho), equal_nan=True)
    assert acc.name == acc.attrs["name"]
    hcoord = "rho"
    scoord = None
    dims = dim_dict[hcoord][scoord]
    axes = axesTYX
    coords = coord_dict[hcoord][scoord]
    coordnames = coordnamesTYX
    for ax, dim in zip(axes, dims):
        assert acc.cf[ax].name == dim
    for coordname, coord in zip(coordnames, coords):
        assert acc.cf[coordname].name == coord


def test_ddxi():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        with pytest.raises(KeyError):
            acc = ds[testvar].xroms.ddxi(grid)

        if testvar == "salt":
            hcoord = "u"
            scoord = "s_w"
        elif testvar == "u":
            hcoord = "rho"
            scoord = "s_w"
        elif testvar == "v":
            hcoord = "psi"
            scoord = "s_w"
        dims = dim_dict[hcoord][scoord]
        axes = axesTZYX
        coords = coord_dict[hcoord][scoord]
        coordnames = coordnamesTZYX

        acc = ds.xroms.ddxi(testvar)
        assert np.allclose(acc, xroms.ddxi(ds[testvar], grid))
        assert acc.name == acc.attrs["name"]
        for ax, dim in zip(axes, dims):
            assert acc.cf[ax].name == dim
        for coordname, coord in zip(coordnames, coords):
            assert acc.cf[coordname].name == coord


def test_ddeta():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        with pytest.raises(KeyError):
            acc = ds[testvar].xroms.ddeta(grid)

        if testvar == "salt":
            hcoord = "v"
            scoord = "s_w"
        elif testvar == "u":
            hcoord = "psi"
            scoord = "s_w"
        elif testvar == "v":
            hcoord = "rho"
            scoord = "s_w"
        dims = dim_dict[hcoord][scoord]
        axes = axesTZYX
        coords = coord_dict[hcoord][scoord]
        coordnames = coordnamesTZYX

        acc = ds.xroms.ddeta(testvar)
        assert np.allclose(acc, xroms.ddeta(ds[testvar], grid))
        assert acc.name == acc.attrs["name"]
        for ax, dim in zip(axes, dims):
            assert acc.cf[ax].name == dim
        for coordname, coord in zip(coordnames, coords):
            assert acc.cf[coordname].name == coord


def test_ddz():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        with pytest.raises(KeyError):
            acc = ds[testvar].xroms.ddz(grid)
        dims = list(ds[testvar].dims)
        axes = axesTZYX
        coords = [ds[testvar].cf[coordname].name for coordname in coordnamesTZYX]
        coordnames = coordnamesTZYX
        # correct dim and coord in derivative direction
        # import pdb; pdb.set_trace()
        if grid._get_dims_from_axis(ds[testvar], "Z")[0] == "s_rho":
            # if grid.axes["Z"]._get_axis_coord(ds[testvar])[1] == "s_rho":
            dims[1] = "s_w"
            coords[1] = coords[1].replace("rho", "w")
        else:
            dims[1] = "s_rho"
            coords[1] = coords[1].replace("w", "rho")

        acc = ds.xroms.ddz(testvar)
        assert np.allclose(acc, xroms.ddz(ds[testvar], grid))
        assert acc.name == acc.attrs["name"]
        for ax, dim in zip(axes, dims):
            assert acc.cf[ax].name == dim
        for coordname, coord in zip(coordnames, coords):
            assert acc.cf[coordname].name == coord


def test_to_grid():

    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        for scoord in ["s_w", "s_rho"]:
            for hcoord in ["rho", "u", "v", "psi"]:
                acc = ds.xroms.to_grid(testvar, hcoord=hcoord, scoord=scoord)
                # acc = ds[testvar].xroms.to_grid(grid, hcoord=hcoord, scoord=scoord)
                assert np.allclose(
                    acc, xroms.to_grid(ds[testvar], grid, hcoord=hcoord, scoord=scoord)
                )
                assert acc.name == acc.attrs["name"]
                dims = dim_dict[hcoord][scoord]
                axes = axesTZYX
                coords = coord_dict[hcoord][scoord]
                coordnames = coordnamesTZYX
                for ax, dim in zip(axes, dims):
                    assert acc.cf[ax].name == dim
                for coordname, coord in zip(coordnames, coords):
                    assert acc.cf[coordname].name == coord

                acc = ds.xroms.to_grid(testvar, hcoord=hcoord, scoord=scoord)
                assert np.allclose(
                    acc, xroms.to_grid(ds[testvar], grid, hcoord=hcoord, scoord=scoord)
                )
                assert acc.name == acc.attrs["name"]
                for ax, dim in zip(axes, dims):
                    assert acc.cf[ax].name == dim
                for coordname, coord in zip(coordnames, coords):
                    assert acc.cf[coordname].name == coord


# can't figure out what is wrong here, will have to come back
# def test_sel2d():
#     lon0, lat0 = -94.8, 28.0
#     testvars = ["salt", "u", "v"]
#     for testvar in testvars:
#         acc = ds[testvar].xroms.sel2d(lon0, lat0)
#         out = xroms.sel2d(
#             ds[testvar],
#             ds[testvar].cf["longitude"],
#             ds[testvar].cf["latitude"],
#             lon0,
#             lat0,
#         )
#         assert np.allclose(acc, out)
#         assert acc.name == testvar
#         dims = ds[testvar].dims
#         axes = axesTZYX
#         coords = [ds[testvar].cf[coordname].name for coordname in coordnamesTZYX]
#         coordnames = coordnamesTZYX
#         # import pdb; pdb.set_trace()
#         for ax, dim in zip(axes, dims):
#             assert acc.cf[ax].name == dim
#         for coordname, coord in zip(coordnames, coords):
#             assert acc.cf[coordname].name == coord


def test_argsel2d():
    lon0, lat0 = -94.8, 28.0
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        inds = ds[testvar].xroms.argsel2d(lon0, lat0)
        outinds = xroms.argsel2d(
            ds[testvar].cf["longitude"], ds[testvar].cf["latitude"], lon0, lat0
        )
        assert np.allclose(inds, outinds)


def test_gridmean():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        for axis in ["Z", "Y", "X"]:
            var1 = ds[testvar].xroms.gridmean(grid, axis)
            var2 = xroms.gridmean(ds[testvar], grid, axis)
            assert np.allclose(var1, var2)


def test_gridsum():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        for axis in ["Z", "Y", "X"]:
            var1 = ds[testvar].xroms.gridsum(grid, axis)
            var2 = xroms.gridsum(ds[testvar], grid, axis)
            assert np.allclose(var1, var2)


def test_interpll():
    XESMF_AVAILABLE = xroms.XESMF_AVAILABLE
    xroms.XESMF_AVAILABLE = False

    with pytest.raises(ModuleNotFoundError):
        ie, ix = 2, 3
        indexer = {"eta_rho": [ie], "xi_rho": [ix]}
        testvars = ["salt", "u", "v"]
        for testvar in testvars:
            var1 = xroms.interpll(
                ds[testvar], ds.lon_rho.isel(indexer), ds.lat_rho.isel(indexer)
            )
            var2 = ds[testvar].xroms.interpll(
                ds.lon_rho.isel(indexer), ds.lat_rho.isel(indexer)
            )
            assert np.allclose(var1, var2)

    # put back the way it was for testing
    xroms.XESMF_AVAILABLE = XESMF_AVAILABLE


def test_zslice():
    testvars = ["salt", "u", "v"]
    for testvar in testvars:
        varin = ds[testvar]
        depths = np.asarray(ds[testvar].cf["vertical"][0, :, 0, 0].values)
        varout = xroms.isoslice(varin, depths, grid, axis="Z")
        varcomp = ds[testvar].xroms.zslice(grid, depths)
        # varcomp = ds[testvar].xroms.isoslice(grid, depths, axis="Z")
        assert np.allclose(
            varout.cf.isel(T=0, Y=0, X=0), varcomp.cf.isel(T=0, Y=0, X=0)
        )

        varcompds = ds.xroms.zslice(testvar, depths)
        # varcomp = ds[testvar].xroms.isoslice(grid, depths, axis="Z")
        assert np.allclose(
            varout.cf.isel(T=0, Y=0, X=0), varcompds.cf.isel(T=0, Y=0, X=0)
        )
