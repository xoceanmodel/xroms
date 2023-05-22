"""Test interpolation functions with known coordinates to make sure results are correct"""

import numpy as np
import pytest
import xarray as xr

import xroms


grid1 = xr.open_dataset("xroms/tests/input/grid.nc")
# ds = xroms.open_netcdf('xroms/tests/input/ocean_his_0001.nc')
ds = xr.open_dataset("xroms/tests/input/ocean_his_0001.nc")
# combine the two:
ds = ds.merge(grid1, overwrite_vars=True, compat="override")
ds, grid = xroms.roms_dataset(ds, include_3D_metrics=True)


def test_interpll():
    XESMF_AVAILABLE = xroms.XESMF_AVAILABLE
    xroms.XESMF_AVAILABLE = False

    with pytest.raises(ModuleNotFoundError):
        # test pairs of points
        ie, ix = 2, 3
        testvars = ["salt", "u", "v", "z_w"]
        for testvar in testvars:
            varin = ds[testvar]
            lon = ds[testvar].cf["longitude"]
            lat = ds[testvar].cf["latitude"]
            indexer = {varin.cf["Y"].name: [ie], varin.cf["X"].name: [ix]}
            varout = xroms.interpll(varin, lon.isel(indexer), lat.isel(indexer))
            assert np.allclose(varout.squeeze(), varin.isel(indexer).squeeze())

        # test grid of points
        ie, ix = [2], [3]
        testvars = ["salt", "u", "v", "z_w"]
        for testvar in testvars:
            varin = ds[testvar]
            lon = ds[testvar].cf["longitude"]
            lat = ds[testvar].cf["latitude"]
            indexer = {varin.cf["Y"].name: ie, varin.cf["X"].name: ix}
            varout = xroms.interpll(
                varin, lon.cf.isel(indexer), lat.cf.isel(indexer), which="grid"
            )
            assert np.allclose(varout.squeeze(), varin.isel(indexer).squeeze())

    # put back the way it was for testing
    xroms.XESMF_AVAILABLE = XESMF_AVAILABLE


def test_zslice():
    testvars = ["salt", "u", "v", "z_w"]
    for testvar in testvars:
        varin = ds[testvar]
        depths = np.asarray(ds[testvar].cf["vertical"][0, :, 0, 0].values)
        varout = xroms.isoslice(varin, depths, grid, axis="Z")
        assert np.allclose(varout.cf.isel(T=0, Y=0, X=0), varin.cf.isel(T=0, Y=0, X=0))
