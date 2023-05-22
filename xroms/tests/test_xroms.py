"""Test xroms functions."""

import os

from glob import glob

import xroms


def test_imports():
    import xroms
    import xroms.roms_seawater


# def test_open_netcdf():
#     """Test xroms.open_netcdf()."""

#     file = os.path.join(xroms.__path__[0], "tests", "input", "ocean_his_0001.nc")
#     ds = xroms.open_netcdf(file)  # , Vtransform=2)

#     assert ds


# def test_open_mfnetcdf():
#     """Test xroms.open_mfnetcdf()."""

#     base = os.path.join(xroms.__path__[0], "tests", "input")
#     files = glob("%s/ocean_his_000?.nc" % base)
#     ds = xroms.open_mfnetcdf(files, Vtransform=2)

#     assert ds


# def test_open_zarr():
#     """Test xroms.open_zarr()."""

#     base = os.path.join(xroms.__path__[0], "tests", "input")
#     files = glob("%s/ocean_his_000?" % base)
#     ds = xroms.open_zarr(files, chunks={"ocean_time": 2}, Vtransform=2)

#     assert ds
