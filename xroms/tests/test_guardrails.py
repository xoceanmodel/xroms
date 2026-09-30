"""Pre-1.0 entry points fail loudly with a message naming the replacement."""

import subprocess
import sys

import pytest

import xroms


@pytest.mark.parametrize("func", ["roms_dataset", "open_netcdf", "open_mfnetcdf", "open_zarr", "grid_interp"])
def test_removed_functions_explain_replacement(func):
    with pytest.raises(RuntimeError, match="removed in 1.0"):
        getattr(xroms, func)("anything")


def test_roms_dataset_message_mentions_new_workflow():
    with pytest.raises(RuntimeError) as err:
        xroms.roms_dataset(None)
    msg = str(err.value)
    assert "xr.open_dataset" in msg and "ds.xroms" in msg and "assign_z" in msg


def test_dataset_accessor_removed_members(rutgers):
    with pytest.raises(AttributeError, match="xgcm_grid"):
        rutgers.xroms.xgrid
    with pytest.raises(AttributeError, match="1.0"):
        rutgers.xroms.set_grid(None)


@pytest.mark.parametrize("method", ["ddxi", "ddeta", "ddz", "zslice", "gridmean", "gridsum"])
def test_dataarray_accessor_grid_methods_point_to_dataset(rutgers, method):
    with pytest.raises(AttributeError, match=f"ds.xroms.{method}"):
        getattr(rutgers.temp.xroms, method)()


def test_xgcm_grid_argument_rejected(rutgers):
    grid = rutgers.xroms.xgcm_grid()
    with pytest.raises(TypeError, match="Dataset"):
        xroms.ddxi(rutgers.temp, grid)
    with pytest.raises(TypeError, match="xroms 1.0"):
        xroms.to_rho(rutgers.u, grid)


def test_import_has_no_side_effects():
    code = (
        "import warnings, xarray as xr\n"
        "before = dict(xr.get_options())\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import xroms\n"
        "assert dict(xr.get_options()) == before, 'xroms changed xarray options'\n"
        "own = [str(x.message) for x in w if 'xroms' in str(x.filename)]\n"
        "assert not own, own\n"
        "import sys\n"
        "assert 'xesmf' not in sys.modules and 'cartopy' not in sys.modules and 'pooch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
