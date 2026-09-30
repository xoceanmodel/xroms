"""Pre-1.0 entry points fail loudly with a message naming the replacement."""

import importlib.util
import os
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


def test_roms_dataset_message_maps_the_include_flags():
    with pytest.raises(RuntimeError) as err:
        xroms.roms_dataset(None, include_Z0=True, include_cell_area=True)
    msg = str(err.value)
    assert "include_Z0=True -> zeta=0" in msg and "ds.xroms.z(zeta=0)" in msg
    assert "include_cell_area -> ds.xroms.dA()" in msg and "include_cell_volume -> ds.xroms.dV()" in msg
    assert "include_3D_metrics" in msg


def test_open_mfnetcdf_message_quotes_the_old_defaults():
    with pytest.raises(RuntimeError) as err:
        xroms.open_mfnetcdf(["a.nc", "b.nc"])
    assert 'xr.open_mfdataset(files, data_vars="minimal", coords="minimal", compat="override")' in str(err.value)


def test_dataset_accessor_removed_members(rutgers):
    with pytest.raises(AttributeError, match="xgcm_grid"):
        rutgers.xroms.xgrid
    with pytest.raises(AttributeError, match="1.0"):
        rutgers.xroms.set_grid(None)


@pytest.mark.parametrize("name", ["w", "omega"])
def test_dataset_accessor_placeholders_removed(rutgers, name):
    with pytest.raises(AttributeError, match=rf"removed ds\.xroms\.{name}\b") as err:
        getattr(rutgers.xroms, name)
    assert f"ds['{name}']" in str(err.value)
    assert not hasattr(rutgers.xroms, name)


@pytest.mark.parametrize("method", ["east_rotated", "north_rotated"])
def test_include_vars_adcp_rejected_with_a_hint(rutgers, method):
    with pytest.raises(TypeError, match="include_vars_adcp") as err:
        getattr(rutgers.xroms, method)(0.3, include_vars_adcp=True)
    assert "removed" in str(err.value) and "east_rotated" in str(err.value)
    with pytest.raises(TypeError, match="unexpected keyword argument 'nope'") as err:
        getattr(rutgers.xroms, method)(0.3, nope=1)
    assert "removed" not in str(err.value)


@pytest.mark.parametrize("method", ["ddxi", "ddeta", "ddz"])
def test_positional_hcoord_scoord_get_a_hint(rutgers, method):
    acc = getattr(rutgers.xroms, method)
    with pytest.raises(TypeError, match="keyword-only") as err:
        acc("temp", "rho", "s_rho")
    assert "hcoord=" in str(err.value) and "scoord=" in str(err.value)
    assert acc("temp", hcoord="rho", scoord="s_rho").dims == rutgers.temp.dims


def test_dataarray_to_grid_rejects_an_xgcm_grid(rutgers):
    grid = rutgers.xroms.xgcm_grid()
    with pytest.raises(TypeError, match="xgcm Grid") as err:
        rutgers.temp.xroms.to_grid(grid, "u")
    assert "da.xroms.to_grid('u')" in str(err.value)
    with pytest.raises(TypeError, match="xgcm Grid"):
        rutgers.temp.xroms.to_grid(xgrid=grid, hcoord="u")
    assert rutgers.temp.xroms.to_grid("u").dims[-1] == "xi_u"


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


def test_namespace_has_no_stray_names():
    assert not hasattr(xroms, "version") and not hasattr(xroms, "PackageNotFoundError")
    assert isinstance(xroms.__version__, str)
    assert not hasattr(xroms, "nope")
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        xroms.__getattr__("nope")


def test_xesmf_available_reports_whether_xesmf_is_installed(monkeypatch):
    assert xroms.XESMF_AVAILABLE is (importlib.util.find_spec("xesmf") is not None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert xroms.XESMF_AVAILABLE is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    from xroms import XESMF_AVAILABLE  # the v0.6.2 spelling

    assert XESMF_AVAILABLE is True


def test_xesmf_available_does_not_import_xesmf(tmp_path):
    (tmp_path / "xesmf.py").write_text("raise RuntimeError('xesmf was imported')\n")
    code = (
        "import sys, xroms\n"
        "assert xroms.XESMF_AVAILABLE is True\n"
        "assert 'xesmf' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *filter(None, [os.environ.get("PYTHONPATH")])])}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
