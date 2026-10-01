"""Importing and using xroms leaves the interpreter as it was, and optional dependencies stay optional.

``test_guardrails.py`` checks the xarray options, xroms' own warnings and a few optional modules
in one subprocess. This goes further, each in a fresh interpreter:

* ``xr.get_options()`` after ``import xroms`` equals a fresh interpreter's (and nothing else global moves:
  numpy's error state and print options, ``sys.path``, the environment, logging, the warning filters
  and the warning hook), also after a run of calculations;
* ``import xroms``, and every one of its modules, emit no warning at all (``python -W error``);
* none of ``xesmf``, ``cartopy``, ``gsw``, ``pyproj``, ``pooch``, ``cf_xarray``, ``matplotlib``, ``numba``,
  ``dask`` (nor ``xgcm``, ``scipy``, ``netCDF4``) is imported by it beyond what xarray itself imports, and
  the calculations bring in only ``xgcm`` and what it needs (``numba``, ``dask``), never the plotting, mapping,
  regridding, seawater, projection or download packages;
* ``xroms.XESMF_AVAILABLE`` looks xesmf up without importing it, even when it cannot be imported;
* each optional feature says what to install when its package is missing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import xroms


ROOT = Path(xroms.__file__).resolve().parents[1]

#: asked for in the issue: must not be imported by ``import xroms`` unless xarray already did
OPTIONAL = ("xesmf", "cartopy", "gsw", "pyproj", "pooch", "cf_xarray", "matplotlib", "numba", "dask")
#: imported lazily too (xgcm only when a grid move or derivative runs)
LAZY = OPTIONAL + ("xgcm", "scipy", "netCDF4")
#: never imported by any calculation: only features that name them (teos10, geodesic, interpll, examples) need them
NEVER = ("xesmf", "cartopy", "gsw", "pyproj", "pooch", "cf_xarray", "matplotlib")


def _run(code, *flags):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
    return subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)


def _ok(code, *flags):
    result = _run(code, *flags)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


SNAPSHOT = """
import json, logging, os, sys, warnings
import numpy as np
import xarray as xr


def snapshot():
    return {
        "xarray_options": {k: repr(v) for k, v in dict(xr.get_options()).items()},
        "numpy_errors": np.geterr(),
        "numpy_print": {k: repr(v) for k, v in np.get_printoptions().items()},
        "sys_path": list(sys.path),
        "environ": dict(os.environ),
        "logging": [repr(logging.getLogger().handlers), logging.getLogger().level, logging.root.manager.disable],
        "warning_filters": [repr(f) for f in warnings.filters],
        "warning_hook": repr(warnings.showwarning),
        "excepthook": repr(sys.excepthook),
    }
"""


def test_the_xarray_options_are_a_fresh_interpreters():
    show = "print(json.dumps({k: repr(v) for k, v in dict(xr.get_options()).items()}, sort_keys=True))"
    fresh = json.loads(_ok(f"import json, xarray as xr\n{show}"))
    with_xroms = json.loads(_ok(f"import json, xarray as xr\nimport xroms\n{show}"))
    assert with_xroms == fresh and fresh  # and there are options to compare


def test_importing_xroms_moves_nothing_global():
    _ok(SNAPSHOT + "before = snapshot()\nimport xroms\nafter = snapshot()\nassert after == before, [k for k in before if before[k] != after[k]]\n")


def test_calculations_move_nothing_global():
    code = SNAPSHOT + """
import xroms
from xroms.tests import _synthetic as syn

ds = syn.make_dataset("rutgers")


def run():
    xroms.z(ds); xroms.ddxi(ds.temp, ds); xroms.ddz(ds.temp, ds); xroms.relative_vorticity(ds.u, ds.v, ds)
    xroms.density(ds.temp, ds.salt, grid=ds); xroms.mld(xroms.potential_density(ds.temp, ds.salt), ds)
    xroms.zslice(ds.temp, [-5.0], ds); xroms.depth_average(ds.temp, ds)
    ds.xroms.N2; ds.xroms.speed; ds.xroms.eastnorth; ds.xroms.xgcm_grid(); ds.chunk({"ocean_time": 1}).xroms.ertel.compute()


before = snapshot()
run()
first = snapshot()
# the first grid move imports xgcm, and with it numba and scipy, which add warning filters of their own
moved = [k for k in before if before[k] != first[k] and k != "warning_filters"]
assert not moved, moved
added = [f for f in first["warning_filters"] if f not in before["warning_filters"]]
assert not [f for f in added if "xroms" in f], added
# after that, nothing at all moves
run()
assert snapshot() == first, [k for k in first if snapshot()[k] != first[k]]
"""
    _ok(code)


def test_importing_xroms_and_all_its_modules_emits_no_warning():
    if _run("import xarray", "-W", "error").returncode != 0:
        pytest.skip("xarray itself warns when imported here")
    code = (
        "import importlib, pkgutil\n"
        "import xroms\n"
        "for m in pkgutil.iter_modules(xroms.__path__):\n"
        "    if m.name != 'tests':\n"
        "        importlib.import_module('xroms.' + m.name)\n"
        "from xroms import XESMF_AVAILABLE\n"
        "assert xroms.__version__\n"
    )
    result = _run(code, "-W", "error")
    assert result.returncode == 0, result.stderr


def test_import_adds_no_optional_package_beyond_xarrays():
    code = (
        "import sys\n"
        "import xarray\n"
        "bare = set(sys.modules)\n"
        "import xroms\n"
        f"added = [name for name in {LAZY!r} if name in sys.modules and name not in bare]\n"
        "assert not added, added\n"
        # nor any package that is not in the standard library: xroms is the only new top-level module
        "new = {m.split('.')[0] for m in set(sys.modules) - bare} - set(sys.stdlib_module_names)\n"
        "assert new <= {'xroms'}, sorted(new)\n"
    )
    _ok(code)


def test_every_module_imports_without_optional_packages():
    code = (
        "import importlib, pkgutil, sys\n"
        "import xarray\n"
        "bare = set(sys.modules)\n"
        "import xroms\n"
        "for m in pkgutil.iter_modules(xroms.__path__):\n"
        "    if m.name != 'tests':\n"
        "        importlib.import_module('xroms.' + m.name)\n"
        f"added = [name for name in {LAZY!r} if name in sys.modules and name not in bare]\n"
        "assert not added, added\n"
    )
    _ok(code)


def test_calculations_import_xgcm_and_what_it_needs_and_nothing_else():
    code = (
        "import sys\n"
        "import xarray\n"
        "import xroms\n"
        "from xroms.tests import _synthetic as syn\n"
        "ds = syn.make_dataset('rutgers')\n"
        "xroms.z(ds); xroms.ddxi(ds.temp, ds); xroms.density(ds.temp, ds.salt, grid=ds)\n"
        "xroms.zslice(ds.temp, [-5.0], ds); xroms.mld(xroms.potential_density(ds.temp, ds.salt), ds)\n"
        "ds.xroms.N2; ds.xroms.xgcm_grid(); ds.temp.xroms.to_grid('u', 's_w')\n"
        f"loaded = [name for name in {NEVER!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
        "assert 'xgcm' in sys.modules\n"
    )
    _ok(code)


# --- XESMF_AVAILABLE ------------------------------------------------------------------------------------


def test_xesmf_available_when_xesmf_cannot_be_imported(tmp_path):
    # present on the path, but importing it fails: it is "available", and nothing tried the import
    (tmp_path / "xesmf.py").write_text("raise RuntimeError('xesmf was imported')\n")
    code = (
        "import sys\n"
        "import xroms\n"
        "assert xroms.XESMF_AVAILABLE is True\n"
        "assert 'xesmf' not in sys.modules\n"
        "from xroms import XESMF_AVAILABLE\n"
        "assert XESMF_AVAILABLE is True and 'xesmf' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.returncode == 0, result.stderr


def test_xesmf_available_is_false_when_its_import_is_blocked():
    code = (
        "import sys\n"
        "sys.modules['xesmf'] = None  # an import of it now fails, as when it is not installed\n"
        "import xroms\n"
        "assert xroms.XESMF_AVAILABLE is False\n"
        "from xroms import XESMF_AVAILABLE\n"
        "assert XESMF_AVAILABLE is False\n"
    )
    _ok(code)


def test_xesmf_available_is_looked_up_on_every_access(monkeypatch):
    import importlib.util

    seen = []
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: seen.append(name) or real(name, *a))
    first, second = xroms.XESMF_AVAILABLE, xroms.XESMF_AVAILABLE
    assert seen == ["xesmf", "xesmf"] and first is second


# --- optional packages that are missing say what to install ------------------------------------------------------------


def _argsel_geodesic():
    xroms.argsel2d([[0.0, 1.0]], [[0.0, 1.0]], 0.0, 0.0, method="geodesic")


def _teos10():
    from xroms.tests import _synthetic as syn

    ds = syn.make_dataset("rutgers")
    xroms.density(ds.temp, ds.salt, grid=ds, eos="teos10")


def _interpll():
    from xroms.tests import _synthetic as syn

    ds = syn.make_dataset("rutgers")
    xroms.interpll(ds.temp, [-90.0], [28.0])


def _example_data():
    from xroms import datasets

    datasets.fetch_ROMS_example_full_grid()


@pytest.mark.parametrize(
    "module,call,error,message",
    [
        ("pyproj", _argsel_geodesic, ModuleNotFoundError, r"pyproj.*xroms\[geodesic\]"),
        ("gsw", _teos10, ImportError, r"teos10.*gsw.*xroms\[teos10\]"),
        ("xesmf", _interpll, ModuleNotFoundError, r"xESMF.*conda install"),
        ("pooch", _example_data, ModuleNotFoundError, r"pooch.*xroms\[examples\]"),
    ],
    ids=["geodesic", "teos10", "interpll", "example-data"],
)
def test_a_missing_optional_package_says_what_to_install(monkeypatch, module, call, error, message):
    monkeypatch.setitem(sys.modules, module, None)  # as if it were not installed
    with pytest.raises(error, match=message):
        call()
