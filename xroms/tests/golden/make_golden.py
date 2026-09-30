"""Generate golden outputs from the released xroms v0.6.2 (git commit be8ea34).

The rewritten xroms package is regression-tested against the numbers recorded
here. To keep the record independent of the package being rewritten, the v0.6.2
sources are exported from git into a temporary directory and imported from
there; xroms is never imported from the working tree or from site-packages.

Run it with a Python environment that has xgcm 0.8.1 (the script asserts this):

    PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/ocean-skill/bin/python xroms/tests/golden/make_golden.py

It writes golden_tiny.nc, golden_window.nc and window_input.nc next to this file.
See README.md in this directory for what they contain.
"""

import filecmp
import importlib
import subprocess
import sys
import tempfile
import warnings

from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple


warnings.filterwarnings("ignore")

import cf_xarray  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import xgcm  # noqa: E402


sys.dont_write_bytecode = True

COMMIT = "be8ea34"
EXPECTED_XGCM = "0.8.1"
CREATED_BY = "xroms/tests/golden/make_golden.py"

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]  # this file lives in <worktree>/xroms/tests/golden/

# Inputs, relative to the repository root.
TINY_FILES = ("xroms/tests/input/ocean_his_0001.nc", "xroms/tests/input/grid.nc")
WINDOW_FILE = "xroms/data/ROMS_example_full_grid.nc"
WINDOW_ISEL = {
    "eta_rho": slice(60, 72),
    "eta_u": slice(60, 72),
    "eta_v": slice(60, 71),
    "xi_rho": slice(100, 116),
    "xi_u": slice(100, 115),
    "xi_v": slice(100, 116),
}
WINDOW_FULL_SIZES = {
    "eta_rho": 191,
    "xi_rho": 300,
    "eta_u": 191,
    "xi_u": 299,
    "eta_v": 190,
    "xi_v": 300,
    "s_rho": 30,
    "s_w": 31,
    "ocean_time": 2,
}

# Depths [m] for the fixed-depth slices and salinities for "temp on salt surfaces".
# The tiny grid is 100 m deep everywhere. The window is only 11-27 m deep, so the
# -50/-20/-5 m that would suit a deep grid lie mostly below its seabed; use depths
# that are inside the water column everywhere in the window instead.
# The window's salinity profiles are often not monotonic, so a salinity value can be
# crossed several times in one column (xgcm then returns the topmost crossing, which
# another implementation might not). 23 and 30 are crossed at most once in every
# column of the window, so the slices onto them are unambiguous.
TINY = {"depths": [-10.0, -2.0, -0.5], "salts": [18.0, 22.0]}
WINDOW = {"depths": [-10.0, -5.0, -1.0], "salts": [23.0, 30.0]}

ROMS_DATASET_CALL = (
    "xroms.roms_dataset(ds, include_Z0=True, include_cell_volume=True, "
    "include_cell_area=True)"
)

# Coordinates and variables that roms_dataset adds, recorded as they come out of it.
GRID_NAMES = [
    "z_rho",
    "z_w",
    "z_rho_u",
    "z_rho_v",
    "z_rho_psi",
    "z_w_u",
    "z_w_v",
    "z_w_psi",
    "z_rho0",
    "z_w0",
    "dx",
    "dy",
    "dx_u",
    "dy_u",
    "dx_v",
    "dy_v",
    "dx_psi",
    "dy_psi",
    "dA",
    "dA_u",
    "dA_v",
    "dA_psi",
    "dz",
    "dz_u",
    "dz_v",
    "dz_w",
    "dV",
]

# Attributes that netCDF encoding handles itself and that would conflict if set here.
DROP_ATTRS = {"_FillValue", "coordinates"}


class Step(NamedTuple):
    """One recorded computation.

    keys: names of the outputs the statement assigns.
    stmt: statement executed with `ds`, `xgrid`, `np` and `xroms` (plus the results
        of earlier steps) in scope; it is stored verbatim as `xroms_call`.
    fallback: only tried if `stmt` raises; same outputs, adapted so that it runs.
    space: which namespace to run in ("main", or "acc" for the accessor, which gets
        its own freshly processed dataset).
    """

    keys: Tuple[str, ...]
    stmt: str
    fallback: Optional[str] = None
    space: str = "main"


def build_steps(depths: List[float], salts: List[float]) -> List[Step]:
    """List every output to record, in the order they are stored."""
    note = f"  # ds, xgrid = {ROMS_DATASET_CALL}"
    steps = [Step((n,), f'{n} = ds["{n}"]{note}') for n in GRID_NAMES]

    # xroms.rotate_vectors without `attrs` needs name/long_name/units attributes on
    # its inputs, which the output of xroms.to_rho does not have (KeyError: 'name').
    # The accessor passes `attrs`, so do the same; the values are not affected.
    rot = (
        "xroms.rotate_vectors(to_rho_u, to_rho_v, ds.angle, isradians=True, "
        'reference="xaxis"'
    )
    rot_attrs = ', attrs={"x": {"name": "rot_x"}, "y": {"name": "rot_y"}}'

    # xroms.isoslice onto salinity fails when var carries a z_rho coordinate: the
    # interpolated z_rho is re-attached next to the new `salt` axis, cf_xarray then
    # reports z_rho (not salt) as the Z axis, and xroms.order leaves `salt` out of
    # its transpose. Dropping the z_* coordinates of var avoids that and does not
    # change the interpolated values.
    temp_no_z = 'ds.temp.drop_vars([c for c in ds.temp.coords if c.startswith("z_")])'
    on_salt = f'np.array({salts!r}), xgrid, iso_array=ds.salt, axis="Z")'

    steps += [
        # grid moves
        Step(("to_rho_u",), "to_rho_u = xroms.to_rho(ds.u, xgrid)"),
        Step(("to_rho_v",), "to_rho_v = xroms.to_rho(ds.v, xgrid)"),
        Step(("to_u_temp",), "to_u_temp = xroms.to_u(ds.temp, xgrid)"),
        Step(("to_v_temp",), "to_v_temp = xroms.to_v(ds.temp, xgrid)"),
        Step(("to_psi_temp",), "to_psi_temp = xroms.to_psi(ds.temp, xgrid)"),
        Step(("to_s_w_temp",), "to_s_w_temp = xroms.to_s_w(ds.temp, xgrid)"),
        Step(("to_s_rho_zw",), "to_s_rho_zw = xroms.to_s_rho(ds.z_w, xgrid)"),
        # derivatives
        Step(("ddxi_temp",), "ddxi_temp = xroms.ddxi(ds.temp, xgrid)"),
        Step(
            ("ddxi_temp_rho",),
            'ddxi_temp_rho = xroms.ddxi(ds.temp, xgrid, hcoord="rho", scoord="s_rho")',
        ),
        Step(("ddeta_temp",), "ddeta_temp = xroms.ddeta(ds.temp, xgrid)"),
        Step(
            ("ddeta_temp_rho",),
            "ddeta_temp_rho = "
            'xroms.ddeta(ds.temp, xgrid, hcoord="rho", scoord="s_rho")',
        ),
        Step(("ddz_temp",), "ddz_temp = xroms.ddz(ds.temp, xgrid)"),
        Step(
            ("ddz_temp_rho",),
            'ddz_temp_rho = xroms.ddz(ds.temp, xgrid, scoord="s_rho")',
        ),
        Step(("ddxi_zeta",), "ddxi_zeta = xroms.ddxi(ds.zeta, xgrid)"),
        Step(("ddeta_zeta",), "ddeta_zeta = xroms.ddeta(ds.zeta, xgrid)"),
        Step(("ddxi_u",), "ddxi_u = xroms.ddxi(ds.u, xgrid)"),
        Step(("ddeta_v",), "ddeta_v = xroms.ddeta(ds.v, xgrid)"),
        # derived quantities
        Step(("speed",), "speed = xroms.speed(ds.u, ds.v, xgrid)"),
        Step(("KE",), "KE = xroms.KE(ds.rho0, speed)"),
        Step(("ug", "vg"), "ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)"),
        Step(("EKE",), "EKE = xroms.EKE(ug, vg, xgrid)"),
        Step(("dudz",), "dudz = xroms.dudz(ds.u, xgrid)"),
        Step(("dvdz",), "dvdz = xroms.dvdz(ds.v, xgrid)"),
        Step(
            ("vertical_shear",),
            "vertical_shear = xroms.vertical_shear(dudz, dvdz, xgrid)",
        ),
        Step(("vort",), "vort = xroms.relative_vorticity(ds.u, ds.v, xgrid)"),
        Step(("convergence",), "convergence = xroms.convergence(ds.u, ds.v, xgrid)"),
        Step(("sig0",), "sig0 = xroms.potential_density(ds.temp, ds.salt, 0)"),
        Step(("buoy",), "buoy = xroms.buoyancy(sig0)"),
        Step(
            ("ertel",),
            'ertel = xroms.ertel(buoy, ds.u, ds.v, ds.f, xgrid, scoord="s_w")',
        ),
        Step(("rho",), "rho = xroms.density(ds.temp, ds.salt, ds.z_rho)"),
        Step(("N2",), "N2 = xroms.N2(rho, xgrid)"),
        Step(("M2",), "M2 = xroms.M2(rho, xgrid)"),
        Step(("mld",), "mld = xroms.mld(sig0, xgrid, ds.h, ds.mask_rho)"),
        # accessor rotations, on a freshly processed dataset
        Step(
            ("east",), "ds.xroms.set_grid(xgrid); east = ds.xroms.east", space="acc"
        ),
        Step(
            ("north",), "ds.xroms.set_grid(xgrid); north = ds.xroms.north", space="acc"
        ),
        # rotation function, on the already computed rho-grid u and v
        Step(
            ("rot_x", "rot_y"),
            f"rot_x, rot_y = {rot})",
            f"rot_x, rot_y = {rot}{rot_attrs})",
        ),
        # interpolation
        Step(
            ("zslice_temp",),
            f"zslice_temp = xroms.isoslice(ds.temp, np.array({depths!r}), xgrid)",
        ),
        Step(
            ("isoslice_temp_on_salt",),
            f"isoslice_temp_on_salt = xroms.isoslice(ds.temp, {on_salt}",
            f"isoslice_temp_on_salt = xroms.isoslice({temp_no_z}, {on_salt}",
        ),
        # aggregates
        Step(
            ("gridmean_temp_YX",),
            'gridmean_temp_YX = xroms.gridmean(ds.temp, xgrid, ("Y", "X"))',
        ),
        Step(
            ("gridsum_temp_Z",), 'gridsum_temp_Z = xroms.gridsum(ds.temp, xgrid, "Z")'
        ),
    ]
    return steps


def fmt_error(exc: BaseException) -> str:
    """One-line, length-limited description of an exception."""
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())
    return msg if len(msg) <= 400 else msg[:397] + "..."


class Recorder:
    """Collects results, the call that made each, and what failed."""

    def __init__(self) -> None:
        self.arrays: Dict[str, xr.DataArray] = {}
        self.calls: Dict[str, str] = {}
        # keys whose first-choice statement failed but a fallback worked
        self.primary_failed: Dict[str, Tuple[str, str]] = {}
        # keys with no output at all
        self.failed: Dict[str, str] = {}

    def run(self, step: Step, spaces: Dict[str, Dict[str, Any]]) -> None:
        """Execute a step, trying its fallback if the first statement raises."""
        ns = spaces[step.space]
        used, first_error = step.stmt, None
        try:
            exec(step.stmt, ns)
        except Exception as exc:
            first_error = fmt_error(exc)
            if step.fallback is None:
                self.failed.update({k: first_error for k in step.keys})
                return
            try:
                exec(step.fallback, ns)
                used = step.fallback
            except Exception as exc2:
                msg = f"{first_error} | fallback also failed: {fmt_error(exc2)}"
                self.failed.update({k: msg for k in step.keys})
                return

        for key in step.keys:
            self.arrays[key] = ns[key]
            self.calls[key] = used
            if first_error is not None:
                self.primary_failed[key] = (step.stmt, first_error)


def netcdf_attr(value: Any) -> Any:
    """Coerce an attribute value to something netCDF can store."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in "iuf":
        return value
    if isinstance(value, (list, tuple)) and len(value) > 0:
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return np.asarray(value)
    return str(value)


def clean_output(key: str, da: Any) -> xr.DataArray:
    """Reduce an xroms result to its values and dims plus its own attributes.

    All non-dimension coordinates are dropped; dimension coordinates are kept and
    the dims keep their order.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"expected a DataArray, got {type(da).__name__}")
    da = da.reset_coords(drop=True)
    da = da.drop_vars([c for c in da.coords if c not in da.dims])
    da = da.copy(deep=True).compute()  # deep copy: never touch the dataset's own
    da.name = key
    da.encoding = {}
    da.attrs = {k: netcdf_attr(v) for k, v in da.attrs.items() if k not in DROP_ATTRS}
    return da


def assemble(
    arrays: Dict[str, xr.DataArray]
) -> Tuple[xr.Dataset, Dict[str, Dict[str, str]]]:
    """Put cleaned outputs into one Dataset without any implicit alignment.

    A dimension is renamed for storage when it cannot be shared as is:

    * it has the name of a stored variable (xroms.isoslice onto depths makes a
      dimension called `z_rho`, which is also the name of the z_rho output):
      it becomes `<dim>_dim`;
    * another output already uses that dimension name with a different size or
      different index values: it becomes `<dim>_<key>`.

    Returns the Dataset and {key: {original dim: stored dim}} for renamed dims.
    """
    sizes: Dict[str, int] = {}
    index: Dict[str, np.ndarray] = {}
    renamed: Dict[str, Dict[str, str]] = {}
    stored = []

    for key, da in arrays.items():
        new_names: Dict[str, str] = {}
        for dim in da.dims:
            size = da.sizes[dim]
            values = da[dim].values if dim in da.coords else None
            name = f"{dim}_dim" if dim in arrays else dim
            clash = name in sizes and sizes[name] != size
            if name in index and values is not None:
                clash = clash or not np.array_equal(index[name], values)
            if clash:
                name = f"{name}_{key}"
            if name != dim:
                new_names[dim] = name
            sizes.setdefault(name, size)
            if values is not None:
                index.setdefault(name, values)
        if new_names:
            da = da.rename(new_names)
            renamed[key] = new_names
        stored.append(da)

    merged = xr.merge(
        stored, join="exact", compat="no_conflicts", combine_attrs="override"
    )
    return merged, renamed


def write_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write with netCDF4, zlib-compressed, dtypes untouched."""
    for var in ds.variables.values():
        var.encoding = {}
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, mode="w", format="NETCDF4", engine="netcdf4", encoding=encoding)


def export_v062(dest: Path) -> str:
    """Export the xroms package at COMMIT into dest, like `git archive | tar -x`.

    Returns the full commit hash.
    """
    git = subprocess.Popen(
        ["git", "-C", str(WORKTREE), "archive", COMMIT, "xroms"],
        stdout=subprocess.PIPE,
    )
    assert git.stdout is not None
    try:
        subprocess.run(["tar", "-x", "-C", str(dest)], stdin=git.stdout, check=True)
    finally:
        git.stdout.close()
        returncode = git.wait()
    if returncode != 0:
        raise RuntimeError(f"git archive {COMMIT} failed (exit code {returncode})")
    full = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", f"{COMMIT}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return full.stdout.strip()


def import_v062(src: Path) -> Any:
    """Import xroms from src (and only from there)."""
    assert "xroms" not in sys.modules, "xroms was imported before the v0.6.2 export"
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    xroms = importlib.import_module("xroms")
    origin = Path(xroms.__file__).resolve()
    assert src.resolve() in origin.parents, f"xroms came from {origin}, not from {src}"
    return xroms


def input_file(rel: str, src: Path) -> Path:
    """Path of an input file: the working-tree copy if it is identical to COMMIT's.

    The goldens must describe COMMIT's code run on COMMIT's inputs, so if the file
    has since been changed or removed in the working tree, use COMMIT's copy.
    """
    worktree_copy, commit_copy = WORKTREE / rel, src / rel
    same = worktree_copy.is_file() and filecmp.cmp(
        worktree_copy, commit_copy, shallow=False
    )
    if same:
        return worktree_copy
    print(f"NOTE: {rel} differs from {COMMIT} in the working tree; using {COMMIT}'s")
    return commit_copy


def load_tiny(src: Path) -> xr.Dataset:
    """The tiny test input, built as the existing tests build it."""
    his = xr.open_dataset(input_file(TINY_FILES[0], src))
    grid = xr.open_dataset(input_file(TINY_FILES[1], src))
    ds = his.merge(grid, overwrite_vars=True, compat="override").load()
    his.close()
    grid.close()
    return ds


def load_window_raw(src: Path) -> xr.Dataset:
    """A small window of the Rutgers-style example grid, before any xroms use."""
    full = xr.open_dataset(input_file(WINDOW_FILE, src))
    sizes = {k: full.sizes[k] for k in WINDOW_FULL_SIZES}
    assert sizes == WINDOW_FULL_SIZES, f"unexpected example grid sizes: {sizes}"
    assert int(full.Vtransform) == 1, "expected an example grid with Vtransform 1"
    window = full.isel(**WINDOW_ISEL).load()
    full.close()
    return window


def land_cells(ds: xr.Dataset) -> int:
    """Number of land points (mask_rho == 0)."""
    return int((ds["mask_rho"] == 0).sum())


def max_crossings(salt: xr.DataArray, value: float) -> int:
    """Most times any water column crosses `value` between adjacent s levels."""
    s = np.moveaxis(salt.values.astype(np.float64), salt.get_axis_num("s_rho"), 0)
    lower, upper = s[:-1], s[1:]
    crossed = ((lower - value) * (upper - value) <= 0) & (lower != upper)
    return int(crossed.sum(axis=0).max())


def verify_golden(
    path: Path,
    cleaned: Dict[str, xr.DataArray],
    renamed: Dict[str, Dict[str, str]],
    failed: Dict[str, str],
) -> None:
    """Re-open the written file and check it holds exactly what was recorded."""
    with xr.open_dataset(path) as opened:
        back = opened.load()
    assert sorted(back.data_vars) == sorted(cleaned), f"{path.name}: variables differ"
    for key, da in cleaned.items():
        got = back[key]
        stored_dims = tuple(renamed.get(key, {}).get(d, d) for d in da.dims)
        assert got.dims == stored_dims, f"{key}: dims {got.dims} != {stored_dims}"
        assert got.dtype == da.dtype, f"{key}: dtype {got.dtype} != {da.dtype}"
        np.testing.assert_array_equal(got.values, da.values, err_msg=key)
        for attr in ("xroms_call", "v0.6.2_dims"):
            assert got.attrs[attr] == da.attrs[attr], f"{key}: attr {attr} differs"
    for key, msg in failed.items():
        assert back.attrs[f"failed_{key}"] == msg, f"{key}: failure note missing"
    assert back.attrs["source_commit"] == COMMIT


def generate(
    make_raw: Callable[[], xr.Dataset],
    params: Dict[str, List[float]],
    xroms: Any,
    path: Path,
    global_attrs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run every step on one input, write `path`, and check it reads back."""
    process = dict(include_Z0=True, include_cell_volume=True, include_cell_area=True)
    ds, xgrid = xroms.roms_dataset(make_raw(), **process)
    ds_acc, xgrid_acc = xroms.roms_dataset(make_raw(), **process)

    # the slice depths/salinities must be inside the water column / salinity range
    h_min = float(ds.h.min())
    assert all(-h_min < d < 0 for d in params["depths"]), "slice depth outside water"
    s_min, s_max = float(ds.salt.min()), float(ds.salt.max())
    assert all(s_min < s < s_max for s in params["salts"]), "salinity outside range"
    assert all(
        max_crossings(ds.salt, s) <= 1 for s in params["salts"]
    ), "a salinity value is crossed more than once in some column: ambiguous slice"

    base = {"np": np, "xroms": xroms}
    spaces = {
        "main": {**base, "ds": ds, "xgrid": xgrid},
        "acc": {**base, "ds": ds_acc, "xgrid": xgrid_acc},
    }
    steps = build_steps(params["depths"], params["salts"])
    expected = [k for step in steps for k in step.keys]
    assert len(expected) == len(set(expected)), "duplicate output keys"
    rec = Recorder()
    for step in steps:
        rec.run(step, spaces)

    cleaned: Dict[str, xr.DataArray] = {}
    for key in expected:  # stored in the order listed in build_steps
        if key not in rec.arrays:
            continue
        try:
            da = clean_output(key, rec.arrays[key])
        except Exception as exc:
            rec.failed[key] = f"could not be recorded: {fmt_error(exc)}"
            continue
        da.attrs["xroms_call"] = rec.calls[key]
        da.attrs["v0.6.2_dims"] = str(rec.arrays[key].dims)
        if key in rec.primary_failed:
            da.attrs["primary_call"] = rec.primary_failed[key][0]
            da.attrs["primary_call_error"] = rec.primary_failed[key][1]
        cleaned[key] = da
    assert sorted(cleaned) == sorted(set(expected) - set(rec.failed))

    golden, renamed = assemble(cleaned)
    for key, dims in renamed.items():
        text = ", ".join(f"{old}->{new}" for old, new in dims.items())
        golden[key].attrs["dims_renamed_for_storage"] = text
    golden.attrs = {
        **global_attrs,
        "roms_dataset_call": f"ds, xgrid = {ROMS_DATASET_CALL}",
        "mask_rho_land_cells": land_cells(ds),
        **{f"failed_{k}": v for k, v in rec.failed.items()},
    }
    write_netcdf(golden, path)
    verify_golden(path, cleaned, renamed, rec.failed)
    return {"cleaned": cleaned, "renamed": renamed, "rec": rec}


def nan_fraction(da: xr.DataArray) -> float:
    """Fraction of NaN values in da."""
    return float(np.isnan(da.values).mean()) if da.dtype.kind == "f" else 0.0


def print_report(label: str, result: Dict[str, Any], path: Path) -> None:
    """Print what was recorded for one input."""
    cleaned, renamed, rec = result["cleaned"], result["renamed"], result["rec"]
    print(f"\n=== {label} -> {path.name}")
    print(f"{len(cleaned)} outputs recorded, {len(rec.failed)} failed")
    for key, da in cleaned.items():
        dims = tuple(renamed.get(key, {}).get(d, d) for d in da.dims)
        nans = int(np.isnan(da.values).sum()) if da.dtype.kind == "f" else 0
        flag = "  [fallback call]" if key in rec.primary_failed else ""
        head = f"  {key:22s} {str(da.dtype):8s} nan {nans:5d}/{da.size:<6d}"
        print(f"{head} {dims}{flag}")
    for key, (_, err) in rec.primary_failed.items():
        print(f"  first call failed for {key}: {err}")
    for key, err in rec.failed.items():
        print(f"  FAILED {key}: {err}")
    for key in ("zslice_temp", "isoslice_temp_on_salt"):
        if key not in cleaned:
            continue
        da = cleaned[key]
        axis = [d for d in da.dims if d in ("z_rho", "salt")][0]
        values = da[axis].values
        fracs = [nan_fraction(da.isel({axis: i})) for i in range(len(values))]
        pairs = ", ".join(f"{v:g}: {f:.1%} NaN" for v, f in zip(values, fracs))
        print(f"  {key} per {axis} value -> {pairs}")


def save_window_input(src: Path, path: Path) -> Tuple[xr.Dataset, str]:
    """Write the raw window to `path`; return it as read back, and its isel text.

    The goldens are computed from the read-back copy, so they correspond exactly to
    what anyone opening window_input.nc gets.
    """
    raw = load_window_raw(src)
    isel_text = ", ".join(f"{k}={v.start}:{v.stop}" for k, v in WINDOW_ISEL.items())
    raw.attrs.update(
        {
            "window_source": WINDOW_FILE,
            "window_isel": isel_text,
            "source_commit": COMMIT,
            "created_by": CREATED_BY,
        }
    )
    write_netcdf(raw.copy(deep=True), path)
    with xr.open_dataset(path) as opened:
        saved = opened.load()
    xr.testing.assert_identical(saved, raw)
    for name, var in saved.variables.items():
        assert var.dtype == raw[name].dtype, f"{path.name}: {name} dtype changed"
    return saved, isel_text


def record_all(src: Path, xroms: Any, commit_full: str) -> None:
    """Record both inputs and print a report."""
    provenance = {
        "source_commit": COMMIT,
        "source_commit_full": commit_full,
        "xgcm_version": xgcm.__version__,
        "xarray_version": xr.__version__,
        "numpy_version": np.__version__,
        "cf_xarray_version": cf_xarray.__version__,
        "created_by": CREATED_BY,
    }

    tiny_path = HERE / "golden_tiny.nc"
    tiny_attrs = {
        **provenance,
        "input": (
            f"{TINY_FILES[0]} merged with {TINY_FILES[1]} "
            "(overwrite_vars=True, compat='override'), as in the existing tests"
        ),
    }
    tiny = generate(lambda: load_tiny(src), TINY, xroms, tiny_path, tiny_attrs)

    raw_path = HERE / "window_input.nc"
    window_saved, isel_text = save_window_input(src, raw_path)
    window_path = HERE / "golden_window.nc"
    window_attrs = {
        **provenance,
        "input": (
            f"{WINDOW_FILE} windowed with isel({isel_text}); "
            "identical to window_input.nc"
        ),
    }
    window = generate(
        lambda: window_saved.copy(deep=True), WINDOW, xroms, window_path, window_attrs
    )

    print_report("tiny", tiny, tiny_path)
    print_report("window", window, window_path)
    n_cells = window_saved["mask_rho"].size
    print(f"\nwindow mask_rho land cells: {land_cells(window_saved)} of {n_cells}")
    print(f"tiny mask_rho land cells: {land_cells(load_tiny(src))}")
    print("\nfiles:")
    for path in (tiny_path, window_path, raw_path):
        print(f"  {path.relative_to(WORKTREE)}  {path.stat().st_size:,} bytes")


def main() -> None:
    assert xgcm.__version__ == EXPECTED_XGCM, (
        f"xgcm {xgcm.__version__} found but {EXPECTED_XGCM} is required; run this "
        "with the environment that has xgcm 0.8.1 (see the module docstring)"
    )
    with tempfile.TemporaryDirectory(prefix="xroms_v062_") as tmpname:
        src = Path(tmpname)
        commit_full = export_v062(src)
        xroms = import_v062(src)
        try:
            print(f"commit {commit_full}")
            print(f"xroms imported from {xroms.__file__}")
            versions = f"xgcm {xgcm.__version__}, xarray {xr.__version__}"
            print(f"{versions}, numpy {np.__version__}")
            record_all(src, xroms, commit_full)
        finally:
            sys.path.remove(str(src))


if __name__ == "__main__":
    main()
