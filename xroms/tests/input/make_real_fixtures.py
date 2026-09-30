"""Make small test fixtures from real UCLA-ROMS files.

The older fixtures in this directory (``grid.nc``, ``ocean_his_000?.nc``; see
``make_files.py``) are synthetic and Rutgers-style. This script adds fixtures that
come from REAL files, so that xroms can be tested against the layout that UCLA-ROMS
and roms-tools really produce. It writes:

``ucla_rst.nc``
    A UCLA-ROMS restart file, cut to the variables ``ocean_time, zeta, ubar, vbar, u,
    v, temp, salt`` and to a small horizontal subset (10 x 8 rho points, see
    ``RST_SUBSET``). Both time records and all 100 vertical levels are kept, and ALL
    global attributes are kept unchanged (``theta_s, theta_b, hc, Cs_r, Cs_w, rho0,
    ...``). As in the original, there are no coordinate variables and ``ocean_time``
    is a plain data variable.
``ucla_grd.nc``
    The matching UCLA grid file (Easy Grid output) with all its variables, cut to the
    same horizontal subset.
``romstools_grid.nc``
    A very small grid made with ``roms_tools.Grid``, with both land and ocean and a
    rotated horizontal grid, written by ``Grid.save()``. This one needs an
    environment in which ``roms_tools`` can be imported.

Sources
-------
The two UCLA files are ``eastpac25km_rst.19980106000000.nc`` and ``epac25km_grd.nc``
from the roms-tools test data (https://github.com/CWorthy-ocean/roms-tools-test-data).
Point ``--src-dir`` (or the environment variable ``ROMS_TOOLS_TEST_DATA``) at a local
clone of that repository; the default is ``~/packages/roms-tools-test-data``. The
roms-tools grid is built from ``EMODnet_C2_coarse100.nc`` (topography) and
``GSHHS_l_L1.*`` (coastlines) from the same clone. Nothing is downloaded.

Usage
-----
From the repository root::

    python xroms/tests/input/make_real_fixtures.py

This makes the two UCLA files, which need only numpy, xarray and netCDF4, and then the
roms-tools grid if ``roms_tools`` can be imported. Otherwise it says so and skips that
file, so run it in an environment in which roms_tools is installed to get all three
(the UCLA files come out byte-for-byte the same in any environment). Options:

``--only ucla`` or ``--only romstools``
    Make just the UCLA files, or just the roms-tools grid.
``--src-dir DIR``
    Local clone of the roms-tools test data (see above).
``--out-dir DIR``
    Where to write the files (default: the directory of this script).

Method
------
The UCLA files are copied with netCDF4-python rather than round-tripped through
xarray, so that nothing is re-interpreted on the way: dtypes, attribute dtypes
(including the float64 arrays ``Cs_r`` and ``Cs_w``), ``_FillValue``, the char array
``spherical``, the unlimited ``time`` dimension and the per-record chunking all carry
over exactly, and no attributes are added.

The only change to the data is the horizontal subset: rho points 0:10 in eta and 0:8 in
xi, and the u and v points that go with them (``xi_u`` 0:7 and ``eta_v`` 0:9), so that
``xi_u = xi_rho - 1`` and ``eta_v = eta_rho - 1`` still hold. It is there for size. The
restart holds real float64 fields, which compress by only about 20%, and the full 15 x 10
restart is 758 KB, over the 500 KB limit of the ``check-added-large-files`` hook in this
repository's pre-commit configuration. Cutting horizontally keeps every variable, every
time record and every vertical level (``Cs_r`` and ``Cs_w`` have 100 and 101 values).

Large variables are compressed losslessly (zlib level 4 with the HDF5 shuffle filter),
which leaves every dtype unchanged. Small variables are left uncompressed because the
HDF5 chunk index costs more than zlib saves (compressing the grid files would make them
larger), so the grid files are not compressed at all.

Everything written is checked: each UCLA file against its source at the netCDF level
(names, dims, dtypes, attributes, and the data of the subset, bytewise) and through
xarray with ``decode_times=False``; that the restart and grid sizes match; that every
file is under the size limit; and the roms-tools grid for the variables, coordinates
and attributes that are expected in it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import warnings

from pathlib import Path
from typing import Any, Literal

import netCDF4
import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DEFAULT_SRC_DIR = Path(
    os.environ.get("ROMS_TOOLS_TEST_DATA", "~/packages/roms-tools-test-data")
).expanduser()

COMPLEVEL = 4  # zlib level
MIN_COMPRESS_BYTES = 4096  # compress only variables at least this big
MAX_BYTES = 500 * 1024  # size limit of pre-commit's check-added-large-files hook

# UCLA files (from the roms-tools test data)
RST_SRC = "eastpac25km_rst.19980106000000.nc"
GRD_SRC = "epac25km_grd.nc"
RST_OUT = "ucla_rst.nc"
GRD_OUT = "ucla_grd.nc"
RST_KEEP = ("ocean_time", "zeta", "ubar", "vbar", "u", "v", "temp", "salt")

# Horizontal subset of the UCLA files, as index ranges: the first 10 rho points in eta
# and the first 8 in xi, and the u and v points between them (u[i] lies between rho[i]
# and rho[i + 1], and likewise v), so that xi_u = xi_rho - 1 and eta_v = eta_rho - 1
# still hold. The grid file has no u or v dimensions. See the module docstring for why.
RHO_SUBSET = {"eta_rho": slice(0, 10), "xi_rho": slice(0, 8)}
RST_SUBSET = {**RHO_SUBSET, "xi_u": slice(0, 7), "eta_v": slice(0, 9)}
GRD_SUBSET = RHO_SUBSET

# roms-tools grid (generated from local files of the same test data)
RT_OUT = "romstools_grid.nc"
RT_TOPO = "EMODnet_C2_coarse100.nc"
RT_COAST = ("GSHHS_l_L1.shp", "GSHHS_l_L1.shx", "GSHHS_l_L1.dbf", "GSHHS_l_L1.prj")
# nx and ny count interior cells: roms-tools adds 2 rho-points in each direction, so
# this grid has (eta_rho, xi_rho) = (8, 10). It is a 240 x 180 km domain off the
# southwest coast of Iceland, rotated by 20 degrees, with the coast along its east side
# and open ocean to the west. theta_s, theta_b and hc differ from the roms-tools
# defaults (5, 2, 300), which the UCLA restart file also uses, so that tests cannot
# pass by falling back on those values.
RT_GRID_KWARGS = dict(
    nx=8,
    ny=6,
    size_x=240,
    size_y=180,
    center_lon=-22.0,
    center_lat=64.2,
    rot=20,
    N=10,
    theta_s=6.0,
    theta_b=1.5,
    hc=50.0,
)


def _check(condition: bool, message: str) -> None:
    """Raise AssertionError with ``message`` unless ``condition`` (survives -O)."""
    if not condition:
        raise AssertionError(message)


def _open_raw(path: Path, mode: Literal["r", "w"] = "r", **kwargs) -> netCDF4.Dataset:
    """Open a netCDF file with all automatic conversions turned off."""
    nc = netCDF4.Dataset(path, mode, **kwargs)
    nc.set_auto_maskandscale(False)  # no masking or scaling
    nc.set_auto_chartostring(False)  # keep char arrays as char arrays
    return nc


def _engines() -> list[str]:
    """xarray engines usable here: netcdf4 always, h5netcdf if installed."""
    engines = ["netcdf4"]
    if importlib.util.find_spec("h5netcdf") is not None:
        engines.append("h5netcdf")
    return engines


def _subset_sizes(subset: dict[str, slice]) -> dict[str, int]:
    """Sizes of the dimensions in ``subset`` (slices with a start and a stop)."""
    return {dim: window.stop - window.start for dim, window in subset.items()}


def _window_length(name: str, window: slice, size: int) -> int:
    """Number of points that ``window`` takes from a dimension of length ``size``."""
    _check(window.step in (None, 1), f"{name}: only slices with step 1 are supported")
    _check(
        window.stop is None or window.stop <= size,
        f"{name}: the slice goes beyond the {size} points of the dimension",
    )
    start, stop, _ = window.indices(size)
    _check(stop > start, f"{name}: the slice is empty")
    return stop - start


def _select(
    nc: netCDF4.Dataset,
    label: str,
    keep: tuple[str, ...] | None,
    subset: dict[str, slice],
) -> tuple[list[str], dict[str, int]]:
    """Get the variables of ``nc`` to copy and the sizes of the dimensions they use.

    Both are in the order of the file, and the sizes are those of the subset.
    """
    names = [n for n in nc.variables if keep is None or n in keep]
    missing = set(keep or ()) - set(names)
    _check(not missing, f"{label}: no such variables: {sorted(missing)}")

    used = {dim for n in names for dim in nc.variables[n].dimensions}
    unknown = set(subset) - used
    _check(
        not unknown, f"{label}: no copied variable uses dimensions {sorted(unknown)}"
    )

    sizes = {}
    for name, dim in nc.dimensions.items():
        if name in used:
            sizes[name] = len(dim)
            if name in subset:
                sizes[name] = _window_length(name, subset[name], len(dim))
    return names, sizes


def _index(dimensions: tuple[str, ...], subset: dict[str, slice]) -> tuple[Any, ...]:
    """Index that takes the subset of a variable with these dimensions."""
    return tuple(subset.get(dim, slice(None)) for dim in dimensions) or (Ellipsis,)


def copy_netcdf(
    src: Path,
    dst: Path,
    keep: tuple[str, ...] | None = None,
    subset: dict[str, slice] | None = None,
    complevel: int = COMPLEVEL,
) -> None:
    """Copy (a subset of) the variables of ``src`` to a new NETCDF4 file ``dst``.

    All global attributes are copied, and so are the attributes of each kept variable.
    Dimensions that no kept variable uses are not created. Dtypes, ``_FillValue`` and
    the unlimited dimension are preserved, as is the source chunking (clipped to the
    dimension sizes). Numeric variables of at least ``MIN_COMPRESS_BYTES`` bytes are
    compressed with zlib.

    Parameters
    ----------
    src, dst : Path
        Source file and file to create (overwritten).
    keep : tuple of str, optional
        Names of the variables to copy, which are written in source order. Default is
        all variables.
    subset : dict of str to slice, optional
        Window (a slice with step 1) to take along each named dimension, for all of the
        variables that have it. Each name must be a dimension of a copied variable.
        Default is to take everything.
    complevel : int, optional
        zlib compression level.
    """
    subset = subset or {}
    with _open_raw(src) as s, _open_raw(dst, "w", format="NETCDF4") as d:
        _check(not s.groups, f"{src.name}: groups are not supported")
        names, sizes = _select(s, src.name, keep, subset)
        for name, size in sizes.items():
            d.createDimension(name, None if s.dimensions[name].isunlimited() else size)

        d.setncatts({a: s.getncattr(a) for a in s.ncattrs()})

        for n in names:
            sv = s.variables[n]
            shape = tuple(sizes[dim] for dim in sv.dimensions)
            attrs = {a: sv.getncattr(a) for a in sv.ncattrs()}
            fill = attrs.pop("_FillValue", None)  # can only be set at creation
            kwargs: dict = {}
            chunks = sv.chunking()
            if chunks != "contiguous":
                kwargs["chunksizes"] = np.minimum(chunks, shape).tolist()
            if (
                sv.dtype.kind in "iuf"
                and int(np.prod(shape)) * sv.dtype.itemsize >= MIN_COMPRESS_BYTES
            ):
                kwargs.update(zlib=True, complevel=complevel, shuffle=True)
                kwargs.setdefault("chunksizes", list(shape))
            dv = d.createVariable(n, sv.dtype, sv.dimensions, fill_value=fill, **kwargs)
            dv.setncatts(attrs)
            with warnings.catch_warnings():
                # netCDF4-python writes char arrays with a call that NumPy >= 2.5
                # deprecates
                warnings.filterwarnings(
                    "ignore",
                    message="Setting the shape on a NumPy array",
                    category=DeprecationWarning,
                )
                dv[...] = sv[_index(sv.dimensions, subset)]


def _same_attr(a: object, b: object) -> bool:
    """True if two attribute values have the same type, shape and value."""
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def verify_copy(
    src: Path,
    dst: Path,
    keep: tuple[str, ...] | None = None,
    subset: dict[str, slice] | None = None,
) -> None:
    """Check at the netCDF level that ``dst`` is an exact (subset) copy of ``src``.

    Compares dimension names, sizes and unlimited flags; variable names, dimensions,
    dtypes, attributes (with their dtypes) and data (bytewise, so NaNs compare equal);
    and the global attributes. The data of ``dst`` must be those of ``src`` within the
    ``subset`` windows, and the sizes of the dimensions those of the windows.
    """
    subset = subset or {}
    with _open_raw(src) as s, _open_raw(dst) as d:
        names, sizes = _select(s, src.name, keep, subset)
        _check(list(d.variables) == names, f"{dst.name}: variables differ")
        _check(list(d.dimensions) == list(sizes), f"{dst.name}: dimensions differ")
        for name, size in sizes.items():
            dd = d.dimensions[name]
            _check(len(dd) == size, f"{dst.name}: {name} has {len(dd)}, not {size}")
            _check(
                s.dimensions[name].isunlimited() == dd.isunlimited(),
                f"{dst.name}: unlimited flag of {name} differs",
            )

        _check(s.ncattrs() == d.ncattrs(), f"{dst.name}: global attribute names differ")
        for a in s.ncattrs():
            _check(
                _same_attr(s.getncattr(a), d.getncattr(a)),
                f"{dst.name}: global attribute {a!r} differs",
            )

        for n in names:
            sv, dv = s.variables[n], d.variables[n]
            _check(
                (sv.dtype, sv.dimensions) == (dv.dtype, dv.dimensions),
                f"{dst.name}: {n}: dtype or dimensions differ",
            )
            _check(
                dv.shape == tuple(sizes[dim] for dim in sv.dimensions),
                f"{dst.name}: {n}: shape is {dv.shape}",
            )
            _check(
                set(sv.ncattrs()) == set(dv.ncattrs()),
                f"{dst.name}: {n}: attribute names differ",
            )
            for a in sv.ncattrs():
                _check(
                    _same_attr(sv.getncattr(a), dv.getncattr(a)),
                    f"{dst.name}: {n}: attribute {a!r} differs",
                )
            expected = sv[_index(sv.dimensions, subset)]
            _check(
                np.ascontiguousarray(expected).tobytes()
                == np.ascontiguousarray(dv[...]).tobytes(),
                f"{dst.name}: {n}: data differ",
            )


def verify_xarray(
    src: Path,
    dst: Path,
    keep: tuple[str, ...] | None = None,
    subset: dict[str, slice] | None = None,
) -> None:
    """Check through xarray that ``dst`` holds (a subset of) the dataset in ``src``.

    Uses ``decode_times=False`` (the UCLA ``ocean_time`` has units of "second", not CF
    time units) and every available engine.
    """
    for engine in _engines():
        with xr.open_dataset(src, decode_times=False, engine=engine) as s:
            with xr.open_dataset(dst, decode_times=False, engine=engine) as d:
                expected = (s if keep is None else s[list(keep)]).isel(subset or {})
                xr.testing.assert_identical(expected, d)
                _check(
                    all(d[n].dtype == expected[n].dtype for n in d.variables),
                    f"{dst.name}: dtypes differ through xarray ({engine})",
                )


def check_ucla_restart(path: Path) -> None:
    """Check the properties of the restart fixture that matter for xroms."""
    sizes = dict(time=2, s_rho=100, **_subset_sizes(RST_SUBSET))
    for engine in _engines():
        with xr.open_dataset(path, decode_times=False, engine=engine) as ds:
            _check(
                len(ds.coords) == 0, f"{path.name}: has coordinates {list(ds.coords)}"
            )
            _check(
                list(ds.data_vars) == list(RST_KEEP), f"{path.name}: variables differ"
            )
            _check(
                dict(ds.sizes) == sizes,
                f"{path.name}: sizes are {dict(ds.sizes)}, not {sizes}",
            )
            _check(ds["ocean_time"].dims == ("time",), "ocean_time is not on time")
            _check(
                ds["ocean_time"].attrs
                == {"long_name": "Time since 1995/01/01", "units": "second"},
                f"{path.name}: ocean_time attributes changed",
            )
            for name in ("theta_s", "theta_b", "hc", "rho0"):
                _check(
                    np.ndim(ds.attrs[name]) == 0, f"{path.name}: {name} not a scalar"
                )
            _check(ds.attrs["Cs_r"].shape == (ds.sizes["s_rho"],), "Cs_r length")
            _check(ds.attrs["Cs_w"].shape == (ds.sizes["s_rho"] + 1,), "Cs_w length")


def check_ucla_grid(path: Path) -> None:
    """Check the properties of the UCLA grid fixture that matter for xroms."""
    sizes = dict(one=1, **_subset_sizes(GRD_SUBSET))
    for engine in _engines():
        with xr.open_dataset(path, decode_times=False, engine=engine) as ds:
            _check(
                len(ds.coords) == 0, f"{path.name}: has coordinates {list(ds.coords)}"
            )
            _check(
                dict(ds.sizes) == sizes,
                f"{path.name}: sizes are {dict(ds.sizes)}, not {sizes}",
            )
            _check(ds["spherical"].dtype == np.dtype("S1"), "spherical is not char")
            _check(ds["spherical"].values.tolist() == [b"T"], "spherical is not 'T'")


def check_ucla_pair(rst: Path, grd: Path) -> None:
    """Check that the restart and grid fixtures describe the same horizontal grid."""
    with xr.open_dataset(rst, decode_times=False) as r:
        with xr.open_dataset(grd, decode_times=False) as g:
            for dim in ("eta_rho", "xi_rho"):
                _check(
                    r.sizes[dim] == g.sizes[dim], f"restart and grid differ in {dim}"
                )
            _check(r.sizes["xi_u"] == r.sizes["xi_rho"] - 1, "xi_u != xi_rho - 1")
            _check(r.sizes["eta_v"] == r.sizes["eta_rho"] - 1, "eta_v != eta_rho - 1")


def check_romstools_grid(path: Path) -> None:
    """Check that the roms-tools grid has the variables and attributes expected."""
    with xr.open_dataset(path) as ds:
        for name in ("sigma_r", "sigma_w", "Cs_r", "Cs_w", "h", "angle", "pm", "pn"):
            _check(name in ds.variables, f"{path.name}: missing {name}")
        for name in ("lon_rho", "lat_rho", "lon_u", "lat_u", "lon_v", "lat_v"):
            _check(name in ds.coords, f"{path.name}: {name} is not a coordinate")
        for name in ("mask_rho", "mask_u", "mask_v"):
            _check(name in ds.variables, f"{path.name}: missing {name}")
            _check(set(np.unique(ds[name])) == {0, 1}, f"{path.name}: {name} not mixed")
        for name in ("theta_s", "theta_b", "hc", "straddle"):
            _check(name in ds.attrs, f"{path.name}: missing attribute {name}")
        _check(ds.attrs["hc"] == RT_GRID_KWARGS["hc"], "hc was not recorded")
        _check(ds.sizes["s_rho"] == RT_GRID_KWARGS["N"], "unexpected number of levels")
        _check(ds.sizes["s_w"] == ds.sizes["s_rho"] + 1, "s_w != s_rho + 1")
        _check(ds["Cs_r"].shape == (ds.sizes["s_rho"],), "Cs_r length")
        _check(ds["Cs_w"].shape == (ds.sizes["s_w"],), "Cs_w length")
        _check(ds.sizes["xi_u"] == ds.sizes["xi_rho"] - 1, "xi_u != xi_rho - 1")
        _check(ds.sizes["eta_v"] == ds.sizes["eta_rho"] - 1, "eta_v != eta_rho - 1")


def make_ucla(src_dir: Path, out_dir: Path) -> list[Path]:
    """Write ``ucla_rst.nc`` and ``ucla_grd.nc``; return their paths."""
    rst_src, grd_src = src_dir / RST_SRC, src_dir / GRD_SRC
    for path in (rst_src, grd_src):
        _check(path.exists(), f"{path} not found (use --src-dir)")
    rst, grd = out_dir / RST_OUT, out_dir / GRD_OUT

    copy_netcdf(rst_src, rst, keep=RST_KEEP, subset=RST_SUBSET)
    verify_copy(rst_src, rst, keep=RST_KEEP, subset=RST_SUBSET)
    verify_xarray(rst_src, rst, keep=RST_KEEP, subset=RST_SUBSET)
    check_ucla_restart(rst)

    copy_netcdf(grd_src, grd, subset=GRD_SUBSET)
    verify_copy(grd_src, grd, subset=GRD_SUBSET)
    verify_xarray(grd_src, grd, subset=GRD_SUBSET)
    check_ucla_grid(grd)

    check_ucla_pair(rst, grd)
    return [rst, grd]


def make_romstools(src_dir: Path, out_dir: Path) -> list[Path]:
    """Write ``romstools_grid.nc``; return its path. Needs roms_tools."""
    import roms_tools

    needed = (RT_TOPO, *RT_COAST)
    for name in needed:
        _check((src_dir / name).exists(), f"{src_dir / name} not found (use --src-dir)")
    out = (out_dir / RT_OUT).resolve()

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        # Run from a scratch directory that holds only links to the inputs. roms-tools
        # records the paths it is given in the global attributes of the file, so this
        # stores bare file names rather than a home directory, and any stray files
        # that the regridding libraries write stay out of the repository.
        for name in needed:
            (Path(tmp) / name).symlink_to((src_dir / name).resolve())
        os.chdir(tmp)
        try:
            grid = roms_tools.Grid(
                **RT_GRID_KWARGS,
                topography_source={"name": "EMOD", "path": RT_TOPO},
                mask_shapefile=RT_COAST[0],
            )
            grid.save(out)  # writes the file as is, small enough not to need zlib
        finally:
            os.chdir(cwd)
    check_romstools_grid(out)
    return [out]


def describe(path: Path) -> str:
    """One-line size report for ``path``."""
    size = path.stat().st_size
    return f"  {path.name:20s} {size:9,d} bytes ({size / 1024:6.1f} KB)"


def main(argv: list[str] | None = None) -> int:
    """Make the fixtures (see the module docstring)."""
    parser = argparse.ArgumentParser(
        description="Make test fixtures from real UCLA-ROMS files."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help="local clone of roms-tools-test-data (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE,
        help="where to write the fixtures (default: %(default)s)",
    )
    parser.add_argument(
        "--only",
        choices=("ucla", "romstools"),
        help="make only the UCLA restart and grid, or only the roms-tools grid "
        "(default: the UCLA files, plus the roms-tools grid if roms_tools can be "
        "imported)",
    )
    args = parser.parse_args(argv)
    src_dir, out_dir = args.src_dir.expanduser(), args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if args.only in (None, "ucla"):
        written += make_ucla(src_dir, out_dir)
    if args.only in (None, "romstools"):
        try:
            import roms_tools  # noqa: F401
        except ImportError as err:
            message = (
                f"roms_tools cannot be imported with this Python ({err}); "
                f"{RT_OUT} was not made.\nRun this script with the Python of an "
                "environment in which roms_tools is installed to make it."
            )
            if args.only == "romstools":
                print(message, file=sys.stderr)
                return 1
            print(message)
        else:
            written += make_romstools(src_dir, out_dir)

    for path in written:
        size = path.stat().st_size
        _check(
            size <= MAX_BYTES,
            f"{path.name} is {size:,d} bytes, over the {MAX_BYTES:,d} bytes that "
            "pre-commit's check-added-large-files hook allows",
        )
    print("Wrote and verified (all under the pre-commit size limit):")
    print("\n".join(describe(path) for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
