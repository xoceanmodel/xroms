"""Shared tables for the contract, laziness and layout sweeps.

``OPS`` lists the public pure functions, each as one call that works on every variant
of a merged Dataset (see :func:`make_variant`); ``ACCESSOR`` lists the ``ds.xroms``
members with the pure function each one wraps. ``p(name)`` returns the Dataset
variable ``name`` the way the variant has it (its top level for ``surface``), so a
single lambda serves every variant.

Some entries compose two functions (``KE`` needs a speed, ``EKE`` geostrophic
velocities, ``ertel`` a buoyancy): the chain is the documented way to call them.
``EKE`` is the plain ``0.5 * (ug**2 + vg**2)`` of v1.0, so it needs no time dim.
"""

import functools
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import xarray as xr

import xroms
from xroms.conventions import canonicalize, time_dim
from xroms.tests.conftest import chunked, merged


K, J = 3, 4  # the water column of the ``column`` variant (valid on every stagger)
#: rho-index window of the ``halo`` variant, and the matching window of each canonical dim
HALO = 3
WINDOW = {"xi_rho": slice(4, 9), "xi_u": slice(4, 8), "eta_rho": slice(3, 6), "eta_v": slice(3, 5)}


@functools.lru_cache(maxsize=None)
def _built(layout, kwargs):
    return merged(layout, **dict(kwargs))


def dataset(layout, **kwargs):
    """``conftest.merged(layout, **kwargs)``, built once per layout and options (every call gets its own shallow copy)."""
    return _built(layout, tuple(sorted(kwargs.items()))).copy(deep=False)


@dataclass(frozen=True)
class Op:
    name: str
    call: Callable  # (ds, p) -> DataArray or tuple of them
    driver: Optional[str]  # Dataset variable whose time dim decides whether the result has one (None: never)
    same: Union[None, str, Tuple[Optional[str], ...]] = None  # Dataset variable(s) at the result's position
    local: bool = True  # False: reduces over the horizontal, so a subset gives another answer


def column(ds):
    """One water column, across every staggered dim (no horizontal dims left)."""
    index = {d: (K if d.startswith("eta_") else J) for d in ds.dims if d.startswith(("eta_", "xi_"))}
    return ds.isel(index)


def make_variant(ds, name):
    """``(grid Dataset, p)`` for variant ``name`` of the merged Dataset ``ds``."""
    tdim = time_dim(ds)
    grid = ds
    if name == "one_time":
        grid = ds.isel({tdim: 0})
    elif name == "time1":  # a time dim of length 1 is still a time dim
        grid = ds.isel({tdim: [0]})
    elif name == "subset":
        grid = xroms.subset(ds, X=slice(2, 9), Y=slice(1, 7))
    elif name == "halo":  # the window of WINDOW plus HALO rho points all round: trim(n=HALO) leaves WINDOW
        grid = xroms.subset(ds, X=WINDOW["xi_rho"], Y=WINDOW["eta_rho"], halo=HALO)
    elif name == "transposed":  # every variable's dims in reverse order: (xi, eta, level, time)
        grid = ds.transpose()
    elif name == "float32":  # as ROMS writes its fields; the grid stays double
        grid = ds.assign({n: ds[n].astype("float32") for n in ("temp", "salt", "u", "v", "zeta")})
    elif name == "member":  # an ensemble: the fields get a leading dim that is not a grid dim
        return grid, lambda n: _with_member(grid[n]) if "s_rho" in grid[n].dims or n == "zeta" else grid[n]
    elif name == "chunked":
        grid = chunked(ds)
    elif name == "column":
        grid = column(ds)
    elif name == "surface":
        return grid, lambda n: xroms.surface(grid[n]) if "s_rho" in grid[n].dims else grid[n]
    elif name != "full":
        raise ValueError(name)
    return grid, lambda n: grid[n]


def _with_member(var):
    out = xr.concat([var, var * 1.01], dim="member")
    out.name, out.attrs = var.name, var.attrs
    return out


def results(out):
    """``out`` as a tuple of DataArrays."""
    return out if isinstance(out, tuple) else (out,)


def has_time(var):
    return time_dim(canonicalize(var)) is not None


def _rho(ds, p):
    return xroms.density(p("temp"), p("salt"), grid=ds)


def _sig0(ds, p):
    return xroms.potential_density(p("temp"), p("salt"))


def _buoy(ds, p):
    return xroms.buoyancy(_sig0(ds, p))


def _geo(ds, p):
    return xroms.uv_geostrophic(p("zeta"), p("f"), ds)


def _earth(ds, p):
    return xroms.grid_to_earth(p("u"), p("v"), p("angle"))


OPS = [
    Op("ddxi", lambda ds, p: xroms.ddxi(p("temp"), ds), "temp", "u"),
    Op("ddeta", lambda ds, p: xroms.ddeta(p("temp"), ds), "temp", "v"),
    Op("ddxi_along_s", lambda ds, p: xroms.ddxi(p("temp"), ds, along_s=True), "temp", "u"),
    Op("ddeta_along_s", lambda ds, p: xroms.ddeta(p("temp"), ds, along_s=True), "temp", "v"),
    Op("ddz", lambda ds, p: xroms.ddz(p("temp"), ds), "temp"),
    Op("hgrad", lambda ds, p: xroms.hgrad(p("temp"), ds), "temp", ("u", "v")),
    Op("to_rho", lambda ds, p: xroms.to_rho(p("u")), "u", "temp"),
    Op("to_u", lambda ds, p: xroms.to_u(p("temp")), "temp", "u"),
    Op("to_v", lambda ds, p: xroms.to_v(p("temp")), "temp", "v"),
    Op("to_psi", lambda ds, p: xroms.to_psi(p("temp")), "temp"),
    Op("to_s_w", lambda ds, p: xroms.to_s_w(p("temp")), "temp"),
    Op("to_s_rho", lambda ds, p: xroms.to_s_rho(xroms.to_s_w(p("temp"))), "temp", "temp"),
    Op("to_grid", lambda ds, p: xroms.to_grid(p("u"), hcoord="v", scoord="s_w"), "u"),
    Op("relative_vorticity", lambda ds, p: xroms.relative_vorticity(p("u"), p("v"), ds), "u"),
    Op("convergence", lambda ds, p: xroms.convergence(p("u"), p("v"), ds), "u", "temp"),
    Op("convergence_along_s", lambda ds, p: xroms.convergence(p("u"), p("v"), ds, along_s=True), "u", "temp"),
    Op("divergence", lambda ds, p: xroms.divergence(p("u"), p("v"), ds), "u", "temp"),
    Op("speed", lambda ds, p: xroms.speed(p("u"), p("v")), "u", "temp"),
    Op("KE", lambda ds, p: xroms.KE(xroms.rho0(ds), xroms.speed(p("u"), p("v"))), "u", "temp"),
    Op("uv_geostrophic", _geo, "zeta", ("u", "v")),
    Op("EKE", lambda ds, p: xroms.EKE(*_geo(ds, p)), "zeta", "temp"),
    Op("dudz", lambda ds, p: xroms.dudz(p("u"), ds), "u"),
    Op("dvdz", lambda ds, p: xroms.dvdz(p("v"), ds), "v"),
    Op("vertical_shear", lambda ds, p: xroms.vertical_shear(xroms.dudz(p("u"), ds), xroms.dvdz(p("v"), ds)), "u"),
    Op("ertel", lambda ds, p: xroms.ertel(_buoy(ds, p), p("u"), p("v"), p("f"), ds), "u", "temp"),
    Op("density", _rho, "temp", "temp"),
    Op("potential_density", _sig0, "temp", "temp"),
    Op("buoyancy", _buoy, "temp", "temp"),
    Op("N2", lambda ds, p: xroms.N2(_rho(ds, p), ds), "temp"),
    Op("M2", lambda ds, p: xroms.M2(_rho(ds, p), ds), "temp", "temp"),
    Op("mld", lambda ds, p: xroms.mld(_sig0(ds, p), ds), "temp", "zeta"),
    Op("z_rho", lambda ds, p: xroms.z(ds), "zeta", "temp"),
    Op("z_w_u", lambda ds, p: xroms.z(ds, hcoord="u", scoord="s_w"), "zeta"),
    Op("dz", lambda ds, p: xroms.dz(ds), "zeta", "temp"),
    Op("dz_w_v", lambda ds, p: xroms.dz(ds, hcoord="v", scoord="s_w"), "zeta"),
    Op("dx", lambda ds, p: xroms.dx(ds), None, "h"),
    Op("dy_u", lambda ds, p: xroms.dy(ds, "u"), None),
    Op("dA_psi", lambda ds, p: xroms.dA(ds, "psi"), None),
    Op("dV", lambda ds, p: xroms.dV(ds), "zeta", "temp"),
    Op("depth_average", lambda ds, p: xroms.depth_average(p("temp"), ds), "temp", "zeta"),
    Op("surface", lambda ds, p: xroms.surface(p("temp")), "temp", "zeta"),
    Op("bottom", lambda ds, p: xroms.bottom(p("u")), "u", "u"),
    Op("zslice", lambda ds, p: xroms.zslice(p("temp"), [-5.0, -2.0], ds), "temp"),
    Op("isoslice", lambda ds, p: xroms.isoslice(p("temp"), [-5.0, -2.0], xroms.z(ds), new_dim="zz"), "temp"),
    Op("gridmean_XY", lambda ds, p: xroms.gridmean(p("temp"), ds, ("X", "Y")), "temp", local=False),
    Op("gridsum_Z", lambda ds, p: xroms.gridsum(p("temp"), ds, "Z"), "temp", "zeta"),
    Op("grid_to_earth", _earth, "u", ("temp", "temp")),
    Op("earth_to_grid", lambda ds, p: xroms.earth_to_grid(*_earth(ds, p), p("angle")), "u", ("u", "v")),
]
OP_IDS = [o.name for o in OPS]

#: results that keep the order their input had instead of putting it in (time, vertical, eta, xi) order: selections
#: and reductions (``isel``, ``mean``) and the metrics, none of which promise an order in their docstrings
KEEPS_INPUT_ORDER = {"dx", "dy_u", "dA_psi", "depth_average", "surface", "bottom"}

#: ops that take no variable, only the grid: their vertical structure comes from the vertical
#: parameters (Cs_r, Cs_w), which are chunked on their own, not from the data's levels
GRID_ONLY = {"z_rho", "z_w_u", "dz", "dz_w_v", "dx", "dy_u", "dA_psi", "dV"}

# --- the ``ds.xroms`` members, with the pure function each wraps ------------------------


@dataclass(frozen=True)
class Acc:
    name: str
    accessor: Callable  # ds -> result (a tuple for eastnorth)
    pure: Callable  # ds -> the same result from the pure function(s)
    pos: Union[None, str, Tuple[str, ...]]  # horizontal position of the result (one per result of a tuple)
    vert: Optional[str]  # vertical position (None: none)
    same: Union[None, str, Tuple[Optional[str], ...]] = None  # variable of ds at that position


def _uv(ds):
    return ds.u, ds.v


def _acc_rho(ds):
    return xroms.density(ds.temp, ds.salt, grid=ds)


def _acc_sig0(ds):
    return xroms.potential_density(ds.temp, ds.salt, 0)


def _acc_buoy(ds):
    return xroms.buoyancy(_acc_sig0(ds), xroms.rho0(ds))


def _acc_conv(ds):
    return xroms.convergence(ds.u, ds.v, ds)


def _acc_sig0_mld(ds):
    return xroms.mld(_acc_sig0(ds), ds)


ACCESSOR = [
    Acc("speed", lambda ds: ds.xroms.speed, lambda ds: xroms.speed(*_uv(ds)), "rho", "s_rho", "temp"),
    Acc("KE", lambda ds: ds.xroms.KE, lambda ds: xroms.KE(xroms.rho0(ds), xroms.speed(*_uv(ds))), "rho", "s_rho", "temp"),
    Acc("ug", lambda ds: ds.xroms.ug, lambda ds: xroms.uv_geostrophic(ds.zeta, ds.f, ds, which="xi"), "u", None, "u"),
    Acc("vg", lambda ds: ds.xroms.vg, lambda ds: xroms.uv_geostrophic(ds.zeta, ds.f, ds, which="eta"), "v", None, "v"),
    Acc("EKE", lambda ds: ds.xroms.EKE, lambda ds: xroms.EKE(*xroms.uv_geostrophic(ds.zeta, ds.f, ds)), "rho", None, "temp"),
    Acc("east", lambda ds: ds.xroms.east, lambda ds: xroms.grid_to_earth(*_uv(ds), ds.angle)[0], "rho", "s_rho", "temp"),
    Acc("north", lambda ds: ds.xroms.north, lambda ds: xroms.grid_to_earth(*_uv(ds), ds.angle)[1], "rho", "s_rho", "temp"),
    Acc("eastnorth", lambda ds: ds.xroms.eastnorth, lambda ds: xroms.grid_to_earth(*_uv(ds), ds.angle), ("rho", "rho"), "s_rho", ("temp", "temp")),
    Acc("east_rotated", lambda ds: ds.xroms.east_rotated(0.3), lambda ds: xroms.rotate_vectors(*xroms.grid_to_earth(*_uv(ds), ds.angle), 0.3)[0], "rho", "s_rho", "temp"),
    Acc("north_rotated", lambda ds: ds.xroms.north_rotated(0.3), lambda ds: xroms.rotate_vectors(*xroms.grid_to_earth(*_uv(ds), ds.angle), 0.3)[1], "rho", "s_rho", "temp"),
    Acc("dudz", lambda ds: ds.xroms.dudz, lambda ds: xroms.dudz(ds.u, ds), "u", "s_w"),
    Acc("dvdz", lambda ds: ds.xroms.dvdz, lambda ds: xroms.dvdz(ds.v, ds), "v", "s_w"),
    Acc("vertical_shear", lambda ds: ds.xroms.vertical_shear, lambda ds: xroms.vertical_shear(xroms.dudz(ds.u, ds), xroms.dvdz(ds.v, ds)), "rho", "s_w"),
    Acc("vort", lambda ds: ds.xroms.vort, lambda ds: xroms.relative_vorticity(ds.u, ds.v, ds), "psi", "s_rho"),
    Acc("convergence", lambda ds: ds.xroms.convergence, _acc_conv, "rho", "s_rho", "temp"),
    Acc("convergence_norm", lambda ds: ds.xroms.convergence_norm, lambda ds: _acc_conv(canonicalize(ds)).isel(s_rho=-1) / canonicalize(ds).f, "rho", None, "temp"),
    Acc("divergence", lambda ds: ds.xroms.divergence, lambda ds: xroms.divergence(ds.u, ds.v, ds), "rho", "s_rho", "temp"),
    Acc("divergence_norm", lambda ds: ds.xroms.divergence_norm, lambda ds: xroms.divergence(*_uv(canonicalize(ds)), canonicalize(ds)).isel(s_rho=-1) / canonicalize(ds).f, "rho", None, "temp"),
    Acc("ertel", lambda ds: ds.xroms.ertel, lambda ds: xroms.ertel(_acc_buoy(ds), ds.u, ds.v, ds.f, ds), "rho", "s_rho", "temp"),
    Acc("rho", lambda ds: ds.xroms.rho, _acc_rho, "rho", "s_rho", "temp"),
    Acc("sig0", lambda ds: ds.xroms.sig0, _acc_sig0, "rho", "s_rho", "temp"),
    Acc("buoyancy", lambda ds: ds.xroms.buoyancy, _acc_buoy, "rho", "s_rho", "temp"),
    Acc("N2", lambda ds: ds.xroms.N2, lambda ds: xroms.N2(_acc_rho(ds), ds, xroms.rho0(ds)), "rho", "s_w"),
    Acc("M2", lambda ds: ds.xroms.M2, lambda ds: xroms.M2(_acc_rho(ds), ds, xroms.rho0(ds)), "rho", "s_rho", "temp"),
    Acc("mld", lambda ds: ds.xroms.mld(), _acc_sig0_mld, "rho", None, "zeta"),
    Acc("ddxi", lambda ds: ds.xroms.ddxi("temp"), lambda ds: xroms.ddxi(ds.temp, ds), "u", "s_rho", "u"),
    Acc("ddeta", lambda ds: ds.xroms.ddeta("temp"), lambda ds: xroms.ddeta(ds.temp, ds), "v", "s_rho", "v"),
    Acc("ddz", lambda ds: ds.xroms.ddz("temp"), lambda ds: xroms.ddz(ds.temp, ds), "rho", "s_w"),
    Acc("hgrad", lambda ds: ds.xroms.hgrad("temp"), lambda ds: xroms.hgrad(ds.temp, ds), ("u", "v"), "s_rho", ("u", "v")),
    Acc("zslice", lambda ds: ds.xroms.zslice("temp", [-5.0, -2.0]), lambda ds: xroms.zslice(ds.temp, [-5.0, -2.0], ds), "rho", None),
    Acc("isoslice", lambda ds: ds.xroms.isoslice("temp", [-5.0, -2.0], ds.xroms.z_rho, new_dim="zz"), lambda ds: xroms.isoslice(ds.temp, [-5.0, -2.0], xroms.z(ds), new_dim="zz"), "rho", None),
    Acc("z_rho", lambda ds: ds.xroms.z_rho, lambda ds: xroms.z(ds), "rho", "s_rho", "temp"),
    Acc("z_w", lambda ds: ds.xroms.z_w, lambda ds: xroms.z(ds, scoord="s_w"), "rho", "s_w"),
    Acc("z(u, s_w)", lambda ds: ds.xroms.z(hcoord="u", scoord="s_w"), lambda ds: xroms.z(ds, hcoord="u", scoord="s_w"), "u", "s_w"),
    Acc("dz", lambda ds: ds.xroms.dz(), lambda ds: xroms.dz(ds), "rho", "s_rho", "temp"),
    Acc("dz(v, s_w)", lambda ds: ds.xroms.dz("v", "s_w"), lambda ds: xroms.dz(ds, hcoord="v", scoord="s_w"), "v", "s_w"),
    Acc("dx", lambda ds: ds.xroms.dx(), lambda ds: xroms.dx(ds), "rho", None, "h"),
    Acc("dy(u)", lambda ds: ds.xroms.dy("u"), lambda ds: xroms.dy(ds, "u"), "u", None),
    Acc("dA(psi)", lambda ds: ds.xroms.dA("psi"), lambda ds: xroms.dA(ds, "psi"), "psi", None),
    Acc("dV", lambda ds: ds.xroms.dV(), lambda ds: xroms.dV(ds), "rho", "s_rho", "temp"),
    Acc("to_grid", lambda ds: ds.xroms.to_grid("u", hcoord="v", scoord="s_w"), lambda ds: xroms.to_grid(ds.u, hcoord="v", scoord="s_w"), "v", "s_w"),
    Acc("depth_average", lambda ds: ds.xroms.depth_average("temp"), lambda ds: xroms.depth_average(ds.temp, ds), "rho", None, "zeta"),
    Acc("gridmean", lambda ds: ds.xroms.gridmean("temp", "Z"), lambda ds: xroms.gridmean(ds.temp, ds, "Z"), "rho", None, "zeta"),
    Acc("gridsum", lambda ds: ds.xroms.gridsum("temp", "Z"), lambda ds: xroms.gridsum(ds.temp, ds, "Z"), "rho", None, "zeta"),
    Acc("surface", lambda ds: ds.xroms.surface("u"), lambda ds: xroms.surface(ds.u), "u", None, "u"),
    Acc("bottom", lambda ds: ds.xroms.bottom("v"), lambda ds: xroms.bottom(ds.v), "v", None, "v"),
]
ACC_IDS = [a.name for a in ACCESSOR]
