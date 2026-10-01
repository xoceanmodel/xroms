"""Grid moves, derivatives, grid-weighted sums/means, selection and subsetting.

All functions are stateless: they compute from the arrays passed in (and, for
metric- or depth-dependent operations, from the grid primitives in ``grid``),
and return new objects in canonical ROMS dim names.

Output placement: a derivative lands where the discrete operation puts it.
``ddxi``/``ddeta`` move the horizontal position (rho -> u, u -> rho, ...) and keep
the input's vertical levels; ``ddz`` moves the vertical position
(``s_rho`` <-> ``s_w``). Pass ``hcoord``/``scoord`` to get the result elsewhere.

Boundaries: ``"extend"`` (default) gives one-sided values at the edges, taken
from the data (the nearest computed derivative, not the nearest difference: grid
spacing varies from point to point); ``"fill"`` puts ``fill_value`` there (NaN by
default; 0 imposes a zero-gradient condition). Nothing is ever padded with the
field's own edge value before differencing, which is what produced spurious zeros
before v1.0.
"""

import numpy as np
import xarray as xr

from . import _xgcm
from ._align import _check_grid, _reject_legacy, require, select_like, with_grid_coords
from .conventions import CANONICAL, canonicalize, hposition, normalize_hcoord, normalize_scoord, time_dim, vposition
from .vertical import dz as _dz
from .vertical import z_like


HDIMS = {"X": ("xi_rho", "xi_u"), "Y": ("eta_rho", "eta_v")}
_HORIZONTAL = {d for pair in HDIMS.values() for d in pair}


# --- moving between grid positions ------------------------------------------------


def to_grid(
    var,
    hcoord=None,
    scoord=None,
    *args,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    attrs=None,
):
    """Move ``var`` to horizontal position ``hcoord`` and/or vertical ``scoord``.

    ``hcoord`` is one of ``rho``, ``u``, ``v``, ``psi``; ``scoord`` is ``s_rho``
    (or ``rho``) or ``s_w`` (or ``w``). Axes already at the requested position
    are left alone. No grid is needed: moving between staggers only averages
    neighbouring points.
    """
    _reject_legacy(args, "to_grid", "Use xroms.to_grid(var, hcoord=..., scoord=...).")
    if type(hcoord).__module__.startswith("xgcm"):
        _reject_legacy((hcoord,), "to_grid", "Use xroms.to_grid(var, hcoord=..., scoord=...).")
    var = canonicalize(var)
    hcoord = normalize_hcoord(hcoord)
    scoord = normalize_scoord(scoord)
    if hcoord is not None:
        eta_t, xi_t = CANONICAL[hcoord]
        for axis, target in (("X", xi_t), ("Y", eta_t)):
            dim, _ = _xgcm.axis_dim(var, axis)
            if dim is not None and dim != target:
                var = _xgcm.interp(var, axis, boundary=hboundary, fill_value=hfill_value)
    if scoord is not None:
        dim, _ = _xgcm.axis_dim(var, "Z")
        if dim is not None and dim != scoord:
            var = _xgcm.interp(var, "Z", boundary=sboundary, fill_value=sfill_value)
    if attrs is not None:
        var = var.copy(deep=False)
        var.attrs = dict(attrs)
        if "name" in attrs:
            var.name = attrs["name"]
    return order(var)


def to_rho(var, *args, hboundary="extend", hfill_value=np.nan):
    """Move ``var`` to rho points horizontally."""
    _reject_legacy(args, "to_rho", "Use xroms.to_rho(var).")
    return to_grid(var, hcoord="rho", hboundary=hboundary, hfill_value=hfill_value)


def to_u(var, *args, hboundary="extend", hfill_value=np.nan):
    """Move ``var`` to u points horizontally."""
    _reject_legacy(args, "to_u", "Use xroms.to_u(var).")
    return to_grid(var, hcoord="u", hboundary=hboundary, hfill_value=hfill_value)


def to_v(var, *args, hboundary="extend", hfill_value=np.nan):
    """Move ``var`` to v points horizontally."""
    _reject_legacy(args, "to_v", "Use xroms.to_v(var).")
    return to_grid(var, hcoord="v", hboundary=hboundary, hfill_value=hfill_value)


def to_psi(var, *args, hboundary="extend", hfill_value=np.nan):
    """Move ``var`` to psi points horizontally."""
    _reject_legacy(args, "to_psi", "Use xroms.to_psi(var).")
    return to_grid(var, hcoord="psi", hboundary=hboundary, hfill_value=hfill_value)


def to_s_rho(var, *args, sboundary="extend", sfill_value=np.nan):
    """Move ``var`` to rho (layer-centre) levels vertically."""
    _reject_legacy(args, "to_s_rho", "Use xroms.to_s_rho(var).")
    return to_grid(var, scoord="s_rho", sboundary=sboundary, sfill_value=sfill_value)


def to_s_w(var, *args, sboundary="extend", sfill_value=np.nan):
    """Move ``var`` to w (layer-interface) levels vertically."""
    _reject_legacy(args, "to_s_w", "Use xroms.to_s_w(var).")
    return to_grid(var, scoord="s_w", sboundary=sboundary, sfill_value=sfill_value)


# --- derivatives -------------------------------------------------------------------


def _derivative_attrs(var, what, attrs):
    if attrs is not None:
        return dict(attrs)
    out = dict(var.attrs)
    name = var.name if var.name is not None else "var"
    out["name"] = f"d{name}{what}"
    out["units"] = "1/m * " + str(var.attrs.get("units", "units"))
    long = var.attrs.get("long_name", name)
    out["long_name"] = {
        "dxi": f"horizontal xi derivative of {long}",
        "deta": f"horizontal eta derivative of {long}",
        "dz": f"vertical derivative of {long}",
    }[what]
    return out


def _finish(result, attrs, hcoord, scoord, hboundary, hfill_value, sboundary, sfill_value):
    result = to_grid(
        result, hcoord, scoord,
        hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value,
    )
    result.attrs = attrs
    result.name = attrs.get("name", result.name)
    return result


def _gradient_kernel(f, z, mode, fill_value):
    """d f / d z along the last axis at the same points (non-uniform, second order).

    Interior: 3-point stencil exact for quadratics. Edges: second-order one-sided
    (``mode="extend"``, needs >= 3 points; first order with 2) or ``fill_value``.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)
    out = np.full(np.broadcast_shapes(f.shape, z.shape), np.nan)
    f, z = np.broadcast_to(f, out.shape), np.broadcast_to(z, out.shape)
    n = out.shape[-1]
    if n < 2:
        return out
    if n == 2:
        g = (f[..., 1] - f[..., 0]) / (z[..., 1] - z[..., 0])
        out[..., 0] = out[..., 1] = g if mode == "extend" else fill_value
        return out
    hm = z[..., 1:-1] - z[..., :-2]
    hp = z[..., 2:] - z[..., 1:-1]
    out[..., 1:-1] = (hm**2 * (f[..., 2:] - f[..., 1:-1]) + hp**2 * (f[..., 1:-1] - f[..., :-2])) / (hm * hp * (hm + hp))
    if mode == "extend":
        for edge, (i0, i1, i2) in (("lo", (0, 1, 2)), ("hi", (-1, -2, -3))):
            d1 = z[..., i1] - z[..., i0]
            d2 = z[..., i2] - z[..., i0]
            # quadratic through three points, derivative at the first
            out[..., i0] = ((f[..., i1] - f[..., i0]) * d2**2 - (f[..., i2] - f[..., i0]) * d1**2) / (d1 * d2 * (d2 - d1))
    else:
        out[..., 0] = fill_value
        out[..., -1] = fill_value
    return out


def _ddz_same_levels(var, zz, sboundary, sfill_value):
    """d var / d z on var's own vertical levels (no staggering).

    Returns ``var``'s coords (and the index coords of ``zz``) on the result.
    """
    dim = vposition(var)
    var, zz = xr.align(var, zz.reset_coords(drop=True), join="exact", copy=False)
    # bare arrays into apply_ufunc, coords back on afterwards: before xarray 2025.8
    # apply_ufunc strips the attrs of the coordinates it merges in place, which
    # would empty the attrs of the user's lon_rho etc.
    work = xr.DataArray(var.variable)
    zwork = xr.DataArray(zz.variable)
    work = work.chunk({dim: -1}) if work.chunks is not None else work
    zwork = zwork.chunk({dim: -1}) if zwork.chunks is not None else zwork
    out = xr.apply_ufunc(
        _gradient_kernel,
        work,
        zwork,
        kwargs={"mode": sboundary, "fill_value": sfill_value},
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[float],
    )
    if var.chunks is not None:
        out = out.chunk({dim: var.chunksizes[dim]})
    out = out.transpose(*[d for d in var.dims if d in out.dims], ...)
    return out.assign_coords({**zz.coords, **var.coords})


@with_grid_coords
def ddz(
    var,
    grid=None,
    *,
    z=None,
    zeta=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    attrs=None,
):
    """Vertical derivative ``d var / dz`` [var units / m].

    By default the result lands on the other vertical grid (``s_rho`` -> ``s_w``,
    ``s_w`` -> ``s_rho``). Asking for the input's own levels (e.g.
    ``scoord="s_rho"`` for a rho-level input) uses a second-order stencil on those
    levels rather than averaging the staggered result.

    ``z`` (at ``var``'s position) or ``grid`` (to compute it, with ``zeta``) is
    required. ``var`` needs at least 2 vertical levels.
    """
    grid = _check_grid(grid, "ddz")
    var = canonicalize(var)
    vpos = vposition(var)
    if vpos is None:
        raise ValueError(f"{var.name!r} has no vertical dimension")
    if var.sizes[vpos] < 2:
        raise ValueError(f"ddz needs at least 2 levels ({vpos} has length {var.sizes[vpos]})")
    if z is None and grid is None:
        raise ValueError("ddz needs z= (depths at var's points) or grid= (to compute them)")
    zz = z_like(var, grid, zeta=zeta, z=z)
    new_attrs = _derivative_attrs(var, "dz", attrs)
    scoord = normalize_scoord(scoord)
    if scoord == vpos:
        result = _ddz_same_levels(var, zz, sboundary, sfill_value)
        scoord = None
    else:
        # slopes, not differences, so that the edges are the derivative (extend) or sfill_value (fill)
        den = _xgcm.diff(zz.reset_coords(drop=True), "Z", boundary="extend")
        result = _xgcm.diff(var, "Z", boundary=sboundary, fill_value=sfill_value, spacing=den)
    return _finish(result, new_attrs, hcoord, scoord, hboundary, hfill_value, sboundary, sfill_value)


def _spacing_at(grid, name, like, dest):
    """Grid spacing ``1 / pm`` (or ``1 / pn``) on position ``dest``, over ``like``'s footprint."""
    require(grid, name, purpose="horizontal derivatives")
    field = grid[name]
    field = field.reset_coords(drop=True) if field.coords else field
    field = select_like(field, like, name=name)
    if dest in ("u", "psi"):
        field = _xgcm.interp(field, "X")
    if dest in ("v", "psi"):
        field = _xgcm.interp(field, "Y")
    return 1.0 / field


def _flip(pos, axis):
    eta, xi = CANONICAL[pos]
    if axis == "X":
        xi = "xi_u" if xi == "xi_rho" else "xi_rho"
    else:
        eta = "eta_v" if eta == "eta_rho" else "eta_rho"
    return {pair: p for p, pair in CANONICAL.items()}[(eta, xi)]


def _single_level(var, grid):
    """Was ``var`` cut out of a 3-D field at one s-level (so it has no vertical dim)?

    ``isel(s_rho=k)`` leaves a scalar s coordinate behind when the file has s labels
    (Rutgers, REMORA). UCLA and CROCO files have none, so there the level is
    recognised by ``var`` having the name of a variable of ``grid`` that has a
    vertical dim, and no dim that variable lacks (other than a horizontal one: the
    level may have been moved to other points). A z-slice, say, has a dim of its own
    and is at constant depth already.
    """
    if any(c in var.coords and var[c].ndim == 0 for c in ("s_rho", "s_w")):
        return True
    if var.name not in grid.variables:
        return False
    dims = grid.variables[var.name].dims
    return any(d in ("s_rho", "s_w") for d in dims) and set(var.dims) <= set(dims) | _HORIZONTAL


def _hderivative(var, grid, axis, *, z, zeta, hcoord, scoord, hboundary, hfill_value, sboundary, sfill_value, along_s, attrs, func):
    grid = _check_grid(grid, func)
    var = canonicalize(var)
    if grid is None:
        raise ValueError(f"{func} needs grid= (a Dataset with pm/pn, h and the s-coordinate parameters)")
    pos = hposition(var)
    if pos is None:
        raise ValueError(f"cannot tell the horizontal grid position of {var.name!r} from dims {var.dims}")
    dest = _flip(pos, axis)
    vpos = vposition(var)
    levels = 0 if vpos is None else var.sizes[vpos]
    if vpos is None and not along_s and _single_level(var, grid):
        raise ValueError(
            f"{var.name!r} is a single selected s-level of a 3-D field. A horizontal derivative "
            "along that s-surface is not a derivative at constant depth; compute on the 3-D "
            "field and then select the level, or pass along_s=True to accept the along-s derivative "
            "(which is also the plain derivative you want for a field that is not on an s-surface, "
            "such as a depth average)."
        )
    if vpos is not None and levels < 2 and not along_s:
        raise ValueError(
            "the chain rule needs at least 2 vertical levels; pass along_s=True for the derivative "
            f"along the single layer ({vpos} has length {levels})"
        )
    chain_rule = levels >= 2
    # name everything that is missing at once, not one variable per attempt
    require(grid, "pm" if axis == "X" else "pn", *(["h"] if chain_rule and z is None else []), purpose=func)
    spacing = _spacing_at(grid, "pm" if axis == "X" else "pn", var, dest)
    new_attrs = _derivative_attrs(var, "dxi" if axis == "X" else "deta", attrs)
    # slopes, not differences, so that "extend" copies the derivative at the edges
    if not chain_rule:
        result = _xgcm.diff(var, axis, boundary=hboundary, fill_value=hfill_value, spacing=spacing)
    else:
        zz = z_like(var, grid, zeta=zeta, z=z).reset_coords(drop=True)
        dqds = _xgcm.diff(var, axis, boundary=hboundary, fill_value=hfill_value, spacing=spacing)
        dzds = _xgcm.diff(zz, axis, boundary=hboundary, fill_value=hfill_value, spacing=spacing)
        dqdz = _ddz_same_levels(var, zz, sboundary, sfill_value)
        dqdz = _xgcm.interp(dqdz, axis, boundary=hboundary, fill_value=hfill_value)
        result = dqds - dqdz * dzds
    return _finish(result, new_attrs, hcoord, scoord, hboundary, hfill_value, sboundary, sfill_value)


@with_grid_coords
def ddxi(
    var,
    grid=None,
    *,
    z=None,
    zeta=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    along_s=False,
    attrs=None,
):
    """Horizontal derivative along xi at constant depth [var units / m].

    ``(dq/dxi)_z = (dq/dxi)_s - (dq/dz) (dz/dxi)_s``, evaluated at the staggered
    point (a rho-point input lands on u points) on the input's own vertical
    levels, so it needs at least 2 of them. ``grid`` supplies ``pm`` and, for 3-D
    inputs, ``h``/``zeta``/s-params (or pass ``z``). Variables without a vertical
    dim get a plain derivative, except a single selected s-level (known by its
    scalar s coordinate, or on UCLA and CROCO output, which have none, by carrying
    the name of a 3-D variable of ``grid``) and a vertical dim of length 1: those
    raise unless ``along_s=True``, which takes the derivative along the layer (also
    the way to take the plain derivative of, say, a depth average that kept its name).
    """
    return _hderivative(
        var, grid, "X", z=z, zeta=zeta, hcoord=hcoord, scoord=scoord, hboundary=hboundary,
        hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value, along_s=along_s,
        attrs=attrs, func="ddxi",
    )


@with_grid_coords
def ddeta(
    var,
    grid=None,
    *,
    z=None,
    zeta=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    along_s=False,
    attrs=None,
):
    """Horizontal derivative along eta at constant depth [var units / m]; see :func:`ddxi`."""
    return _hderivative(
        var, grid, "Y", z=z, zeta=zeta, hcoord=hcoord, scoord=scoord, hboundary=hboundary,
        hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value, along_s=along_s,
        attrs=attrs, func="ddeta",
    )


@with_grid_coords
def hgrad(var, grid=None, which="both", **kwargs):
    """Both horizontal derivatives (``which="both"``), or ``"xi"``/``"eta"`` only."""
    if which == "xi":
        return ddxi(var, grid, **kwargs)
    if which == "eta":
        return ddeta(var, grid, **kwargs)
    if which == "both":
        return ddxi(var, grid, **kwargs), ddeta(var, grid, **kwargs)
    raise ValueError(f"which must be 'both', 'xi' or 'eta', not {which!r}")


# --- grid-weighted sums and means ----------------------------------------------------


_AXIS_NAMES = {"X": "X", "Y": "Y", "Z": "Z", "xi_rho": "X", "xi_u": "X", "eta_rho": "Y", "eta_v": "Y", "s_rho": "Z", "s_w": "Z"}


def _grid_weights(var, grid, dims, zeta):
    """Metric weights (dx, dy and/or dz at var's position) and the dims to reduce."""
    var = canonicalize(var)
    items = [dims] if isinstance(dims, str) else list(dims)
    axes = []
    for d in items:
        if d not in _AXIS_NAMES:
            raise ValueError(f"dims must be axis letters X/Y/Z or ROMS dim names, not {d!r}")
        axes.append(_AXIS_NAMES[d])
    pos = hposition(var) or "rho"
    weight = xr.DataArray(1.0)
    reduce = []
    for axis in axes:
        if axis == "X":
            weight = weight * _spacing_at(grid, "pm", var, pos)
            reduce.append(_xgcm.axis_dim(var, "X")[0])
        elif axis == "Y":
            weight = weight * _spacing_at(grid, "pn", var, pos)
            reduce.append(_xgcm.axis_dim(var, "Y")[0])
        else:
            vpos = vposition(var)
            if vpos is None:
                raise ValueError(f"{var.name!r} has no vertical dimension")
            weight = weight * _dz(grid, hcoord=pos, scoord=vpos, zeta=zeta, like=var).reset_coords(drop=True)
            reduce.append(vpos)
    if any(r is None for r in reduce):
        raise ValueError(f"{var.name!r} lacks a dimension for axes {axes}")
    return var, weight, reduce


@with_grid_coords
def gridsum(var, grid, dims, *, zeta=None):
    """Grid-weighted sum over ``dims`` (axis letters ``X``/``Y``/``Z`` or dim names).

    Multiplies by the grid spacing at ``var``'s position (``dx``, ``dy``, ``dz``)
    before summing: e.g. ``gridsum(u, ds, "Z")`` is depth-integrated u.
    """
    grid = _check_grid(grid, "gridsum")
    var, weight, reduce = _grid_weights(var, grid, dims, zeta)
    out = (var * weight).sum(reduce)
    out.attrs = dict(var.attrs)
    out.attrs["long_name"] = f"{var.attrs.get('long_name', var.name)}, grid sum over {', '.join(reduce)}"
    return order(out.rename(var.name))


@with_grid_coords
def gridmean(var, grid, dims, *, zeta=None):
    """Grid-weighted mean over ``dims``; NaN points (e.g. land) carry no weight."""
    grid = _check_grid(grid, "gridmean")
    var, weight, reduce = _grid_weights(var, grid, dims, zeta)
    weight = weight.broadcast_like(var).where(var.notnull(), 0.0)
    total = weight.sum(reduce)
    out = (var * weight).sum(reduce) / total.where(total > 0)
    out.attrs = dict(var.attrs)
    out.attrs["long_name"] = f"{var.attrs.get('long_name', var.name)}, grid mean over {', '.join(reduce)}"
    return order(out.rename(var.name))


# --- ordering, selection, subsetting ---------------------------------------------------


def order(var):
    """Transpose to (time, vertical, eta, xi) followed by any other dims."""
    tdim = time_dim(var)
    lead = [tdim, "s_rho", "s_w", "eta_rho", "eta_u", "eta_v", "eta_psi", "eta_vert", "xi_rho", "xi_u", "xi_v", "xi_psi", "xi_vert"]
    front = [d for d in lead if d is not None and d in var.dims]
    return var.transpose(*front, ...)


R_EARTH = 6371315.0  # m


def _haversine(lon, lat, lon0, lat0):
    lon, lat, lon0, lat0 = map(np.deg2rad, (lon, lat, lon0, lat0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _canonical_xy(lons, lats):
    """``lons`` and ``lats`` with canonical dim names (plain arrays pass through)."""
    return tuple(canonicalize(a) if isinstance(a, xr.DataArray) else a for a in (lons, lats))


def argsel2d(lons, lats, lon0, lat0, *, method="haversine"):
    """Index (or indices) of the grid point(s) nearest to ``(lon0, lat0)``.

    ``method``: ``"haversine"`` (great circle on a sphere, default),
    ``"geodesic"`` (WGS84 via pyproj), or ``"cartesian"`` (x/y grids such as
    REMORA's). ``lon0``/``lat0`` may be scalars or 1-D arrays; the result is a
    tuple of indices in ``lons``' shape (arrays for array input).
    """
    lons, lats = (np.asarray(a) for a in _canonical_xy(lons, lats))
    pts_lon, pts_lat = np.atleast_1d(np.asarray(lon0, dtype=float)), np.atleast_1d(np.asarray(lat0, dtype=float))
    flat_lon, flat_lat = lons.reshape(-1), lats.reshape(-1)
    idx = np.empty(pts_lon.size, dtype=int)
    if method == "geodesic":
        try:
            import pyproj
        except ImportError:
            raise ModuleNotFoundError(
                'method="geodesic" needs pyproj: pip install "xroms[geodesic]" or conda install pyproj'
            ) from None
        geod = pyproj.Geod(ellps="WGS84")
    for i, (x0, y0) in enumerate(zip(pts_lon, pts_lat)):
        if method == "haversine":
            dist = _haversine(flat_lon, flat_lat, x0, y0)
        elif method == "geodesic":
            _, _, dist = geod.inv(flat_lon, flat_lat, np.full_like(flat_lon, x0), np.full_like(flat_lat, y0))
        elif method == "cartesian":
            dist = np.hypot(flat_lon - x0, flat_lat - y0)
        else:
            raise ValueError(f"method must be 'haversine', 'geodesic' or 'cartesian', not {method!r}")
        idx[i] = int(np.nanargmin(dist))
    inds = np.unravel_index(idx, lons.shape)
    if np.ndim(lon0) == 0:
        return tuple(int(i[0]) for i in inds)
    return inds


def sel2d(var, lons, lats, lon0, lat0, **kwargs):
    """Values of ``var`` at the grid point(s) nearest to ``(lon0, lat0)``.

    ``lons``/``lats`` are the 2-D coordinates at ``var``'s grid position; their
    dims name the dims of ``var`` to index. Rutgers-style names (``eta_u``,
    ``xi_v``, ...) may be mixed with canonical ones: all are canonicalized first.
    """
    if not isinstance(var, xr.DataArray):
        raise TypeError("var must be a DataArray")
    var = canonicalize(var)
    lons, lats = _canonical_xy(lons, lats)
    inds = argsel2d(lons, lats, lon0, lat0, **kwargs)
    dims = lons.dims if isinstance(lons, xr.DataArray) else [d for d in var.dims if d.startswith(("eta", "xi"))][-2:]
    if np.ndim(lon0) == 0:
        return var.isel({dims[0]: inds[0], dims[1]: inds[1]})
    points = xr.DataArray(np.arange(len(inds[0])), dims="points")
    return var.isel({dims[0]: xr.DataArray(inds[0], dims="points"), dims[1]: xr.DataArray(inds[1], dims="points")}).assign_coords(points=points)


def _check_slice(sl, name, n):
    if sl is None:
        return None
    if not isinstance(sl, slice):
        raise TypeError(f"{name} must be a slice, e.g. slice(20, 40)")
    if sl.step not in (None, 1):
        raise ValueError(
            f"{name}={sl}: strided subsets break the staggered-grid relationships and are not "
            "supported; subset contiguously, then thin the result yourself"
        )
    start, stop, _ = sl.indices(n)
    if stop - start < 2:
        raise ValueError(f"{name}={sl} keeps fewer than 2 rho points")
    return start, stop


_X_DIMS = {"center": ("xi_rho", "xi_v"), "inner": ("xi_u", "xi_psi"), "corner": ("xi_vert",)}
_Y_DIMS = {"center": ("eta_rho", "eta_u"), "inner": ("eta_v", "eta_psi"), "corner": ("eta_vert",)}


def subset(ds, X=None, Y=None, *, halo=0):
    """Subset horizontally by rho indices, keeping every stagger consistent.

    ``X``/``Y`` are contiguous slices of rho indices (``None``/negative bounds
    allowed; steps other than 1 raise). u/v/psi dims keep the points strictly
    between the kept rho points (the ROMS relationship), and corner dims the
    points around them. ``halo=n`` keeps ``n`` extra rho points on each side
    (clipped at the domain edge) so that staggered operations on the subset are
    exact; remove it afterwards with :func:`trim`. Metadata-only (lazy).
    """
    isel = {}
    record = {}
    for axis, sl, dims in (("X", X, _X_DIMS), ("Y", Y, _Y_DIMS)):
        center = next((d for d in dims["center"] if d in ds.dims), None)
        if sl is None or center is None:
            continue
        n = ds.sizes[center]
        start, stop = _check_slice(sl, axis, n)
        lo, hi = max(start - halo, 0), min(stop + halo, n)
        record[axis] = (start - lo, hi - stop)
        for d in dims["center"]:
            if d in ds.dims:
                isel[d] = slice(lo, hi)
        for d in dims["inner"]:
            if d in ds.dims:
                isel[d] = slice(lo, hi - 1)
        for d in dims["corner"]:
            if d in ds.dims:
                isel[d] = slice(lo, hi + 1)
    out = ds.isel(isel)
    if halo:
        out.attrs = dict(out.attrs)
        out.attrs["xroms_halo"] = [record.get("X", (0, 0))[0], record.get("X", (0, 0))[1], record.get("Y", (0, 0))[0], record.get("Y", (0, 0))[1]]
    return out


def trim(obj, n=None):
    """Remove a halo added by :func:`subset` (or ``n`` points from every horizontal edge)."""
    if n is None:
        halo = obj.attrs.get("xroms_halo")
        if halo is None:
            raise ValueError("no halo recorded on this object; pass n=")
        xl, xr_, yl, yr = (int(v) for v in halo)
    else:
        xl = xr_ = yl = yr = int(n)
    isel = {}
    for dims, left, right in ((_X_DIMS, xl, xr_), (_Y_DIMS, yl, yr)):
        for group in dims.values():
            for d in group:
                if d in obj.dims:
                    isel[d] = slice(left, obj.sizes[d] - right if right else None)
    out = obj.isel(isel)
    if n is None:
        out.attrs = {k: v for k, v in out.attrs.items() if k != "xroms_halo"}
    return out


def xisoslice(iso_array, iso_value, projected_array, coord):
    """Calculate an isosurface.

    This function has been possibly superseded by isoslice
    that wraps `xgcm.grid.transform` for the following reasons,
    but more testing is needed:

    * The implementation of `xgcm.grid.transform` is more robust
      than `xisoslice` which has extra code for in case iso_value
      is exactly in iso_array.
    * For a 5-day model file, the run time for the same call for
      was approximately the same for xisolice and isoslice.
    * isoslice might be more computationally robust for not
      breaking mid-way, but this is still unclear.

    This function calculates the value of projected_array on
    an isosurface in the array iso_array defined by iso_value.

    Parameters
    ----------
    iso_array: DataArray, ndarray
        Array in which the isosurface is defined
    iso_value: float
        Value of the isosurface in iso_array
    projected_array: DataArray, ndarray
        Array in which to project values on the isosurface. This can have
        multiple time outputs. Needs to be broadcastable from iso_array?
    coord: string
        Name of coordinate associated with the dimension along which to project

    Returns
    -------
    DataArray or ndarray of values of projected_array on the isosurface

    Notes
    -----
    Performs lazy evaluation.

    `xisoslice` requires that iso_array be monotonic. If iso_value is not monotonic
    it will still run but values may be incorrect where not monotonic.
    If iso_value is exactly in iso_array or the value is passed twice in iso_array,
    a message will be printed. iso_value is changed a tiny amount in this case to
    account for it being in iso_array exactly. The latter case is not deal with.

    Examples
    --------

    Calculate lat-z slice of salinity along a constant longitude value (-91.5):

    >>> sl = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.salt, 'xi_rho')

    Calculate a lon-lat slice at a constant z value (-10):

    >>> sl = xroms.utilities.xisoslice(xroms.z(ds), -10, ds.temp, 's_rho')

    Calculate a lon-lat slice at a constant z value (-10) but without zeta changing in time:

    (resting heights, relative to mean sea level and not varying in time)
    >>> sl = xroms.utilities.xisoslice(xroms.z(ds, zeta=0), -10, ds.temp, 's_rho')

    Calculate the depth of a specific isohaline (33):

    >>> sl = xroms.utilities.xisoslice(ds.salt, 33, xroms.z(ds), 's_rho')

    Calculate the salt 10 meters above the seabed. Either do this on the vertical
    rho grid, or first change to the w grid and then use `xisoslice`. You may prefer
    to do the latter if there is a possibility that the distance above the seabed you are
    interpolating to (10 m) could be below the deepest rho grid depth.

    * on rho grid directly:

      >>> sl = xroms.xisoslice(xroms.z(ds, reference="bottom"), 10., ds.salt, 's_rho')

    * on w grid:

      >>> var_w = xroms.to_s_w(ds.salt)
      >>> sl = xroms.xisoslice(xroms.z(ds, scoord="s_w", reference="bottom"), 10., var_w, 's_w')

    In addition to calculating the slices themselves, you may need to calculate
    related coordinates for plotting. For example, to accompany the lat-z slice,
    you may want the following:

    calculate z values (s_rho)

    >>> slz = xroms.utilities.xisoslice(ds.lon_rho, -91.5, xroms.z(ds), 'xi_rho')

    calculate latitude values (eta_rho)

    >>> sllat = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.lat_rho, 'xi_rho')

    assign these as coords to be used in plot

    >>> sl = sl.assign_coords(z=slz, lat=sllat)

    points that should be masked

    >>> slmask = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.mask_rho, 'xi_rho')

    drop masked values

    >>> sl = sl.where(slmask==1, drop=True)
    """

    # length of the projected coordinate, minus one
    Nm = len(iso_array[coord]) - 1

    # A 'lower' slice including all but the last value, and an
    # 'upper' slice including all but the first value
    lslice = {coord: slice(None, -1)}
    uslice = {coord: slice(1, None)}

    # prop is now the array on which to calculate the isosurface, with
    # the iso_value subtracted so that the isosurface is defined by
    # prop == 0
    prop = iso_array - iso_value

    # propl are the prop values in the lower slice
    propl = prop.isel(**lslice)
    propl.coords[coord] = np.arange(Nm)
    # propu in the upper slice
    propu = prop.isel(**uslice)
    propu.coords[coord] = np.arange(Nm)

    # Find the location where prop changes sign, meaning it bounds the
    # desired isosurface. zc has a length of Nm in the projected dimension
    # and may be considered to be an array in between the values in the
    # projected dimension. zc==1 means the prop changed signs crossing this
    # value, so that the isovalue occurs between those two values.
    zc = xr.where((propu * propl) <= 0.0, 1.0, 0.0)

    # Get the upper and lower slices of the array that will be projected
    # on the isosurface
    varl = projected_array.isel(**lslice)
    varl.coords[coord] = np.arange(Nm)
    varu = projected_array.isel(**uslice)
    varu.coords[coord] = np.arange(Nm)

    # propl*zc extracts the value of prop below the iso_surface.
    # propu*zc above. Extract similar values for the projected array.
    propl = (propl * zc).sum(coord)
    propu = (propu * zc).sum(coord)
    varl = (varl * zc).sum(coord)
    varu = (varu * zc).sum(coord)

    # A linear fit to of the projected array to the isosurface; NaN where there is
    # no crossing (propu == propl == 0), without dividing by zero (dask would warn)
    den = propu - propl
    out = varl - propl * (varu - varl) / den.where(den != 0)

    # If the sum == 2, that means iso_value is exactly in iso_array
    check = zc.sum(coord) == 2

    # it's too slow for large arrays to check this, so just always
    # divide and it will happen where necessary.
    # where iso_value is located in iso_array, divide result by 2
    out = xr.where(check, out / 2, out)

    return out
