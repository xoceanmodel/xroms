"""Vertical coordinates of terrain-following ROMS grids, computed on demand.

Nothing is stored: every call derives z from the primitives in hand (``h``,
``zeta``, and the s-coordinate parameters found by
:func:`xroms.conventions.vertical_params`).

Vertical references (``reference=``) and signs (``positive=``):

====================  ====================================================
``"mean_sea_level"``  height relative to the model's zero level (ROMS z)
``"surface"``         relative to the moving free surface (``z - zeta``)
``"bottom"``          relative to the seabed (``z + h``)
====================  ====================================================

``positive="up"`` gives heights (negative below the reference);
``positive="down"`` gives depths. Every output is labelled with ``units``,
``positive``, a CF ``standard_name`` where one exists, and
``vertical_reference``.
"""

import warnings

import numpy as np
import xarray as xr

from . import _xgcm
from ._align import _check_grid, GridMismatchError, is_time_varying, level_positions, require, select_like, with_grid_coords
from .conventions import (
    canonicalize,
    free_surface_name,
    hposition,
    normalize_hcoord,
    normalize_scoord,
    time_dim,
    vertical_params,
    vposition,
)


REFERENCES = ("mean_sea_level", "surface", "bottom")

#: (reference, positive) -> CF standard_name
STANDARD_NAMES = {
    ("mean_sea_level", "up"): "height_above_mean_sea_level",
    ("mean_sea_level", "down"): "depth_below_geoid",
    ("surface", "up"): "height",
    ("surface", "down"): "depth",
    ("bottom", "up"): "height_above_sea_floor",
    ("bottom", "down"): "depth_below_sea_floor",
}
_FROM_STANDARD_NAME = {v: k for k, v in STANDARD_NAMES.items()}
_FROM_STANDARD_NAME.update(
    {
        "height_above_geoid": ("mean_sea_level", "up"),
        "altitude": ("mean_sea_level", "up"),
        "depth_below_mean_sea_level": ("mean_sea_level", "down"),
        "sea_floor_depth_below_geoid": ("mean_sea_level", "down"),
    }
)


def _check_reference(reference, positive):
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, not {reference!r}")
    if positive not in ("up", "down"):
        raise ValueError(f"positive must be 'up' or 'down', not {positive!r}")


def infer_reference(attrs, default=("mean_sea_level", "up")):
    """``(reference, positive)`` implied by CF attrs, or ``default``.

    Uses ``vertical_reference`` (xroms' own label), then ``standard_name``, then
    ``positive``. Contradictory metadata raises ``ValueError``.
    """
    ref = attrs.get("vertical_reference")
    pos = attrs.get("positive")
    std = attrs.get("standard_name")
    from_std = _FROM_STANDARD_NAME.get(std)
    if from_std is not None:
        if ref is not None and ref != from_std[0]:
            raise ValueError(f"vertical_reference={ref!r} contradicts standard_name={std!r}")
        if pos is not None and pos != from_std[1]:
            raise ValueError(f"positive={pos!r} contradicts standard_name={std!r}")
        return from_std
    if ref is None and pos is None:
        return default
    return (ref or default[0], pos or default[1])


def label(da, reference="mean_sea_level", positive="up", name=None, long_name=None):
    """``da`` labelled with xroms' vertical metadata (returns a new object).

    The attrs are replaced, not extended: whatever ``da`` carried (typically
    what arithmetic inherited from ``h``) describes another quantity.
    """
    _check_reference(reference, positive)
    attrs = {"units": "m", "positive": positive, "vertical_reference": reference}
    std = STANDARD_NAMES.get((reference, positive))
    if std:
        attrs["standard_name"] = std
    if long_name:
        attrs["long_name"] = long_name
    out = da.copy(deep=False)
    out.attrs = attrs
    if name is not None:
        out = out.rename(name)
    return out


def _order(da):
    tdim = time_dim(da)
    front = [d for d in (tdim, "s_rho", "s_w", "eta_rho", "eta_v", "xi_rho", "xi_u") if d is not None and d in da.dims]
    rest = [d for d in da.dims if d not in front]
    return da.transpose(*front, *rest)


def _hc_above_h(h, hc):
    """``(hc, min(h))`` if ``hc`` exceeds the shallowest ``h``, else None.

    Dask-backed input is skipped (None): the check would have to compute it.
    """
    if any(getattr(a, "chunks", None) is not None for a in (h, hc)):
        return None
    h = np.asarray(h, dtype=float)
    if h.size == 0 or np.isnan(h).all():
        return None
    h_min, hc = float(np.nanmin(h)), float(np.max(np.asarray(hc, dtype=float)))
    return (hc, h_min) if hc > h_min else None


def compute_depth(h, zeta=0, *, hc, Cs, sigma, Vtransform=2, positive="up"):
    """ROMS z at the points of ``h``/``zeta`` and levels of ``Cs``/``sigma``.

    Pure arithmetic on anything broadcastable (full fields, slices, single
    columns, scalars). ``positive="down"`` returns depth below the model's zero
    level (the roms-tools convention). Lazy for dask inputs.

    Vtransform 1 gives non-monotonic z (layers of negative thickness) where
    ``hc`` exceeds the water depth, a configuration ROMS refuses to run. A
    ``UserWarning`` flags it when ``h`` is in memory; dask-backed ``h`` is not
    checked, since that would compute it.
    """
    if Vtransform == 1:
        above = _hc_above_h(h, hc)
        if above is not None:
            warnings.warn(
                f"Vtransform 1 needs hc <= min(h) for monotonic depths, but hc={above[0]:g} m exceeds "
                f"the shallowest h={above[1]:g} m, so z is non-monotonic (layers get negative "
                "thickness) where the water is shallower than hc. ROMS stops on this configuration: "
                "lower hc (or theta_s/theta_b), or use Vtransform 2.",
                UserWarning,
                stacklevel=2,
            )
        zo = hc * (sigma - Cs) + Cs * h
        z = zo + zeta * (1.0 + zo / h)
    elif Vtransform == 2:
        zo = (hc * sigma + Cs * h) / (hc + h)
        z = zeta + (zeta + h) * zo
    else:
        raise ValueError(f"Vtransform must be 1 or 2, not {Vtransform}")
    if isinstance(z, xr.DataArray):
        z = _order(z)
        if positive == "down":
            z = -z
        return label(z, "mean_sea_level", positive, long_name="vertical position (ROMS z)")
    return -z if positive == "down" else z


def _resolve_zeta(grid, zeta, like=None):
    """The free-surface field to use, already matched to ``like`` if given."""
    if isinstance(zeta, xr.DataArray):
        zeta = canonicalize(zeta)
        if hposition(zeta) not in (None, "rho"):
            raise ValueError(
                f"zeta is on {hposition(zeta)} points; pass it on rho points, as ROMS stores it "
                "(z is averaged onto other points from there)."
            )
        return select_like(zeta, like, name="zeta") if like is not None else zeta
    if zeta is None:
        name = free_surface_name(grid)
        if name is None:
            if like is not None and is_time_varying(like):
                raise GridMismatchError(
                    f"{like.name or 'variable'!r} varies in time but the grid has no 'zeta' (nor a CF "
                    "sea_surface_height_above_geoid), so its "
                    "depths would silently assume a flat free surface. Choose the free surface "
                    "explicitly: zeta=0 (static, resting depths), zeta=<DataArray> (e.g. the output's "
                    "zeta), or merge the output and grid Datasets so the grid has a zeta, e.g. "
                    "xroms.merge_grid(out, grid)."
                )
            return 0.0
        field = grid[name]
        return select_like(field, like, name="zeta") if like is not None else canonicalize(field)
    if isinstance(zeta, str):
        if zeta != "mean":
            raise ValueError(f"zeta must be None, a number, 'mean', or a DataArray, not {zeta!r}")
        name = free_surface_name(grid)
        if name is None:
            require(grid, "zeta", purpose="zeta='mean'")
        field = canonicalize(grid[name])
        tdim = time_dim(field)
        field = field.mean(tdim) if tdim else field
        return select_like(field, like, name="zeta") if like is not None else field
    return float(zeta)


def _to_hcoord(field, hcoord):
    """Average a rho-point field onto ``hcoord``."""
    if hcoord in ("u", "psi"):
        field = _xgcm.interp(field, "X")
    if hcoord in ("v", "psi"):
        field = _xgcm.interp(field, "Y")
    return field


@with_grid_coords
def z(
    grid,
    *,
    hcoord="rho",
    scoord="s_rho",
    zeta=None,
    reference="mean_sea_level",
    positive="up",
    method="average",
    like=None,
    Vtransform=None,
):
    """Vertical position of every point of ``grid`` at (``hcoord``, ``scoord``).

    Parameters
    ----------
    grid : Dataset
        Holds ``h`` (and ``zeta`` unless ``zeta`` is given) plus the s-coordinate
        parameters, in any ROMS-family layout.
    hcoord, scoord : str
        Horizontal (``rho``/``u``/``v``/``psi``) and vertical (``s_rho``/``s_w``)
        position.
    zeta : None, float, "mean", or DataArray
        Free surface: the grid's ``zeta`` (None; if the grid has none, 0 for a
        ``like`` without time and an error for a time-varying one), a constant
        (``0`` gives static depths), its time mean, or an explicit field. With
        ``like``, an explicit field is matched to it like any grid field.
    reference, positive : str
        See the module docstring.
    method : "average" or "interp_inputs"
        Away from rho points, ``"average"`` computes z at rho points and averages
        it onto ``hcoord`` (ROMS' own practice); ``"interp_inputs"`` averages ``h``
        and ``zeta`` first (roms-tools' practice).
    like : DataArray, optional
        Restrict the computation to the footprint, times and vertical levels of
        this variable. Levels are matched by label where both have an index, else
        they must be equally many.
    """
    hcoord = normalize_hcoord(hcoord) or "rho"
    scoord = normalize_scoord(scoord) or "s_rho"
    _check_reference(reference, positive)
    if method not in ("average", "interp_inputs"):
        raise ValueError(f"method must be 'average' or 'interp_inputs', not {method!r}")
    grid = _check_grid(grid, "z")
    require(grid, "h", purpose="depths")
    params = vertical_params(grid, Vtransform=Vtransform)
    cs, sigma = (params.Cs_r, params.sigma_r) if scoord == "s_rho" else (params.Cs_w, params.sigma_w)

    rho_like = canonicalize(like) if like is not None else None
    if rho_like is not None:
        # a variable cut vertically on its own has fewer levels than the parameters describe
        levels = level_positions(rho_like, scoord, cs.sizes[scoord], grid.indexes.get(scoord))
        if levels is not None:
            cs, sigma = cs.isel({scoord: levels}), sigma.isel({scoord: levels})
    h = grid["h"].reset_coords(drop=True) if "h" in grid.coords else grid["h"]
    h = select_like(h, rho_like, name="h") if rho_like is not None else h
    zeta_field = _resolve_zeta(grid, zeta, rho_like)

    if hcoord != "rho" and method == "interp_inputs":
        h = _to_hcoord(h, hcoord)
        if isinstance(zeta_field, xr.DataArray):
            zeta_field = _to_hcoord(zeta_field, hcoord)
    out = compute_depth(h, zeta_field, hc=params.hc, Cs=cs, sigma=sigma, Vtransform=params.Vtransform)
    if hcoord != "rho" and method == "average":
        out = _to_hcoord(out, hcoord)

    if reference == "surface":
        zeta_here = zeta_field if isinstance(zeta_field, xr.DataArray) else zeta_field
        if isinstance(zeta_here, xr.DataArray) and hcoord != "rho" and method == "average":
            zeta_here = _to_hcoord(zeta_here, hcoord)
        out = out - zeta_here
    elif reference == "bottom":
        h_here = h if (hcoord == "rho" or method == "interp_inputs") else _to_hcoord(h, hcoord)
        out = out + h_here
    if positive == "down":
        out = -out
    out = _order(out)
    name = {"s_rho": "z_rho", "s_w": "z_w"}[scoord] + ("" if hcoord == "rho" else f"_{hcoord}")
    return label(out, reference, positive, name=name, long_name=f"vertical position at {hcoord}/{scoord} points")


_z = z  # z_like's ``z=`` argument shadows the function name


def z_like(var, grid, *, zeta=None, z=None, reference="mean_sea_level", positive="up", method="average"):
    """z at the grid position of ``var``, restricted to its footprint, times and levels.

    A given ``z`` must be on ``var``'s vertical levels, and at its horizontal
    points or at rho points (then averaged onto ``var``'s points, as `z` does).
    It is matched to ``var`` like any grid field (see
    :func:`xroms._align.select_like`): by labels where both have them, through
    ``var``'s scalar coords for dims ``var`` was selected along, and otherwise
    by size, with an error that says to subset the Dataset instead.
    """
    var = canonicalize(var)
    if z is not None:
        z = canonicalize(z)
        have, want = hposition(z), hposition(var)
        if have is not None and want is not None and have != want and have != "rho":
            raise ValueError(
                f"z is at {have} points but {var.name!r} is at {want} points. Pass z at rho "
                "points (it is averaged onto the variable's points) or at the variable's own points."
            )
        zlev, vlev = vposition(z), vposition(var)
        if zlev is not None and vlev is not None and zlev != vlev:
            raise ValueError(f"z is on {zlev} levels but {var.name!r} is on {vlev} levels; pass z on the variable's own levels.")
        z = select_like(z, var, name="z")
        if have is not None and want is not None and have != want:
            if want in ("u", "psi"):
                z = _xgcm.interp(z, "X")
            if want in ("v", "psi"):
                z = _xgcm.interp(z, "Y")
        return z
    hcoord = hposition(var) or "rho"
    scoord = vposition(var)
    if scoord is None:
        # a single selected level: z on all levels, then the one its scalar coord names
        level = next((d for d in ("s_rho", "s_w") if d in var.coords and var[d].ndim == 0), None)
        if level is None or level not in getattr(grid, "indexes", {}):
            raise ValueError(
                f"{var.name!r} has no vertical dimension, or is one level without an s-coordinate "
                "label to find it by, so there is no depth to compute for it. For one selected "
                "level, compute on the 3-D fields and select afterwards (e.g. "
                "xroms.density(ds.temp, ds.salt, grid=ds).isel(s_rho=-1)), or pass z= (e.g. "
                "xroms.surface(xroms.z(ds)))."
            )
        full = _z(
            grid, hcoord=hcoord, scoord=level, zeta=zeta, reference=reference,
            positive=positive, method=method, like=var,
        )
        return select_like(full, var, name="z")
    return _z(
        grid, hcoord=hcoord, scoord=scoord, zeta=zeta, reference=reference,
        positive=positive, method=method, like=var,
    )


@with_grid_coords
def dz(grid, *, hcoord="rho", scoord="s_rho", zeta=None, method="average", like=None):
    """Layer thicknesses (positive, metres).

    On ``s_rho`` this is ``diff(z_w)``. On ``s_w`` it is the spacing between
    neighbouring rho levels, with the half cells ``z_rho[0] - z_w[0]`` at the
    bottom and ``z_w[N] - z_rho[N-1]`` at the top, so w-level sums and means are
    correct at the boundaries.

    With ``like``, the footprint and times are restricted to that variable's, and
    so are its levels when it is on ``scoord`` (matched by label where both have an
    index, else they must be equally many): a variable cut vertically on its own
    gets the thicknesses of its own layers.
    """
    grid = _check_grid(grid, "dz")
    scoord = normalize_scoord(scoord) or "s_rho"
    like = canonicalize(like) if like is not None else None
    # differences and half cells need every level of the grid: select the levels afterwards
    vdim = vposition(like) if like is not None else None
    like_hz = like.isel({vdim: 0}, drop=True) if vdim is not None else like
    kwargs = dict(hcoord=hcoord, zeta=zeta, method=method, like=like_hz)
    z_w = z(grid, scoord="s_w", **kwargs)
    if scoord == "s_rho":
        out = _xgcm.diff(z_w, "Z")
    else:
        z_r = z(grid, scoord="s_rho", **kwargs)
        interior = _xgcm.diff(z_r, "Z", boundary="fill")
        n = interior.sizes["s_w"]
        k = xr.DataArray(np.arange(n), dims="s_w")
        bottom = (z_r.isel(s_rho=0) - z_w.isel(s_w=0)).reset_coords(drop=True)
        top = (z_w.isel(s_w=-1) - z_r.isel(s_rho=-1)).reset_coords(drop=True)
        out = xr.where(k == 0, bottom, xr.where(k == n - 1, top, interior))
        out = _order(out)
    if like is not None:
        levels = level_positions(like, scoord, out.sizes[scoord], grid.indexes.get(scoord))
        if levels is not None:
            out = out.isel({scoord: levels})
    out.attrs = {"units": "m", "long_name": f"layer thickness at {hcoord}/{scoord} points"}
    return out.rename(f"dz_{scoord}" if hcoord == "rho" else f"dz_{scoord}_{hcoord}")


def _level(var, last):
    """The top (``last=True``) or bottom layer of ``var``, marked with the level it is.

    Labels along the vertical dim stay behind as a scalar coord, as ``isel`` leaves
    them. Without any (UCLA, CROCO), the level's non-negative index is left instead,
    so later calls (``utilities._single_level``, ``select_like``) can tell a single
    selected level from a field that never had a vertical dim.
    """
    var = canonicalize(var)
    dim = vposition(var)
    if dim is None:
        raise ValueError(f"{var.name!r} has no vertical dimension")
    index = var.sizes[dim] - 1 if last else 0
    out = var.isel({dim: index})
    if dim not in out.coords:
        out = out.assign_coords({dim: index})
    return out


def surface(var):
    """The top (surface) layer of ``var``.

    The level stays marked by a scalar ``s_rho``/``s_w`` coord: its label, or its
    index when the vertical dim has no labels.
    """
    return _level(var, last=True)


def bottom(var):
    """The bottom layer of ``var``; marked like :func:`surface`."""
    return _level(var, last=False)


def depth_band_weights(z_w, shallow, deep, *, dim_w="s_w", dim_rho="s_rho"):
    """Thickness of each layer inside a depth band (metres), on the rho levels.

    ``z_w`` holds interface positions relative to the mean sea level or the free
    surface, e.g. ``xroms.z(grid, scoord="w", reference="surface")`` for bands
    measured below the moving free surface. Its ``positive`` attr says which way
    it counts (up if it has none); a reference relative to the seabed, or one this
    function cannot interpret, raises. ``shallow``/``deep`` are depths (positive
    down) below that same reference, so ``shallow=0, deep=10`` is its upper 10 m.
    Layers outside the band get weight 0.
    """
    reference, positive = infer_reference(z_w.attrs)
    _check_reference(reference, positive)
    if reference == "bottom":
        raise ValueError(
            "depth_band_weights measures depth below a reference, but z_w is relative to the seabed "
            "(reference='bottom'), so there is no depth below it. Pass z_w relative to the "
            "mean sea level or the free surface: xroms.z(grid, scoord='w', reference='surface')."
        )
    # positional pairing of interfaces: drop sigma labels so they don't misalign
    depth_w = (z_w if positive == "down" else -z_w).drop_vars(dim_w, errors="ignore")
    upper = depth_w.isel({dim_w: slice(1, None)}).rename({dim_w: dim_rho})
    lower = depth_w.isel({dim_w: slice(None, -1)}).rename({dim_w: dim_rho})
    overlap = (lower.clip(max=float(deep)) - upper.clip(min=float(shallow))).clip(min=0.0)
    overlap.attrs = {"units": "m", "long_name": f"layer thickness within {shallow}-{deep} m"}
    return overlap.rename("dz_band")


@with_grid_coords
def depth_average(var, grid, *, shallow=None, deep=None, zeta=None, reference="mean_sea_level"):
    """Thickness-weighted vertical mean of a rho-level ``var``.

    With no band limits this is the full-column (barotropic) average. Band limits
    are depths (positive down) below ``reference``: use ``reference="surface"``
    for "the upper 10 m of the water column" (``shallow=0, deep=10``). A ``var``
    cut vertically on its own is averaged over its own layers (see :func:`dz`).
    """
    grid = _check_grid(grid, "depth_average")
    var = canonicalize(var)
    if "s_rho" not in var.dims:
        raise ValueError("depth_average needs a variable on s_rho levels")
    z_w = z(grid, hcoord=hposition(var) or "rho", scoord="s_w", zeta=zeta, reference=reference, like=var)
    if shallow is None and deep is None:
        w = _xgcm.diff(z_w, "Z")
    else:
        w = depth_band_weights(z_w, 0.0 if shallow is None else shallow, np.inf if deep is None else deep)
    levels = level_positions(var, "s_rho", w.sizes["s_rho"], grid.indexes.get("s_rho"))
    if levels is not None:
        w = w.isel(s_rho=levels)
    # missing values (land, a missing level) carry no weight; no weight at all (land, a band
    # below the bottom) gives NaN, divided as NaN so that dask does not warn about 0/0
    w = w.where(var.notnull(), 0.0)
    total = w.sum("s_rho")
    out = (var * w).sum("s_rho") / total.where(total > 0)
    out.attrs = dict(var.attrs)
    out.attrs["long_name"] = f"depth average of {var.attrs.get('long_name', var.name)}"
    return out.rename(var.name)
