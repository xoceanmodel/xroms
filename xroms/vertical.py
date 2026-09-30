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

import numpy as np
import xarray as xr

from . import _xgcm
from ._align import require, select_like
from .conventions import canonicalize, normalize_hcoord, normalize_scoord, time_dim, vertical_params


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
    """Attach xroms' vertical metadata to ``da`` (returns a new object)."""
    _check_reference(reference, positive)
    attrs = dict(da.attrs)
    attrs.update(units="m", positive=positive, vertical_reference=reference)
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


def compute_depth(h, zeta=0, *, hc, Cs, sigma, Vtransform=2, positive="up"):
    """ROMS z at the points of ``h``/``zeta`` and levels of ``Cs``/``sigma``.

    Pure arithmetic on anything broadcastable (full fields, slices, single
    columns, scalars). ``positive="down"`` returns depth below the model's zero
    level (the roms-tools convention). Lazy for dask inputs.
    """
    if Vtransform == 1:
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
        return label(z, "mean_sea_level", positive)
    return -z if positive == "down" else z


def _resolve_zeta(grid, zeta, like=None):
    """The free-surface field to use, already matched to ``like`` if given."""
    if isinstance(zeta, xr.DataArray):
        return canonicalize(zeta)
    if zeta is None:
        if "zeta" not in grid.variables:
            return 0.0
        field = grid["zeta"]
        return select_like(field, like, name="zeta") if like is not None else canonicalize(field)
    if isinstance(zeta, str):
        if zeta != "mean":
            raise ValueError(f"zeta must be None, a number, 'mean', or a DataArray, not {zeta!r}")
        require(grid, "zeta", purpose="zeta='mean'")
        field = canonicalize(grid["zeta"])
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
        Free surface: the grid's ``zeta`` (None; 0 if absent), a constant (``0``
        gives static depths), its time mean, or an explicit field.
    reference, positive : str
        See the module docstring.
    method : "average" or "interp_inputs"
        Away from rho points, ``"average"`` computes z at rho points and averages
        it onto ``hcoord`` (ROMS' own practice); ``"interp_inputs"`` averages ``h``
        and ``zeta`` first (roms-tools' practice).
    like : DataArray, optional
        Restrict the computation to the footprint and times of this variable.
    """
    hcoord = normalize_hcoord(hcoord) or "rho"
    scoord = normalize_scoord(scoord) or "s_rho"
    _check_reference(reference, positive)
    if method not in ("average", "interp_inputs"):
        raise ValueError(f"method must be 'average' or 'interp_inputs', not {method!r}")
    grid = canonicalize(grid)
    require(grid, "h", purpose="depths")
    params = vertical_params(grid, Vtransform=Vtransform)
    cs, sigma = (params.Cs_r, params.sigma_r) if scoord == "s_rho" else (params.Cs_w, params.sigma_w)

    rho_like = canonicalize(like) if like is not None else None
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
    """z at the grid position of ``var``, restricted to its footprint and times.

    A given ``z`` must be on ``var``'s vertical levels, and at its horizontal
    points or at rho points (then averaged onto ``var``'s points, as `z` does).
    """
    from .conventions import hposition, vposition

    var = canonicalize(var)
    if z is not None:
        z = canonicalize(z)
        have, want = hposition(z), hposition(var)
        if have is not None and want is not None and have != want:
            if have != "rho":
                raise ValueError(
                    f"z is at {have} points but {var.name!r} is at {want} points. Pass z at rho "
                    "points (it is averaged onto the variable's points) or at the variable's own points."
                )
            if want in ("u", "psi"):
                z = _xgcm.interp(z, "X")
            if want in ("v", "psi"):
                z = _xgcm.interp(z, "Y")
        zlev, vlev = vposition(z), vposition(var)
        if zlev is not None and vlev is not None and zlev != vlev:
            raise ValueError(f"z is on {zlev} levels but {var.name!r} is on {vlev} levels; pass z on the variable's own levels.")
        return z
    hcoord = hposition(var) or "rho"
    scoord = vposition(var)
    if scoord is None:
        raise ValueError(f"{var.name!r} has no vertical dimension")
    return _z(
        grid, hcoord=hcoord, scoord=scoord, zeta=zeta, reference=reference,
        positive=positive, method=method, like=var,
    )


def dz(grid, *, hcoord="rho", scoord="s_rho", zeta=None, method="average", like=None):
    """Layer thicknesses (positive, metres).

    On ``s_rho`` this is ``diff(z_w)``. On ``s_w`` it is the spacing between
    neighbouring rho levels, with the half cells ``z_rho[0] - z_w[0]`` at the
    bottom and ``z_w[N] - z_rho[N-1]`` at the top, so w-level sums and means are
    correct at the boundaries.
    """
    scoord = normalize_scoord(scoord) or "s_rho"
    kwargs = dict(hcoord=hcoord, zeta=zeta, method=method, like=like)
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
    out.attrs = {"units": "m", "long_name": f"layer thickness at {hcoord}/{scoord} points"}
    return out.rename(f"dz_{scoord}" if hcoord == "rho" else f"dz_{scoord}_{hcoord}")


def surface(var):
    """The top (surface) layer of ``var``."""
    var = canonicalize(var)
    dim = "s_rho" if "s_rho" in var.dims else "s_w"
    return var.isel({dim: -1})


def bottom(var):
    """The bottom layer of ``var``."""
    var = canonicalize(var)
    dim = "s_rho" if "s_rho" in var.dims else "s_w"
    return var.isel({dim: 0})


def depth_band_weights(z_w, shallow, deep, *, dim_w="s_w", dim_rho="s_rho"):
    """Thickness of each layer inside a depth band (metres), on the rho levels.

    ``z_w`` holds interface heights (positive up) relative to some reference, e.g.
    ``xroms.z(grid, scoord="w", reference="surface")`` for bands measured below
    the moving free surface. ``shallow``/``deep`` are depths (positive down)
    below that same reference, so ``shallow=0, deep=10`` is its upper 10 m.
    Layers outside the band get weight 0.
    """
    # positional pairing of interfaces: drop sigma labels so they don't misalign
    depth_w = -z_w.drop_vars(dim_w, errors="ignore")
    upper = depth_w.isel({dim_w: slice(1, None)}).rename({dim_w: dim_rho})
    lower = depth_w.isel({dim_w: slice(None, -1)}).rename({dim_w: dim_rho})
    overlap = (lower.clip(max=float(deep)) - upper.clip(min=float(shallow))).clip(min=0.0)
    overlap.attrs = {"units": "m", "long_name": f"layer thickness within {shallow}-{deep} m"}
    return overlap.rename("dz_band")


def depth_average(var, grid, *, shallow=None, deep=None, zeta=None, reference="mean_sea_level"):
    """Thickness-weighted vertical mean of a rho-level ``var``.

    With no band limits this is the full-column (barotropic) average. Band limits
    are depths (positive down) below ``reference``: use ``reference="surface"``
    for "the upper 10 m of the water column" (``shallow=0, deep=10``).
    """
    var = canonicalize(var)
    if "s_rho" not in var.dims:
        raise ValueError("depth_average needs a variable on s_rho levels")
    from .conventions import hposition

    z_w = z(grid, hcoord=hposition(var) or "rho", scoord="s_w", zeta=zeta, reference=reference, like=var)
    if shallow is None and deep is None:
        w = _xgcm.diff(z_w, "Z")
    else:
        w = depth_band_weights(z_w, 0.0 if shallow is None else shallow, np.inf if deep is None else deep)
    out = (var * w).sum("s_rho") / w.sum("s_rho")
    out.attrs = dict(var.attrs)
    out.attrs["long_name"] = f"depth average of {var.attrs.get('long_name', var.name)}"
    return out.rename(var.name)
