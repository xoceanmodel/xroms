"""Horizontal grid metrics, areas, volumes and masks at any stagger, on demand."""

import numpy as np
import xarray as xr

from . import _xgcm
from ._align import require, select_like
from .conventions import canonicalize, normalize_hcoord


def _at(field, hcoord):
    """Average a rho-point field onto ``hcoord``."""
    if hcoord in ("u", "psi"):
        field = _xgcm.interp(field, "X")
    if hcoord in ("v", "psi"):
        field = _xgcm.interp(field, "Y")
    return field


def _metric(grid, name, hcoord, like):
    grid = canonicalize(grid)
    require(grid, name, purpose="grid metrics")
    field = grid[name]
    field = field.reset_coords(drop=True) if field.coords else field
    if like is not None:
        field = select_like(field, like, name=name)
    return _at(field, normalize_hcoord(hcoord) or "rho")


def dx(grid, hcoord="rho", *, like=None):
    """Grid spacing along xi (metres), ``1 / pm`` averaged onto ``hcoord``."""
    out = 1.0 / _metric(grid, "pm", hcoord, like)
    out.attrs = {"units": "m", "long_name": f"grid spacing in xi at {hcoord or 'rho'} points"}
    return out.rename(f"dx_{hcoord or 'rho'}")


def dy(grid, hcoord="rho", *, like=None):
    """Grid spacing along eta (metres), ``1 / pn`` averaged onto ``hcoord``."""
    out = 1.0 / _metric(grid, "pn", hcoord, like)
    out.attrs = {"units": "m", "long_name": f"grid spacing in eta at {hcoord or 'rho'} points"}
    return out.rename(f"dy_{hcoord or 'rho'}")


def dA(grid, hcoord="rho", *, like=None):
    """Cell area (m²) at ``hcoord``: ``dx * dy``."""
    out = dx(grid, hcoord, like=like) * dy(grid, hcoord, like=like)
    out.attrs = {"units": "m2", "long_name": f"cell area at {hcoord or 'rho'} points"}
    return out.rename(f"dA_{hcoord or 'rho'}")


def dV(grid, hcoord="rho", scoord="s_rho", *, zeta=None, like=None):
    """Cell volume (m³) at (``hcoord``, ``scoord``): ``dz * dA``."""
    from .vertical import dz

    out = dz(grid, hcoord=hcoord, scoord=scoord, zeta=zeta, like=like) * dA(grid, hcoord, like=like)
    out.attrs = {"units": "m3", "long_name": f"cell volume at {hcoord or 'rho'}/{scoord} points"}
    return out.rename(f"dV_{scoord}_{hcoord or 'rho'}")


R_EARTH = 6371315.0  # m, as in ROMS and roms-tools


def nominal_resolution(grid, units="m", lat=None):
    """Mean grid spacing ``(mean(1/pm) + mean(1/pn)) / 2``.

    ``units="degrees"`` converts to degrees of longitude at ``lat`` (default: the
    middle of the domain's latitude range), as roms-tools does.
    """
    grid = canonicalize(grid)
    require(grid, "pm", "pn", purpose="nominal resolution")
    res = float(((1.0 / grid.pm).mean() + (1.0 / grid.pn).mean()) / 2)
    if units == "m":
        return res
    if units in ("degrees", "deg"):
        if lat is None:
            require(grid, "lat_rho", purpose="resolution in degrees")
            lat = float((grid.lat_rho.max() + grid.lat_rho.min()) / 2)
        meters_per_degree = 2 * np.pi * R_EARTH / 360
        return res / (meters_per_degree * np.cos(np.deg2rad(lat)))
    raise ValueError(f"units must be 'm' or 'degrees', not {units!r}")


def mask_at(mask_rho, hcoord):
    """Land/sea mask at ``hcoord``: water only where every neighbouring rho point is water.

    Equivalent to the products ROMS uses (``mask_u = mask_rho[i] * mask_rho[i+1]``),
    for 0/1 masks of any dtype, including time-varying masks.
    """
    hcoord = normalize_hcoord(hcoord)
    mask = canonicalize(mask_rho)
    if hcoord in (None, "rho"):
        return mask
    avg = _at(mask.astype(float), hcoord)
    out = xr.where(avg == 1.0, 1, 0).astype(mask.dtype)
    out.attrs = {"long_name": f"mask on {hcoord}-points", "flag_values": [0, 1], "flag_meanings": "land water"}
    return out.rename(f"mask_{hcoord}")
