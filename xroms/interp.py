"""Interpolation: onto iso-surfaces or depths (vertical), and to lon/lat points."""

import numpy as np
import xarray as xr

from . import _xgcm
from .conventions import canonicalize, hposition, horizontal_coords, vposition
from .utilities import order
from .vertical import infer_reference, label, z_like


_AXIS_DIMS = {"Z": ("s_rho", "s_w"), "X": ("xi_rho", "xi_u"), "Y": ("eta_rho", "eta_v")}


def _resolve_dim(var, dim):
    if dim is None:
        dim = "Z"
    if dim in _AXIS_DIMS:
        found = [d for d in _AXIS_DIMS[dim] if d in var.dims]
        if not found:
            raise ValueError(f"{var.name!r} has no {dim!r} dimension")
        return found[0]
    if dim not in var.dims:
        raise ValueError(f"{var.name!r} has no dimension {dim!r}")
    return dim


def _nearest(var, iso_values, iso_array, dim, new_dim, mask_edges):
    values = iso_values if isinstance(iso_values, xr.DataArray) else xr.DataArray(
        np.atleast_1d(np.asarray(iso_values, dtype=float)), dims=[new_dim]
    )
    if new_dim not in values.dims:
        values = values.expand_dims(new_dim)
    arr = iso_array.reset_coords(drop=True)
    work_var = var.chunk({dim: -1}) if var.chunks is not None else var
    arr = arr.chunk({dim: -1}) if arr.chunks is not None else arr
    dist = abs(arr - values)
    idx = dist.fillna(np.inf).argmin(dim)
    out = work_var.isel({dim: idx})
    if mask_edges:
        inside = (values >= arr.min(dim)) & (values <= arr.max(dim))
        out = out.where(inside)
    return out.drop_vars(dim, errors="ignore")


def isoslice(var, iso_values, iso_array, *, dim=None, new_dim=None, method="linear", mask_edges=True):
    """Values of ``var`` where ``iso_array`` equals each of ``iso_values``.

    Parameters
    ----------
    var, iso_array : DataArray
        Share the dimension ``dim`` along which to interpolate (default: the
        vertical; ``"X"``/``"Y"``/``"Z"`` or a dim name).
    iso_values : sequence or DataArray
        Target values; a DataArray may vary in space (N-D targets, e.g. another
        grid's depths), with extra dim ``new_dim``.
    new_dim : str, optional
        Name of the new dimension (default: ``iso_array``'s name, else ``"iso"``).
    method : ``"linear"`` or ``"nearest"``
        Linear interpolation (xgcm ``transform``) or the nearest native level.
    mask_edges : bool
        NaN where a target lies outside the column's range (True), or hold the
        edge value (False).
    """
    var, iso_array = canonicalize(var), canonicalize(iso_array)
    dim = _resolve_dim(var, dim)
    new_dim = new_dim or (iso_array.name if isinstance(iso_array.name, str) and iso_array.name not in var.dims else "iso")
    if method == "linear":
        out = _xgcm.transform(var, iso_values, iso_array, dim, new_dim=new_dim, mask_edges=mask_edges)
    elif method == "nearest":
        out = _nearest(var, iso_values, iso_array, dim, new_dim, mask_edges)
        out.attrs = dict(var.attrs)
    else:
        raise ValueError(f"method must be 'linear' or 'nearest', not {method!r}")
    if not isinstance(iso_values, xr.DataArray):
        out = out.assign_coords({new_dim: np.atleast_1d(np.asarray(iso_values, dtype=float))})
    elif new_dim in iso_values.coords:
        out = out.assign_coords({new_dim: iso_values[new_dim]})
    if isinstance(iso_array, xr.DataArray) and new_dim in out.coords:
        out[new_dim].attrs.update({k: v for k, v in iso_array.attrs.items() if k in ("units", "standard_name", "positive", "vertical_reference", "long_name")})
    return _order_with(out, new_dim, dim)


def _order_with(out, new_dim, replaced):
    """Put ``new_dim`` where the replaced dim would be in (T, Z, Y, X) order."""
    ordered = order(out)
    if replaced in ("s_rho", "s_w") and new_dim in ordered.dims:
        dims = list(ordered.dims)
        dims.remove(new_dim)
        tpos = 1 if dims and dims[0] in ("ocean_time", "time", "scrum_time") else 0
        dims.insert(tpos, new_dim)
        ordered = ordered.transpose(*dims)
    return ordered


def zslice(
    var,
    depths,
    grid=None,
    *,
    z=None,
    zeta=None,
    reference=None,
    positive=None,
    method="linear",
    mask_edges=True,
    new_dim="z",
):
    """Interpolate ``var`` to fixed vertical positions ``depths``.

    ``depths`` are measured relative to ``reference`` (``"mean_sea_level"``,
    ``"surface"`` or ``"bottom"``) with sign ``positive`` (``"up"``/``"down"``).
    If not given they are inferred from CF metadata on ``depths`` (when it is a
    DataArray: ``standard_name``/``positive``), else default to heights relative
    to mean sea level (ROMS z, negative below). E.g. ``zslice(temp, [10],
    ds, reference="surface", positive="down")`` is 10 m below the moving surface.
    """
    var = canonicalize(var)
    attrs = depths.attrs if isinstance(depths, xr.DataArray) else {}
    ref0, pos0 = infer_reference(attrs)
    reference = reference or ref0
    positive = positive or pos0
    if z is None and grid is None:
        raise ValueError("zslice needs grid= (to compute depths) or z=")
    zz = z_like(var, grid, zeta=zeta, z=z, reference=reference, positive=positive)
    if z is not None:
        zz = label(zz, reference, positive) if "vertical_reference" not in zz.attrs else zz
    out = isoslice(var, depths, zz.rename(new_dim), dim=vposition(var), new_dim=new_dim, method=method, mask_edges=mask_edges)
    out[new_dim] = label(out[new_dim], reference, positive)
    return out


def interpll(var, lons, lats, which="pairs", regridder=None, **kwargs):
    """Interpolate ``var`` horizontally to lon/lat points with xESMF.

    ``which="pairs"`` treats ``lons``/``lats`` as point pairs (dim
    ``locations``); ``"grid"`` builds the lat x lon grid. Pass a previously
    returned ``regridder`` (``out.attrs`` is not used for this; keep the object
    from :func:`make_regridder`) to reuse weights across variables and times.
    Extra ``kwargs`` go to ``xesmf.Regridder``.
    """
    var = canonicalize(var)
    if regridder is None:
        regridder = make_regridder(var, lons, lats, which=which, **kwargs)
    src = _xesmf_input(var)
    out = regridder(src, keep_attrs=True)
    if which == "pairs":
        out = out.assign_coords(locations=("locations", np.arange(out.sizes["locations"]), {"axis": "X"}))
    return out


def _xesmf_input(var):
    pos = hposition(var) or "rho"
    xname, yname = horizontal_coords(var, pos)
    if xname is None or not xname.startswith("lon"):
        raise ValueError(f"{var.name!r} needs lon_{pos}/lat_{pos} coordinates for xESMF interpolation")
    return var.rename({xname: "lon", yname: "lat"})


def make_regridder(var, lons, lats, which="pairs", method="bilinear", **kwargs):
    """Build (and return, for reuse) the xESMF regridder used by :func:`interpll`."""
    try:
        import xesmf as xe
    except ImportError:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("interpll needs xESMF (conda install -c conda-forge xesmf)") from None
    var = canonicalize(var)
    lats = np.asarray(lats).flatten()
    lons = np.asarray(lons).flatten()
    if which == "pairs":
        target = xr.Dataset({"lat": (["locations"], lats), "lon": (["locations"], lons)})
        locstream_out = True
    elif which == "grid":
        target = xr.Dataset({"lat": (["lat"], lats), "lon": (["lon"], lons)})
        locstream_out = False
    else:
        raise ValueError(f"which must be 'pairs' or 'grid', not {which!r}")
    return xe.Regridder(_xesmf_input(var), target, method, locstream_out=locstream_out, **kwargs)
