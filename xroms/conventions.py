"""ROMS-family conventions, detected feature by feature.

Covers Rutgers ROMS, UCLA ROMS (and roms-tools grids), CROCO and REMORA. Nothing
here computes data; every function reads names and metadata only.

Canonical dims (what xroms computes in, and what pure functions return):

========  =========================
position  dims
========  =========================
rho       ``(eta_rho, xi_rho)``
u         ``(eta_rho, xi_u)``
v         ``(eta_v, xi_rho)``
psi       ``(eta_v, xi_u)``
corners   ``(eta_vert, xi_vert)``
vertical  ``s_rho`` (N), ``s_w`` (N+1)
========  =========================

Rutgers ROMS and REMORA name every stagger separately (``eta_u``, ``xi_v``,
``eta_psi``, ``xi_psi``); those aliases are mapped to the canonical names.
"""

import re

from dataclasses import dataclass

import numpy as np
import xarray as xr


ALIASES = {"eta_u": "eta_rho", "xi_v": "xi_rho", "eta_psi": "eta_v", "xi_psi": "xi_u"}
CANONICAL = {
    "rho": ("eta_rho", "xi_rho"),
    "u": ("eta_rho", "xi_u"),
    "v": ("eta_v", "xi_rho"),
    "psi": ("eta_v", "xi_u"),
}
RUTGERS = {
    "rho": ("eta_rho", "xi_rho"),
    "u": ("eta_u", "xi_u"),
    "v": ("eta_v", "xi_v"),
    "psi": ("eta_psi", "xi_psi"),
}
HCOORDS = tuple(CANONICAL)
TIME_NAMES = ("ocean_time", "time", "scrum_time")
_HDIMS = {"xi_rho", "xi_u", "eta_rho", "eta_v"}


def normalize_hcoord(hcoord):
    """Validate a horizontal position name (``rho``, ``u``, ``v``, ``psi``)."""
    if hcoord is None or hcoord in HCOORDS:
        return hcoord
    raise ValueError(f"hcoord must be one of {HCOORDS} or None, not {hcoord!r}")


def normalize_scoord(scoord):
    """Map ``s_rho``/``rho`` to ``s_rho`` and ``s_w``/``w`` to ``s_w``."""
    mapping = {"s_rho": "s_rho", "rho": "s_rho", "s_w": "s_w", "w": "s_w", None: None}
    try:
        return mapping[scoord]
    except (KeyError, TypeError):
        raise ValueError(f"scoord must be 's_rho', 'rho', 's_w', 'w' or None, not {scoord!r}") from None


# --- dims and positions ---------------------------------------------------


def _alias_map(obj):
    """Aliases present in ``obj`` and the canonical names they map to."""
    dims = obj.dims if isinstance(obj, xr.DataArray) else obj.sizes
    mapping = {a: c for a, c in ALIASES.items() if a in dims}
    sizes = dict(obj.sizes)
    # a psi-named dim one larger than rho holds cell corners, not psi points
    for psi, rho, vert in (("xi_psi", "xi_rho", "xi_vert"), ("eta_psi", "eta_rho", "eta_vert")):
        if psi in mapping and rho in sizes and sizes[psi] == sizes[rho] + 1:
            mapping[psi] = vert
    return mapping


def canonicalize(obj):
    """Rename Rutgers/REMORA alias dims to canonical names (metadata only).

    Index coords on alias dims are dropped where a canonical index already
    exists (they describe the same positions). Works on Datasets and DataArrays.
    """
    mapping = _alias_map(obj)
    if not mapping:
        return obj
    drop = [a for a, c in mapping.items() if a in obj.indexes and c in obj.indexes]
    return obj.drop_vars(drop).rename({k: v for k, v in mapping.items()})


def convention(ds):
    """``"rutgers"`` if ``ds`` uses alias dims, else ``"canonical"``."""
    return "rutgers" if any(a in ds.dims for a in ALIASES) else "canonical"


def hposition(da):
    """Horizontal position (``rho``/``u``/``v``/``psi``) of ``da``, or None.

    Scalar coords count too, so a u-point section that was ``isel``-ed along xi
    (keeping a scalar ``xi_u`` label) is still recognised as a u-point field.
    """
    if isinstance(da, xr.DataArray):
        da = canonicalize(da)
        dims = set(da.dims) | {c for c in da.coords if da[c].ndim == 0 and c in _HDIMS}
    else:
        dims = set(da)
    xi = "xi_rho" if "xi_rho" in dims else "xi_u" if "xi_u" in dims else None
    eta = "eta_rho" if "eta_rho" in dims else "eta_v" if "eta_v" in dims else None
    for pos, pair in CANONICAL.items():
        if pair == (eta, xi):
            return pos
    return None


def vposition(da):
    """Vertical position (``s_rho``/``s_w``) of ``da``, or None."""
    return "s_rho" if "s_rho" in da.dims else "s_w" if "s_w" in da.dims else None


def rename_like(da, ds):
    """Return ``da`` using ``ds``'s naming for its grid position.

    Pure functions return canonical names; for Rutgers/REMORA-named data, this
    maps e.g. a u-point result ``(eta_rho, xi_u)`` to ``(eta_u, xi_u)`` so it
    combines with ``ds.u`` without broadcasting. Only names present in ``ds``
    are used.
    """
    if convention(ds) != "rutgers":
        return da
    pos = hposition(da)
    if pos is None or pos == "rho":
        return da
    rename = {}
    for canon, rut in zip(CANONICAL[pos], RUTGERS[pos]):
        if canon != rut and canon in da.dims and rut in ds.dims:
            rename[canon] = rut
    return da.rename(rename) if rename else da


def time_dim(obj):
    """Name of the time dimension of ``obj``, or None."""
    for name in TIME_NAMES:
        if name in obj.dims:
            return name
    for dim in obj.dims:
        if dim in obj.coords and np.issubdtype(obj[dim].dtype, np.datetime64):
            return dim
    return None


# --- SGRID ------------------------------------------------------------------


def sgrid_topology(ds):
    """The SGRID ``grid_topology`` variable's attrs, parsed, or None."""
    for name, var in ds.variables.items():
        if var.attrs.get("cf_role") == "grid_topology":
            parsed = {"variable": name}
            for key in ("face_dimensions", "edge1_dimensions", "edge2_dimensions", "vertical_dimensions", "node_dimensions"):
                if key in var.attrs:
                    parsed[key] = var.attrs[key]
            return parsed
    return None


def sgrid_attrs(ds):
    """SGRID topology attributes describing ``ds``'s own dim names."""
    rut = convention(ds) == "rutgers"
    if rut:
        face = "xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: both)"
        edge1 = "xi_u: xi_psi eta_u: eta_psi (padding: both)"
        edge2 = "xi_v: xi_psi (padding: both) eta_v: eta_psi"
        node = "xi_psi eta_psi"
    else:
        face = "xi_rho: xi_u (padding: both) eta_rho: eta_v (padding: both)"
        edge1 = "xi_u: xi_u eta_rho: eta_v (padding: both)"
        edge2 = "xi_rho: xi_u (padding: both) eta_v: eta_v"
        node = "xi_u eta_v"
    attrs = {
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": node,
        "face_dimensions": face,
        "edge1_dimensions": edge1,
        "edge2_dimensions": edge2,
    }
    if "s_rho" in ds.dims and "s_w" in ds.dims:
        attrs["vertical_dimensions"] = "s_rho: s_w (padding: none)"
    return attrs


# --- horizontal coordinates -------------------------------------------------


def horizontal_coords(obj, hcoord="rho"):
    """``(x, y)`` coordinate names at ``hcoord``: lon/lat, else Cartesian x/y.

    ``obj`` may be a Dataset or a DataArray (its coords are searched).
    """
    names = set(obj.variables) if isinstance(obj, xr.Dataset) else set(obj.coords)
    for xname, yname in ((f"lon_{hcoord}", f"lat_{hcoord}"), (f"x_{hcoord}", f"y_{hcoord}")):
        if xname in names and yname in names:
            return xname, yname
    return None, None


def is_spherical(ds):
    """True when lon/lat are available (rather than Cartesian x/y only)."""
    return "lon_rho" in ds.variables and "lat_rho" in ds.variables


# --- vertical parameters ------------------------------------------------------


def stretching(sigma, theta_s, theta_b, Vstretching=4):
    """ROMS vertical stretching curve ``C(sigma)``.

    ``Vstretching=4`` is Shchepetkin & McWilliams (2009), the UCLA/roms-tools
    default; ``Vstretching=2`` is Shchepetkin (2005).
    """
    sigma = xr.DataArray(sigma) if not isinstance(sigma, xr.DataArray) else sigma
    if Vstretching == 4:
        if theta_s > 0:
            c = (1.0 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1.0)
        else:
            c = -(sigma**2)
        if theta_b > 0:
            c = (np.exp(theta_b * c) - 1.0) / (1.0 - np.exp(-theta_b))
        return c
    if Vstretching == 2:
        alfa, beta = 1.0, 1.0
        if theta_s > 0:
            csur = (1.0 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1.0)
            if theta_b > 0:
                cbot = -1.0 + np.sinh(theta_b * (sigma + 1.0)) / np.sinh(theta_b)
                weight = (sigma + 1.0) ** alfa * (1.0 + (alfa / beta) * (1.0 - (sigma + 1.0) ** beta))
                return weight * csur + (1.0 - weight) * cbot
            return csur
        return sigma
    raise ValueError(f"Vstretching={Vstretching} is not supported (use 2 or 4, or provide Cs_r/Cs_w)")


def sigma_levels(N, loc="rho"):
    """Evenly spaced sigma values: ``s_rho`` (N, mid-layer) or ``s_w`` (N+1)."""
    if loc in ("rho", "s_rho", "r"):
        return xr.DataArray((np.arange(N) - N + 0.5) / N, dims="s_rho", name="s_rho")
    if loc in ("w", "s_w"):
        return xr.DataArray((np.arange(N + 1) - N) / N, dims="s_w", name="s_w")
    raise ValueError(f"loc must be 'rho' or 'w', not {loc!r}")


@dataclass(frozen=True)
class VerticalParams:
    """Terrain-following vertical coordinate parameters of a ROMS dataset."""

    Vtransform: int
    hc: float
    Cs_r: xr.DataArray
    Cs_w: xr.DataArray
    sigma_r: xr.DataArray
    sigma_w: xr.DataArray


def _sources(ds, grid):
    return [s for s in (ds, grid) if s is not None]


def _var(sources, *names):
    for src in sources:
        for name in names:
            if name in src.variables:
                return src[name].reset_coords(drop=True) if name not in src.dims else src[name].variable
    return None


def _attr(sources, *names):
    for src in sources:
        for name in names:
            if name in src.attrs:
                return src.attrs[name]
    return None


def _scalar(sources, name):
    value = _var(sources, name)
    if value is not None:
        return float(np.asarray(value.values if hasattr(value, "values") else value).reshape(-1)[0])
    value = _attr(sources, name)
    if value is not None:
        return float(np.asarray(value).reshape(-1)[0])
    return None


def _levels(sources, dim):
    for src in sources:
        if dim in src.dims:
            return src.sizes[dim]
    return None


def _as_level_array(value, dim):
    if isinstance(value, xr.Variable):
        return xr.DataArray(value.values, dims=dim)
    if isinstance(value, xr.DataArray):
        arr = value.reset_coords(drop=True)
        return arr.rename({arr.dims[0]: dim}) if arr.dims and arr.dims[0] != dim else arr
    return xr.DataArray(np.asarray(value, dtype=float).reshape(-1), dims=dim)


def vertical_params(ds, grid=None, *, Vtransform=None):
    """Find the s-coordinate parameters of ``ds`` (optionally with ``grid``).

    Looks, per item, in variables then global attributes of ``ds`` and then
    ``grid``, covering Rutgers ROMS, UCLA ROMS output and roms-tools grids,
    CROCO and REMORA:

    * sigma: ``s_rho``/``s_w`` coordinate values, then ``sigma_r``/``sigma_w``,
      then ``sc_r``/``sc_w``; otherwise computed from the number of levels.
    * ``Cs_r``/``Cs_w``: variables, then attributes; otherwise computed from
      ``theta_s``/``theta_b`` with ``Vstretching`` (4 unless stated).
    * ``hc``: variable or attribute.
    * ``Vtransform``: the argument, a variable, an attribute, ``VertCoordType``
      (``"NEW"`` -> 2, ``"OLD"`` -> 1), else 2 for UCLA-style files (stretching
      only in attributes, or roms-tools ``sigma_r`` variables) and REMORA.
    """
    src = _sources(ds, grid)
    N = _levels(src, "s_rho")
    if N is None:
        raise ValueError("dataset has no 's_rho' dimension; it is not a 3-D ROMS dataset")

    sigma_r = _var(src, "s_rho", "sigma_r", "sc_r")
    sigma_w = _var(src, "s_w", "sigma_w", "sc_w")
    sigma_r = sigma_levels(N, "rho") if sigma_r is None else _as_level_array(sigma_r, "s_rho")
    sigma_w = sigma_levels(N, "w") if sigma_w is None else _as_level_array(sigma_w, "s_w")

    theta_s, theta_b = _scalar(src, "theta_s"), _scalar(src, "theta_b")
    vstretching = _scalar(src, "Vstretching")
    vstretching = 4 if vstretching is None else int(vstretching)

    cs_attr_only = False
    cs = {}
    for name, sig, dim in (("Cs_r", sigma_r, "s_rho"), ("Cs_w", sigma_w, "s_w")):
        value = _var(src, name)
        if value is None:
            value = _attr(src, name)
            cs_attr_only = cs_attr_only or value is not None
        if value is None:
            if theta_s is None or theta_b is None:
                raise ValueError(f"cannot find {name} (variable or attribute) or theta_s/theta_b to compute it")
            value = stretching(sig, theta_s, theta_b, vstretching)
        cs[name] = _as_level_array(value, dim)

    hc = _scalar(src, "hc")
    if hc is None:
        raise ValueError("cannot find the critical depth 'hc' (variable or attribute)")

    vt = Vtransform
    if vt is None:
        vt = _scalar(src, "Vtransform")
    if vt is None:
        vct = _attr(src, "VertCoordType")
        if vct is not None:
            vt = {"NEW": 2, "OLD": 1}.get(str(vct).strip().upper())
    if vt is None and (cs_attr_only or _var(src, "sigma_r") is not None or any(sgrid_topology(s) for s in src)):
        vt = 2  # UCLA ROMS output / roms-tools grids / REMORA only use Vtransform 2
    if vt is None:
        raise ValueError("cannot determine Vtransform; pass Vtransform=1 or 2")
    vt = int(vt)
    if vt not in (1, 2):
        raise ValueError(f"Vtransform must be 1 or 2, not {vt}")

    for name, arr, dim in (("Cs_r", cs["Cs_r"], "s_rho"), ("Cs_w", cs["Cs_w"], "s_w"), ("sigma_r", sigma_r, "s_rho"), ("sigma_w", sigma_w, "s_w")):
        n_data = _levels(src, dim)
        if n_data is not None and arr.sizes[dim] != n_data:
            raise ValueError(
                f"{name} has {arr.sizes[dim]} levels but the data have {n_data} along {dim!r}. "
                "If the data were subset vertically, subset the Dataset (not just the variable) "
                "so the vertical parameters stay matched, or pass them explicitly."
            )
    return VerticalParams(Vtransform=vt, hc=hc, Cs_r=cs["Cs_r"], Cs_w=cs["Cs_w"], sigma_r=sigma_r, sigma_w=sigma_w)


def rho0(ds, grid=None, default=1025.0):
    """Reference density: ``rho0`` variable, then attribute, then ``default``."""
    value = _scalar(_sources(ds, grid), "rho0")
    return default if value is None else value


# --- time ---------------------------------------------------------------------


_EPOCH = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_UNIT_SECONDS = {"second": 1, "seconds": 1, "s": 1, "sec": 1, "day": 86400, "days": 86400}


def _parse_epoch(text):
    match = _EPOCH.search(text or "")
    if not match:
        return None
    y, m, d, hh, mm, ss = match.groups()
    return np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(hh or 0):02d}:{int(mm or 0):02d}:{int(ss or 0):02d}")


def decode_time(ds, reference_date=None, time_var=None):
    """Return ``ds`` with a decoded datetime index on its time dimension.

    For UCLA ROMS output, the ``time`` dim has no coordinate and ``ocean_time``
    holds seconds with the epoch only in its ``long_name`` ("Time since
    1995/01/01"). The epoch is parsed from ``long_name``/``units`` (or taken
    from ``reference_date``; if both exist they must agree) and a datetime64
    coordinate is attached on the time dim; ``ocean_time`` is kept unchanged.
    Datasets whose time is already decoded are returned as is.
    """
    tdim = time_dim(ds)
    if tdim is None:
        raise ValueError("no time dimension found")
    if tdim in ds.coords and np.issubdtype(ds[tdim].dtype, np.datetime64):
        return ds
    name = time_var
    if name is None:
        candidates = [v for v in ("ocean_time", tdim, "scrum_time") if v in ds.variables and ds[v].dims == (tdim,)]
        if not candidates:
            raise ValueError(f"no time variable on dim {tdim!r}")
        name = candidates[0]
    var = ds[name]
    if np.issubdtype(var.dtype, np.datetime64):
        return ds.assign_coords({tdim: var.variable})
    units = str(var.attrs.get("units", "seconds")).strip().lower()
    epoch = _parse_epoch(var.attrs.get("long_name")) or _parse_epoch(units)
    if reference_date is not None:
        ref = np.datetime64(reference_date, "s")
        if epoch is not None and ref != epoch:
            raise ValueError(f"reference_date {ref} disagrees with the file's epoch {epoch}")
        epoch = ref
    if epoch is None:
        raise ValueError(f"cannot find a reference date for {name!r}; pass reference_date=")
    unit = units.split(" since ")[0].strip()
    scale = _UNIT_SECONDS.get(unit)
    if scale is None:
        raise ValueError(f"unsupported time units {units!r}")
    seconds = np.asarray(var.values, dtype="float64") * scale
    times = epoch.astype("datetime64[ns]") + (seconds * 1e9).astype("timedelta64[ns]")
    return ds.assign_coords({tdim: (tdim, times, {"long_name": "time"})})


# --- CF / SGRID decoration ------------------------------------------------------


def add_cf_attrs(ds, *, index_coords=False, sgrid=True):
    """Return a copy of ``ds`` decorated for cf-xarray and xgcm (metadata only).

    * ``axis``/``standard_name`` attributes on the coordinates that exist;
    * an SGRID ``grid`` topology variable if none is present (``sgrid=True``);
    * integer index coords on horizontal dims **only if** ``index_coords=True``
      and the dim has none (existing labels are never renumbered).
    """
    ds = ds.copy()
    for pos in HCOORDS:
        for suffix, (std, units) in {"lon": ("longitude", "degrees_east"), "lat": ("latitude", "degrees_north")}.items():
            name = f"{suffix}_{pos}"
            if name in ds.variables:
                ds[name].attrs.setdefault("standard_name", std)
                ds[name].attrs.setdefault("units", units)
        for suffix, std in {"x": "projection_x_coordinate", "y": "projection_y_coordinate"}.items():
            name = f"{suffix}_{pos}"
            if name in ds.variables:
                ds[name].attrs.setdefault("standard_name", std)
    for dim in ("s_rho", "s_w"):
        if dim in ds.coords:
            ds[dim].attrs.setdefault("axis", "Z")
    tdim = time_dim(ds)
    if tdim is not None and tdim in ds.coords:
        ds[tdim].attrs.setdefault("axis", "T")
        ds[tdim].attrs.setdefault("standard_name", "time")
    if index_coords:
        for dim in [d for d in ds.dims if d.startswith(("xi_", "eta_"))]:
            if dim not in ds.coords:
                ds = ds.assign_coords({dim: (dim, np.arange(ds.sizes[dim]), {"axis": "X" if dim.startswith("xi") else "Y"})})
    if sgrid and sgrid_topology(ds) is None:
        ds["grid"] = ((), 0, sgrid_attrs(ds))
    return ds
