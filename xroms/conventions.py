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
# attrs of an SGRID topology variable that name dims
_SGRID_DIM_ATTRS = ("face_dimensions", "edge1_dimensions", "edge2_dimensions", "vertical_dimensions", "node_dimensions")
# attr that marks the topology variable add_cf_attrs writes, to tell it from one that came with a file
_GENERATED = "xroms_generated"


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


def _rename_topology(ds, mapping):
    """``ds`` with the dim names in its SGRID topology attrs mapped through ``mapping``."""
    token = re.compile(r"\b(" + "|".join(mapping) + r")\b")
    renamed = {}
    for name, var in ds.variables.items():
        if var.attrs.get("cf_role") == "grid_topology":
            for key in _SGRID_DIM_ATTRS:
                old = var.attrs.get(key)
                new = token.sub(lambda match: mapping[match.group(1)], old) if isinstance(old, str) else old
                if new != old:
                    renamed[name, key] = new
    if renamed:
        ds = ds.copy()  # own attrs dicts, so the caller's Dataset is left alone
        for (name, key), value in renamed.items():
            ds[name].attrs[key] = value
    return ds


def canonicalize(obj):
    """Rename Rutgers/REMORA alias dims to canonical names (metadata only).

    Index coords on alias dims are dropped where a canonical index already
    exists (they describe the same positions). The dim names in an SGRID
    topology variable's attrs are renamed too, so they keep describing the
    dims. Works on Datasets and DataArrays.
    """
    mapping = _alias_map(obj)
    if not mapping:
        return obj
    drop = [a for a, c in mapping.items() if a in obj.indexes and c in obj.indexes]
    out = obj.drop_vars(drop).rename(dict(mapping))
    return _rename_topology(out, mapping) if isinstance(out, xr.Dataset) else out


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
    are used. If ``ds`` has canonical names, ``da`` is canonicalized, so a
    Rutgers-named ``da`` comes back with canonical names too.
    """
    if convention(ds) != "rutgers":
        return canonicalize(da)
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
            for key in _SGRID_DIM_ATTRS:
                if key in var.attrs:
                    parsed[key] = var.attrs[key]
            return parsed
    return None


def _has_file_topology(ds):
    """True if ``ds`` has an SGRID topology variable that did not come from :func:`add_cf_attrs`.

    REMORA writes one, so it says which model made the file. The one xroms adds
    only describes dim names (any Dataset can get it) and says nothing about the model.
    """
    return any(
        var.attrs.get("cf_role") == "grid_topology" and _GENERATED not in var.attrs for var in ds.variables.values()
    )


def sgrid_attrs(ds):
    """SGRID topology attributes describing ``ds``'s own dim names.

    They are marked as written by xroms (``xroms_generated``), so that
    :func:`vertical_params` does not mistake them for a REMORA file's own topology.
    """
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
        _GENERATED: "xroms.add_cf_attrs",
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
    """Terrain-following vertical coordinate parameters of a ROMS dataset.

    The level arrays of a pair :func:`vertical_params` was not asked for (``levels=``) are None.
    """

    Vtransform: int
    hc: float
    Cs_r: xr.DataArray | None = None
    Cs_w: xr.DataArray | None = None
    sigma_r: xr.DataArray | None = None
    sigma_w: xr.DataArray | None = None


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


def _first(value, keep=()):
    """``value`` with every dim not in ``keep`` cut to its first element.

    ``xr.open_mfdataset`` with its default ``data_vars="all"`` gives every
    parameter (``hc``, ``Cs_r``, ...) a leading time dim, one identical copy per
    record. The first copy stands for all of them; comparing them would read every file.
    """
    return value.isel({dim: 0 for dim in value.dims if dim not in keep})


def _scalar(sources, name):
    value = _var(sources, name)
    if value is not None:
        return float(np.asarray(_first(value).values))
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
    """``value`` as a 1-D array along the level dim ``dim`` (``s_rho`` or ``s_w``).

    The level dim is picked by name; any other dim (time, see :func:`_first`) is
    cut to its first element. A 1-D variable on a differently named dim is taken
    to be the profile.
    """
    if isinstance(value, xr.Variable):
        return xr.DataArray(value.values, dims=dim)
    if isinstance(value, xr.DataArray):
        if dim in value.dims:
            return _first(value, keep=(dim,)).reset_coords(drop=True)
        if value.ndim != 1:
            raise ValueError(
                f"{value.name!r} has dims {value.dims} but no {dim!r} dim; rename its vertical dim to {dim!r}."
            )
        return value.reset_coords(drop=True).rename({value.dims[0]: dim})
    return xr.DataArray(np.asarray(value, dtype=float).reshape(-1), dims=dim)


#: per level dim: the Cs and sigma names, where sigma values are read from, and ``sigma_levels``' loc
_PAIRS = {
    "s_rho": ("Cs_r", "sigma_r", ("s_rho", "sigma_r", "sc_r"), "rho"),
    "s_w": ("Cs_w", "sigma_w", ("s_w", "sigma_w", "sc_w"), "w"),
}


def _wanted_levels(levels):
    """The level dims whose arrays ``levels=`` asks :func:`vertical_params` for."""
    if levels is True:
        return ("s_rho", "s_w")
    if levels is False:
        return ()
    try:
        dim = normalize_scoord(levels) if isinstance(levels, str) else None
    except ValueError:
        dim = None
    if dim is None:
        raise ValueError(
            "levels must be True (both pairs of level arrays), 's_rho' (or 'rho'), 's_w' (or 'w'), or False "
            f"(only hc and Vtransform), not {levels!r}"
        )
    return (dim,)


def _level_arrays(src, wanted):
    """The ``Cs``/``sigma`` arrays of the level dims in ``wanted``, checked against the levels of the data."""
    n_r, n_w = _levels(src, "s_rho"), _levels(src, "s_w")
    if n_r is not None and n_w is not None and n_w != n_r + 1:
        raise ValueError(
            f"'s_rho' has {n_r} levels but 's_w' has {n_w}; 's_w' must have one more level than 's_rho'. "
            "To subset vertically, select both together, e.g. ds.isel(s_rho=slice(a, b), s_w=slice(a, b + 1))."
        )
    if n_r is None and n_w is None:
        raise ValueError("dataset has no 's_rho' dimension; it is not a 3-D ROMS dataset")
    if n_r is None and "s_rho" in wanted:
        raise ValueError(
            "dataset has an 's_w' dimension but no 's_rho' one, as when one level is selected with an "
            "integer (ds.isel(s_rho=3) drops the dim). Compute on the full Dataset and select afterwards, "
            "e.g. xroms.z(ds).isel(s_rho=3)."
        )
    n_levels = n_r if n_r is not None else n_w - 1
    theta_s, theta_b = _scalar(src, "theta_s"), _scalar(src, "theta_b")
    vstretching = _scalar(src, "Vstretching")
    vstretching = 4 if vstretching is None else int(vstretching)

    out = {}
    for dim in wanted:
        cs_name, sigma_name, sigma_sources, loc = _PAIRS[dim]
        sigma = _var(src, *sigma_sources)
        sigma = sigma_levels(n_levels, loc) if sigma is None else _as_level_array(sigma, dim)
        cs = _var(src, cs_name)
        if cs is None:
            cs = _attr(src, cs_name)
        if cs is None:
            if theta_s is None or theta_b is None:
                raise ValueError(f"cannot find {cs_name} (variable or attribute) or theta_s/theta_b to compute it")
            cs = stretching(sigma, theta_s, theta_b, vstretching)
        out[cs_name], out[sigma_name] = _as_level_array(cs, dim), sigma

    for name, arr in out.items():
        dim = arr.dims[0]
        n_data = _levels(src, dim)
        # without an s_w dim, the w arrays still need one level more than the data's s_rho
        n_need = n_data if n_data is not None else n_levels + 1
        if arr.sizes[dim] != n_need:
            have = f"{n_data} along {dim!r}" if n_data is not None else f"{n_levels} along 's_rho', so {n_need} interfaces"
            raise ValueError(
                f"{name} has {arr.sizes[dim]} levels but the data have {have}. "
                "If the data were subset vertically, subset the Dataset (not just the variable) "
                "so the vertical parameters stay matched, or pass them explicitly."
            )
    return out


def _vtransform_value(value, name, source):
    """``value`` as the integer 1 or 2, else a ``ValueError`` naming ``source``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number not in (1.0, 2.0):  # also rejects 2.5, which int() would silently turn into 2
        raise ValueError(f"{name} must be 1 or 2, not {value!r}. Correct {source}.")
    return int(number)


def _vtransform(src, Vtransform, default_Vtransform):
    """Vtransform: the argument, else what the Dataset states or implies, else ``default_Vtransform``."""
    if default_Vtransform is not None:
        default_Vtransform = _vtransform_value(default_Vtransform, "default_Vtransform", "the default_Vtransform= argument")
    if Vtransform is not None:
        return _vtransform_value(Vtransform, "Vtransform", "the Vtransform= argument")
    value = _scalar(src, "Vtransform")
    if value is not None:
        return _vtransform_value(value, "Vtransform", "the dataset's 'Vtransform' variable or attribute, or pass Vtransform=")
    vct = _attr(src, "VertCoordType")
    if vct is not None:
        value = {"NEW": 2, "OLD": 1}.get((vct.decode() if isinstance(vct, bytes) else str(vct)).strip().upper())
        if value is None:
            raise ValueError(
                f"cannot determine Vtransform: VertCoordType is {vct!r}, neither 'NEW' (2) nor 'OLD' (1). Set it "
                "on the Dataset, e.g. ds['Vtransform'] = 1 (or 2), or pass Vtransform=."
            )
        return value
    cs_attr_only = any(_var(src, name) is None and _attr(src, name) is not None for name in ("Cs_r", "Cs_w"))
    if cs_attr_only or _var(src, "sigma_r") is not None or any(_has_file_topology(s) for s in src):
        return 2  # UCLA ROMS output / roms-tools grids / REMORA only use Vtransform 2
    if default_Vtransform is not None:
        return default_Vtransform
    raise ValueError(
        "cannot determine Vtransform (no variable, attribute or VertCoordType gives it). Set it on the "
        "Dataset, e.g. ds['Vtransform'] = 1 (or 2), or pass Vtransform= (which wins over the Dataset) or "
        "default_Vtransform= (used only when the Dataset states none)."
    )


def vertical_params(ds, grid=None, *, levels=True, hc=None, Vtransform=None, default_Vtransform=None):
    """Find the s-coordinate parameters of ``ds`` (optionally with ``grid``).

    Looks, per item, in variables then global attributes of ``ds`` and then
    ``grid``, covering Rutgers ROMS, UCLA ROMS output and roms-tools grids,
    CROCO and REMORA:

    * sigma: ``s_rho``/``s_w`` coordinate values, then ``sigma_r``/``sigma_w``,
      then ``sc_r``/``sc_w``; otherwise computed from the number of levels.
    * ``Cs_r``/``Cs_w``: variables, then attributes; otherwise computed from
      ``theta_s``/``theta_b`` with ``Vstretching`` (4 unless stated).
    * ``hc``: the argument, else a variable or attribute.
    * ``Vtransform``: the argument, a variable, an attribute, ``VertCoordType``
      (``"NEW"`` -> 2, ``"OLD"`` -> 1), else 2 for UCLA-style files (stretching
      only in attributes, or roms-tools ``sigma_r`` variables) and REMORA (its
      own SGRID topology variable; the one :func:`add_cf_attrs` writes does not
      count), else ``default_Vtransform``.

    Parameters
    ----------
    levels : True, "s_rho", "s_w" or False
        The level arrays to find: both pairs (True), ``Cs_r``/``sigma_r`` only
        (``"s_rho"``: all that depths on rho levels need), ``Cs_w``/``sigma_w``
        only (``"s_w"``), or none (False). With False only ``hc`` and
        ``Vtransform`` are looked up, so the Dataset may be cut vertically in
        any way, or be 2-D. Arrays not asked for are None.
    hc : float, optional
        Critical depth (m). Wins over the Dataset's.
    Vtransform : 1 or 2, optional
        Wins over the Dataset's.
    default_Vtransform : 1 or 2, optional
        Used only when nothing in the Dataset states or implies a Vtransform.

    Parameters that carry a time dim (as after ``xr.open_mfdataset`` with its
    default ``data_vars="all"``) are read from their first record. Raises
    ``ValueError`` if ``s_w`` does not have one more level than ``s_rho``, if the
    level arrays asked for do not match the levels of the data, or if
    ``Vtransform`` is not 1 or 2.
    """
    src = _sources(ds, grid)
    wanted = _wanted_levels(levels)
    arrays = _level_arrays(src, wanted) if wanted else {}
    if hc is not None:
        try:
            value = float(hc)
        except (TypeError, ValueError):
            value = np.nan
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"hc must be a finite number of metres, 0 or more, not {hc!r}")
        hc = value
    else:
        hc = _scalar(src, "hc")
        if hc is None:
            raise ValueError("cannot find the critical depth 'hc' (variable or attribute); pass hc= if you know it")
    vt = _vtransform(src, Vtransform, default_Vtransform)
    return VerticalParams(Vtransform=vt, hc=hc, **arrays)


def rho0(ds, grid=None, default=1025.0):
    """Reference density: ``rho0`` variable, then attribute, then ``default``."""
    value = _scalar(_sources(ds, grid), "rho0")
    return default if value is None else value


#: CF names a free surface goes by once a Dataset is renamed to them (ocean-skill renames zeta to the first)
FREE_SURFACE_NAMES = (
    "sea_surface_height_above_geoid",
    "sea_surface_height_above_mean_sea_level",
    "sea_surface_height",
    "sea_surface_elevation",
)


def free_surface_name(ds):
    """Name of ``ds``'s free surface, or None: ``zeta``, else the one variable named or
    standard-named as a CF sea surface height (:data:`FREE_SURFACE_NAMES`) on rho points.

    On a Dataset that has been cut to a transect, a row or a column, positions can't be
    read from the dims, so there it is the one on ``h``'s points (the same dims as ``h``,
    besides time); a sea surface height elsewhere, such as a tide gauge's series in a
    full Dataset, is not the free surface. Raises ``ValueError`` when several variables
    could be it.
    """
    if "zeta" in ds.variables:
        return "zeta"
    h_dims = set(ds["h"].dims) if "h" in ds.variables else None

    def on_h_points(var):
        if hposition(var) == "rho":
            return True
        return h_dims is not None and hposition(var) is None and set(var.dims) - {time_dim(var)} == h_dims

    found = [
        name
        for name, var in ds.data_vars.items()
        if (name in FREE_SURFACE_NAMES or var.attrs.get("standard_name") in FREE_SURFACE_NAMES) and on_h_points(var)
    ]
    if len(found) > 1:
        raise ValueError(f"several variables could be the free surface ({found}); choose one with zeta=<DataArray>")
    return found[0] if found else None


# --- time ---------------------------------------------------------------------


_EPOCH = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_SINCE = re.compile(r"\s*(\S+)\s+since\s+(\S.*)", re.IGNORECASE)
# time units that come without an epoch, as CF names, and their length in seconds
_UNIT_NAMES = {
    "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds", "seconds": "seconds",
    "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
    "d": "days", "day": "days", "days": "days",
}
_UNIT_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
# calendars whose dates datetime64 can hold (proleptic Gregorian); the others need cftime
_DATETIME64_CALENDARS = ("standard", "gregorian", "proleptic_gregorian")
# the dates datetime64[ns] can hold
_NS_RANGE = (np.datetime64("1677-09-22", "us"), np.datetime64("2262-04-11", "us"))


def _parse_epoch(text):
    match = _EPOCH.search("" if text is None else str(text))
    if not match:
        return None
    y, m, d, hh, mm, ss = match.groups()
    return np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(hh or 0):02d}:{int(mm or 0):02d}:{int(ss or 0):02d}")


def _is_decoded(var):
    """True if ``var`` holds dates already: datetime64, or cftime objects."""
    if np.issubdtype(var.dtype, np.datetime64):
        return True
    if var.dtype != object or var.size == 0:
        return False
    first = var.isel({dim: 0 for dim in var.dims}).values.item()
    return type(first).__module__.split(".")[0] == "cftime"


def _decode_cf(values, units, calendar, name):
    """CF-decode ``values`` with xarray, which honours ``calendar``.

    Gives datetime64 where the calendar and the dates allow it, else cftime dates.
    """
    attrs = {"units": units} if calendar is None else {"units": units, "calendar": calendar}
    try:
        decoded = xr.decode_cf(xr.Dataset({"t": ("t", values, attrs)}), decode_timedelta=False)
    except (ValueError, OverflowError) as err:
        raise ValueError(
            f"cannot decode the times of {name!r} with units {units!r} and calendar {calendar or 'standard'!r}. "
            f"Correct the units/calendar attributes of ds[{name!r}], or pass reference_date= to supply the epoch."
        ) from err
    return decoded["t"].values


def _add_offsets(epoch, values, unit, name):
    """``epoch`` plus ``values`` (in ``unit``) as datetime64[ns], proleptic Gregorian.

    Adds in microseconds, so an epoch outside the datetime64[ns] range (years 1678
    to 2262) still works when the dates themselves fall inside it.
    """
    micro = np.rint(np.asarray(values, dtype="float64") * (_UNIT_SECONDS[unit] * 1e6))
    try:
        if (np.abs(micro) >= 1e18).any():
            raise OverflowError
        times = epoch.astype("datetime64[us]") + micro.astype("timedelta64[us]")
        if ((times < _NS_RANGE[0]) | (times > _NS_RANGE[1])).any():
            raise OverflowError  # numpy < 2.3 wraps around silently when casting to ns
        return times.astype("datetime64[ns]")
    except OverflowError:
        raise ValueError(
            f"the times of {name!r} ({unit} since {epoch}) fall outside the range datetime64 can hold "
            "(years 1678 to 2262). If that epoch is not the real start date, correct it in the variable's "
            "long_name/units and pass reference_date=. Otherwise decode with cftime: give "
            f"ds[{name!r}] CF units such as 'seconds since 0001-01-01' and call xr.decode_cf(ds, use_cftime=True)."
        ) from None


def decode_time(ds, reference_date=None, time_var=None):
    """Return ``ds`` with a decoded datetime index on its time dimension.

    * Times with CF ``units`` ("hours since 2013-12-17 00:00:00"), for example
      data opened with ``decode_times=False``, are decoded by xarray's CF
      decoding, which honours the ``calendar`` attribute: datetime64 where the
      calendar and the dates allow it, else cftime dates.
    * For UCLA ROMS output, the ``time`` dim has no coordinate and ``ocean_time``
      holds seconds (``units`` "second") with the epoch only in its ``long_name``
      ("Time since 1995/01/01"). The epoch is then taken from ``long_name`` (or
      from ``reference_date``; if both exist they must agree), seconds, minutes,
      hours or days are added to it, and a datetime64 coordinate is attached on
      the time dim; ``ocean_time`` is kept unchanged. Dates outside the datetime64
      range raise an error that suggests cftime.

    If ``units`` and ``long_name`` both give an epoch, they must agree. Datasets whose
    time is already decoded (datetime64 or cftime) are returned as is.
    """
    tdim = time_dim(ds)
    if tdim is None:
        raise ValueError("no time dimension found")
    if tdim in ds.coords and _is_decoded(ds[tdim]):
        return ds
    name = time_var
    if name is None:
        candidates = [v for v in ("ocean_time", tdim, "scrum_time") if v in ds.variables and ds[v].dims == (tdim,)]
        if not candidates:
            raise ValueError(f"no time variable on dim {tdim!r}")
        name = candidates[0]
    var = ds[name]
    if _is_decoded(var):
        return ds.assign_coords({tdim: var.variable})

    units = str(var.attrs.get("units", "seconds")).strip()
    in_units, in_long_name = _parse_epoch(units), _parse_epoch(var.attrs.get("long_name"))
    if in_units is not None and in_long_name is not None and in_units != in_long_name:
        raise ValueError(
            f"the units of {name!r} ({units!r}) and its long_name ({var.attrs['long_name']!r}) give different epochs. "
            f"Correct or remove one of them in ds[{name!r}].attrs."
        )
    epoch = in_units if in_units is not None else in_long_name
    if reference_date is not None:
        ref = np.datetime64(reference_date, "s")
        if epoch is not None and ref != epoch:
            raise ValueError(
                f"reference_date {ref} disagrees with the file's epoch {epoch}. "
                f"Drop reference_date, or correct the epoch in the units/long_name of ds[{name!r}]."
            )
        epoch = ref
    values = np.asarray(var.values)
    calendar = str(var.attrs["calendar"]).strip().lower() if "calendar" in var.attrs else None

    since = _SINCE.fullmatch(units)
    times = None
    if since:
        try:
            times = _decode_cf(values, f"{since[1].lower()} since {since[2]}", calendar, name)
        except ValueError:
            if epoch is None:  # the units are all there is
                raise
    if times is None:
        if epoch is None:
            raise ValueError(f"cannot find a reference date for {name!r}; pass reference_date=")
        unit = _UNIT_NAMES.get((since[1] if since else units).lower())
        if unit is None:
            raise ValueError(
                f"unsupported time units {units!r} for {name!r}; use seconds, minutes, hours or days, "
                "or CF units such as 'seconds since 2000-01-01'"
            )
        if calendar is None or calendar in _DATETIME64_CALENDARS:
            times = _add_offsets(epoch, values, unit, name)
        else:
            times = _decode_cf(values, f"{unit} since {str(epoch).replace('T', ' ')}", calendar, name)
    return ds.assign_coords({tdim: (tdim, times, {"long_name": "time"})})


# --- CF / SGRID decoration ------------------------------------------------------


def _position_coords(ds):
    """``ds`` with the lon/lat and x/y at every horizontal position as coordinates.

    Each variable then carries those at its own position, and writing ``ds`` lists
    them in its CF ``coordinates`` attribute.
    """
    names = [f"{prefix}_{pos}" for pos in HCOORDS for prefix in ("lon", "lat", "x", "y")]
    promoted = [name for name in names if name in ds.data_vars]
    ds = ds.set_coords(promoted)
    for var in ds.data_vars.values():
        listed = var.encoding.get("coordinates")
        if listed is not None and any(name in var.coords and name not in listed.split() for name in promoted):
            # the file's own list leaves them out: on writing, xarray lists every coordinate the variable has
            del var.encoding["coordinates"]
    return ds


def merge_grid(ds, grid):
    """Model output and its separate grid as one Dataset, with lon/lat as coordinates.

    UCLA ROMS, roms-tools and often CROCO keep the grid (``h``, ``pm``/``pn``,
    masks, ``angle``, ``f``, longitudes and latitudes) in a file of its own. This
    merges it into the output like ``xr.merge([ds, grid], compat="override")``:
    where both have a variable or an attribute, ``ds``'s is kept. It also:

    * puts the grid in ``ds``'s dim naming, if ``ds`` is canonical and the grid
      uses Rutgers aliases;
    * checks that both cover the same points: same sizes and, where both have
      them, the same labels, so nothing is padded with NaN;
    * makes the lon/lat (or x/y) at every position coordinates (see
      :func:`add_cf_attrs`), so that every variable and every xroms result carries
      them and results can be stored back.

    Only metadata changes: data stays lazy, and neither input is modified.

    Parameters
    ----------
    ds : Dataset
        Model output.
    grid : Dataset
        Its grid.

    Returns
    -------
    Dataset

    Raises
    ------
    ValueError
        If the two cover different points (a subdomain, a tile), or ``ds`` uses
        Rutgers aliases while the grid uses canonical names.
    """
    if not isinstance(ds, xr.Dataset) or not isinstance(grid, xr.Dataset):
        raise TypeError("merge_grid takes the output and the grid as Datasets")
    if convention(grid) == "rutgers" and convention(ds) != "rutgers":
        grid = canonicalize(grid)
    elif convention(ds) == "rutgers" and convention(grid) != "rutgers" and {"xi_u", "eta_v"} & set(grid.dims):
        # canonical u/v/psi fields (eta_rho, xi_u) next to Rutgers ones (eta_u, xi_u) would mix two namings
        raise ValueError(
            "the output uses Rutgers dim names (eta_u, xi_v, ...) and the grid canonical ones: "
            "merge_grid(xroms.canonicalize(ds), grid) merges them in canonical naming."
        )
    for dim in sorted(set(ds.dims) & set(grid.dims)):
        labels_differ = dim in ds.indexes and dim in grid.indexes and not ds.indexes[dim].equals(grid.indexes[dim])
        if ds.sizes[dim] != grid.sizes[dim] or labels_differ:
            raise ValueError(
                f"the output and the grid cover different {dim} points ({ds.sizes[dim]} vs {grid.sizes[dim]}"
                f"{', other labels' if labels_differ else ''}): subset the grid the same way as the output first, "
                "e.g. with xroms.subset or isel on both."
            )
    merged = xr.merge([ds, grid], compat="override", join="exact", combine_attrs="override")
    merged.attrs = {**grid.attrs, **ds.attrs}
    return _position_coords(merged)


def add_cf_attrs(ds, *, index_coords=False, sgrid=True):
    """Return a copy of ``ds`` decorated for cf-xarray (metadata only).

    * the longitudes and latitudes (and x/y) at every position as coordinates,
      with ``standard_name``/``units``, so that every xroms result carries them.
      UCLA grid files keep them as data variables, which results cannot carry:
      they would not merge back into ``ds``. Each variable gets the ones at its
      own position, and writing ``ds`` lists them in its CF ``coordinates``
      attribute;
    * ``axis``/``standard_name`` attributes on the coordinates that exist;
    * an SGRID ``grid`` topology variable if none is present (``sgrid=True``),
      marked ``xroms_generated`` so that it is never taken for a REMORA file's own;
    * integer index coords on horizontal dims **only if** ``index_coords=True``
      and the dim has none (existing labels are never renumbered).

    This does not prepare ``ds`` for ``xgcm.Grid(ds)``, whose autoparse needs
    ``c_grid_axis_shift`` attributes that are not added here. For an xgcm Grid
    use ``ds.xroms.xgcm_grid()``.
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
    ds = _position_coords(ds)
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
