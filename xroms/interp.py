"""Interpolation: onto iso-surfaces or depths (vertical), and to lon/lat points."""

import numpy as np
import xarray as xr

from . import _xgcm
from ._align import _check_grid
from .conventions import canonicalize, hposition, horizontal_coords, vposition
from .utilities import order
from .vertical import _check_reference, infer_reference, label, z_like


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


def _is_grid_dim(name):
    """Grid dims (``xi_*``, ``eta_*``, ``s_*``) name positions; others (time, ...) do not."""
    return str(name).startswith(("xi_", "eta_", "s_"))


def _check_pairing(var, iso_values, iso_array, dim):
    """Raise unless ``iso_array`` (and DataArray targets) sit on ``var``'s points along ``dim``.

    Dims are matched by name, so a u-point variable against rho-point depths would
    come back with both ``xi_u`` and ``xi_rho``: broadcast, not sliced.
    """
    if dim not in iso_array.dims:
        raise ValueError(f"both the variable and the iso array need dim {dim!r}")
    if var.sizes[dim] != iso_array.sizes[dim]:
        raise ValueError(
            f"{var.name!r} has {var.sizes[dim]} points along {dim!r} but the iso array has "
            f"{iso_array.sizes[dim]}; subset both the same way."
        )
    have = hposition(var)
    for other, what in ((iso_array, "the iso array"), (iso_values, "iso_values")):
        there = hposition(other) if isinstance(other, xr.DataArray) else None
        if have is not None and there is not None and have != there:
            raise ValueError(
                f"{var.name!r} is at {have} points but {what} is at {there} points, so slicing would "
                f"broadcast one across the other. Put both on the same points first, e.g. "
                f"xroms.to_grid(..., hcoord={have!r}); to slice at depths use xroms.zslice(var, depths, ds), "
                "which builds z at the variable's points."
            )
    lacking = [d for d in iso_array.dims if d not in var.dims and _is_grid_dim(d)]
    if lacking:
        raise ValueError(
            f"the iso array has dims {lacking} that {var.name!r} lacks, so they are not on the same points. "
            "Select the same points from both (e.g. with isel), or, if the variable really is the same "
            "everywhere along those dims, broadcast it over them first (xr.broadcast)."
        )


def _single_chunk(da, dim):
    """``da`` with ``dim`` in one chunk; only a dim dask has split is rechunked."""
    if da.chunks is not None and len(da.chunksizes[dim]) > 1:
        return da.chunk({dim: -1})
    return da


def _nearest_kernel(var, iso, values, mask_edges, dtype):
    """Level of ``iso`` (last axis) nearest each of ``values`` (last axis), applied to ``var``.

    Leading axes broadcast. NaN distances never win, so a column without valid
    ``iso`` points reads its first level; ``mask_edges`` then blanks it, like any
    target outside the column's range. Ties go to the first level.
    """
    lead = np.broadcast_shapes(var.shape[:-1], iso.shape[:-1], values.shape[:-1])
    var, iso, values = (np.broadcast_to(a, lead + a.shape[-1:]) for a in (var, iso, values))
    out = np.empty(lead + values.shape[-1:], dtype=dtype)
    if mask_edges:
        valid = ~np.isnan(iso)
        low = np.where(valid, iso, np.inf).min(axis=-1)
        high = np.where(valid, iso, -np.inf).max(axis=-1)
        blank = np.asarray(np.nan).astype(dtype)  # NaN, or NaT for times
    for k in range(values.shape[-1]):
        target = values[..., k]
        dist = np.abs(iso - target[..., None])
        nearest = np.where(np.isnan(dist), np.inf, dist).argmin(axis=-1)
        picked = np.take_along_axis(var, nearest[..., None], axis=-1)[..., 0]
        if mask_edges:
            picked = np.where((target >= low) & (target <= high), picked, blank)
        out[..., k] = picked
    return out


def _nearest(var, iso_values, iso_array, dim, new_dim, mask_edges):
    values = iso_values if isinstance(iso_values, xr.DataArray) else xr.DataArray(
        np.atleast_1d(np.asarray(iso_values, dtype=float)), dims=[new_dim]
    )
    if new_dim not in values.dims:
        values = values.expand_dims(new_dim)
    # NaN has to fit where a target can fall outside the column
    dtype = var.dtype if not mask_edges or var.dtype.kind in "fcMm" else np.result_type(var.dtype, np.float32)
    # bare arrays, so positions pair up (as in the linear method) whatever the labels;
    # the searched dim and the targets are core dims, each a single chunk
    args = [
        _single_chunk(xr.DataArray(a.variable), core)
        for a, core in ((var, dim), (iso_array, dim), (values, new_dim))
    ]
    out = xr.apply_ufunc(
        _nearest_kernel,
        *args,
        kwargs={"mask_edges": mask_edges, "dtype": dtype},
        input_core_dims=[[dim], [dim], [new_dim]],
        output_core_dims=[[new_dim]],
        dask="parallelized",
        output_dtypes=[dtype],
    )
    keep = {k: v for k, v in var.coords.items() if dim not in v.dims and k != dim}
    keep.update({
        k: v for k, v in iso_array.coords.items()
        if k not in keep and k != dim and dim not in v.dims and set(v.dims) <= set(out.dims)
    })
    out = out.assign_coords(keep)
    out.attrs = dict(var.attrs)
    out.name = var.name
    return out


def isoslice(var, iso_values, iso_array, *, dim=None, new_dim=None, method="linear", mask_edges=True):
    """Values of ``var`` where ``iso_array`` equals each of ``iso_values``.

    Parameters
    ----------
    var, iso_array : DataArray
        Share the dimension ``dim`` along which to interpolate (default: the
        vertical; ``"X"``/``"Y"``/``"Z"`` or a dim name) and sit on the same
        horizontal points: dims are matched by name, so a u-point ``var`` needs a
        u-point ``iso_array`` (move one with :func:`xroms.to_grid`; to slice at
        depths use :func:`zslice`, which builds z at ``var``'s points). Dims that
        are not grid dims (time, say) and only ``iso_array`` has are broadcast.
    iso_values : sequence or DataArray
        Target values; a DataArray may vary in space (N-D targets, e.g. another
        grid's depths), with extra dim ``new_dim``.
    new_dim : str, optional
        Name of the new dimension (default: ``iso_array``'s name, else ``"iso"``).
    method : ``"linear"`` or ``"nearest"``
        Linear interpolation (xgcm ``transform``) or the nearest native level
        (ties go to the first level). Both are lazy under dask and rechunk only
        ``dim``, to a single chunk.
    mask_edges : bool
        NaN where a target lies outside the column's range (True), or hold the
        edge value (False).

    Notes
    -----
    Linear results have the dtype numpy promotes ``var`` and ``iso_array`` to (at
    least float32) and nearest results keep ``var``'s, lazy or computed alike.
    """
    if not isinstance(iso_array, xr.DataArray):
        legacy = type(iso_array).__module__.startswith("xgcm")
        found = "an xgcm Grid (xroms 1.0 no longer takes one)" if legacy else type(iso_array).__name__
        raise TypeError(
            f"iso_array must be a DataArray, not {found}. Slice on a field or on depths with "
            "xroms.isoslice(var, values, xroms.z(ds)), or at fixed depths with xroms.zslice(var, depths, ds)."
        )
    if method not in ("linear", "nearest"):
        raise ValueError(f"method must be 'linear' or 'nearest', not {method!r}")
    var, iso_array = canonicalize(var), canonicalize(iso_array)
    if isinstance(iso_values, xr.DataArray):
        iso_values = canonicalize(iso_values)
    dim = _resolve_dim(var, dim)
    _check_pairing(var, iso_values, iso_array, dim)
    new_dim = new_dim or (iso_array.name if isinstance(iso_array.name, str) and iso_array.name not in var.dims else "iso")
    if method == "linear":
        out = _xgcm.transform(var, iso_values, iso_array, dim, new_dim=new_dim, mask_edges=mask_edges)
    else:
        out = _nearest(var, iso_values, iso_array, dim, new_dim, mask_edges)
    if not isinstance(iso_values, xr.DataArray):
        out = out.assign_coords({new_dim: np.atleast_1d(np.asarray(iso_values, dtype=float))})
    elif new_dim in iso_values.coords:
        out = out.assign_coords({new_dim: iso_values[new_dim]})
    if new_dim in out.coords:
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


def _in_requested_labels(z, reference, positive):
    """``z`` expressed in ``reference``/``positive``, going by its own CF labels.

    A z in the requested reference and sign is returned as is, one that differs
    only in sign is negated, and one in another reference is refused (its values
    are not that reference's positions; it has to be built again). A z carrying no
    label is taken at its word.
    """
    try:
        have_ref, have_pos = infer_reference(z.attrs, default=(None, None))
    except ValueError as err:
        raise ValueError(
            f"{err} (in the attrs of z); fix them, or build z with xroms.z(ds, reference=..., positive=...)"
        ) from None
    if have_ref is not None and have_ref != reference:
        raise ValueError(
            f"z is labelled relative to {have_ref!r} but {reference!r} was requested. Pass "
            f"reference={have_ref!r} to slice in z's own reference, or build z in the requested one: "
            f"xroms.z(ds, reference={reference!r}, positive={positive!r})."
        )
    if have_pos is not None and have_pos != positive:
        flipped = -z
        flipped.attrs = dict(z.attrs)  # arithmetic drops them
        return label(flipped, reference, positive)
    return z


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
    A flag left unset is inferred from CF metadata on ``depths`` (when it is a
    DataArray: ``standard_name``/``positive``), else defaults to heights relative
    to mean sea level (ROMS z, negative below); a flag that is given always wins
    over the metadata. E.g. ``zslice(temp, [10], ds, reference="surface",
    positive="down")`` is 10 m below the moving surface.

    ``z`` (instead of ``grid``) holds the vertical positions of ``var``'s points.
    If it carries xroms/CF vertical labels they are honoured: a z in the requested
    reference and sign is used as is, one with the opposite sign is negated, and
    one in another reference raises (build it with ``xroms.z(ds, reference=...)``).
    A z with no labels is taken to be in the requested reference already.
    """
    grid = _check_grid(grid, "zslice")
    var = canonicalize(var)
    if reference is None or positive is None:
        attrs = depths.attrs if isinstance(depths, xr.DataArray) else {}
        try:
            ref0, pos0 = infer_reference(attrs)
        except ValueError as err:
            raise ValueError(
                f"{err} (in the attrs of depths); pass both reference= and positive= to ignore them"
            ) from None
        reference = reference or ref0
        positive = positive or pos0
    _check_reference(reference, positive)
    if z is None and grid is None:
        raise ValueError("zslice needs grid= (to compute depths) or z=")
    if z is not None:
        z = _in_requested_labels(z, reference, positive)
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
