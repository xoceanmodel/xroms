"""The only place xroms calls xgcm.

Every call builds a throwaway ``xgcm.Grid`` sized from the input's own dims, so
nothing about an earlier, larger, or differently shaped dataset can leak in.
Around each xgcm call this module:

* strips the input to a bare array (coords are re-attached afterwards, minus any
  that span the operated dim, since those describe the old grid position);
* rechunks only the operated dim to a single chunk when dask has split it
  (xgcm refuses chunked "inner"/"outer" core dims, issues #16/#77), and restores
  the original chunk layout along that dim afterwards (last chunk +/- 1);
* carries integer index labels across the stagger when the input had them.

Inputs must use canonical ROMS dim names (see ``conventions.canonicalize``).
"""

import numpy as np
import xarray as xr


#: axis -> (center dim, staggered dim, xgcm position of the staggered dim)
AXES = {
    "X": ("xi_rho", "xi_u", "inner"),
    "Y": ("eta_rho", "eta_v", "inner"),
    "Z": ("s_rho", "s_w", "outer"),
}


def _xgcm():
    import xgcm

    return xgcm


def axis_dim(da, axis):
    """Return ``(dim, is_center)`` for ``axis`` in ``da``, or ``(None, None)``."""
    center, stag, _ = AXES[axis]
    if center in da.dims:
        return center, True
    if stag in da.dims:
        return stag, False
    return None, None


def target_dim(axis, from_center):
    """Dim name after moving along ``axis`` from center (True) or staggered."""
    center, stag, _ = AXES[axis]
    return stag if from_center else center


def _grid(n_center, axis, padding):
    # fill_value is passed per call: xgcm 0.10 warns whenever Grid gets one
    center, stag, kind = AXES[axis]
    n_stag = n_center - 1 if kind == "inner" else n_center + 1
    ds = xr.Dataset(coords={center: np.arange(n_center), stag: np.arange(n_stag)})
    return _xgcm().Grid(
        ds,
        coords={axis: {"center": center, kind: stag}},
        padding={axis: padding},
        autoparse_metadata=False,
    )


def _n_center(da, axis):
    center, stag, kind = AXES[axis]
    if center in da.dims:
        return da.sizes[center]
    n = da.sizes[stag]
    return n + 1 if kind == "inner" else n - 1


def _restore_chunks(out, in_chunks, new_dim, delta):
    """Re-split ``new_dim`` like the input was, with the last chunk +/- delta."""
    if in_chunks is None or len(in_chunks) <= 1:
        return out
    chunks = list(in_chunks)
    chunks[-1] += delta
    if chunks[-1] <= 0:
        extra = chunks.pop()
        chunks[-1] += extra
    return out.chunk({new_dim: tuple(chunks)})


def _new_labels(labels, axis, from_center):
    """Integer labels along the new dim (inner[i] sits between center[i], [i+1])."""
    if labels is None or axis == "Z":
        return None
    labels = np.asarray(labels)
    if from_center:
        return labels[:-1]
    return np.concatenate([labels, [labels[-1] + 1]]) if labels.size else labels


def _apply(func, da, axis, padding, fill_value):
    """Run ``grid.<func>(da, axis)`` statelessly and put coords/labels/chunks back."""
    dim, from_center = axis_dim(da, axis)
    if dim is None:
        raise ValueError(f"{da.name!r} has no {axis!r} dimension; dims are {da.dims}")
    new_dim = target_dim(axis, from_center)
    labels = da[dim].values if dim in da.indexes else None
    keep_coords = {k: v for k, v in da.coords.items() if dim not in v.dims and k != dim}
    bare = xr.DataArray(da.variable, name=da.name)
    in_chunks = None
    if bare.chunks is not None:
        in_chunks = bare.chunksizes.get(dim)
        if len(in_chunks) > 1:
            bare = bare.chunk({dim: -1})
    grid = _grid(_n_center(da, axis), axis, padding)
    out = getattr(grid, func)(bare, axis, fill_value=fill_value)
    out = out.drop_vars([c for c in out.coords], errors="ignore")
    delta = out.sizes[new_dim] - da.sizes[dim]
    out = _restore_chunks(out, in_chunks, new_dim, delta)
    new_labels = _new_labels(labels, axis, from_center)
    if new_labels is not None:
        out = out.assign_coords({new_dim: new_labels})
    out = out.assign_coords(keep_coords)
    out.attrs = dict(da.attrs)
    out.name = da.name
    return out


def interp(da, axis, boundary="extend", fill_value=np.nan):
    """Average neighbours along ``axis`` onto the other stagger position.

    ``boundary`` applies only where padding is needed (center -> inner and
    outer -> center are interior-only; inner -> center and center -> outer are padded):
    ``"extend"`` repeats the edge value, ``"fill"`` pads with ``fill_value``.
    """
    return _apply("interp", da, axis, _padding(boundary), fill_value)


def diff(da, axis, boundary="extend", fill_value=np.nan):
    """Difference neighbours along ``axis`` onto the other stagger position.

    Where the result needs values outside the data (the two edges when moving
    inner -> center or center -> outer), ``boundary="extend"`` copies the nearest
    *computed* difference (a one-sided estimate), never the zero that padding the
    field with its own edge value would produce. ``boundary="fill"`` sets those
    edges to ``fill_value`` (NaN by default; 0 imposes a zero-gradient condition).
    """
    dim, from_center = axis_dim(da, axis)
    padded_edges = (axis == "Z" and from_center) or (axis != "Z" and not from_center)
    out = _apply("diff", da, axis, "fill", np.nan)
    if padded_edges:
        out = _set_edges(out, target_dim(axis, from_center), boundary, fill_value)
    return out


def _padding(boundary):
    if boundary not in ("extend", "fill"):
        raise ValueError(f"boundary must be 'extend' or 'fill', not {boundary!r}")
    return boundary


def _edge_kernel(a, mode, fill_value):
    a = np.array(a, copy=True)
    if a.shape[-1] >= 3 and mode == "extend":
        a[..., 0] = a[..., 1]
        a[..., -1] = a[..., -2]
    else:
        a[..., 0] = fill_value
        a[..., -1] = fill_value
    return a


def _set_edges(da, dim, boundary, fill_value):
    """Replace the two edge values along ``dim`` per ``boundary``."""
    _padding(boundary)
    in_chunks = da.chunksizes.get(dim) if da.chunks is not None else None
    work = da.chunk({dim: -1}) if in_chunks is not None and len(in_chunks) > 1 else da
    out = xr.apply_ufunc(
        _edge_kernel,
        work,
        kwargs={"mode": boundary, "fill_value": fill_value},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[np.result_type(da.dtype, np.float64)],
        keep_attrs=True,
    ).transpose(*da.dims)
    if in_chunks is not None and len(in_chunks) > 1:
        out = out.chunk({dim: in_chunks})
    return out


def transform(da, iso_values, iso_array, dim, *, new_dim="z", method="linear", mask_edges=True):
    """Interpolate ``da`` onto values of ``iso_array`` along ``dim`` (xgcm transform).

    ``iso_values`` may be 1-D (numpy/list) or an N-D DataArray whose extra dim
    is ``new_dim``. Only ``dim`` is rechunked to a single chunk.
    """
    if dim not in da.dims or dim not in iso_array.dims:
        raise ValueError(f"both the variable and the iso array need dim {dim!r}")
    work = xr.DataArray(da.variable, name=da.name or "var")
    target_data = xr.DataArray(iso_array.variable, name="iso_array")
    # xgcm needs the variable to carry every dim of the iso array (static depths
    # sliced on a time-varying density, say): broadcast it, lazily
    extra = {d: target_data.sizes[d] for d in target_data.dims if d not in work.dims}
    if extra:
        work = work.expand_dims(extra)
    if work.chunks is not None:
        work = work.chunk({dim: -1})
    if target_data.chunks is not None:
        target_data = target_data.chunk({dim: -1})
    grid = _xgcm().Grid(
        xr.Dataset(coords={dim: np.arange(da.sizes[dim])}),
        coords={"Z": {"center": dim}},
        padding={"Z": "fill"},
        autoparse_metadata=False,
    )
    if isinstance(iso_values, xr.DataArray):
        target = iso_values
        if new_dim not in target.dims:
            target = target.expand_dims(new_dim)
        values = target[new_dim].values if new_dim in target.coords else None
    else:
        values = np.atleast_1d(np.asarray(iso_values, dtype=float))
        target = xr.DataArray(values, dims=[new_dim], name=new_dim)
    out = grid.transform(
        work,
        "Z",
        target,
        target_data=target_data,
        method=method,
        mask_edges=mask_edges,
        target_dim=new_dim,
    )
    out = out.drop_vars([c for c in out.coords if c != new_dim], errors="ignore")
    if values is not None:
        out = out.assign_coords({new_dim: values})
    keep = {k: v for k, v in da.coords.items() if dim not in v.dims and k != dim}
    keep.update({
        k: v for k, v in iso_array.coords.items()
        if k not in keep and k != dim and dim not in v.dims and set(v.dims) <= set(out.dims)
    })
    out = out.assign_coords(keep)
    out.attrs = dict(da.attrs)
    out.name = da.name
    return out


def grid_for(ds, hcoords=True, vertical=True, padding="extend"):
    """A fresh, correctly configured ``xgcm.Grid`` for the (canonical) ``ds``.

    Exposed to users as ``ds.xroms.xgcm_grid()`` for their own xgcm work. Only
    axes whose dims are present are included; no metrics are attached, because
    ROMS metrics depend on position and are computed by xroms on demand.
    """
    coords = {}
    for axis, (center, stag, kind) in AXES.items():
        if axis == "Z" and not vertical:
            continue
        if axis != "Z" and not hcoords:
            continue
        if center in ds.dims and stag in ds.dims:
            coords[axis] = {"center": center, kind: stag}
    if not coords:
        raise ValueError("dataset has no complete ROMS axis (center + staggered dims)")
    base = xr.Dataset(coords={d: np.arange(ds.sizes[d]) for pair in coords.values() for d in pair.values()})
    return _xgcm().Grid(base, coords=coords, padding={a: padding for a in coords}, autoparse_metadata=False)
