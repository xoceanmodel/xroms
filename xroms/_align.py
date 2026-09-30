"""Select grid fields so they match the variable being computed on.

Core array functions never check whole-domain consistency (roms-tools passes
boundary lines and margins). These helpers are used only when xroms reads grid
primitives (``h``, ``zeta``, ``pm``, ...) out of a ``grid`` Dataset on behalf of a
variable, and they make that pairing explicit:

* a dim the grid field has but the variable lacks (typically time after
  ``isel``/``sel``) is matched through the variable's scalar coord, or else
  raises a guided error;
* shared dims of different lengths are matched by index labels when both sides
  have them, otherwise raise an error saying to subset the Dataset instead;
* staggered footprints are handled: a u-point variable needs the rho-point
  fields on both sides of each u point.
"""

import numpy as np

from .conventions import TIME_NAMES, canonicalize


HAXES = {"X": ("xi_rho", "xi_u"), "Y": ("eta_rho", "eta_v")}


class GridMismatchError(ValueError):
    """The variable and the grid data cannot be paired unambiguously."""


def _is_time_dim(obj, dim):
    if dim in TIME_NAMES:
        return True
    return dim in obj.coords and np.issubdtype(obj[dim].dtype, np.datetime64)


def time_mismatch_message(var_name, dim, field_name):
    return (
        f"{field_name!r} varies along {dim!r} but {var_name!r} has no {dim!r} dim or scalar "
        f"{dim!r} coord to match it (e.g. after a time mean, resample, groupby, or selecting a "
        "time on UCLA output without a decoded time coordinate). Choose the free surface "
        "explicitly: zeta=0 (static depths), zeta='mean' (time-mean zeta), zeta=<DataArray>, "
        "or pass z= directly. For UCLA ROMS output, run xroms.decode_time(ds) before selecting times."
    )


def select_like(field, like, *, name=None):
    """Select ``field`` (a grid primitive) to line up with variable ``like``.

    Both are canonicalized first. Returns the selected field.
    """
    field = canonicalize(field)
    like = canonicalize(like)
    fname = name or field.name or "grid field"
    vname = like.name or "variable"

    # dims on the field that the variable lacks
    for dim in list(field.dims):
        if dim in like.dims:
            continue
        if any(dim in pair for pair in HAXES.values()) and any(
            other in like.dims for pair in HAXES.values() if dim in pair for other in pair
        ):
            continue  # a different stagger of the same axis: handled below
        if dim in like.coords and like[dim].ndim == 0:
            if dim in field.indexes:
                field = field.sel({dim: like[dim].values})
                continue
            raise GridMismatchError(
                f"{vname!r} was selected along {dim!r} but {fname!r} has no {dim!r} index to "
                "match it; subset the Dataset instead of the variable."
            )
        if _is_time_dim(field, dim):
            raise GridMismatchError(time_mismatch_message(vname, dim, fname))
        if dim.startswith(("xi_", "eta_")):
            raise GridMismatchError(
                f"{vname!r} has no {dim!r} dimension (it was probably indexed along it without "
                f"index coordinates), so {fname!r} cannot be matched; subset the Dataset instead."
            )

    # horizontal footprint per axis
    for axis, (center, stag) in HAXES.items():
        vdim = center if center in like.dims else stag if stag in like.dims else None
        fdim = center if center in field.dims else stag if stag in field.dims else None
        if vdim is None or fdim is None:
            continue
        offset = 0 if vdim == fdim else (1 if (vdim == stag and fdim == center) else -1)
        expected = like.sizes[vdim] + offset
        if vdim in like.indexes and fdim in field.indexes:
            labels = np.asarray(like[vdim].values)
            if labels.size == 0:
                continue
            if offset == 1:
                need = np.concatenate([labels, [labels[-1] + 1]])
            elif offset == -1:
                need = labels[:-1]
            else:
                need = labels
            missing = np.setdiff1d(need, field[fdim].values)
            if missing.size:
                raise GridMismatchError(
                    f"{fname!r} lacks {fdim!r} labels {missing[:5].tolist()} needed for {vname!r}; "
                    "the grid does not cover the variable's footprint."
                )
            if field.sizes[fdim] != need.size or not np.array_equal(field[fdim].values, need):
                field = field.sel({fdim: need})
        elif field.sizes[fdim] != expected:
            raise GridMismatchError(
                f"{vname!r} and {fname!r} have incompatible sizes along axis {axis} "
                f"({vdim}={like.sizes[vdim]}, {fdim}={field.sizes[fdim]}) and no index coordinates "
                "to align them. Subset the Dataset (e.g. with xroms.subset) rather than a single "
                "variable, or add index coords with xroms.add_cf_attrs(ds, index_coords=True)."
            )
    return field


def require(grid, *names, purpose=None):
    """Raise a clear error listing which grid variables are missing."""
    missing = [n for n in names if n not in grid.variables]
    if missing:
        why = f" for {purpose}" if purpose else ""
        raise GridMismatchError(
            f"grid variables {missing} are needed{why} but not found. For UCLA ROMS or CROCO "
            "output the grid is often a separate file: merge it with "
            "xr.merge([ds, grid], compat='override') or pass grid=<grid Dataset>."
        )
