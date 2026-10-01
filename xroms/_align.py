"""Select grid fields so they match the variable being computed on.

Core array functions never check whole-domain consistency (roms-tools passes
boundary lines and margins). These helpers are used only when xroms reads grid
primitives (``h``, ``zeta``, ``pm``, ...) out of a ``grid`` Dataset on behalf of a
variable, or takes an explicit ``zeta=``/``z=`` for one, and they make that
pairing explicit:

* a dim the grid field has but the variable lacks (typically time after
  ``isel``/``sel``) is matched through the variable's scalar coord, or else
  raises a guided error;
* shared dims of different lengths are matched by index labels when both sides
  have them, otherwise raise an error saying to subset the Dataset instead;
* staggered footprints are handled: a u-point variable needs the rho-point
  fields on both sides of each u point;
* integer index labels that do not step by 1 (a strided subset) raise, because
  ``pm``/``pn`` describe neighbouring cells;
* arrays computed from the vertical parameters are matched to a variable cut
  vertically on its own (:func:`level_positions`), by label or else by count.
"""

import functools
import inspect

import numpy as np
import xarray as xr

from .conventions import TIME_NAMES, canonicalize, horizontal_coords, hposition, time_dim


HAXES = {"X": ("xi_rho", "xi_u"), "Y": ("eta_rho", "eta_v")}

_LEVELS_ADVICE = (
    "select levels on the Dataset rather than on the variable alone (e.g. ds.isel({dim}=slice(1, 4)))."
)


class GridMismatchError(ValueError):
    """The variable and the grid data cannot be paired unambiguously."""


def _reject_legacy(args, func, hint):
    """Guardrail for pre-1.0 calls that passed an ``xgcm.Grid`` positionally."""
    if args:
        raise TypeError(f"xroms 1.0: {func} no longer takes an xgcm grid argument. {hint}")


def _check_grid(grid, func):
    if grid is None:
        return None
    if type(grid).__module__.startswith("xgcm"):
        raise TypeError(
            f"xroms 1.0: pass the Dataset holding the grid variables instead of an xgcm Grid, "
            f"in its place (e.g. xroms.{func}(..., ds)). xroms no longer builds or stores xgcm grids."
        )
    if not isinstance(grid, xr.Dataset):
        raise TypeError(f"grid must be an xarray Dataset, not {type(grid).__name__}")
    return canonicalize(grid)


def _is_time_dim(obj, dim):
    if dim in TIME_NAMES:
        return True
    return dim in obj.coords and np.issubdtype(obj[dim].dtype, np.datetime64)


def is_time_varying(like):
    """True if ``like`` varies in time: it has a time dim, or the scalar time coord
    that selecting one time leaves behind."""
    if time_dim(like) is not None:
        return True
    return any(like[c].ndim == 0 and _is_time_dim(like, c) for c in like.coords)


def time_mismatch_message(var_name, dim, field_name):
    return (
        f"{field_name!r} varies along {dim!r} but {var_name!r} has no {dim!r} dim or scalar "
        f"{dim!r} coord to match it (e.g. after a time mean, resample, groupby, or selecting a "
        "time on UCLA output without a decoded time coordinate). Choose the free surface "
        "explicitly: zeta=0 (static depths), zeta='mean' (time-mean zeta), or a zeta=<DataArray> "
        f"or z=<DataArray> without a {dim!r} dim (e.g. a time mean, or one selected time). "
        "For UCLA ROMS output, run xroms.decode_time(ds) before selecting times."
    )


def check_unstrided(labels, dim, what="variable"):
    """Raise ``GridMismatchError`` if integer ``labels`` of horizontal ``dim`` do not step by 1.

    A step other than 1 means the points were subsampled, e.g.
    ``isel(xi_rho=slice(None, None, 2))``. ``pm``/``pn`` and the stagger
    relationships describe neighbouring cells, so a derivative, average or metric
    over strided points would be silently wrong (too large by the stride). Labels
    that are not integers may be coordinates such as longitude and are not
    checked; a strided subset without labels cannot be detected.
    """
    labels = np.asarray(labels)
    if labels.size < 2 or not np.issubdtype(labels.dtype, np.integer):
        return
    if (np.diff(labels.astype("int64")) != 1).any():
        shown = labels[:5].tolist()
        raise GridMismatchError(
            f"{what!r} has {dim!r} index labels {shown}{'...' if labels.size > 5 else ''} that do not "
            "step by 1: strided subsets change the spacing that pm/pn describe, so derivatives, "
            "averages and metrics on them would be wrong; subset with step 1 (e.g. "
            "xroms.subset(ds, X=slice(...))) and thin the result afterwards."
        )


def level_positions(like, dim, n, labels=None, *, name=None):
    """Positions among the grid's ``n`` levels along ``dim`` of the levels of ``like``.

    For arrays computed from the grid's vertical parameters, which cover every
    level the grid has, when the variable was cut vertically on its own
    (``temp.isel(s_rho=slice(1, 4))``). ``labels`` is the grid's index along
    ``dim`` (or None). Levels are matched by label when both sides have an index
    and by position otherwise, which needs equal counts. Returns None when
    ``like`` has all the levels in order, so nothing needs selecting.
    """
    like = canonicalize(like)
    if dim not in like.dims:
        return None
    vname = name or like.name or "variable"
    if labels is not None and dim in like.indexes:
        wanted = like.indexes[dim]
        if wanted.equals(labels):
            return None
        positions = labels.get_indexer(wanted)
        if (positions < 0).any():
            raise GridMismatchError(
                f"{vname!r} has {dim!r} labels {wanted[positions < 0][:5].tolist()} that the grid "
                "lacks, so its vertical levels cannot be matched to the vertical parameters; "
                + _LEVELS_ADVICE.format(dim=dim)
            )
        return positions
    if like.sizes[dim] != n:
        raise GridMismatchError(
            f"{vname!r} has {like.sizes[dim]} {dim!r} levels but the grid's vertical parameters "
            f"have {n}, and without {dim!r} index labels on both they cannot be matched; "
            + _LEVELS_ADVICE.format(dim=dim)
        )
    return None


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
                try:
                    field = field.sel({dim: like[dim].values})
                except KeyError:
                    raise GridMismatchError(
                        f"{vname!r} was selected at {dim}={like[dim].values}, but {fname!r} has no such "
                        f"{dim!r} label to match it; subset the Dataset instead of the variable, or "
                        f"pass one that covers it."
                    ) from None
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

    # shared non-horizontal dims (typically time after isel/sel with a list or slice)
    for dim in field.dims:
        if dim not in like.dims or dim.startswith(("xi_", "eta_")):
            continue
        if dim in like.indexes and dim in field.indexes:
            labels = like.indexes[dim]
            if not field.indexes[dim].equals(labels):
                missing = labels.difference(field.indexes[dim])
                if missing.size:
                    raise GridMismatchError(
                        f"{fname!r} lacks {dim!r} labels {list(missing[:5])} of {vname!r}; the grid "
                        "data does not cover the variable."
                    )
                field = field.sel({dim: labels})
        elif field.sizes[dim] != like.sizes[dim]:
            if dim in ("s_rho", "s_w"):
                advice = "; " + _LEVELS_ADVICE.format(dim=dim)
            else:
                advice = ". Subset the Dataset rather than a single variable"
                advice += ", or run xroms.decode_time(ds) first for UCLA ROMS output." if _is_time_dim(field, dim) else "."
            raise GridMismatchError(
                f"{vname!r} and {fname!r} have different lengths along {dim!r} "
                f"({like.sizes[dim]} vs {field.sizes[dim]}) and no index coordinates to align them"
                + advice
            )

    # horizontal footprint per axis
    for axis, (center, stag) in HAXES.items():
        vdim = center if center in like.dims else stag if stag in like.dims else None
        fdim = center if center in field.dims else stag if stag in field.dims else None
        if vdim is None or fdim is None:
            continue
        offset = 0 if vdim == fdim else (1 if (vdim == stag and fdim == center) else -1)
        expected = like.sizes[vdim] + offset
        if vdim in like.indexes:
            check_unstrided(like[vdim].values, vdim, vname)
        if fdim in field.indexes:
            check_unstrided(field[fdim].values, fdim, fname)
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


def attach_grid_coords(out, grid):
    """``out`` with ``grid``'s horizontal coordinates at its position: lon/lat, else x/y.

    A result that lands somewhere its inputs were not (a derivative on psi
    points, z at u points) loses their coordinates; this puts the grid's own
    back, matched to the result like any grid field. Only the grid's
    coordinates are attached, never its data variables: a file that keeps
    lon/lat as data variables (UCLA's grid) would otherwise get results that
    cannot be stored back in it (``ds.set_coords`` makes them coordinates).
    Coordinates the result already has are kept, and ones that cannot be
    matched are left out. Tuples of results are handled element by element.
    """
    if isinstance(out, tuple):
        return tuple(attach_grid_coords(item, grid) for item in out)
    if not isinstance(out, xr.DataArray) or not isinstance(grid, xr.Dataset):
        return out
    pos = hposition(out)
    if pos is None:
        return out
    coords = {}
    for name in horizontal_coords(grid, pos):
        if name is None or name in out.coords or name not in grid.coords:
            continue
        try:
            field = select_like(grid[name], out, name=name)
        except GridMismatchError:
            continue
        if set(field.dims) <= set(out.dims):
            # the bare variable: no index coords are added to the result
            coords[name] = field.variable
    return out.assign_coords(coords) if coords else out


def with_grid_coords(func):
    """Give the result of ``func``, which reads a ``grid``, that grid's coordinates at its position."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return attach_grid_coords(out, signature.bind_partial(*args, **kwargs).arguments.get("grid"))

    return wrapper


def require(grid, *names, purpose=None):
    """Raise a clear error listing which grid variables are missing."""
    missing = [n for n in names if n not in grid.variables]
    if missing:
        why = f" for {purpose}" if purpose else ""
        raise GridMismatchError(
            f"grid variables {missing} are needed{why} but not found. For UCLA ROMS or CROCO "
            "output the grid is often a separate file: merge it with "
            "xroms.merge_grid(ds, grid) or pass grid=<grid Dataset>."
        )
