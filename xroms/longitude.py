"""Longitude conventions, and the longitude and latitude of ROMS grid points.

Grids store longitude as ``-180..180`` or as ``0..360``, and a domain that crosses
the prime meridian or the dateline is one contiguous span in only one of them.
:func:`wrap_longitude` moves longitudes between the conventions, :func:`straddles`
says whether a domain crosses the prime meridian (roms-tools' ``straddle`` flag),
and :func:`lonlat_at` gives the longitude and latitude of a grid at any stagger,
averaging longitudes where they are contiguous.

Everything here is pure: inputs are never modified, attrs are kept, and dask-backed
arrays stay lazy. Reading the convention off the data (``wrap_longitude`` without
``convention``, :func:`straddles`, interpolating in :func:`lonlat_at`) computes the
extent of the longitudes, never the longitudes themselves.
"""

import warnings

import numpy as np
import xarray as xr

from ._align import GridMismatchError, _check_grid
from .conventions import HCOORDS, horizontal_coords
from .utilities import order, to_grid


CONVENTIONS = ("-180-180", "0-360")

#: Degrees within which two spans, or a longitude and 180, count as equal. A domain nowhere
#: near the dateline is a tie between the conventions, and rounding must not break the tie
#: (the same tolerance ocean-skill's ``natural_convention`` uses).
_TOL = 1e-6


# --- finding longitudes ------------------------------------------------------------


def _is_longitude(name, var):
    """True for a numeric variable named ``lon``, ``longitude`` or ``lon_*``, or with ``standard_name`` longitude."""
    if var.dtype.kind not in "if":
        return False
    named = isinstance(name, str) and (name in ("lon", "longitude") or name.startswith("lon_"))
    return named or var.attrs.get("standard_name") == "longitude"


def _longitudes_in(obj):
    """Names of the longitude variables of a Dataset, or of the longitude coordinates of a DataArray."""
    pool = obj.variables if isinstance(obj, xr.Dataset) else obj.coords
    return [name for name in pool if _is_longitude(name, pool[name])]


def _reference(obj):
    """The longitudes a convention is read from: ``lon_rho`` if there is one, else the first found.

    A DataArray without longitude coordinates is its own reference, and so is anything
    that is not xarray; a Dataset without longitudes has none (None).
    """
    if not isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj
    names = _longitudes_in(obj)
    if names:
        return obj["lon_rho" if "lon_rho" in names else names[0]]
    return obj if isinstance(obj, xr.DataArray) else None


# --- which convention a domain is contiguous in ------------------------------------------


def _frames(lon):
    """``(span in -180..180, span in 0..360, any negative)`` of the finite values of ``lon``.

    Spans are NaN without finite values. A value at +180 counts as 180 (the seam itself)
    in the -180..180 frame instead of wrapping to -180, which would stretch a domain that
    reaches the dateline without crossing it over the whole globe.
    """
    if isinstance(lon, xr.DataArray):
        values = xr.DataArray(lon.data, dims=lon.dims)  # no coords, so the reductions below stay scalars
    else:
        values = xr.DataArray(np.asarray(lon))
    if values.dtype.kind not in "if":
        raise TypeError(f"longitudes must be numbers, not {values.dtype}")
    if values.size == 0:
        return np.nan, np.nan, False
    values = values.astype("float64")
    values = values.where(np.isfinite(values))
    w180 = xr.where(abs(values - 180.0) <= _TOL, 180.0, ((values + 180.0) % 360.0) - 180.0)
    w360 = values % 360.0
    extent = xr.Dataset(
        {"lo180": w180.min(), "hi180": w180.max(), "lo360": w360.min(), "hi360": w360.max(), "negative": (values < 0).any()}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice": nothing finite to measure
        extent = extent.compute()  # one pass over lazy data
    return (
        float(extent["hi180"] - extent["lo180"]),
        float(extent["hi360"] - extent["lo360"]),
        bool(extent["negative"]),
    )


def _pick(span180, span360):
    """The convention with the shorter span, None for a tie (or when there is nothing to measure)."""
    if span180 < span360 - _TOL:
        return "-180-180"
    if span360 < span180 - _TOL:
        return "0-360"
    return None


def straddles(lon):
    """True if the domain crosses the prime meridian, so it is one span only in ``-180..180``.

    This is roms-tools' ``straddle`` flag, measured from the values rather than from
    jumps in the stored longitudes: it is True when the span of the longitudes in
    ``-180..180`` is shorter than in ``0..360``. A domain that is contiguous in both
    conventions (``10..20E``, or a global one) and one that crosses the dateline
    instead (``77E..316E``, contiguous only in ``0..360``) give False.

    Parameters
    ----------
    lon : number, array, DataArray or Dataset
        Longitudes in degrees east in any convention, or a Dataset or DataArray that
        carries them: ``lon_rho`` is used if there is one, else the first longitude
        variable found (see `wrap_longitude`). NaN is ignored.

    Returns
    -------
    bool
        Computes the extent of the longitudes if they are lazy.

    Raises
    ------
    ValueError
        If a Dataset has no longitude variable.

    Examples
    --------
    >>> xroms.straddles([350.0, 355.0, 5.0, 10.0])
    True
    >>> xroms.straddles([150.0, 200.0, 250.0])
    False
    """
    reference = _reference(lon)
    if reference is None:
        raise ValueError(
            "no longitude found: straddles needs a variable named lon, longitude or lon_*, or with standard_name 'longitude'"
        )
    return _pick(*_frames(reference)[:2]) == "-180-180"


# --- wrapping ------------------------------------------------------------------------


def _wrap(lon, convention):
    """``lon`` (a numpy array) in ``convention``; integers stay integers and floats keep their dtype."""
    with np.errstate(invalid="ignore"):  # inf has no wrapped value, and comes out NaN
        if convention == "0-360":
            out = np.mod(lon, 360)
            out = np.where(out >= 360, 0, out)  # a tiny negative rounds up to 360.0
        else:
            out = 180 - np.mod(180 - lon, 360)
            out = np.where(out <= -180, 180, out)  # the same rounding at -180
            out = np.where((lon > -180) & (lon <= 180), lon + 0, out)  # in range is left bit for bit as it was (-0.0 becomes 0.0)
    return out.astype(lon.dtype, copy=False)


def _wrap_xr(var, convention):
    """Wrap a Variable or DataArray of longitudes, lazily for dask, keeping attrs (and name and coords)."""
    if var.dtype.kind not in "if":
        raise TypeError(f"longitudes must be numbers, not {var.dtype}")
    return xr.apply_ufunc(
        _wrap, var, kwargs={"convention": convention}, dask="parallelized", output_dtypes=[var.dtype], keep_attrs=True
    )


def _resort(obj, names):
    """Re-sort along longitude index coordinates that wrapping left non-monotonic (2-D coordinates never are)."""
    for name in names:
        if name in obj.indexes and name in obj.dims:
            index = obj.indexes[name]
            if not (index.is_monotonic_increasing or index.is_monotonic_decreasing):
                obj = obj.sortby(name)
    return obj


def wrap_longitude(obj, convention=None):
    """Longitudes in the ``-180..180`` or the ``0..360`` convention.

    Parameters
    ----------
    obj : number, array, DataArray or Dataset
        Longitudes in degrees east, in any convention and any number of turns away.
        A Dataset, or a DataArray with longitude coordinates, has every longitude
        variable and coordinate wrapped with one shared convention; a DataArray
        without longitude coordinates is treated as longitudes itself. A variable is
        a longitude if its name is ``lon``, ``longitude`` or starts with ``lon_``, or
        its ``standard_name`` is ``"longitude"``.
    convention : {"-180-180", "0-360"} or None
        ``"-180-180"`` gives values in (-180, 180] (180 stays 180, -180 becomes 180),
        ``"0-360"`` values in [0, 360). None (default) picks the convention in which the
        longitudes form one contiguous span: ``"-180-180"`` for a domain that crosses
        the prime meridian (see `straddles`), ``"0-360"`` for one that crosses the
        dateline, such as a Pacific domain stored as ``77..180`` and ``-180..-44``.
        Longitudes that are contiguous in both (``10..20E``, a global grid, a single
        value) are returned unchanged. The convention comes from ``lon_rho`` if there
        is one, else from the first longitude found, and may compute the extent of
        lazy longitudes.

    Returns
    -------
    number, array, DataArray or Dataset
        Of the type of ``obj``, with names, attrs and (floating) dtypes kept, lazy if
        ``obj`` was, and NaN left as NaN. A 1-D longitude index coordinate that wrapping
        leaves out of order is sorted again (along with the data that depend on it); 2-D
        coordinates are never reordered. A Dataset without longitudes is returned as it is.

    Raises
    ------
    ValueError
        If ``convention`` is anything but ``"-180-180"``, ``"0-360"`` or None.

    Examples
    --------
    >>> xroms.wrap_longitude(190.0)
    190.0
    >>> xroms.wrap_longitude(190.0, "-180-180")
    -170.0
    >>> xroms.wrap_longitude(ds, "0-360")  # every lon_* of ds, as roms-tools stores them
    """
    if convention is not None and convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS} or None, not {convention!r}")
    if isinstance(obj, xr.Dataset):
        return _wrap_dataset(obj, convention)
    if isinstance(obj, xr.DataArray):
        return _wrap_dataarray(obj, convention)
    values = np.asarray(obj)
    if values.dtype.kind not in "if":
        raise TypeError(f"longitudes must be numbers, not {values.dtype}")
    convention = convention or _pick(*_frames(values)[:2])
    out = values.copy() if convention is None else _wrap(values, convention)
    if isinstance(obj, (int, float)) and not isinstance(obj, np.generic):
        return out.item()  # a Python number in, a Python number out
    return out[()] if out.ndim == 0 else out


def _wrap_dataset(ds, convention):
    names = _longitudes_in(ds)
    if not names:
        return ds
    convention = convention or _pick(*_frames(_reference(ds))[:2])
    if convention is None:
        return ds
    wrapped = {name: _wrap_xr(ds.variables[name], convention) for name in names}
    out = ds.assign_coords({n: v for n, v in wrapped.items() if n in ds.coords})
    out = out.assign({n: v for n, v in wrapped.items() if n not in ds.coords})
    return _resort(out, names)


def _wrap_dataarray(da, convention):
    names = _longitudes_in(da)
    convention = convention or _pick(*_frames(_reference(da))[:2])
    if convention is None:
        return da
    out = _wrap_xr(da, convention) if (not names or _is_longitude(da.name, da)) else da
    out = out.assign_coords({name: _wrap_xr(da.coords[name].variable, convention) for name in names})
    return _resort(out, names)


# --- longitude and latitude at the points of a grid ------------------------------------


def _labelled(da, name, long_name, units=None, standard_name=None):
    attrs = {"long_name": long_name}
    if units is not None:
        attrs["units"] = units
    if standard_name is not None:
        attrs["standard_name"] = standard_name
    out = da.rename(name)
    out.attrs = attrs
    return out


def _lon_at(lon, hcoord):
    """Rho longitudes averaged onto ``hcoord`` where they are contiguous, then back in their stored convention."""
    span180, span360, negative = _frames(lon)
    contiguous = _pick(span180, span360)
    if contiguous is not None:  # e.g. 359.5 and 0.5 are averaged as -0.5 and 0.5, not left to give 180
        lon = _wrap_xr(lon, contiguous)
    return _wrap_xr(to_grid(lon, hcoord), "-180-180" if negative else "0-360")


def lonlat_at(grid, hcoord="u"):
    """Longitude and latitude (or Cartesian x and y) of the ``hcoord`` points of ``grid``.

    Parameters
    ----------
    grid : Dataset
        Holds ``lon_rho`` and ``lat_rho`` (or ``x_rho`` and ``y_rho``) and, optionally,
        the same at ``hcoord``, in any ROMS-family layout.
    hcoord : {"rho", "u", "v", "psi"}
        Horizontal position of the points. Default ``"u"``.

    Returns
    -------
    lon, lat : DataArray
        Named ``lon_<hcoord>`` and ``lat_<hcoord>`` (``x_<hcoord>`` and ``y_<hcoord>``
        for a Cartesian grid), on canonical dims: rho ``(eta_rho, xi_rho)``, u
        ``(eta_rho, xi_u)``, v ``(eta_v, xi_rho)``, psi ``(eta_v, xi_u)``, with
        ``long_name``, ``units`` and, for lon/lat, ``standard_name`` attrs.

    Raises
    ------
    ValueError
        If ``hcoord`` is not a horizontal position, or ``grid`` has neither the pair at
        ``hcoord`` nor lon/lat or x/y at rho points to average from.

    Notes
    -----
    A pair stored in ``grid`` at ``hcoord`` is returned as it is. Otherwise the rho
    positions are averaged onto ``hcoord`` (`xroms.to_grid`), which is approximate: prefer
    the positions a grid file holds. Longitudes are averaged in the convention in which the
    rho longitudes are contiguous, so on a grid stored in ``0..360`` that crosses Greenwich
    359.5 and 0.5 average to 0.0, not 180.0. The result is then written in the convention
    the rho longitudes are stored in: ``-180..180`` if any of them is negative, else
    ``0..360``. Lazy positions stay lazy, but their extent is computed to find the convention.

    Examples
    --------
    >>> lon_u, lat_u = xroms.lonlat_at(grid, "u")
    """
    if hcoord not in HCOORDS:
        raise ValueError(f"hcoord must be one of {HCOORDS}, not {hcoord!r}")
    grid = _check_grid(grid, "lonlat_at")
    stored = horizontal_coords(grid, hcoord)
    if stored[0] is not None:
        return tuple(order(grid[name].reset_coords(drop=True)) for name in stored)
    xname, yname = horizontal_coords(grid, "rho")
    if xname is None:
        raise GridMismatchError(
            f"the grid has neither lon_rho/lat_rho nor x_rho/y_rho, so the {hcoord} points cannot be placed. For UCLA "
            "ROMS or CROCO output the grid is often a separate file: pass it, or merge it with "
            "xr.merge([ds, grid], compat='override')."
        )
    x, y = grid[xname].reset_coords(drop=True), grid[yname].reset_coords(drop=True)
    if xname.startswith("lon_"):
        return (
            _labelled(_lon_at(x, hcoord), f"lon_{hcoord}", f"longitude of {hcoord}-points", "degrees_east", "longitude"),
            _labelled(to_grid(y, hcoord), f"lat_{hcoord}", f"latitude of {hcoord}-points", "degrees_north", "latitude"),
        )
    return (
        _labelled(to_grid(x, hcoord), f"x_{hcoord}", f"x-location of {hcoord}-points", x.attrs.get("units")),
        _labelled(to_grid(y, hcoord), f"y_{hcoord}", f"y-location of {hcoord}-points", y.attrs.get("units")),
    )
