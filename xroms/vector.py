"""Functions related to vectors.

:func:`rotate_vectors` rotates any pair of components; :func:`grid_to_earth` and
:func:`earth_to_grid` are the two common uses for ROMS velocities, between the
grid-aligned ``u``/``v`` on their staggered points and eastward/northward
components on rho points. Nothing needs a grid object: positions come from the
dimension names of the inputs.
"""

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from .conventions import hposition
from .utilities import _where_data, order, to_grid, to_u, to_v


def _rotated_attrs(src, axis):
    """Default attrs of a rotated component, from the component it was made from."""
    base = src.attrs.get("name") or src.name or axis
    long_name = src.attrs.get("long_name")
    return {
        "name": f"{base}_rot",
        "long_name": f"{long_name}, rotated" if long_name else f"rotated {axis} component",
        "units": src.attrs.get("units", ""),
    }


def _check_uv_points(u, v):
    """Raise a ValueError if ``u`` is on v points or ``v`` is on u points.

    That is what passing the two the wrong way round looks like. Both on rho points
    is fine (velocities that have already been averaged there), and so is any
    position that cannot be told from the dims.
    """
    found = hposition(u), hposition(v)
    if found[0] == "v" or found[1] == "u":
        where = [
            f"{name} is on {pos} points" if pos else f"{name} has dims {da.dims}, which do not give its points"
            for name, pos, da in (("u", found[0], u), ("v", found[1], v))
        ]
        raise ValueError(
            f"grid_to_earth needs u on u points (or rho points) and v on v points (or rho points), but "
            f"{where[0]} and {where[1]}. Check that u (the xi component) and v (the eta component) are "
            "not swapped."
        )


def rotate_vectors(
    x: Union[float, np.ndarray, xr.DataArray],
    y: Union[float, np.ndarray, xr.DataArray],
    angle: Union[float, np.ndarray, xr.DataArray],
    isradians: bool = True,
    reference: str = "xaxis",
    *,
    hcoord="rho",
    hboundary="extend",
    hfill_value=np.nan,
    attrs: Optional[dict] = None,
    xgrid=None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Rotate vectors according to reference.

    Parameters
    ----------
    x : Union[float, np.ndarray, xr.DataArray]
        x component of vector to be rotated
    y : Union[float, np.ndarray, xr.DataArray]
        y component of vector to be rotated
    angle : Union[float, np.ndarray, xr.DataArray]
        Angle by which to rotate x and y.
    isradians : bool, optional
        True if angle is in radians, False for degrees, by default True
    reference : str, optional
        Which reference is angle coming from? "xaxis" if angle is the angle between the x-axis and x (positive going counter clockwise, 0 at the x axis), or "compass" if angle is 0 at north on a compass and is positive going clockwise, by default "xaxis".
    hcoord : string, optional.
        Name of horizontal grid to move DataArray inputs to before rotating, so
        that the components and the angle are at the same points.
        Options are 'rho', 'psi', 'u', 'v', or None to leave them where they are.
        Default 'rho'. Numbers and arrays are not moved.
    hboundary : {"extend", "fill"}, optional
        Horizontal boundary treatment when moving DataArray inputs to ``hcoord``
        (see `xroms.to_grid`): "extend" (default) repeats the nearest value at the
        domain edge, "fill" uses ``hfill_value`` there.
    hfill_value : float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.
    attrs : Optional[dict], optional
        Dict containing two keys, "x" and "y", each a dict of attributes, by default None. Attributes should include "name", "standard_name", "long_name", "units", if possible. "name" is required.
        Only applied when the results are DataArrays. If None and both
        components are DataArrays, the results are named from them
        (``<name>_rot``).
    xgrid : None
        Removed in xroms 1.0, which needs no grid object: passing one raises a
        `TypeError`.

    Returns
    -------
    Tuple[xr.DataArray]
        x and y, rotated by angle. Nothing passed in is modified. DataArrays are
        ordered (time, vertical, eta, xi, then any other dimensions).

    Examples
    --------
    >>> xroms.rotate_vectors(1, 0, 90, isradians=False)
    >>> xroms.rotate_vectors(ds.u, ds.v, ds.angle)  # u and v are moved to rho points first
    """
    if xgrid is not None:
        raise TypeError("xroms 1.0: rotate_vectors no longer needs xgrid; remove it")
    if reference not in (None, "xaxis", "compass"):
        raise ValueError(f"reference must be 'xaxis' or 'compass', not {reference!r}")
    if attrs is not None and (("x" not in attrs) or ("y" not in attrs)):
        raise KeyError("if you input attributes, make a dict for each of x and y attributes.")

    # make sure components are on the same grid
    moves = dict(hcoord=hcoord, hboundary=hboundary, hfill_value=hfill_value)
    if isinstance(x, xr.DataArray):
        x = to_grid(x, **moves)
    if isinstance(y, xr.DataArray):
        y = to_grid(y, **moves)
    if isinstance(angle, xr.DataArray):
        angle = to_grid(angle, **moves)

    # everything is in radians after this
    if not isradians:
        angle = np.deg2rad(angle)

    # not `angle *= -1`: that would change the caller's angle in place
    if reference == "compass":
        angle = -angle

    # perform rotation
    xrot = x * np.cos(angle) - y * np.sin(angle)
    yrot = x * np.sin(angle) + y * np.cos(angle)
    if isinstance(xrot, xr.DataArray):
        # the angle may bring dimensions of its own
        xrot, yrot = order(xrot), order(yrot)

    if attrs is not None:
        if isinstance(xrot, xr.DataArray) and isinstance(yrot, xr.DataArray):
            for rot, key in ((xrot, "x"), (yrot, "y")):
                if "name" not in attrs[key]:
                    raise KeyError(f'attrs["{key}"] needs a "name".')
                rot.attrs = dict(attrs[key])
                rot.name = rot.attrs["name"]
    elif isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        for rot, src, key in ((xrot, x, "x"), (yrot, y, "y")):
            rot.attrs = _rotated_attrs(src, key)
            rot.name = rot.attrs["name"]

    return xrot, yrot


def grid_to_earth(u, v, angle, *, hcoord="rho", hboundary="extend"):
    """Rotate grid-aligned velocity (u, v) to eastward and northward components.

    Parameters
    ----------
    u : DataArray
        Velocity along xi (on u points, or on rho points if already averaged there).
    v : DataArray
        Velocity along eta (on v points, or on rho points if already averaged there).
    angle : DataArray, float
        Angle [radians] between the xi axis and east, as in the ROMS grid
        variable ``angle`` (positive counterclockwise).
    hcoord : string, optional
        Horizontal grid the results are on: 'rho' (default), 'psi', 'u' or 'v'.
    hboundary : string, optional
        Boundary treatment ("extend" or "fill") when averaging u and v from
        their own points to ``hcoord``.

    Returns
    -------
    east, north : DataArray
        Eastward and northward velocity [m/s] on ``hcoord`` points (named
        ``east`` and ``north``). Points without any u or v data around them
        (land) are NaN.

    Raises
    ------
    ValueError
        If u is on v points or v is on u points (for example, if they are swapped).

    Notes
    -----
    Masked (NaN) u and v are set to zero before they are averaged onto
    ``hcoord``, so that cells next to land keep the neighbouring value instead of
    becoming masked themselves.

    Examples
    --------
    >>> east, north = xroms.vector.grid_to_earth(ds.u, ds.v, ds.angle)
    """
    if not isinstance(u, xr.DataArray) or not isinstance(v, xr.DataArray):
        raise TypeError("u and v must be DataArrays")
    _check_uv_points(u, v)

    east_attrs = {
        "name": "east",
        "standard_name": "eastward_sea_water_velocity",
        "long_name": "u rotated to eastward axis",
        "units": "m/s",
    }
    north_attrs = {
        "name": "north",
        "standard_name": "northward_sea_water_velocity",
        "long_name": "v rotated to northward axis",
        "units": "m/s",
    }

    # need to fill nans with zeros so that the masked locations in
    # velocity fields are not fully brought forward into the rho mask
    # but are instead interpolated over. By making them 0, they are
    # calculated into the mask_rho positions by combining them with
    # neighboring cells. If this wasn't done, the fact that they are masked
    # would supersede the neighboring cells and they would be masked in mask_rho.
    # this needs to be done anytime the velocities are moved from their native
    # grids to the rho or other grids to preserve their locations around masked cells.
    east, north = rotate_vectors(
        u.fillna(0),
        v.fillna(0),
        angle,
        isradians=True,
        reference="xaxis",
        hcoord=hcoord,
        hboundary=hboundary,
        attrs={"x": east_attrs, "y": north_attrs},
    )
    # points without any u or v data around them (land) are NaN, not the 0 of the fill
    return tuple(_where_data(c, u, v, hcoord=hcoord, hboundary=hboundary) for c in (east, north))


def earth_to_grid(east, north, angle, *, hcoord="native"):
    """Rotate eastward and northward velocity to grid-aligned (u, v) components.

    The inverse of :func:`grid_to_earth`.

    Parameters
    ----------
    east : DataArray
        Eastward velocity.
    north : DataArray
        Northward velocity.
    angle : DataArray, float
        Angle [radians] between the xi axis and east, as in the ROMS grid
        variable ``angle`` (positive counterclockwise).
    hcoord : string, optional
        Horizontal grid the results are on. "native" (default) puts u on u
        points and v on v points; 'rho', 'psi', 'u' or 'v' put both there.

    Returns
    -------
    u, v : DataArray
        Velocity along xi and along eta [m/s] (named ``u`` and ``v``).

    Notes
    -----
    The components are rotated on rho points, where ``east``, ``north`` and
    ``angle`` are brought if they are elsewhere, and then averaged to u and v
    points for ``hcoord="native"``. The averaging means the first and last u
    (v) points along xi (eta) are not recovered exactly by a round trip from
    :func:`grid_to_earth`; the interior is exact for fields that vary linearly.

    Examples
    --------
    >>> u, v = xroms.vector.earth_to_grid(east, north, ds.angle)
    """
    if not isinstance(east, xr.DataArray) or not isinstance(north, xr.DataArray):
        raise TypeError("east and north must be DataArrays")
    if hcoord not in ("native", "rho", "u", "v", "psi"):
        raise ValueError(f"hcoord must be 'native', 'rho', 'u', 'v' or 'psi', not {hcoord!r}")

    u_attrs = {
        "name": "u",
        "standard_name": "sea_water_x_velocity",
        "long_name": "u-momentum component",
        "units": "m/s",
    }
    v_attrs = {
        "name": "v",
        "standard_name": "sea_water_y_velocity",
        "long_name": "v-momentum component",
        "units": "m/s",
    }

    u, v = rotate_vectors(
        east,
        north,
        -angle,
        isradians=True,
        reference="xaxis",
        hcoord="rho" if hcoord == "native" else hcoord,
        attrs={"x": u_attrs, "y": v_attrs},
    )
    if hcoord == "native":
        u, v = to_u(u), to_v(v)
    return u, v
