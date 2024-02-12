"""Functions related to vectors."""

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
import xgcm

from .utilities import to_grid


def rotate_vectors(
    x: Union[float, np.array, xr.DataArray],
    y: Union[float, np.array, xr.DataArray],
    angle: Union[float, np.array, xr.DataArray],
    isradians: bool = True,
    reference: str = "xaxis",
    xgrid: Optional[xgcm.grid.Grid] = None,
    hcoord="rho",
    attrs: Optional[dict] = None,
    **kwargs,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Rotate vectors according to reference.

    Parameters
    ----------
    x : Union[float, np.array, xr.DataArray]
        x component of vector to be rotated
    y : Union[float, np.array, xr.DataArray]
        y component of vector to be rotated
    angle : Union[float,np.array,xr.DataArray]
        Angle by which to rotate x and y.
    isradians : bool, optional
        True if angle is in radians, False for degrees, by default True
    reference : str, optional
        Which reference is angle coming from? "xaxis" if angle is the angle between the x-axis and x (positive going counter clockwise, 0 at the x axis), or "compass" if angle is 0 at north on a compass and is positive going clockwise, by default "xaxis".
    xgrid : Optional[xgcm.grid.Grid], optional
        xgcm grid, by default None. If not input, any grid changing using hcoord or kwargs is ignored.
    hcoord: string, optional.
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'. Default 'rho'.
    attrs : Optional[dict], optional
        Dict containing two keys, "x" and "y", each a dict of attributes, by default None. Attributes should include "name", "standard_name", "long_name", "units", if possible. "name" is required.
    kwargs :
        will be passed on to `xroms.to_grid()`.

    Returns
    -------
    Tuple[xr.DataArray]
        x and y, rotated by angle.
    """

    if xgrid is not None:
        # make sure components are on the same grid
        if isinstance(x, xr.DataArray):
            x = to_grid(x, xgrid, hcoord=hcoord, **kwargs)
        if isinstance(y, xr.DataArray):
            y = to_grid(y, xgrid, hcoord=hcoord, **kwargs)
        if isinstance(angle, xr.DataArray):
            angle = to_grid(angle, xgrid, hcoord=hcoord, **kwargs)

    # everything is in radians after this
    if not isradians:
        angle = np.deg2rad(angle)

    if reference == "compass":
        angle *= -1

    # perform rotation
    xrot = x * np.cos(angle) - y * np.sin(angle)
    yrot = x * np.sin(angle) + y * np.cos(angle)

    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray) and attrs is None:
        xrot.attrs["name"] = f'{x.attrs["name"]}_rot'
        xrot.attrs["long_name"] = (
            f'{x.attrs["long_name"]}, rotated' or "rotated x component"
        )
        xrot.attrs["units"] = x.attrs["units"] or ""
        xrot.name = xrot.attrs["name"]

        yrot.attrs["name"] = f'{y.attrs["name"]}_rot'
        yrot.attrs["long_name"] = (
            f'{y.attrs["long_name"]}, rotated' or "rotated y component"
        )
        yrot.attrs["units"] = y.attrs["units"] or ""
        yrot.name = yrot.attrs["name"]
    elif attrs is not None:
        if ("x" not in attrs) or ("y" not in attrs):
            raise KeyError(
                "if you input attributes, make a dict for each of x and y attributes."
            )
        xrot.attrs, yrot.attrs = attrs["x"], attrs["y"]
        xrot.name = xrot.attrs["name"]
        yrot.name = yrot.attrs["name"]

    return xrot, yrot
