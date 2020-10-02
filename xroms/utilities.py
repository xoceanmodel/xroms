import warnings

import numpy as np
import xarray as xr

import xroms


try:
    import cartopy.geodesic
except ImportError:
    warnings.warn(
        "cartopy is not installed, so `sel2d` and `argsel2d` will not run.",
        ImportWarning,
    )


def hgrad(
    q,
    grid,
    which="both",
    z=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
    attrs=None,
):
    """Return gradients of property q accounting for s coordinates.

    Inputs
    ------

    q: DataArray
        Property to take gradients of.
    grid: xgcm.grid
        Grid object associated with q.
    which: string, optional
        Which components of gradient to return.
        * 'both': return both components of hgrad.
        * 'xi': return only xi-direction.
        * 'eta': return only eta-direction.
    z: DataArray, ndarray, optional
        Depth [m]. If None, use z coordinate attached to q.
    hcoord: string, optional
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivatives of q. This same value
        will be used for all horizontal grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for calculating horizontal derivatives of q. This same value will
        be used for all vertical grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    attrs: dict, optional
        Dictionary of attributes to add to resultant arrays. Requires that
        q is DataArray.

    Returns
    -------
    DataArray(s) of dqdxi and/or dqdeta, the gradients of q
    in the xi- and eta-directions with attributes altered to reflect calculation.

    Notes
    -----
    dqdxi = dqdx*dzdz - dqdz*dzdx

    dqdeta = dqdy*dzdz - dqdz*dzdy

    Derivatives are taken in the ROMS curvilinear grid native xi- and eta- directions.

    These derivatives properly account for the fact that ROMS vertical coordinates are
    s coordinates and therefore can vary in time and space.

    The xi derivative will alter the number of points in the xi and s dimensions.
    The eta derivative will alter the number of points in the eta and s dimensions.

    Example usage
    -------------
    >>> dtempdxi, dtempdeta = xroms.hgrad(ds.temp, grid)
    """

    assert isinstance(q, xr.DataArray), "var must be DataArray"

    if z is None:
        try:
            coords = list(q.coords)
            z_coord_name = coords[[coord[:2] == "z_" for coord in coords].index(True)]
            z = q[z_coord_name]
            is3D = True
        except:
            # if we get here that means that q doesn't have z coords (like zeta)
            is3D = False
    else:
        is3D = True

    if which in ["both", "xi"]:

        if is3D:
            dqdx = grid.interp(
                grid.derivative(q, "X", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dqdz = grid.interp(
                grid.derivative(q, "Z", boundary=sboundary, fill_value=sfill_value),
                "X",
                boundary=hboundary,
                fill_value=hfill_value,
            )
            dzdx = grid.interp(
                grid.derivative(z, "X", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dzdz = grid.interp(
                grid.derivative(z, "Z", boundary=sboundary, fill_value=sfill_value),
                "X",
                boundary=hboundary,
                fill_value=hfill_value,
            )

            dqdxi = dqdx * dzdz - dqdz * dzdx

        else:  # 2D variables
            dqdxi = grid.derivative(q, "X", boundary=hboundary, fill_value=hfill_value)

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs["name"] = "d" + q.name + "dxi"
            attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
            attrs["long_name"] = "horizontal xi derivative of " + attrs.setdefault(
                "long_name", "var"
            )
            attrs["grid"] = grid
        dqdxi = xroms.to_grid(
            dqdxi,
            grid,
            hcoord=hcoord,
            scoord=scoord,
            attrs=attrs,
            hboundary=hboundary,
            hfill_value=hfill_value,
            sboundary=sboundary,
            sfill_value=sfill_value,
        )

    if which in ["both", "eta"]:

        if is3D:
            dqdy = grid.interp(
                grid.derivative(q, "Y", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dqdz = grid.interp(
                grid.derivative(q, "Z", boundary=sboundary, fill_value=sfill_value),
                "Y",
                boundary=hboundary,
                fill_value=hfill_value,
            )
            dzdy = grid.interp(
                grid.derivative(z, "Y", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dzdz = grid.interp(
                grid.derivative(z, "Z", boundary=sboundary, fill_value=sfill_value),
                "Y",
                boundary=hboundary,
                fill_value=hfill_value,
            )

            dqdeta = dqdy * dzdz - dqdz * dzdy

        else:  # 2D variables
            dqdeta = grid.derivative(q, "Y", boundary=hboundary, fill_value=hfill_value)

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs["name"] = "d" + q.name + "deta"
            attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
            attrs["long_name"] = "horizontal eta derivative of " + attrs.setdefault(
                "long_name", "var"
            )
            attrs["grid"] = grid
        dqdeta = xroms.to_grid(
            dqdeta,
            grid,
            hcoord=hcoord,
            scoord=scoord,
            attrs=attrs,
            hboundary=hboundary,
            hfill_value=hfill_value,
            sboundary=sboundary,
            sfill_value=sfill_value,
        )

    if which == "both":
        return dqdxi, dqdeta
    elif which == "xi":
        return dqdxi
    elif which == "eta":
        return dqdeta
    else:
        print("nothing being returned from hgrad")


def ddxi(
    var,
    grid,
    z=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    attrs=None,
):
    """Calculate d/dxi for a variable.

    Inputs
    ------
    var: DataArray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var.
    z: DataArray, ndarray, optional
        Depth [m]. If None, use z coordinate attached to var.
    hcoord: string, optional
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivative of var. This same value
        will be used for all horizontal grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for calculating horizontal derivative of var. This same value will
        be used for all vertical grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    attrs: dict, optional
        Dictionary of attributes to add to resultant arrays. Requires that
        q is DataArray. For example:
        `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

    Returns
    -------
    DataArray of dqdxi, the gradient of q in the xi-direction with
    attributes altered to reflect calculation.

    Notes
    -----
    dqdxi = dqdx*dzdz - dqdz*dzdx

    Derivatives are taken in the ROMS curvilinear grid native xi-direction.

    These derivatives properly account for the fact that ROMS vertical coordinates are
    s coordinates and therefore can vary in time and space.

    This will alter the number of points in the xi and s dimensions.

    Example usage
    -------------
    >>> xroms.ddxi(ds.salt, grid)
    """

    var = xroms.hgrad(
        var,
        grid,
        which="xi",
        z=z,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
        attrs=attrs,
    )
    return var


def ddeta(
    var,
    grid,
    z=None,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    attrs=None,
):
    """Calculate d/deta for a variable.

    Inputs
    ------
    var: DataArray, ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var.
    z: DataArray, ndarray, optional
        Depth [m]. If None, use z coordinate attached to var.
    hcoord: string, optional
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivative of var. This same value
        will be used for grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for calculating horizontal derivative of var. This same value will
        be used for vertical grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    attrs: dict, optional
        Dictionary of attributes to add to resultant arrays. Requires that
        q is DataArray. For example:
        `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

    Returns
    -------
    DataArray or ndarray of dqdeta, the gradient of q in the eta-direction with
    attributes altered to reflect calculation.

    Notes
    -----
    dqdeta = dqdy*dzdz - dqdz*dzdy

    Derivatives are taken in the ROMS curvilinear grid native eta-direction.

    These derivatives properly account for the fact that ROMS vertical coordinates are
    s coordinates and therefore can vary in time and space.

    This will alter the number of points in the eta and s dimensions.

    Example usage
    -------------
    >>> xroms.ddeta(ds.salt, grid)
    """

    var = xroms.hgrad(
        var,
        grid,
        which="eta",
        z=z,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
        attrs=attrs,
    )
    return var


def ddz(
    var,
    grid,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
    attrs=None,
):
    """Calculate d/dz for a variable.

    Inputs
    ------
    var: DataArray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hcoord: string, optional.
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional.
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivative of var. This same value
        will be used for grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection for
        calculating z derivative. This same value
        will be used for grid changes too.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary fill value
        associated with sboundary input.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    attrs: dict, optional
        Dictionary of attributes to add to resultant arrays. Requires that
        q is DataArray. For example:
        `attrs={'name': 'varname', 'long_name': 'longvarname', 'units': 'units'}`

    Returns
    -------
    DataArray of vertical derivative of variable with
    attributes altered to reflect calculation.

    Notes
    -----
    This will alter the number of points in the s dimension.

    Example usage
    -------------
    >>> xroms.ddz(ds.salt, grid)
    """

    assert isinstance(var, xr.DataArray), "var must be DataArray"

    if attrs is None:
        attrs = var.attrs.copy()
        attrs["name"] = "d" + var.name + "dz"
        attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
        attrs["long_name"] = "vertical derivative of " + attrs.setdefault(
            "long_name", "var"
        )
        attrs["grid"] = grid

    var = grid.derivative(var, "Z", boundary=sboundary, fill_value=sfill_value)
    var = to_grid(
        var,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        attrs=attrs,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    return var


def argsel2d(lons, lats, lon0, lat0):
    """Find the indices of coordinate pair closest to another point.

    Inputs
    ------
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.

    Returns
    -------
    Index or indices of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. Number of dimensions of
    returned indices will correspond to the shape of input lons.

    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.

    Example usage
    -------------
    >>> xroms.argsel2d(ds.lon_rho, ds.lat_rho, -96, 27)
    """

    # input lons and lats can be multidimensional and might be DataArrays or lists
    pts = list(zip(*(np.asarray(lons).flatten(), np.asarray(lats).flatten())))
    endpts = list(zip(*(np.asarray(lon0).flatten(), np.asarray(lat0).flatten())))

    G = cartopy.geodesic.Geodesic()  # set up class
    dist = np.asarray(G.inverse(pts, endpts)[:, 0])  # select distances specifically
    iclosest = abs(np.asarray(dist)).argmin()  # find indices of closest point
    # return index or indices in input array shape
    inds = np.unravel_index(iclosest, np.asarray(lons).shape)

    return inds


def sel2d(var, lons, lats, lon0, lat0):
    """Find the value of the var at closest location to lon0,lat0.

    Inputs
    ------
    var: DataArray, ndarray
        Variable to operate on.
    lons: DataArray, ndarray, list
        Longitudes of points to search through for closest point.
    lats: DataArray, ndarray, list
        Latitudes of points to search through for closest point.
    lon0: float, int
        Longitude of comparison point.
    lat0: float, int
        Latitude of comparison point.

    Returns
    -------
    Value in var of location in coordinate pairs made up of lons, lats
    that is closest to location lon0, lat0. If var has other
    dimensions, they are brought along.

    Notes
    -----
    This function uses Great Circle distance to calculate distances assuming
    longitudes and latitudes as point coordinates. Uses cartopy function
    `Geodesic`: https://scitools.org.uk/cartopy/docs/latest/cartopy/geodesic.html

    If searching for the closest grid node to a lon/lat location, be sure to
    use the correct horizontal grid (rho, u, v, or psi). This is accounted for
    if this function is used through the accessor.

    This is meant to be used by the accessor to conveniently wrap
    `argsel2d`.

    Example usage
    -------------
    >>> xroms.sel2d(ds.temp, ds.lon_rho, ds.lat_rho, -96, 27)
    """

    assert isinstance(var, xr.DataArray), "Input a DataArray"
    inds = argsel2d(lons, lats, lon0, lat0)
    return var.cf.isel(Y=inds[0], X=inds[1])


def to_rho(var, grid, hboundary="extend", hfill_value=None):
    """Change var to rho horizontal grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto rho horizontal grid.

    Notes
    -----
    If var is already on rho grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_rho('salt', grid)
    """
    if "xi_rho" not in var.dims:
        var = grid.interp(
            var, "X", to="center", boundary=hboundary, fill_value=hfill_value
        )
    if "eta_rho" not in var.dims:
        var = grid.interp(
            var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        )
    return var


def to_psi(var, grid, hboundary="extend", hfill_value=None):
    """Change var to psi horizontal grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto psi horizontal grid.

    Notes
    -----
    If var is already on psi grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_psi('salt', grid)
    """

    if "xi_u" not in var.dims:
        var = grid.interp(
            var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        )
    if "eta_v" not in var.dims:
        var = grid.interp(
            var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        )
    return var


def to_u(var, grid, hboundary="extend", hfill_value=None):
    """Change var to u horizontal grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto u horizontal grid.

    Notes
    -----
    If var is already on u grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_u('salt', grid)
    """
    if "xi_u" not in var.dims:
        var = grid.interp(
            var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        )
    if "eta_rho" not in var.dims:
        var = grid.interp(
            var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        )
    return var


def to_v(var, grid, hboundary="extend", hfill_value=None):
    """Change var to v horizontal grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto v horizontal grid.

    Notes
    -----
    If var is already on v grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_v('salt', grid)
    """
    if "xi_rho" not in var.dims:
        var = grid.interp(
            var, "X", to="center", boundary=hboundary, fill_value=hfill_value
        )
    if "eta_v" not in var.dims:
        var = grid.interp(
            var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        )
    return var


def to_s_rho(var, grid, sboundary="extend", sfill_value=None):
    """Change var to rho vertical grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto rho vertical grid.

    Notes
    -----
    If var is already on rho grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_s_rho('salt', grid)
    """

    # only change if not already on s_rho
    if "s_rho" not in var.dims:
        var = grid.interp(
            var, "Z", to="center", boundary=sboundary, fill_value=sfill_value
        )
    return var


def to_s_w(var, grid, sboundary="extend", sfill_value=None):
    """Change var to w vertical grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto rho vertical grid.

    Notes
    -----
    If var is already on w grid, nothing happens.

    `to_grid` function wraps all of the `to_*` functions so one function
    can be used for all grid changes.

    Example usage
    -------------
    >>> xroms.to_s_w('salt', grid)
    """

    # only change if not already on s_w
    if "s_w" not in var.dims:
        var = grid.interp(
            var, "Z", to="outer", boundary=sboundary, fill_value=sfill_value
        )
    return var


def to_grid(
    var,
    grid,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
    attrs=None,
):
    """Implement grid changes.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    hcoord: string, optional.
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional.
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection
        for grid changes.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    sfill_value: float, optional
        Passed to `grid` method calls; vertical boundary selection
        fill value.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray or ndarray interpolated onto hcoord horizontal and scoord
    vertical grids.

    Notes
    -----
    If var is already on selected grid, nothing happens.

    Example usage
    -------------
    >>> xroms.to_grid(ds.salt, grid, hcoord='rho', scoord='w')
    """

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs["name"] = var.name
        attrs["units"] = attrs.setdefault("units", "units")
        attrs["long_name"] = attrs.setdefault("long_name", "var")
        attrs["grid"] = grid

    if hcoord is not None:
        assert hcoord in ["rho", "psi", "u", "v"], (
            'hcoord should be "rho" or "psi" or "u" or "v" but is "%s"' % hcoord
        )
        if hcoord == "rho":
            var = to_rho(var, grid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "psi":
            var = to_psi(var, grid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "u":
            var = to_u(var, grid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "v":
            var = to_v(var, grid, hboundary=hboundary, hfill_value=hfill_value)

    if scoord is not None:
        assert scoord in ["s_rho", "rho", "s_w", "w"], (
            'scoord should be "s_rho", "rho", "s_w", or "w" but is "%s"' % scoord
        )
        if scoord in ["s_rho", "rho"]:
            var = to_s_rho(var, grid, sboundary=sboundary, sfill_value=sfill_value)
        elif scoord in ["s_w", "w"]:
            var = to_s_w(var, grid, sboundary=sboundary, sfill_value=sfill_value)

    if isinstance(var, xr.DataArray):
        var.attrs = attrs
        var.name = var.attrs["name"]

    return var


def gridmean(var, grid, dim):
    """Calculate mean accounting for variable spatial grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    dim: str, list, tuple
        Spatial dimension names to average over. In the `xgcm`
        convention, the allowable names are 'Z', 'Y', or 'X'.

    Returns
    -------
    DataArray or ndarray of average calculated over dim accounting
    for variable spatial grid.

    Notes
    -----
    If result is DataArray, long name attribute is modified to describe
    calculation.

    Example usage
    -------------
    Note that the following two approaches are equivalent:
    >>> app1 = xroms.gridmean(ds.u, ds.attrs['grid'], ('Y','X'))
    >>> app2 = (ds.u*ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))/(ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))
    >>> np.allclose(app1, app2)
    """

    assert isinstance(
        dim, (str, list, tuple)
    ), 'dim must be a string of or a list or tuple containing "X", "Y", and/or "Z"'

    if isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs["name"] = attrs.setdefault("name", "var")
        attrs["units"] = attrs.setdefault("units", "units")
        dimstr = dim if isinstance(dim, str) else ", ".join(dim)
        attrs["long_name"] = (
            attrs.setdefault("long_name", "var") + ", grid mean over dim " + dimstr
        )
        attrs["grid"] = grid

    var = grid.average(var, dim)

    return var


def gridsum(var, grid, dim):
    """Calculate sum accounting for variable spatial grid.

    Inputs
    ------
    var: DataArray or ndarray
        Variable to operate on.
    grid: xgcm.grid
        Grid object associated with var
    dim: str, list, tuple
        Spatial dimension names to sum over. In the `xgcm`
        convention, the allowable names are 'Z', 'Y', or 'X'.

    Returns
    -------
    DataArray or ndarray of sum calculated over dim accounting
    for variable spatial grid.

    Notes
    -----
    If result is DataArray, long name attribute is modified to describe
    calculation.

    Example usage
    -------------
    Note that the following two approaches are equivalent:
    >>> app1 = xroms.gridsum(ds.u, ds.attrs['grid'], ('Z','X'))
    >>> app2 = (ds.u*ds.dz_u * ds.dx_u).sum(('s_rho','xi_u'))
    >>> np.allclose(app1, app2)
    """

    assert isinstance(
        dim, (str, list, tuple)
    ), 'dim must be a string of or a list or tuple containing "X", "Y", and/or "Z"'

    if isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs["name"] = attrs.setdefault("name", "var")
        attrs["units"] = attrs.setdefault("units", "units")
        dimstr = dim if isinstance(dim, str) else ", ".join(dim)
        attrs["long_name"] = (
            attrs.setdefault("long_name", "var") + ", grid sum over dim " + dimstr
        )
        attrs["grid"] = grid

    var = grid.integrate(var, dim)

    return var


def xisoslice(iso_array, iso_value, projected_array, coord):
    """Calculate an isosurface.

    This function has been possibly superceded by isoslice
    that wraps `xgcm.grid.transform` for the following reasons,
    but more testing is needed:
    * The implementation of `xgcm.grid.transform` is more robust
      than `xisoslice` which has extra code for in case iso_value
      is exactly in iso_array.
    * For a 5-day model file, the run time for the same call for
      was approximately the same for xisolice and isoslice.
    * isoslice might be more computationally robust for not
      breaking mid-way, but this is still unclear.

    This function calculates the value of projected_array on
    an isosurface in the array iso_array defined by iso_value.

    Inputs
    ------
    iso_array: DataArray, ndarray
        Array in which the isosurface is defined
    iso_value: float
        Value of the isosurface in iso_array
    projected_array: DataArray, ndarray
        Array in which to project values on the isosurface. This can have
        multiple time outputs. Needs to be broadcastable from iso_array?
    coord: string
        Name of coordinate associated with the dimension along which to project

    Returns
    -------
    DataArray or ndarray of values of projected_array on the isosurface

    Notes
    -----
    Performs lazy evaluation.

    `xisoslice` requires that iso_array be monotonic. If iso_value is not monotonic
    it will still run but values may be incorrect where not monotonic.
    If iso_value is exactly in iso_array or the value is passed twice in iso_array,
    a message will be printed. iso_value is changed a tiny amount in this case to
    account for it being in iso_array exactly. The latter case is not deal with.

    Example usage
    -------------

    Calculate lat-z slice of salinity along a constant longitude value (-91.5):
    >>> sl = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.salt, 'xi_rho')

    Calculate a lon-lat slice at a constant z value (-10):
    >>> sl = xroms.utilities.xisoslice(ds.z_rho, -10, ds.temp, 's_rho')

    Calculate a lon-lat slice at a constant z value (-10) but without zeta changing in time:
    (use ds.z_rho0 which is relative to mean sea level and does not vary in time)
    >>> sl = xroms.utilities.xisoslice(ds.z_rho0, -10, ds.temp, 's_rho')

    Calculate the depth of a specific isohaline (33):
    >>> sl = xroms.utilities.xisoslice(ds.salt, 33, ds.z_rho, 's_rho')

    Calculate the salt 10 meters above the seabed. Either do this on the vertical
    rho grid, or first change to the w grid and then use `xisoslice`. You may prefer
    to do the latter if there is a possibility that the distance above the seabed you are
    interpolating to (10 m) could be below the deepest rho grid depth.
    * on rho grid directly:
      >>> sl = xroms.xisoslice(ds.z_rho + ds.h, 10., ds.salt, 's_rho')
    * on w grid:
      >>> var_w = xroms.to_s_w(ds.salt, ds.xroms.grid)
      >>> sl = xroms.xisoslice(ds.z_w + ds.h, 10., var_w, 's_w')

    In addition to calculating the slices themselves, you may need to calculate
    related coordinates for plotting. For example, to accompany the lat-z slice,
    you may want the following:

        # calculate z values (s_rho)
    >>> slz = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.z_rho, 'xi_rho')

        # calculate latitude values (eta_rho)
    >>> sllat = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.lat_rho, 'xi_rho')

        # assign these as coords to be used in plot
    >>> sl = sl.assign_coords(z=slz, lat=sllat)

        # points that should be masked
    >>> slmask = xroms.utilities.xisoslice(ds.lon_rho, -91.5, ds.mask_rho, 'xi_rho')

        # drop masked values
    >>> sl = sl.where(slmask==1, drop=True)
    """

    # length of the projected coordinate, minus one
    Nm = len(iso_array[coord]) - 1

    # A 'lower' slice including all but the last value, and an
    # 'upper' slice including all but the first value
    lslice = {coord: slice(None, -1)}
    uslice = {coord: slice(1, None)}

    # prop is now the array on which to calculate the isosurface, with
    # the iso_value subtracted so that the isosurface is defined by
    # prop == 0
    prop = iso_array - iso_value

    # propl are the prop values in the lower slice
    propl = prop.isel(**lslice)
    propl.coords[coord] = np.arange(Nm)
    # propu in the upper slice
    propu = prop.isel(**uslice)
    propu.coords[coord] = np.arange(Nm)

    # Find the location where prop changes sign, meaning it bounds the
    # desired isosurface. zc has a length of Nm in the projected dimension
    # and may be considered to be an array in between the values in the
    # projected dimension. zc==1 means the prop changed signs crossing this
    # value, so that the isovalue occurs between those two values.
    zc = xr.where((propu * propl) <= 0.0, 1.0, 0.0)

    # Get the upper and lower slices of the array that will be projected
    # on the isosurface
    varl = projected_array.isel(**lslice)
    varl.coords[coord] = np.arange(Nm)
    varu = projected_array.isel(**uslice)
    varu.coords[coord] = np.arange(Nm)

    # propl*zc extracts the value of prop below the iso_surface.
    # propu*zc above. Extract similar values for the projected array.
    propl = (propl * zc).sum(coord)
    propu = (propu * zc).sum(coord)
    varl = (varl * zc).sum(coord)
    varu = (varu * zc).sum(coord)

    # A linear fit to of the projected array to the isosurface.
    out = varl - propl * (varu - varl) / (propu - propl)

    # If the sum == 2, that means iso_value is exactly in iso_array
    check = zc.sum(coord) == 2

    # it's too slow for large arrays to check this, so just always
    # divide and it will happen where necessary.
    # where iso_value is located in iso_array, divide result by 2
    out = xr.where(check, out / 2, out)

    return out
