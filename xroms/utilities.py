"""
Functions that act on DataArrays or Datasets.
"""


import warnings

import numpy as np
import xarray as xr


try:
    import cartopy.geodesic
except ImportError:
    warnings.warn(
        "cartopy is not installed, so `sel2d` and `argsel2d` will not run.",
        ImportWarning,
    )


def grid_interp(xgrid, da, dim, which_xgcm_function="interp", **kwargs):
    """Interpolate da in dim

    This function is necessary because of weirdness with chunking
    with xgcm.
    More info: https://github.com/xgcm/xgcm/issues/522

    Parameters
    ----------
    xgrid : xgcm grid object
        _description_
    da : DataArray
        interpolating from this dataarray
    dim : str
        interpolating grids in this dimension
    which_xgcm_function : "interp"
        But could instead be "integrate"

    Returns
    -------
    DataArray
        interpolated down one dimension in dim
    """

    # which dimension chunk to save?
    dim_name = da.cf[dim].name  # e.g. dim_name = xi_rho
    i_chunk_dim = list(da.dims).index(
        dim_name
    )  # chunk_dim is e.g. 2, the index in dims for the chunks

    # need to unchunk then rechunk, so save chunk
    if da.chunks is not None:
        chunk = list(da.chunks[i_chunk_dim])

        # to interpolate, first remove chunking to 1 chunk
        new_coord = getattr(xgrid, which_xgcm_function)(
            da.chunk({dim_name: -1}), dim, **kwargs
        )
        # new_coord = xgrid.interp(da.chunk({dim_name: -1}), dim, **kwargs)

        if (
            which_xgcm_function == "interp"
            and new_coord.shape[i_chunk_dim] + 1 == da.shape[i_chunk_dim]
        ):

            # take one off the last chunk in this dimension since interpolation
            # loses one in size
            chunk[-1] -= 1
            # reconstitute chunks after intepolation but minus one in downsized dim
            return new_coord.chunk({new_coord.cf[dim].name: tuple(chunk)})

        elif (
            which_xgcm_function == "interp"
            and new_coord.shape[i_chunk_dim] == da.shape[i_chunk_dim] + 1
        ):

            # add one on the last chunk in this dimension since interpolation
            # loses one in size
            chunk[-1] += 1
            # reconstitute chunks after intepolation but minus one in downsized dim
            return new_coord.chunk({new_coord.cf[dim].name: tuple(chunk)})

        elif which_xgcm_function == "integrate":
            return new_coord

        else:
            raise ValueError("chunks probably are not being dealt with properly")

    else:
        new_coord = getattr(xgrid, which_xgcm_function)(da, dim, **kwargs)
        # new_coord = xgrid.interp(da, dim, **kwargs)
        return new_coord


def hgrad(
    q,
    xgrid,
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

    Note that you need the 3D metrics for horizontal derivatives for ROMS, so ``include_3D_metrics=True`` in ``xroms.roms_dataset()``.

    Parameters
    ----------

    q: DataArray
        Property to take gradients of.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> dtempdxi, dtempdeta = xroms.hgrad(ds.temp, xgrid)
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
            dqdx = xgrid.interp(
                xgrid.derivative(q, "X", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dqdz = xgrid.interp(
                xgrid.derivative(q, "Z", boundary=sboundary, fill_value=sfill_value),
                "X",
                boundary=hboundary,
                fill_value=hfill_value,
            )
            dzdx = xgrid.interp(
                xgrid.derivative(z, "X", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dzdz = xgrid.interp(
                xgrid.derivative(z, "Z", boundary=sboundary, fill_value=sfill_value),
                "X",
                boundary=hboundary,
                fill_value=hfill_value,
            )

            dqdxi = dqdx * dzdz - dqdz * dzdx

        else:  # 2D variables
            dqdxi = xgrid.derivative(q, "X", boundary=hboundary, fill_value=hfill_value)

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs["name"] = "d" + q.name + "dxi"
            attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
            attrs["long_name"] = "horizontal xi derivative of " + attrs.setdefault(
                "long_name", "var"
            )
        dqdxi = to_grid(
            dqdxi,
            xgrid,
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
            dqdy = xgrid.interp(
                xgrid.derivative(q, "Y", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dqdz = xgrid.interp(
                xgrid.derivative(q, "Z", boundary=sboundary, fill_value=sfill_value),
                "Y",
                boundary=hboundary,
                fill_value=hfill_value,
            )
            dzdy = xgrid.interp(
                xgrid.derivative(z, "Y", boundary=hboundary, fill_value=hfill_value),
                "Z",
                boundary=sboundary,
                fill_value=sfill_value,
            )
            dzdz = xgrid.interp(
                xgrid.derivative(z, "Z", boundary=sboundary, fill_value=sfill_value),
                "Y",
                boundary=hboundary,
                fill_value=hfill_value,
            )

            dqdeta = dqdy * dzdz - dqdz * dzdy

        else:  # 2D variables
            dqdeta = xgrid.derivative(
                q, "Y", boundary=hboundary, fill_value=hfill_value
            )

        if attrs is None and isinstance(q, xr.DataArray):
            attrs = q.attrs.copy()
            attrs["name"] = "d" + q.name + "deta"
            attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
            attrs["long_name"] = "horizontal eta derivative of " + attrs.setdefault(
                "long_name", "var"
            )
        dqdeta = to_grid(
            dqdeta,
            xgrid,
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
    xgrid,
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

    Note that you need the 3D metrics for horizontal derivatives for ROMS, so ``include_3D_metrics=True`` in ``xroms.roms_dataset()``.

    Parameters
    ----------
    var: DataArray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.ddxi(ds.salt, xgrid)
    """

    var = hgrad(
        var,
        xgrid,
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
    xgrid,
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

    Note that you need the 3D metrics for horizontal derivatives for ROMS, so ``include_3D_metrics=True`` in ``xroms.roms_dataset()``.

    Parameters
    ----------
    var: DataArray, ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.ddeta(ds.salt, xgrid)
    """

    var = hgrad(
        var,
        xgrid,
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
    xgrid,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
    attrs=None,
):
    """Calculate d/dz for a variable.

    Parameters
    ----------
    var: DataArray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.ddz(ds.salt, xgrid)
    """

    assert isinstance(var, xr.DataArray), "var must be DataArray"

    if attrs is None:
        attrs = var.attrs.copy()
        attrs["name"] = "d" + var.name + "dz"
        attrs["units"] = "1/m * " + attrs.setdefault("units", "units")
        attrs["long_name"] = "vertical derivative of " + attrs.setdefault(
            "long_name", "var"
        )

    var = xgrid.derivative(var, "Z", boundary=sboundary, fill_value=sfill_value)
    var = to_grid(
        var,
        xgrid,
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

    Parameters
    ----------
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

    Examples
    --------
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

    Parameters
    ----------
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

    Examples
    --------
    >>> xroms.sel2d(ds.temp, ds.lon_rho, ds.lat_rho, -96, 27)
    """

    assert isinstance(var, xr.DataArray), "Input a DataArray"
    inds = argsel2d(lons, lats, lon0, lat0)
    return var.cf.isel(Y=inds[0], X=inds[1])


def to_rho(var, xgrid, hboundary="extend", hfill_value=None):
    """Change var to rho horizontal grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_rho('salt', xgrid)
    """
    if "xi_rho" not in var.dims:
        var = grid_interp(
            xgrid, var, "X", to="center", boundary=hboundary, fill_value=hfill_value
        )

    #      var = xgrid.interp(
    #         var, "X", to="center", boundary=hboundary, fill_value=hfill_value
    #     )
    if "eta_rho" not in var.dims:
        var = grid_interp(
            xgrid, var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        )

        #  var = xgrid.interp(
        #     var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        # )
    return var


def to_psi(var, xgrid, hboundary="extend", hfill_value=None):
    """Change var to psi horizontal grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_psi('salt', xgrid)
    """

    if "xi_u" not in var.dims:

        var = grid_interp(
            xgrid, var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        )

        # var = xgrid.interp(
        #     var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        # )
    if "eta_v" not in var.dims:
        var = grid_interp(
            xgrid, var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        )
        # var = xgrid.interp(
        #     var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        # )
    return var


def to_u(var, xgrid, hboundary="extend", hfill_value=None):
    """Change var to u horizontal grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_u('salt', xgrid)
    """
    if "xi_u" not in var.dims:
        var = grid_interp(
            xgrid, var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        )

        #  var = xgrid.interp(
        #     var, "X", to="inner", boundary=hboundary, fill_value=hfill_value
        # )
    if "eta_rho" not in var.dims:
        var = grid_interp(
            xgrid, var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        )

        #  var = xgrid.interp(
        #     var, "Y", to="center", boundary=hboundary, fill_value=hfill_value
        # )
    return var


def to_v(var, xgrid, hboundary="extend", hfill_value=None):
    """Change var to v horizontal grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_v('salt', xgrid)
    """
    if "xi_rho" not in var.dims:
        var = grid_interp(
            xgrid, var, "X", to="center", boundary=hboundary, fill_value=hfill_value
        )

        # var = xgrid.interp(
        #     var, "X", to="center", boundary=hboundary, fill_value=hfill_value
        # )
    if "eta_v" not in var.dims:
        var = grid_interp(
            xgrid, var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        )

        #  var = xgrid.interp(
        #     var, "Y", to="inner", boundary=hboundary, fill_value=hfill_value
        # )
    return var


def to_s_rho(var, xgrid, sboundary="extend", sfill_value=None):
    """Change var to rho vertical grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_s_rho('salt', xgrid)
    """

    # only change if not already on s_rho
    if "s_rho" not in var.dims:
        var = xgrid.interp(
            var, "Z", to="center", boundary=sboundary, fill_value=sfill_value
        )
    return var


def to_s_w(var, xgrid, sboundary="extend", sfill_value=None):
    """Change var to w vertical grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_s_w('salt', xgrid)
    """

    # only change if not already on s_w
    if "s_w" not in var.dims:
        var = xgrid.interp(
            var, "Z", to="outer", boundary=sboundary, fill_value=sfill_value
        )
    return var


def to_grid(
    var,
    xgrid,
    hcoord=None,
    scoord=None,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
    attrs=None,
):
    """Implement grid changes.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    >>> xroms.to_grid(ds.salt, xgrid, hcoord='rho', scoord='w')
    """

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs["name"] = var.name
        attrs["units"] = attrs.setdefault("units", "units")
        attrs["long_name"] = attrs.setdefault("long_name", "var")

    if hcoord is not None:
        assert hcoord in ["rho", "psi", "u", "v"], (
            'hcoord should be "rho" or "psi" or "u" or "v" but is "%s"' % hcoord
        )
        if hcoord == "rho":
            var = to_rho(var, xgrid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "psi":
            var = to_psi(var, xgrid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "u":
            var = to_u(var, xgrid, hboundary=hboundary, hfill_value=hfill_value)
        elif hcoord == "v":
            var = to_v(var, xgrid, hboundary=hboundary, hfill_value=hfill_value)

    if scoord is not None:
        assert scoord in ["s_rho", "rho", "s_w", "w"], (
            'scoord should be "s_rho", "rho", "s_w", or "w" but is "%s"' % scoord
        )
        if scoord in ["s_rho", "rho"]:
            var = to_s_rho(var, xgrid, sboundary=sboundary, sfill_value=sfill_value)
        elif scoord in ["s_w", "w"]:
            var = to_s_w(var, xgrid, sboundary=sboundary, sfill_value=sfill_value)

    if isinstance(var, xr.DataArray):
        var.attrs = attrs
        var.name = var.attrs["name"]

    return var


def gridmean(var, xgrid, dim):
    """Calculate mean accounting for variable spatial grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    Note that the following two approaches are equivalent:
    >>> app1 = xroms.gridmean(ds.u, xgrid, ('Y','X'))
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

    var = xgrid.average(var, dim)

    return var


def gridsum(var, xgrid, dim):
    """Calculate sum accounting for variable spatial grid.

    Parameters
    ----------
    var: DataArray or ndarray
        Variable to operate on.
    xgrid: xgcm.grid
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

    Examples
    --------
    Note that the following two approaches are equivalent:
    >>> app1 = xroms.gridsum(ds.u, xgrid, ('Z','X'))
    >>> app2 = (ds.u*ds.dz_u * ds.dx_u).sum(('s_rho','xi_u'))
    >>> np.allclose(app1, app2)
    """
    # for now with xgcm bug only allow for one dim at a time
    assert isinstance(dim, str), 'dim must be a string containing "X", "Y", and/or "Z"'
    # assert isinstance(
    #     dim, (str, list, tuple)
    # ), 'dim must be a string of or a list or tuple containing "X", "Y", and/or "Z"'

    if isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs["name"] = attrs.setdefault("name", "var")
        attrs["units"] = attrs.setdefault("units", "units")
        dimstr = dim if isinstance(dim, str) else ", ".join(dim)
        attrs["long_name"] = (
            attrs.setdefault("long_name", "var") + ", grid sum over dim " + dimstr
        )

    # if isinstance(dim, str):
    #     var = grid_interp(xgrid, var, dim, which_xgcm_function="integrate")
    # else:
    #     for d in dim:
    #         var = grid_interp(xgrid, var, d, which_xgcm_function="integrate")
    #         import pdb; pdb.set_trace()
    # var = xgrid.integrate(var, dim)
    var = grid_interp(xgrid, var, dim, which_xgcm_function="integrate")

    return var


def xisoslice(iso_array, iso_value, projected_array, coord):
    """Calculate an isosurface.

    This function has been possibly superseded by isoslice
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

    Parameters
    ----------
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

    Examples
    --------

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

      >>> var_w = xroms.to_s_w(ds.salt, ds.xroms.xgrid)
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


def subset(ds, X=None, Y=None):
    """Subset model output horizontally using isel, properly accounting for horizontal grids.

    Parameters
    ----------
    ds: xarray Dataset
        Dataset of ROMS model output. Assumes that full regular grid setup is
        available and has been read in using xroms so that dimension names
        have been updated.
    X: slice, optional
        Slice in X dimension using form `X=slice(start, stop, step)`. For example,
        >>> X=slice(20,40,2)
        Indices are used for rho grid, and psi grid is reduced accordingly.
    Y: slice, optional
        Slice in Y dimension using form `Y=slice(start, stop, step)`. For example,
        >>> Y=slice(20,40,2)
        Indices are used for rho grid, and psi grid is reduced accordingly.

    Returns
    -------
    Dataset with form as if model had been run at the subsetted size. That is, the outermost
    cells of the rho grid are like ghost cells and the psi grid is one inward from this size
    in each direction.

    Notes
    -----
    X and Y must be slices, not single numbers.

    Examples
    --------
    Subset only in Y direction:
    >>> xroms.subset(ds, Y=slice(50,100))
    Subset in X and Y:
    >>> xroms.subset(ds, X=slice(20,40), Y=slice(50,100))
    """

    if X is not None:
        assert isinstance(X, slice), "X must be a slice, e.g., slice(50,100)"
        ds = ds.isel(xi_rho=X, xi_u=slice(X.start, X.stop - 1))
        if "xi_v" in ds.coords:
            ds = ds.isel(xi_v=X)
        if "xi_psi" in ds.coords:
            ds = ds.isel(xi_psi=slice(X.start, X.stop - 1))

    if Y is not None:
        assert isinstance(Y, slice), "Y must be a slice, e.g., slice(50,100)"
        ds = ds.isel(eta_rho=Y, eta_v=slice(Y.start, Y.stop - 1))
        if "eta_u" in ds.coords:
            ds = ds.isel(eta_u=Y)
        if "eta_psi" in ds.coords:
            ds = ds.isel(eta_psi=slice(Y.start, Y.stop - 1))

    return ds


def order(var):
    """Reorder var to typical dimensional ordering.

    Parameters
    ----------
    var: DataArray
        Variable to operate on.

    Returns
    -------
    DataArray with dimensional order ['T', 'Z', 'Y', 'X'], or whatever subset of
    dimensions are present in var.

    Notes
    -----
    Do not consider previously-selected dimensions that are kept on as coordinates but
    cannot be transposed anymore. This is accomplished with `.reset_coords(drop=True)`.

    Examples
    --------
    >>> xroms.order(var)
    """

    # this looks at the dims on var directly which tends to be more accurate
    # since the DataArray can have extra coordinates included (but dropping
    # can drop too many variables).
    return var.cf.transpose(
        *[
            dim
            for dim in ["T", "Z", "Y", "X"]
            if dim in var.cf.axes and var.cf.axes[dim][0] in var.dims
        ]
    )
    # return var.cf.transpose(
    #     *[
    #         dim
    #         for dim in ["T", "Z", "Y", "X"]
    #         if dim in var.reset_coords(drop=True).cf.axes
    #     ]
    # )
