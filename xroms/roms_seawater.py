"""
Functions related to density of seawater.
"""

import numpy as np
import xarray as xr

import xroms


g = 9.81


def density(temp, salt, z=None):
    """Calculate the density [kg/m^3] as calculated in ROMS.

    Parameters
    ----------
    temp: DataArray, ndarray
        Temperature [Celsius]
    salt: DataArray, ndarray
        Salinity
    z: DataArray, ndarray, int, float, optional
        Depth [m]. To specify a reference depth, use a constant. If None,
        use z coordinate attached to temperature.

    Returns
    -------
    DataArray or ndarray of calculated density on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    Equation of state based on ROMS Nonlinear/rho_eos.F

    Examples
    --------
    >>> xroms.density(ds.temp, ds.salt)
    """

    if z is None:
        coords = list(temp.coords)
        z_coord_name = coords[[coord[:2] == "z_" for coord in coords].index(True)]
        z = temp[z_coord_name]

    A00 = +19092.56
    A01 = +209.8925
    A02 = -3.041638
    A03 = -1.852732e-3
    A04 = -1.361629e-5
    B00 = +104.4077
    B01 = -6.500517
    B02 = +0.1553190
    B03 = +2.326469e-4
    D00 = -5.587545
    D01 = +0.7390729
    D02 = -1.909078e-2
    E00 = +4.721788e-1
    E01 = +1.028859e-2
    E02 = -2.512549e-4
    E03 = -5.939910e-7
    F00 = -1.571896e-2
    F01 = -2.598241e-4
    F02 = +7.267926e-6
    G00 = +2.042967e-3
    G01 = +1.045941e-5
    G02 = -5.782165e-10
    G03 = +1.296821e-7
    H00 = -2.595994e-7
    H01 = -1.248266e-9
    H02 = -3.508914e-9
    Q00 = +999.842594
    Q01 = +6.793952e-2
    Q02 = -9.095290e-3
    Q03 = +1.001685e-4
    Q04 = -1.120083e-6
    Q05 = +6.536332e-9
    U00 = +0.824493e0
    U01 = -4.08990e-3
    U02 = +7.64380e-5
    U03 = -8.24670e-7
    U04 = +5.38750e-9
    V00 = -5.72466e-3
    V01 = +1.02270e-4
    V02 = -1.65460e-6
    W00 = +4.8314e-4
    g = 9.81
    sqrtS = np.sqrt(salt)
    den1 = (
        Q00
        + Q01 * temp
        + Q02 * temp**2
        + Q03 * temp**3
        + Q04 * temp**4
        + Q05 * temp**5
        + U00 * salt
        + U01 * salt * temp
        + U02 * salt * temp**2
        + U03 * salt * temp**3
        + U04 * salt * temp**4
        + V00 * salt * sqrtS
        + V01 * salt * sqrtS * temp
        + V02 * salt * sqrtS * temp**2
        + W00 * salt**2
    )
    K0 = (
        A00
        + A01 * temp
        + A02 * temp**2
        + A03 * temp**3
        + A04 * temp**4
        + B00 * salt
        + B01 * salt * temp
        + B02 * salt * temp**2
        + B03 * salt * temp**3
        + D00 * salt * sqrtS
        + D01 * salt * sqrtS * temp
        + D02 * salt * sqrtS * temp**2
    )
    K1 = (
        E00
        + E01 * temp
        + E02 * temp**2
        + E03 * temp**3
        + F00 * salt
        + F01 * salt * temp
        + F02 * salt * temp**2
        + G00 * salt * sqrtS
    )
    K2 = (
        G01
        + G02 * temp
        + G03 * temp**2
        + H00 * salt
        + H01 * salt * temp
        + H02 * salt * temp**2
    )
    bulk = K0 - K1 * z + K2 * z**2
    var = (den1 * bulk) / (bulk + 0.1 * z)

    if isinstance(var, xr.DataArray):
        var.attrs["name"] = "rho"
        var.attrs["long_name"] = "density"
        var.attrs["units"] = "kg/m^3"
        var.name = var.attrs["name"]
        if "lon_rho" in var.coords:
            var.coords["lon_rho"].attrs["standard_name"] = "longitude"
            var.coords["lat_rho"].attrs["standard_name"] = "latitude"

    return var


def potential_density(temp, salt, z=0):
    """Calculate potential density [kg/m^3] with constant depth reference.

    Parameters
    ----------
    temp: DataArray, ndarray
        Temperature [Celsius]
    salt: DataArray, ndarray
        Salinity
    z: int, float, optional
        Reference depth [m].

    Returns
    -------
    DataArray or ndarray of calculated potential density on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    Uses equation of state based on ROMS Nonlinear/rho_eos.F

    Examples
    --------
    >>> xroms.potential_density(ds.temp, ds.salt)
    """

    var = density(temp, salt, z)

    if isinstance(var, xr.DataArray):
        var.attrs["name"] = "sig0"
        var.attrs["long_name"] = "potential density"
        var.attrs["units"] = "kg/m^3"
        var.name = var.attrs["name"]

    return var


def buoyancy(sig0, rho0=1025.0):
    """Calculate buoyancy [m/s^2] based on potential density.

    Parameters
    ----------
    sig0: DataArray, ndarray
        Potential density [kg/m^3]
    rho0: int, float, optional
        Reference density [kg/m^3].

    Returns
    -------
    DataArray or ndarray of calculated buoyancy on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    buoyancy = -g * rho / rho0

    Uses equation of state based on ROMS Nonlinear/rho_eos.F

    g=9.81 [m/s^2]

    Examples
    --------
    >>> xroms.potential_density(ds.temp, ds.salt)
    """

    var = -g * sig0 / rho0

    if isinstance(var, xr.DataArray):
        var.attrs["name"] = "buoyancy"
        var.attrs["long_name"] = "buoyancy"
        var.attrs["units"] = "m/s^2"
        var.name = var.attrs["name"]

    return var


def N2(rho, xgrid, rho0=1025.0, sboundary="fill", sfill_value=np.nan):
    """Calculate buoyancy frequency squared (vertical buoyancy gradient).

    Parameters
    ----------
    rho: DataArray
        Density [kg/m^3]
    xgrid: xgcm.grid
        Grid object associated with rho
    rho0: int, float
        Reference density [kg/m^3].
    sboundary: string, optional
        Passed to `grid` method calls; vertical boundary selection for
        calculating z derivative.
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

    Returns
    -------
    DataArray of buoyancy frequency squared on rho/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    N2 = -g d(rho)/dz / rho0

    Examples
    --------
    >>> xroms.N2(rho, xgrid)
    """

    assert isinstance(rho, xr.DataArray), "rho must be DataArray"

    drhodz = xroms.ddz(rho, xgrid, sboundary=sboundary, sfill_value=sfill_value)
    var = -g * drhodz / rho0

    var.attrs["name"] = "N2"
    var.attrs["long_name"] = "buoyancy frequency squared, or vertical buoyancy gradient"
    var.attrs["units"] = "1/s^2"
    var.name = var.attrs["name"]

    return var


def M2(
    rho,
    xgrid,
    rho0=1025.0,
    hboundary="extend",
    hfill_value=None,
    sboundary="fill",
    sfill_value=np.nan,
    z=None,
):
    """Calculate the horizontal buoyancy gradient.

    Parameters
    ----------
    rho: DataArray
        Density [kg/m^3]
    xgrid: xgcm.grid
        Grid object associated with rho
    rho0: int, float, optional
        Reference density [kg/m^3].
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivatives of rho.
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
        calculating horizontal derivatives of rho.
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
    z: DataArray, optional
        Depths [m] associated with rho. If None, use z coordinate attached to temperature.

    Returns
    -------
    DataArray of the horizontal buoyancy gradient on rho/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    M2 = g/rho0 * sqrt(d(rho)/dxi^2 + d(rho)deta^2)

    g=9.81 [m/s^2]

    Examples
    --------
    >>> xroms.M2(rho, xgrid)
    """

    assert isinstance(rho, xr.DataArray), "rho must be DataArray"

    # calculate spatial derivatives of density
    drhodxi, drhodeta = xroms.hgrad(
        rho,
        xgrid,
        which="both",
        hcoord="rho",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    # combine
    var = np.sqrt(drhodxi**2 + drhodeta**2) * g / rho0

    var.attrs["name"] = "M2"
    var.attrs["long_name"] = "horizontal buoyancy gradient"
    var.attrs["units"] = "1/s^2"
    var.name = var.attrs["name"]

    return var


def mld(sig0, xgrid, h, mask, z=None, thresh=0.03):
    """Calculate the mixed layer depth [m].

    Parameters
    ----------
    sig0: DataArray
        Potential density [kg/m^3]
    xgrid
        xgcm grid
    h: DataArray, ndarray
        Depths [m].
    mask: DataArray, ndarray
        mask to match sig0
    z: DataArray, ndarray, optional
        The vertical depths associated with sig0. Should be on 'rho'
        grid horizontally and vertically. Use z coords associated with DataArray sig0
        if not input.
    thresh: float, optional
        For detection of mixed layer [kg/m^3]

    Returns
    -------
    DataArray of mixed layer depth on rho horizontal grid.
    Output is `[T,Y,X]`.

    Notes
    -----
    Mixed layer depth is based on the fixed Potential Density (PD) threshold.

    Converted to xroms by K. Thyng Aug 2020 from:

    Update history:
    v1.0 DL 2020Jun07

    References:
    ncl mixed_layer_depth function at https://github.com/NCAR/ncl/blob/ed6016bf579f8c8e8f77341503daef3c532f1069/ni/src/lib/nfpfort/ocean.f
    de Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004). Mixed layer depth over the global ocean: An examination of profile data and a profile‐based climatology. Journal of  Geophysical Research: Oceans, 109(C12).

    Examples
    --------
    >>> xroms.mld(sig0, h, mask)
    """

    if h.mean() > 0:  # if depths are positive, change to negative
        h = -h

    # xisoslice will operate over the relevant s dimension
    skey = sig0.dims[[dim[:2] == "s_" for dim in sig0.dims].index(True)]

    if z is None:
        z = sig0.z_rho

    # the mixed layer depth is the isosurface of depth where the potential density equals the surface - a threshold
    mld = xroms.isoslice(
        z,
        np.array([0.0]),
        xgrid,
        iso_array=sig0 - sig0.isel(s_rho=-1) - thresh,
        axis="Z",
    )
    #     mld = xroms.xisoslice(sig0 - sig0.isel(s_rho=-1) - thresh, 0.0, z, skey)

    # Replace nan's that are not masked with the depth of the water column.
    cond = (mld.isnull()) & (mask == 1)
    mld = mld.where(~cond, h)

    mld.attrs["name"] = "mld"
    mld.attrs["long_name"] = "mixed layer depth"
    mld.attrs["units"] = "m"
    mld.name = mld.attrs["name"]

    return mld.squeeze()
