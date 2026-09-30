"""Density of seawater, stratification (N2, M2) and mixed layer depth.

Every function is pure: it computes from the arrays passed in and returns a new
object, never touching its inputs. Heights ``z`` are not looked up by coordinate
name; they are either passed in (``z=``) or computed on demand from ``grid``, a
Dataset holding ``h``, ``zeta`` and the s-coordinate parameters (see
:func:`xroms.z`).

Output lands where the calculation puts it: ``N2`` of a rho-level density is on
the ``s_w`` levels, ``M2`` is on rho points and the input's own levels, and
``mld`` is on rho points with no vertical dimension.
"""

import numpy as np
import xarray as xr

from . import conventions
from ._align import require, select_like
from .conventions import canonicalize, vposition
from .interp import isoslice
from .utilities import _check_grid, ddeta, ddxi, ddz
from .vertical import z_like


g = 9.81  # m/s^2


def _label(var, name, long_name, units):
    """Give a result xroms' name, long_name and units.

    The attrs are replaced, not updated: arithmetic on DataArrays carries the
    inputs' attrs forward (xarray's default ``keep_attrs``), and those describe
    temperature or salinity, not the result.
    """
    var.attrs = {"name": name, "long_name": long_name, "units": units}
    var.name = name
    return var


def _with_cf_standard_names(var):
    """Copy of ``var`` whose lon_rho/lat_rho coordinates have CF standard names.

    The coordinates are copied first so that nothing is written to the
    coordinate objects of the inputs, which results share.
    """
    var = var.copy(deep=False)
    for name, standard_name in (("lon_rho", "longitude"), ("lat_rho", "latitude")):
        if name in var.coords:
            var.coords[name].attrs["standard_name"] = standard_name
    return var


def density(temp, salt, z=None, *, grid=None, zeta=None):
    """Calculate the density [kg/m^3] as calculated in ROMS.

    Parameters
    ----------
    temp : DataArray, ndarray
        Temperature [Celsius]
    salt : DataArray, ndarray
        Salinity
    z : DataArray, ndarray, int, float, optional
        Height of the points [m], as in ROMS: zero at the mean sea level and
        negative below it. This sets the pressure in the equation of state. To
        specify a reference depth, use a constant. If None, it is computed at
        the points of ``temp`` from ``grid``.
    grid : Dataset, optional
        Dataset with ``h``, ``zeta`` and the s-coordinate parameters, used only
        to compute ``z`` when it is not given (see :func:`xroms.z`).
    zeta : None, float, "mean" or DataArray, optional
        Free surface used when computing ``z`` from ``grid``: the grid's
        ``zeta`` (None), a constant (0 for static depths), its time mean, or a
        field (see :func:`xroms.z`).

    Returns
    -------
    DataArray or ndarray of calculated density, on the points of the inputs
    (rho/rho by default).

    Raises
    ------
    ValueError
        If neither ``z`` nor ``grid`` is given, or ``grid`` is given but ``temp``
        is not a DataArray (there is nothing to locate the points with).

    Notes
    -----
    Equation of state based on ROMS Nonlinear/rho_eos.F.

    Examples
    --------
    >>> xroms.density(ds.temp, ds.salt, grid=ds)
    >>> xroms.density(ds.temp, ds.salt, z=xroms.z(ds))
    """
    if isinstance(temp, xr.DataArray):
        temp = canonicalize(temp)
    if isinstance(salt, xr.DataArray):
        salt = canonicalize(salt)
    if isinstance(z, xr.DataArray):
        z = canonicalize(z)

    if z is None:
        if grid is None:
            raise ValueError(
                "density needs the height of each point: pass z= (a DataArray, or a constant "
                "reference depth; negative below the surface) or grid= (the Dataset with h, "
                "zeta and the s-coordinate parameters, to compute it)"
            )
        if not isinstance(temp, xr.DataArray):
            raise ValueError("density can compute z from grid= only when temp is a DataArray; pass z= instead")
        z = z_like(temp, _check_grid(grid, "density"), zeta=zeta)

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
        var = _with_cf_standard_names(var)
        _label(var, "rho", "density", "kg/m^3")

    return var


def potential_density(temp, salt, z=0):
    """Calculate potential density [kg/m^3] with constant depth reference.

    Parameters
    ----------
    temp : DataArray, ndarray
        Temperature [Celsius]
    salt : DataArray, ndarray
        Salinity
    z : int, float, optional
        Reference height [m] (0 is the mean sea level; negative is below it).

    Returns
    -------
    DataArray or ndarray of calculated potential density, on the points of the
    inputs (rho/rho by default).

    Notes
    -----
    Uses equation of state based on ROMS Nonlinear/rho_eos.F

    Examples
    --------
    >>> xroms.potential_density(ds.temp, ds.salt)
    """
    var = density(temp, salt, z)

    if isinstance(var, xr.DataArray):
        _label(var, "sig0", "potential density", "kg/m^3")

    return var


def buoyancy(sig0, rho0=1025.0):
    """Calculate buoyancy [m/s^2] based on potential density.

    Parameters
    ----------
    sig0 : DataArray, ndarray
        Potential density [kg/m^3]
    rho0 : int, float, optional
        Reference density [kg/m^3].

    Returns
    -------
    DataArray or ndarray of calculated buoyancy, on the points of ``sig0``.

    Notes
    -----
    buoyancy = -g * rho / rho0

    g=9.81 [m/s^2]

    Examples
    --------
    >>> xroms.buoyancy(xroms.potential_density(ds.temp, ds.salt))
    """
    var = -g * sig0 / rho0

    if isinstance(var, xr.DataArray):
        _label(var, "buoyancy", "buoyancy", "m/s^2")

    return var


def N2(rho, grid, rho0=None, *, z=None, zeta=None, sboundary="fill", sfill_value=np.nan):
    """Calculate buoyancy frequency squared (vertical buoyancy gradient).

    Parameters
    ----------
    rho : DataArray
        Density [kg/m^3]
    grid : Dataset or None
        Dataset with ``h``, ``zeta`` and the s-coordinate parameters, to compute
        the heights of ``rho``'s points. May be None if ``z`` is given.
    rho0 : int, float, DataArray, optional
        Reference density [kg/m^3]. If None, it is taken from ``grid`` (a
        ``rho0`` variable, then a ``rho0`` attribute) or else is 1025.
    z : DataArray, optional
        Heights [m] at the points of ``rho``, instead of computing them from
        ``grid``.
    zeta : None, float, "mean" or DataArray, optional
        Free surface used when computing the heights from ``grid`` (see
        :func:`xroms.z`).
    sboundary : string, optional
        Vertical boundary treatment of the z derivative: "fill" sets the two
        edge values to ``sfill_value``, "extend" takes the nearest computed
        (one-sided) difference.
    sfill_value : float, optional
        Value used at the vertical edges with ``sboundary="fill"``.

    Returns
    -------
    DataArray of buoyancy frequency squared. For ``rho`` on the ``s_rho`` levels
    it is on the ``s_w`` levels (the derivative is taken across layers), and
    the top and bottom w levels are NaN unless ``sboundary`` says otherwise.

    Notes
    -----
    N2 = -g d(rho)/dz / rho0

    g=9.81 [m/s^2]

    Examples
    --------
    >>> xroms.N2(rho, ds)
    """
    if not isinstance(rho, xr.DataArray):
        raise TypeError("rho must be a DataArray")
    grid = _check_grid(grid, "N2")
    if rho0 is None:
        rho0 = conventions.rho0(grid)

    drhodz = ddz(rho, grid, z=z, zeta=zeta, sboundary=sboundary, sfill_value=sfill_value)
    var = -g * drhodz / rho0

    return _label(var, "N2", "buoyancy frequency squared, or vertical buoyancy gradient", "1/s^2")


def M2(
    rho,
    grid,
    rho0=None,
    *,
    z=None,
    zeta=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="fill",
    sfill_value=np.nan,
    along_s=False,
):
    """Calculate the horizontal buoyancy gradient.

    Parameters
    ----------
    rho : DataArray
        Density [kg/m^3]
    grid : Dataset
        Dataset with the horizontal metrics ``pm`` and ``pn``, ``h``, ``zeta``
        and the s-coordinate parameters.
    rho0 : int, float, DataArray, optional
        Reference density [kg/m^3]. If None, it is taken from ``grid`` (a
        ``rho0`` variable, then a ``rho0`` attribute) or else is 1025.
    z : DataArray, optional
        Heights [m] at the points of ``rho``, instead of computing them from
        ``grid``.
    zeta : None, float, "mean" or DataArray, optional
        Free surface used when computing the heights from ``grid`` (see
        :func:`xroms.z`).
    hboundary : string, optional
        Horizontal boundary treatment when moving the derivatives back to rho
        points: "extend" copies the nearest value to the domain edge, "fill"
        puts ``hfill_value`` there.
    hfill_value : float, optional
        Value used at the horizontal edges with ``hboundary="fill"``.
    sboundary : string, optional
        Vertical boundary treatment of the z derivative in the correction to a
        constant-depth gradient: "fill" sets the two edge levels to
        ``sfill_value``, "extend" uses one-sided second-order differences there.
    sfill_value : float, optional
        Value used at the vertical edges with ``sboundary="fill"``.
    along_s : bool, optional
        For a single selected s-level only: accept derivatives along the
        s-surface instead of at constant depth (see :func:`xroms.ddxi`).

    Returns
    -------
    DataArray of the horizontal buoyancy gradient, on rho points and on the
    vertical levels of ``rho``. With the default ``sboundary="fill"`` the top
    and bottom levels are NaN.

    Notes
    -----
    M2 = g/rho0 * sqrt(d(rho)/dxi^2 + d(rho)/deta^2), with the derivatives at
    constant depth.

    g=9.81 [m/s^2]

    Examples
    --------
    >>> xroms.M2(rho, ds)
    """
    if not isinstance(rho, xr.DataArray):
        raise TypeError("rho must be a DataArray")
    grid = _check_grid(grid, "M2")
    if rho0 is None:
        rho0 = conventions.rho0(grid)

    kwargs = dict(
        z=z,
        zeta=zeta,
        hcoord="rho",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
        along_s=along_s,
    )
    drhodxi = ddxi(rho, grid, **kwargs)
    drhodeta = ddeta(rho, grid, **kwargs)
    var = g / rho0 * np.sqrt(drhodxi**2 + drhodeta**2)

    return _label(var, "M2", "horizontal buoyancy gradient", "1/s^2")


def mld(sig0, grid, *, thresh=0.03, z=None, zeta=None):
    """Calculate the mixed layer depth [m], positive, and the water depth if none is found.

    Parameters
    ----------
    sig0 : DataArray
        Potential density [kg/m^3], on rho points and with a vertical dimension.
    grid : Dataset
        Dataset with ``h`` (positive water depth), ``zeta`` and the s-coordinate
        parameters, and ``mask_rho`` if the grid has land.
    thresh : float, optional
        Density increase over the surface value [kg/m^3] that marks the base of
        the mixed layer.
    z : DataArray, optional
        Heights [m] at the points of ``sig0``, instead of computing them from
        ``grid``.
    zeta : None, float, "mean" or DataArray, optional
        Free surface used when computing the heights from ``grid`` (see
        :func:`xroms.z`).

    Returns
    -------
    DataArray of mixed layer depth [m, positive] on the rho horizontal grid,
    without a vertical dimension.

    Notes
    -----
    The mixed layer depth is based on the fixed potential density (PD)
    threshold: it is the depth where ``sig0`` first exceeds its value at the
    surface (the top level) by ``thresh``, linearly interpolated between
    levels. Where that never happens over water (``mask_rho == 1`` in ``grid``;
    where the surface ``sig0`` is valid if ``grid`` has no ``mask_rho``), the
    mixed layer is taken to be the whole water column and the result is ``h``.
    Land stays NaN.

    Like :func:`xroms.isoslice`, which does the interpolation, this expects
    each column of ``sig0`` to increase monotonically with depth: in a column
    with density inversions the depth found is not guaranteed to be the
    shallowest crossing.

    Converted to xroms by K. Thyng Aug 2020 from:

    Update history:
    v1.0 DL 2020Jun07

    References:
    ncl mixed_layer_depth function at https://github.com/NCAR/ncl/blob/ed6016bf579f8c8e8f77341503daef3c532f1069/ni/src/lib/nfpfort/ocean.f
    de Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004). Mixed layer depth over the global ocean: An examination of profile data and a profile‐based climatology. Journal of  Geophysical Research: Oceans, 109(C12).

    Useful resources:

    * Climate Data Toolbox documentation: https://www.chadagreene.com/CDT/mld_documentation.html
    * MLD calculation from MDTF: https://github.com/NOAA-GFDL/MDTF-diagnostics/blob/437d30590c45e8b7dd0cd01a3dc67066a2137115/diagnostics/mixed_layer_depth/mixed_layer_depth.py#L147

    Examples
    --------
    >>> xroms.mld(xroms.potential_density(ds.temp, ds.salt), ds)
    """
    if not isinstance(sig0, xr.DataArray):
        raise TypeError("sig0 must be a DataArray")
    if grid is None:
        raise ValueError("mld needs grid= (a Dataset with h, and mask_rho if it has land)")
    grid = _check_grid(grid, "mld")
    sig0 = canonicalize(sig0)
    vdim = vposition(sig0)
    if vdim is None:
        raise ValueError(f"{sig0.name!r} has no vertical dimension; mld needs density profiles")
    require(grid, "h", purpose="the depth of mixed layers that reach the bottom")

    zz = z_like(sig0, grid, zeta=zeta, z=z)
    surface = sig0.isel({vdim: -1}, drop=True)

    # the mixed layer depth is the isosurface of depth where the potential density equals the surface + a threshold
    depth = isoslice(zz, [0.0], sig0 - surface - thresh, dim=vdim, new_dim="iso")
    depth = depth.squeeze("iso", drop=True)

    # Replace nans that are not masked with the depth of the water column.
    h = select_like(grid["h"], sig0, name="h").reset_coords(drop=True)
    if "mask_rho" in grid.variables:
        water = select_like(grid["mask_rho"], sig0, name="mask_rho").reset_coords(drop=True) == 1
    else:
        water = surface.notnull()
    depth = depth.fillna(h.where(water))

    # Take absolute value so as to return positive MLD values
    return _label(abs(depth), "mld", "mixed layer depth", "m")
