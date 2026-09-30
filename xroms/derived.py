"""
Variables derived from ROMS output are here.

Every function is stateless: it computes from the arrays (and the grid Dataset)
passed in, never modifies them, and returns a new object in canonical ROMS dim
names. Results land where the calculation naturally puts them:

* `speed`, `EKE` and `convergence` on rho points;
* `uv_geostrophic` on u points (``ug``) and v points (``vg``);
* `dudz`, `dvdz` on (u or v, w) points and `vertical_shear` on (rho, w) points;
* `relative_vorticity` on psi points;
* `ertel` on ``hcoord``/``scoord`` (rho/s_rho by default).

Horizontal derivatives keep the vertical levels of their inputs.

`relative_vorticity`, `convergence` and `ertel` difference u along eta and v along
xi, which only means something at their own points, so they need u on u points and
v on v points and raise a ValueError otherwise (see :func:`xroms.to_u` and
:func:`xroms.to_v` to move velocities that are elsewhere).
"""

import numpy as np
import xarray as xr

from .conventions import hposition, normalize_hcoord, normalize_scoord
from .utilities import (
    _check_grid,
    _reject_legacy,
    ddeta,
    ddxi,
    ddz,
    order,
    to_grid,
    to_rho,
    to_u,
    to_v,
)


g = 9.81  # m/s^2


def _check_dataarrays(**arrays):
    """Raise a clear TypeError unless every named argument is a DataArray."""
    for name, value in arrays.items():
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"{name} must be a DataArray, not {type(value).__name__}")


def _check_uv_positions(u, v, func):
    """Raise a ValueError unless ``u`` is on u points and ``v`` on v points.

    ``func`` differences u along eta and v along xi. With the two swapped, or moved
    to other points, the derivatives land on different points and the result is
    silently wrong (or broadcasts to extra dimensions). A position that cannot be
    told from the dims (None) is left to the derivative that needs it, which says so.
    """
    found = hposition(u), hposition(v)
    if found[0] in (None, "u") and found[1] in (None, "v"):
        return
    where = [
        f"{name} is on {pos} points" if pos else f"{name} has dims {da.dims}, which do not give its points"
        for name, pos, da in (("u", found[0], u), ("v", found[1], v))
    ]
    raise ValueError(
        f"{func} needs u on u points and v on v points, but {where[0]} and {where[1]}. Check that u "
        "(the xi component) and v (the eta component) are not swapped; velocities on other points "
        "can be moved with xroms.to_u and xroms.to_v."
    )


def _label(var, name, long_name, units):
    """Name ``var`` and give it xroms' attrs.

    ``var`` is always a result computed here, never an input. Its attrs are
    replaced rather than merged, so nothing stale is inherited from the inputs
    (a ``standard_name`` of ``u``, say).
    """
    var.attrs = {"name": name, "long_name": long_name, "units": units}
    var.name = name
    return var


def speed(u, v, *args, hboundary="extend", hfill_value=np.nan):
    """Calculate horizontal speed [m/s] from u and v components

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment when moving u and v to rho points:
        ``"extend"`` (default) repeats the nearest value at the domain edge,
        ``"fill"`` uses ``hfill_value`` there.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.

    Returns
    -------
    DataArray of speed calculated on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    speed = np.sqrt(u^2 + v^2)

    Masked (NaN) velocities are set to 0 before moving to rho points so that land
    does not spread into neighboring water points.

    Before xroms 1.0 the xgcm grid was passed here as a third, positional argument;
    it is no longer needed and passing it raises a `TypeError`.

    Example usage
    -------------
    >>> xroms.speed(ds.u, ds.v)
    """

    _reject_legacy(args, "speed", "Use xroms.speed(u, v).")
    _check_dataarrays(u=u, v=v)

    # need to fill nans with zeros so that the masked locations in
    # velocity fields are not fully brought forward into the rho mask
    # but are instead interpolated over. By making them 0, they are
    # calculated into the mask_rho positions by combining them with
    # neighboring cells. If this wasn't done, the fact that they are masked
    # would supersede the neighboring cells and they would be masked in mask_rho.
    # this needs to be done anytime the velocities are moved from their native
    # grids to the rho or other grids to preserve their locations around masked cells.
    u = to_rho(u.fillna(0), hboundary=hboundary, hfill_value=hfill_value)
    v = to_rho(v.fillna(0), hboundary=hboundary, hfill_value=hfill_value)
    var = np.sqrt(u**2 + v**2)

    return _label(var, "speed", "horizontal speed", "m/s")


def KE(rho0, speed):
    """Calculate kinetic energy [kg/(m*s^2)]

    Parameters
    ----------
    rho0: float
        background density of the water [kg/m^3]
    speed: DataArray
        magnitude of horizontal velocity vector [m/s]

    Returns
    -------
    DataArray of kinetic energy on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    KE = 0.5*rho*(u^2 + v^2)

    Examples
    --------
    >>> speed = xroms.speed(ds.u, ds.v)
    >>> xroms.KE(xroms.rho0(ds), speed)
    """

    _check_dataarrays(speed=speed)

    var = 0.5 * rho0 * speed**2

    # speed is whatever DataArray was passed in, and rho0 may have dimensions of its own
    return _label(order(var), "KE", "kinetic energy", "kg/(m*s^2)")


def uv_geostrophic(
    zeta, f, grid, *, hboundary="extend", hfill_value=np.nan, which="both"
):
    """Calculate geostrophic velocities from zeta [m/s]

    Parameters
    ----------
    zeta: DataArray
        sea surface height [m]
    f: DataArray
        Coriolis parameter [1/s]
    grid: Dataset
        Dataset holding the grid variables (``pm`` and ``pn``) associated with zeta.
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment when differentiating zeta and moving
        the result and f between grid positions: ``"extend"`` (default) repeats the
        nearest value at the domain edge, ``"fill"`` uses ``hfill_value`` there.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.
    which: string, optional
        Which components of geostrophic velocity to return.

        * 'both': return both components, ``(ug, vg)``.
        * 'xi': return only the xi component, ``ug``.
        * 'eta': return only the eta component, ``vg``.

    Returns
    -------
    DataArrays of components of geostrophic velocity
    calculated on their respective grids: ``ug`` on u points and ``vg`` on v points.
    Output is `[T,Y,X]`.

    Notes
    -----

    ug = -g * zeta_eta / f  # on u grid

    vg = g * zeta_xi / f  # on v grid

    The derivatives are taken along the model's eta and xi directions, so ug and vg
    are along xi and eta, not east and north.

    Translation to Python of Matlab copy of surf_geostr_vel of IRD Roms_Tools.

    Good resourcefor more information:
    https://uw.pressbooks.pub/ocean285/chapter/geostrophic-balance/

    Examples
    --------
    >>> xroms.uv_geostrophic(ds.zeta, ds.f, ds)
    """

    grid = _check_grid(grid, "uv_geostrophic")
    _check_dataarrays(zeta=zeta, f=f)
    if which not in ("both", "xi", "eta"):
        raise ValueError(f"which must be 'both', 'xi' or 'eta', not {which!r}")
    hopts = dict(hboundary=hboundary, hfill_value=hfill_value)

    if which in ["both", "xi"]:

        # calculate derivatives of zeta
        dzetadeta = ddeta(zeta, grid, hcoord="u", **hopts)

        # calculate geostrophic velocities
        ug = -g * dzetadeta / to_u(f, **hopts)

        ug = _label(ug, "ug", "geostrophic u velocity", "m/s")

    if which in ["both", "eta"]:

        # calculate derivatives of zeta
        dzetadxi = ddxi(zeta, grid, hcoord="v", **hopts)

        # calculate geostrophic velocities
        vg = g * dzetadxi / to_v(f, **hopts)

        vg = _label(vg, "vg", "geostrophic v velocity", "m/s")

    if which == "both":
        return ug, vg
    elif which == "xi":
        return ug
    else:
        return vg


def EKE(ug, vg, *args, hboundary="extend", hfill_value=np.nan):
    """Calculate EKE [m^2/s^2]

    Parameters
    ----------
    ug: DataArray
        Geostrophic or other xi component velocity [m/s]
    vg: DataArray
        Geostrophic or other eta component velocity [m/s]
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment when moving ug and vg to rho points:
        ``"extend"`` (default) repeats the nearest value at the domain edge,
        ``"fill"`` uses ``hfill_value`` there.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.

    Returns
    -------
    DataArray of eddy kinetic energy on rho grid.
    Output is `[T,Y,X]`.

    Notes
    -----
    EKE = 0.5*(ug^2 + vg^2)

    Before xroms 1.0 the xgcm grid was passed here as a third, positional argument;
    it is no longer needed and passing it raises a `TypeError`.

    Examples
    --------
    >>> ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, ds)
    >>> xroms.EKE(ug, vg)
    """

    _reject_legacy(args, "EKE", "Use xroms.EKE(ug, vg).")
    _check_dataarrays(ug=ug, vg=vg)

    # make sure velocities are on rho grid
    ug = to_rho(ug, hboundary=hboundary, hfill_value=hfill_value)
    vg = to_rho(vg, hboundary=hboundary, hfill_value=hfill_value)

    var = 0.5 * (ug**2 + vg**2)

    return _label(var, "EKE", "eddy kinetic energy", "m^2/s^2")


def dudz(u, grid=None, *, z=None, zeta=None, sboundary="extend", sfill_value=np.nan):
    """Calculate the xi component of vertical shear [1/s]

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    grid: Dataset, optional
        Dataset holding the grid variables (``h`` and the s-coordinate parameters,
        and ``zeta`` unless ``zeta`` is given) associated with u. Needed unless
        ``z`` is given.
    z: DataArray, optional
        Depths [m] at the points of ``u`` to use instead of computing them from
        ``grid``. Depths at rho points are averaged onto the u points.
    zeta: None, float, "mean" or DataArray, optional
        Free surface used to compute depths: the grid's ``zeta`` (default), a
        constant (``0`` for static depths), its time mean, or an explicit field.
    sboundary: {"extend", "fill"}, optional
        Vertical boundary treatment at the top and bottom w levels:
        ``"extend"`` (default) copies the nearest computed value (a one-sided
        estimate), ``"fill"`` uses ``sfill_value``.
    sfill_value: float, optional
        Edge value used with ``sboundary="fill"``. Default NaN.

    Returns
    -------
    DataArray of xi component of vertical shear on u/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    u_z = ddz(u)
    Wrapper of `ddz`

    Examples
    --------
    >>> xroms.dudz(ds.u, ds)
    """

    grid = _check_grid(grid, "dudz")
    _check_dataarrays(u=u)

    attrs = {
        "name": "dudz",
        "long_name": "u component of vertical shear",
        "units": "1/s",
    }
    return ddz(
        u,
        grid,
        z=z,
        zeta=zeta,
        attrs=attrs,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )


def dvdz(v, grid=None, *, z=None, zeta=None, sboundary="extend", sfill_value=np.nan):
    """Calculate the eta component of vertical shear [1/s]

    Parameters
    ----------
    v: DataArray
        eta component of velocity [m/s]
    grid: Dataset, optional
        Dataset holding the grid variables (``h`` and the s-coordinate parameters,
        and ``zeta`` unless ``zeta`` is given) associated with v. Needed unless
        ``z`` is given.
    z: DataArray, optional
        Depths [m] at the points of ``v`` to use instead of computing them from
        ``grid``. Depths at rho points are averaged onto the v points.
    zeta: None, float, "mean" or DataArray, optional
        Free surface used to compute depths: the grid's ``zeta`` (default), a
        constant (``0`` for static depths), its time mean, or an explicit field.
    sboundary: {"extend", "fill"}, optional
        Vertical boundary treatment at the top and bottom w levels:
        ``"extend"`` (default) copies the nearest computed value (a one-sided
        estimate), ``"fill"`` uses ``sfill_value``.
    sfill_value: float, optional
        Edge value used with ``sboundary="fill"``. Default NaN.

    Returns
    -------
    DataArray of eta component of vertical shear on v/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    v_z = ddz(v)
    Wrapper of `ddz`

    Examples
    --------
    >>> xroms.dvdz(ds.v, ds)
    """

    grid = _check_grid(grid, "dvdz")
    _check_dataarrays(v=v)

    attrs = {
        "name": "dvdz",
        "long_name": "v component of vertical shear",
        "units": "1/s",
    }
    return ddz(
        v,
        grid,
        z=z,
        zeta=zeta,
        attrs=attrs,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )


def vertical_shear(dudz, dvdz, *args, hboundary="extend", hfill_value=np.nan):
    """Calculate the vertical shear [1/s]

    Parameters
    ----------
    dudz: DataArray
        xi component of vertical shear [1/s]
    dvdz: DataArray
        eta compoenent of vertical shear [1/s]
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment when moving dudz and dvdz to rho points:
        ``"extend"`` (default) repeats the nearest value at the domain edge,
        ``"fill"`` uses ``hfill_value`` there.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.

    Returns
    -------
    DataArray of vertical shear on rho/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    vertical_shear = np.sqrt(u_z^2 + v_z^2)

    Before xroms 1.0 the xgcm grid was passed here as a third, positional argument;
    it is no longer needed and passing it raises a `TypeError`.

    Examples
    --------
    >>> xroms.vertical_shear(xroms.dudz(ds.u, ds), xroms.dvdz(ds.v, ds))
    """

    _reject_legacy(args, "vertical_shear", "Use xroms.vertical_shear(dudz, dvdz).")
    _check_dataarrays(dudz=dudz, dvdz=dvdz)

    # make sure velocities are on rho grid
    dudz = to_rho(dudz, hboundary=hboundary, hfill_value=hfill_value)
    dvdz = to_rho(dvdz, hboundary=hboundary, hfill_value=hfill_value)

    var = np.sqrt(dudz**2 + dvdz**2)

    return _label(var, "shear", "vertical shear", "1/s")


def relative_vorticity(
    u,
    v,
    grid,
    *,
    z=None,
    zeta=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    along_s=False,
):
    """Calculate the vertical component of the relative vorticity [1/s]

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s], on u points
    v: DataArray
        eta component of velocity [m/s], on v points
    grid: Dataset
        Dataset holding the grid variables (``pm`` and ``pn``, and for 3D inputs
        ``h`` and the s-coordinate parameters) associated with u, v.
    z: DataArray, optional
        Depths [m] at rho points (on the vertical levels of u and v) to use
        instead of computing them from ``grid``. They are averaged onto the u and
        v points.
    zeta: None, float, "mean" or DataArray, optional
        Free surface used to compute depths: the grid's ``zeta`` (default), a
        constant (``0`` for static depths), its time mean, or an explicit field.
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment where calculating and moving horizontal
        derivatives needs values outside the data: ``"extend"`` (default) uses the
        nearest computed value, ``"fill"`` uses ``hfill_value``.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.
    sboundary: {"extend", "fill"}, optional
        Same as ``hboundary``, for the vertical derivative that converts the
        derivatives from along the s-levels to at constant depth.
    sfill_value: float, optional
        Edge value used with ``sboundary="fill"``. Default NaN.
    along_s: bool, optional
        For single selected s-levels only: accept derivatives along the s-surface
        instead of at constant depth (see `xroms.ddxi`). Default False.

    Returns
    -------
    DataArray of vertical component of relative vorticity on psi grid, on the
    vertical levels of u and v.
    Output is `[T,Z,Y,X]`.

    Raises
    ------
    ValueError
        If u is not on u points or v is not on v points (for example, if they are
        swapped).

    Notes
    -----
    relative_vorticity = v_x - u_y

    Derivatives are taken at constant depth.

    Examples
    --------
    >>> xroms.relative_vorticity(ds.u, ds.v, ds)
    """

    grid = _check_grid(grid, "relative_vorticity")
    _check_dataarrays(u=u, v=v)
    _check_uv_positions(u, v, "relative_vorticity")
    opts = dict(
        zeta=zeta,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
        along_s=along_s,
    )

    dvdxi = ddxi(v, grid, z=z, **opts)
    dudeta = ddeta(u, grid, z=z, **opts)

    var = dvdxi - dudeta

    return _label(var, "vort", "vertical component of vorticity", "1/s")


def convergence(
    u: xr.DataArray,
    v: xr.DataArray,
    grid,
    *,
    z=None,
    zeta=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    along_s=False,
) -> xr.DataArray:
    """Calculate 2D convergence from u and v [1/s].

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s], on u points
    v: DataArray
        eta component of velocity [m/s], on v points
    grid: Dataset
        Dataset holding the grid variables (``pm`` and ``pn``, and for 3D inputs
        ``h`` and the s-coordinate parameters) associated with u, v.
    z: DataArray, optional
        Depths [m] at rho points (on the vertical levels of u and v) to use
        instead of computing them from ``grid``. They are averaged onto the u and
        v points.
    zeta: None, float, "mean" or DataArray, optional
        Free surface used to compute depths: the grid's ``zeta`` (default), a
        constant (``0`` for static depths), its time mean, or an explicit field.
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment where calculating and moving horizontal
        derivatives needs values outside the data: ``"extend"`` (default) uses the
        nearest computed value, ``"fill"`` uses ``hfill_value``.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.
    sboundary: {"extend", "fill"}, optional
        Same as ``hboundary``, for the vertical derivative that converts the
        derivatives from along the s-levels to at constant depth.
    sfill_value: float, optional
        Edge value used with ``sboundary="fill"``. Default NaN.
    along_s: bool, optional
        For single selected s-levels only: accept derivatives along the s-surface
        instead of at constant depth (see `xroms.ddxi`). Default False.

    Returns
    -------
    DataArray of 2D convergence of horizontal currents on rho grid, on the
    vertical levels of u and v.
    Output is `[T,Z,Y,X]`.

    Raises
    ------
    ValueError
        If u is not on u points or v is not on v points (for example, if they are
        swapped).

    Notes
    -----
    2D convergence = u_x + v_y

    Derivatives are taken at constant depth.

    Resource for more information: https://uw.pressbooks.pub/ocean285/chapter/the-divergence/

    Examples
    --------
    >>> xroms.convergence(ds.u, ds.v, ds)
    """

    grid = _check_grid(grid, "convergence")
    _check_dataarrays(u=u, v=v)
    _check_uv_positions(u, v, "convergence")
    opts = dict(
        zeta=zeta,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
        along_s=along_s,
    )

    dudxi = ddxi(u, grid, z=z, **opts)
    dvdeta = ddeta(v, grid, z=z, **opts)

    var = dudxi + dvdeta

    return _label(var, "convergence", "horizontal convergence", "1/s")


def ertel(
    phi,
    u,
    v,
    f,
    grid,
    *,
    hcoord="rho",
    scoord="s_rho",
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
    sfill_value=np.nan,
    zeta=None,
):
    """Calculate Ertel potential vorticity of phi.

    Parameters
    ----------
    phi: DataArray
        Conservative tracer. Usually this would be the buoyancy but
        could be another approximately conservative tracer. The
        buoyancy can be calculated as:
        >>> xroms.buoyancy(xroms.potential_density(temp, salt))
        and then input as `phi`.
    u: DataArray
        xi component of velocity [m/s], on u points
    v: DataArray
        eta component of velocity [m/s], on v points
    f: DataArray
        Coriolis parameter [1/s], at any horizontal grid position (it is moved
        to ``hcoord``).
    grid: Dataset
        Dataset holding the grid variables (``pm`` and ``pn``, ``h`` and the
        s-coordinate parameters) associated with phi, u, v.
    hcoord: string, optional.
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional.
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: {"extend", "fill"}, optional
        Horizontal boundary treatment where calculating horizontal derivatives
        of phi and relative vorticity, and moving between horizontal grids,
        needs values outside the data: ``"extend"`` (default) uses the nearest
        computed value, ``"fill"`` uses ``hfill_value``. The same choice is used
        for all horizontal grid changes.
    hfill_value: float, optional
        Edge value used with ``hboundary="fill"``. Default NaN.
    sboundary: {"extend", "fill"}, optional
        Same as ``hboundary``, for vertical derivatives and vertical grid changes.
    sfill_value: float, optional
        Edge value used with ``sboundary="fill"``. Default NaN.
    zeta: None, float, "mean" or DataArray, optional
        Free surface used to compute depths: the grid's ``zeta`` (default), a
        constant (``0`` for static depths), its time mean, or an explicit field.

    Returns
    -------
    DataArray of the Ertel potential vorticity for the input tracer.
    Output is `[T,Z,Y,X]`.

    Raises
    ------
    ValueError
        If u is not on u points or v is not on v points (for example, if they are
        swapped), or ``hcoord`` or ``scoord`` is None.

    Notes
    -----
    epv = -v_z * phi_x + u_z * phi_y + (f + v_x - u_y) * phi_z

    Horizontal derivatives are taken at constant depth. Each term is moved to
    ``hcoord``/``scoord`` before they are combined.

    This is not set up to accept different boundary choices for different variables.

    Example usage:
    >>> xroms.ertel(ds.dye_01, ds.u, ds.v, ds.f, ds, scoord='s_w');
    """

    grid = _check_grid(grid, "ertel")
    _check_dataarrays(phi=phi, u=u, v=v, f=f)
    _check_uv_positions(u, v, "ertel")
    hcoord, scoord = normalize_hcoord(hcoord), normalize_scoord(scoord)
    if hcoord is None or scoord is None:
        raise ValueError(
            "ertel combines terms from different grid positions, so hcoord and "
            "scoord must be given (the defaults are 'rho' and 's_rho')"
        )
    hopts = dict(hboundary=hboundary, hfill_value=hfill_value)
    sopts = dict(sboundary=sboundary, sfill_value=sfill_value)
    where = dict(hcoord=hcoord, scoord=scoord, zeta=zeta, **hopts, **sopts)

    phi_xi = ddxi(phi, grid, **where)
    phi_eta = ddeta(phi, grid, **where)
    phi_z = ddz(phi, grid, **where)

    # vertical shear (horizontal components of vorticity)
    u_z = ddz(u, grid, **where)
    v_z = ddz(v, grid, **where)

    # vertical component of vorticity
    vort = relative_vorticity(u, v, grid, zeta=zeta, **hopts, **sopts)
    vort = to_grid(vort, hcoord, scoord, **hopts, **sopts)

    # planetary vorticity, wherever f was given
    f = to_grid(f, hcoord, **hopts)

    # combine terms to get the ertel potential vorticity
    epv = -v_z * phi_xi + u_z * phi_eta + (f + vort) * phi_z

    attrs = {
        "name": "ertel",
        "long_name": "ertel potential vorticity",
        "units": "tracer/(m*s)",
    }
    return to_grid(epv, hcoord, scoord, attrs=attrs, **hopts, **sopts)
