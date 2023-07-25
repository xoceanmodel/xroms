"""
Variables derived from ROMS output are here.
"""

import numpy as np
import xarray as xr

# import xroms
from .utilities import ddz, hgrad, to_grid, to_rho, to_u, to_v


g = 9.81  # m/s^2


def speed(u, v, xgrid, hboundary="extend", hfill_value=None):
    """Calculate horizontal speed [m/s] from u and v components

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with u, v
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for moving to rho grid.
        From xgcm documentation:
        A flag indicating how to handle boundaries:
        * None:  Do not apply any boundary conditions. Raise an error if
          boundary conditions are required for the operation.
        * 'fill':  Set values outside the array boundary to fill_value
          (i.e. a Neumann boundary condition.)
        * 'extend': Set values outside the array to the nearest array
          value. (i.e. a limited form of Dirichlet boundary condition.
    hfill_value: float, optional
        Passed to `grid` method calls; horizontal boundary fill value
        selection for moving to rho grid.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray of speed calculated on rho/rho grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    speed = np.sqrt(u^2 + v^2)

    Example usage
    -------------
    >>> xroms.speed(ds.u, ds.v, xgrid)
    """

    assert isinstance(u, xr.DataArray), "var must be DataArray"
    assert isinstance(v, xr.DataArray), "var must be DataArray"

    u = to_rho(u, xgrid, hboundary=hboundary, hfill_value=hfill_value)
    v = to_rho(v, xgrid, hboundary=hboundary, hfill_value=hfill_value)
    var = np.sqrt(u**2 + v**2)

    var.attrs["name"] = "speed"
    var.attrs["long_name"] = "horizontal speed"
    var.attrs["units"] = "m/s"
    var.name = var.attrs["name"]

    return var


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
    >>> speed = xroms.speed(ds.u, ds.v, xgrid)
    >>> xroms.KE(ds.rho0, speed)
    """

    assert isinstance(speed, xr.DataArray), "speed must be DataArray"

    var = 0.5 * rho0 * speed**2

    var.attrs["name"] = "KE"
    var.attrs["long_name"] = "kinetic energy"
    var.attrs["units"] = "kg/(m*s^2)"
    var.name = var.attrs["name"]

    return var


def uv_geostrophic(zeta, f, xgrid, hboundary="extend", hfill_value=None, which="both"):
    """Calculate geostrophic velocities from zeta [m/s]

    Parameters
    ----------
    zeta: DataArray
        sea surface height [m]
    f: DataArray or ndarray
        Coriolis parameter [1/s]
    xgrid: xgcm.grid
        Grid object associated with zeta
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for moving f to rho grid.
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
        for moving f to rho grid.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.
    which: string, optional
        Which components of geostrophic velocity to return.
        * 'both': return both components of hgrad
        * 'xi': return only xi-direction.
        * 'eta': return only eta-direction.

    Returns
    -------
    DataArrays of components of geostrophic velocity
    calculated on their respective grids.
    Output is `[T,Y,X]`.

    Notes
    -----
    vg = g * zeta_eta / (d eta * f)  # on v grid
    ug = -g * zeta_xi / (d xi * f)  # on u grid
    Translation to Python of Matlab copy of surf_geostr_vel of IRD Roms_Tools.

    Examples
    --------
    >>> xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)
    """

    assert isinstance(zeta, xr.DataArray), "zeta must be DataArray"
    assert isinstance(f, xr.DataArray), "f must be DataArray"

    if which in ["both", "xi"]:

        # calculate derivatives of zeta
        dzetadxi = hgrad(zeta, xgrid, which="xi")

        # calculate geostrophic velocities
        ug = (
            -g * dzetadxi / to_u(f, xgrid, hboundary=hboundary, hfill_value=hfill_value)
        )

        ug.attrs["name"] = "ug"
        ug.attrs["long_name"] = "geostrophic u velocity"
        ug.attrs["units"] = "m/s"
        ug.name = ug.attrs["name"]

    if which in ["both", "eta"]:
        # calculate derivatives of zeta
        dzetadeta = hgrad(zeta, xgrid, which="eta")

        # calculate geostrophic velocities
        vg = (
            g * dzetadeta / to_v(f, xgrid, hboundary=hboundary, hfill_value=hfill_value)
        )

        vg.attrs["name"] = "vg"
        vg.attrs["long_name"] = "geostrophic v velocity"
        vg.attrs["units"] = "m/s"
        vg.name = vg.attrs["name"]

    if which == "both":
        return ug, vg
    elif which == "xi":
        return ug
    elif which == "eta":
        return vg
    else:
        print("nothing being returned from uv_geostrophic")


def EKE(ug, vg, xgrid, hboundary="extend", hfill_value=None):
    """Calculate EKE [m^2/s^2]

    Parameters
    ----------
    ug: DataArray
        Geostrophic or other xi component velocity [m/s]
    vg: DataArray
        Geostrophic or other eta component velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with ug, vg
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for moving to rho grid.
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
        for moving to rho grid.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray of eddy kinetic energy on rho grid.
    Output is `[T,Y,X]`.

    Notes
    -----
    EKE = 0.5*(ug^2 + vg^2)

    Examples
    --------
    >>> ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)
    >>> xroms.EKE(ug, vg, xgrid)
    """

    assert isinstance(ug, xr.DataArray), "ug must be DataArray"
    assert isinstance(vg, xr.DataArray), "vg must be DataArray"

    # make sure velocities are on rho grid
    ug = to_rho(ug, xgrid, hboundary=hboundary, hfill_value=hfill_value)
    vg = to_rho(vg, xgrid, hboundary=hboundary, hfill_value=hfill_value)

    var = 0.5 * (ug**2 + vg**2)

    var.attrs["name"] = "EKE"
    var.attrs["long_name"] = "eddy kinetic energy"
    var.attrs["units"] = "m^2/s^2"
    var.name = var.attrs["name"]

    return var


def dudz(u, xgrid, sboundary="extend", sfill_value=None):
    """Calculate the xi component of vertical shear [1/s]

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with u
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
    DataArray of xi component of vertical shear on u/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    u_z = ddz(u)
    Wrapper of `ddz`

    Examples
    --------
    >>> xroms.dudz(u, xgrid)
    """

    attrs = {
        "name": "dudz",
        "long_name": "u component of vertical shear",
        "units": "1/s",
    }
    return ddz(u, xgrid, attrs=attrs, sboundary=sboundary, sfill_value=sfill_value)


def dvdz(v, xgrid, sboundary="extend", sfill_value=None):
    """Calculate the eta component of vertical shear [1/s]

    Parameters
    ----------
    v: DataArray
        eta component of velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with v
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
    DataArray of eta component of vertical shear on v/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    v_z = ddz(v)
    Wrapper of `ddz`

    Examples
    --------
    >>> xroms.dvdz(v, xgrid)
    """

    attrs = {
        "name": "dvdz",
        "long_name": "v component of vertical shear",
        "units": "1/s",
    }
    return ddz(v, xgrid, attrs=attrs, sboundary=sboundary, sfill_value=sfill_value)


def vertical_shear(dudz, dvdz, xgrid, hboundary="extend", hfill_value=None):
    """Calculate the vertical shear [1/s]

    Parameters
    ----------
    dudz: DataArray
        xi component of vertical shear [1/s]
    dvdz: DataArray
        eta compoenent of vertical shear [1/s]
    xgrid: xgcm.grid
        Grid object associated with dudz, dvdz
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for moving dudz and dvdz to rho grid.
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
        for moving to rho grid.
        From xgcm documentation:
        The value to use in the boundary condition with `boundary='fill'`.

    Returns
    -------
    DataArray of vertical shear on rho/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    vertical_shear = np.sqrt(u_z^2 + v_z^2)

    Examples
    --------
    >>> xroms.vertical_shear(dudz, dvdz, xgrid)
    """

    assert isinstance(dudz, xr.DataArray), "dudz must be DataArray"
    assert isinstance(dvdz, xr.DataArray), "dvdz must be DataArray"

    # make sure velocities are on rho grid
    dudz = to_rho(dudz, xgrid, hboundary=hboundary, hfill_value=hfill_value)
    dvdz = to_rho(dvdz, xgrid, hboundary=hboundary, hfill_value=hfill_value)

    var = np.sqrt(dudz**2 + dvdz**2)

    var.attrs["name"] = "shear"
    var.attrs["long_name"] = "vertical shear"
    var.attrs["units"] = "1/s"
    var.name = var.attrs["name"]

    return var


def relative_vorticity(
    u,
    v,
    xgrid,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
):
    """Calculate the vertical component of the relative vorticity [1/s]

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with u, v
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivatives of u and v.
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
        for calculating horizontal derivatives of u and v.
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
    DataArray of vertical component of relative vorticity psi/w grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    relative_vorticity = v_x - u_y

    Examples
    --------
    >>> xroms.relative_vorticity(u, v, xgrid)
    """

    assert isinstance(u, xr.DataArray), "u must be DataArray"
    assert isinstance(v, xr.DataArray), "v must be DataArray"

    dvdxi = hgrad(
        v,
        xgrid,
        which="xi",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    dudeta = hgrad(
        u,
        xgrid,
        which="eta",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    var = dvdxi - dudeta

    var.attrs["name"] = "vort"
    var.attrs["long_name"] = "vertical component of vorticity"
    var.attrs["units"] = "1/s"
    var.name = var.attrs["name"]

    return var


def divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    xgrid,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
) -> xr.DataArray:
    """Calculate 2D divergence from u and v [1/s].

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    xgrid: xgcm.grid
        Grid object associated with u, v
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivatives of u and v.
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
        for calculating horizontal derivatives of u and v.
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
    DataArray of 2D divergence of horizontal currents on rho/rho grids.
    Output is `[T,Z,Y,X]`.


    Notes
    -----
    2D divergence = u_x + v_y
    Resource for more information: https://uw.pressbooks.pub/ocean285/chapter/the-divergence/

    Examples
    --------
    >>> ds, xgrid = xroms.roms_dataset(ds)
    >>> xroms.divergence(u, v, xgrid)
    """

    assert isinstance(u, xr.DataArray), "u must be DataArray"
    assert isinstance(v, xr.DataArray), "v must be DataArray"

    dudxi = hgrad(
        u,
        xgrid,
        which="xi",
        scoord="s_rho",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    dvdeta = hgrad(
        v,
        xgrid,
        which="eta",
        scoord="s_rho",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    var = dudxi + dvdeta

    var.attrs["name"] = "div"
    var.attrs["long_name"] = "horizontal divergence"
    var.attrs["units"] = "1/s"
    var.name = var.attrs["name"]

    return var


def ertel(
    phi,
    u,
    v,
    f,
    xgrid,
    hcoord="rho",
    scoord="s_rho",
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
):
    """Calculate Ertel potential vorticity of phi.

    Parameters
    ----------
    phi: DataArray
        Conservative tracer. Usually this would be the buoyancy but
        could be another approximately conservative tracer. The
        buoyancy can be calculated as:
        >>> xroms.buoyancy(temp, salt, 0)
        and then input as `phi`.
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    f: DataArray
        Coriolis parameter [1/s]
    xgrid: xgcm.grid
        Grid object associated with u, v
    hcoord: string, optional.
        Name of horizontal grid to interpolate output to.
        Options are 'rho', 'psi', 'u', 'v'.
    scoord: string, optional.
        Name of vertical grid to interpolate output to.
        Options are 's_rho', 's_w', 'rho', 'w'.
    hboundary: string, optional
        Passed to `grid` method calls; horizontal boundary selection
        for calculating horizontal derivatives of phi and for calculating
        relative vorticity. This same value will be used for all
        horizontal grid changes too.
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
        for calculating horizontal and vertical derivatives of phi, and
        for calculating relative vorticity. This same value will be used for
        all vertical grid changes too.
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
    DataArray of the Ertel potential vorticity for the input tracer.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    epv = -v_z * phi_x + u_z * phi_y + (f + v_x - u_y) * phi_z

    This is not set up to accept different boundary choices for different variables.

    Example usage:
    >>> xroms.ertel(ds.dye_01, ds.u, ds.v, ds.f, xgrid, scoord='s_w');
    """

    assert isinstance(phi, xr.DataArray), "phi must be DataArray"
    assert isinstance(u, xr.DataArray), "u must be DataArray"
    assert isinstance(v, xr.DataArray), "v must be DataArray"
    assert isinstance(f, xr.DataArray), "f must be DataArray"

    phi_xi, phi_eta = hgrad(
        phi,
        xgrid,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_xi = to_grid(
        phi_xi,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_eta = to_grid(
        phi_eta,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_z = ddz(
        phi,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    # vertical shear (horizontal components of vorticity)
    u_z = ddz(
        u,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    v_z = ddz(
        v,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    # vertical component of vorticity
    vort = relative_vorticity(
        u,
        v,
        xgrid,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    vort = to_grid(
        vort,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    # combine terms to get the ertel potential vorticity
    epv = -v_z * phi_xi + u_z * phi_eta + (f + vort) * phi_z

    attrs = {
        "name": "ertel",
        "long_name": "ertel potential vorticity",
        "units": "tracer/(m*s)",
    }
    epv = to_grid(
        epv,
        xgrid,
        hcoord=hcoord,
        scoord=scoord,
        attrs=attrs,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    return epv


def w(u, v):
    """Calculate vertical velocity from u and v [m/s]

    TO BE INPUT BY VRX.

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]

    Returns
    -------
    DataArray of vertical component of velocity on [horizontal]/[vertical] grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    [Give calculation]

    Examples
    --------
    >>> xroms.w(u, v)
    """


def omega(u, v):
    """Calculate s-grid vertical velocity from u and v [m/s]

    TO BE INPUT BY VRX.

    Parameters
    ----------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]

    Returns
    -------
    DataArray of vertical component of velocity with respect to the s grid
    on [horizontal]/[vertical] grids.
    Output is `[T,Z,Y,X]`.

    Notes
    -----
    [Give calculation]

    Examples
    --------
    >>> xroms.omega(u, v)
    """
