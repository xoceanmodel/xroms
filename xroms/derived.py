import numpy as np
import xarray as xr

import xroms


g = 9.81  # m/s^2


def speed(u, v, grid, hboundary="extend", hfill_value=None):
    """Calculate horizontal speed [m/s] from u and v components

    Inputs
    ------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    grid: xgcm.grid
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

    Notes
    -----
    speed = np.sqrt(u^2 + v^2)

    Example usage
    -------------
    >>> xroms.speed(ds.u, ds.v, grid)
    """

    assert isinstance(u, xr.DataArray), "var must be DataArray"
    assert isinstance(v, xr.DataArray), "var must be DataArray"

    u = xroms.to_rho(u, grid, hboundary=hboundary, hfill_value=hfill_value)
    v = xroms.to_rho(v, grid, hboundary=hboundary, hfill_value=hfill_value)
    var = np.sqrt(u ** 2 + v ** 2)

    var.attrs["name"] = "s"
    var.attrs["long_name"] = "horizontal speed"
    var.attrs["units"] = "m/s"
    var.attrs["grid"] = grid
    var.name = var.attrs["name"]

    return var


def KE(rho0, speed):
    """Calculate kinetic energy [kg/(m*s^2)]

    Inputs
    ------
    rho0: float
        background density of the water [kg/m^3]
    speed: DataArray
        magnitude of horizontal velocity vector [m/s]

    Returns
    -------
    DataArray of kinetic energy on rho/rho grids.

    Notes
    -----
    KE = 0.5*rho*(u^2 + v^2)

    Example usage
    -------------
    >>> speed = xroms.speed(ds.u, ds.v, ds.attrs['grid'])
    >>> xroms.KE(ds.rho0, speed)
    """

    assert isinstance(speed, xr.DataArray), "speed must be DataArray"

    var = 0.5 * rho0 * speed ** 2

    var.attrs["name"] = "KE"
    var.attrs["long_name"] = "kinetic energy"
    var.attrs["units"] = "kg/(m*s^2)"
    if "grid" in speed.attrs:
        var.attrs["grid"] = speed.attrs["grid"]
    var.name = var.attrs["name"]

    return var


def uv_geostrophic(zeta, f, grid, hboundary="extend", hfill_value=None, which="both"):
    """Calculate geostrophic velocities from zeta [m/s]

    Inputs
    ------
    zeta: DataArray
        sea surface height [m]
    f: DataArray or ndarray
        Coriolis parameter [1/s]
    grid: xgcm.grid
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

    Notes
    -----
    vg = g * zeta_eta / (d eta * f)  # on v grid
    ug = -g * zeta_xi / (d xi * f)  # on u grid
    Translation to Python of Matlab copy of surf_geostr_vel of IRD Roms_Tools.

    Example usage
    -------------
    >>> xroms.uv_geostrophic(ds.zeta, ds.f, grid)
    """

    assert isinstance(zeta, xr.DataArray), "zeta must be DataArray"
    assert isinstance(f, xr.DataArray), "f must be DataArray"

    if which in ["both", "xi"]:

        # calculate derivatives of zeta
        dzetadxi = xroms.hgrad(zeta, grid, which="xi")

        # calculate geostrophic velocities
        ug = (
            -g
            * dzetadxi
            / xroms.to_u(f, grid, hboundary=hboundary, hfill_value=hfill_value)
        )

        ug.attrs["name"] = "u_geo"
        ug.attrs["long_name"] = "geostrophic u velocity"
        ug.attrs["units"] = "m/s"  # inherits grid from T
        ug.name = ug.attrs["name"]

    if which in ["both", "eta"]:
        # calculate derivatives of zeta
        dzetadeta = xroms.hgrad(zeta, grid, which="eta")

        # calculate geostrophic velocities
        vg = (
            g
            * dzetadeta
            / xroms.to_v(f, grid, hboundary=hboundary, hfill_value=hfill_value)
        )

        vg.attrs["name"] = "v_geo"
        vg.attrs["long_name"] = "geostrophic v velocity"
        vg.attrs["units"] = "m/s"  # inherits grid from T
        vg.name = vg.attrs["name"]

    if which == "both":
        return ug, vg
    elif which == "xi":
        return ug
    elif which == "eta":
        return vg
    else:
        print("nothing being returned from uv_geostrophic")


def EKE(ug, vg, grid, hboundary="extend", hfill_value=None):
    """Calculate EKE [m^2/s^2]

    Inputs
    ------
    ug: DataArray
        Geostrophic or other xi component velocity [m/s]
    vg: DataArray
        Geostrophic or other eta component velocity [m/s]
    grid: xgcm.grid
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

    Notes
    -----
    EKE = 0.5*(ug^2 + vg^2)

    Example usage
    -------------
    >>> ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, grid)
    >>> xroms.EKE(ug, vg, grid)
    """

    assert isinstance(ug, xr.DataArray), "ug must be DataArray"
    assert isinstance(vg, xr.DataArray), "vg must be DataArray"

    # make sure velocities are on rho grid
    ug = xroms.to_rho(ug, grid, hboundary=hboundary, hfill_value=hfill_value)
    vg = xroms.to_rho(vg, grid, hboundary=hboundary, hfill_value=hfill_value)

    var = 0.5 * (ug ** 2 + vg ** 2)

    var.attrs["name"] = "EKE"
    var.attrs["long_name"] = "eddy kinetic energy"
    var.attrs["units"] = "m^2/s^2"
    var.name = var.attrs["name"]

    return var


def dudz(u, grid, sboundary="extend", sfill_value=None):
    """Calculate the xi component of vertical shear [1/s]

    Inputs
    ------
    u: DataArray
        xi component of velocity [m/s]
    grid: xgcm.grid
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

    Notes
    -----
    u_z = ddz(u)
    Wrapper of `ddz`

    Example usage
    -------------
    >>> xroms.dudz(u, grid)
    """

    attrs = {
        "name": "dudz",
        "long_name": "u component of vertical shear",
        "units": "1/s",
        "grid": grid,
    }
    return xroms.ddz(u, grid, attrs=attrs, sboundary=sboundary, sfill_value=sfill_value)


def dvdz(v, grid, sboundary="extend", sfill_value=None):
    """Calculate the eta component of vertical shear [1/s]

    Inputs
    ------
    v: DataArray
        eta component of velocity [m/s]
    grid: xgcm.grid
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

    Notes
    -----
    v_z = ddz(v)
    Wrapper of `ddz`

    Example usage
    -------------
    >>> xroms.dvdz(v, grid)
    """

    attrs = {
        "name": "dvdz",
        "long_name": "v component of vertical shear",
        "units": "1/s",
        "grid": grid,
    }
    return xroms.ddz(v, grid, attrs=attrs, sboundary=sboundary, sfill_value=sfill_value)


def vertical_shear(dudz, dvdz, grid, hboundary="extend", hfill_value=None):
    """Calculate the vertical shear [1/s]

    Inputs
    ------
    dudz: DataArray
        xi component of vertical shear [1/s]
    dvdz: DataArray
        eta compoenent of vertical shear [1/s]
    grid: xgcm.grid
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

    Notes
    -----
    vertical_shear = np.sqrt(u_z^2 + v_z^2)

    Example usage
    -------------
    >>> xroms.vertical_shear(dudz, dvdz, grid)
    """

    assert isinstance(dudz, xr.DataArray), "dudz must be DataArray"
    assert isinstance(dvdz, xr.DataArray), "dvdz must be DataArray"

    # make sure velocities are on rho grid
    dudz = xroms.to_rho(dudz, grid, hboundary=hboundary, hfill_value=hfill_value)
    dvdz = xroms.to_rho(dvdz, grid, hboundary=hboundary, hfill_value=hfill_value)

    var = np.sqrt(dudz ** 2 + dvdz ** 2)

    var.attrs["name"] = "shear"
    var.attrs["long_name"] = "vertical shear"
    var.attrs["units"] = "1/s"
    var.attrs["grid"] = grid
    var.name = var.attrs["name"]

    return var


def relative_vorticity(
    u,
    v,
    grid,
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
):
    """Calculate the vertical component of the relative vorticity [1/s]

    Inputs
    ------
    u: DataArray
        xi component of velocity [m/s]
    v: DataArray
        eta component of velocity [m/s]
    grid: xgcm.grid
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

    Notes
    -----
    relative_vorticity = v_x - u_y

    Example usage
    -------------
    >>> xroms.relative_vorticity(u, v, grid)
    """

    assert isinstance(u, xr.DataArray), "u must be DataArray"
    assert isinstance(v, xr.DataArray), "v must be DataArray"

    dvdxi = xroms.hgrad(
        v,
        grid,
        which="xi",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    dudeta = xroms.hgrad(
        u,
        grid,
        which="eta",
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    var = dvdxi - dudeta

    var.attrs["name"] = "vort"
    var.attrs["long_name"] = "vertical component of vorticity"
    var.attrs["units"] = "1/s"  # inherits grid from T
    var.name = var.attrs["name"]

    return var


def ertel(
    phi,
    u,
    v,
    f,
    grid,
    hcoord="rho",
    scoord="s_rho",
    hboundary="extend",
    hfill_value=None,
    sboundary="extend",
    sfill_value=None,
):
    """Calculate Ertel potential vorticity of phi.

    Inputs
    ------
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
    grid: xgcm.grid
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

    Notes
    -----
    epv = -v_z * phi_x + u_z * phi_y + (f + v_x - u_y) * phi_z

    This is not set up to accept different boundary choices for different variables.

    Example usage:
    >>> xroms.ertel(ds.dye_01, ds.u, ds.v, ds.f, ds.attrs['grid'], scoord='s_w');
    """

    assert isinstance(phi, xr.DataArray), "phi must be DataArray"
    assert isinstance(u, xr.DataArray), "u must be DataArray"
    assert isinstance(v, xr.DataArray), "v must be DataArray"
    assert isinstance(f, xr.DataArray), "f must be DataArray"

    phi_xi, phi_eta = xroms.hgrad(
        phi,
        grid,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_xi = xroms.to_grid(
        phi_xi,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_eta = xroms.to_grid(
        phi_eta,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    phi_z = xroms.ddz(
        phi,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    # vertical shear (horizontal components of vorticity)
    u_z = xroms.ddz(
        u,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    v_z = xroms.ddz(
        v,
        grid,
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
        grid,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )
    vort = xroms.to_grid(
        vort,
        grid,
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
        "grid": grid,
    }
    epv = xroms.to_grid(
        epv,
        grid,
        hcoord=hcoord,
        scoord=scoord,
        attrs=attrs,
        hboundary=hboundary,
        hfill_value=hfill_value,
        sboundary=sboundary,
        sfill_value=sfill_value,
    )

    return epv
