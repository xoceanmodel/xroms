"""Density of seawater, stratification (N2, M2) and mixed layer depth.

Every function is pure: it computes from the arrays passed in and returns a new
object, never touching its inputs. Heights ``z`` are not looked up by coordinate
name; they are either passed in (``z=``) or computed on demand from ``grid``, a
Dataset holding ``h``, ``zeta`` and the s-coordinate parameters (see
:func:`xroms.z`).

Output lands where the calculation puts it: ``N2`` of a rho-level density is on
the ``s_w`` levels, ``M2`` is on rho points and the input's own levels, and
``mld`` is on rho points with no vertical dimension. DataArray results are
ordered (time, vertical, eta, xi, then any other dimensions), whatever the order
of the inputs' dimensions.
"""

import warnings

import numpy as np
import xarray as xr

from . import conventions
from ._align import _reject_legacy, require, select_like, with_grid_coords
from .conventions import canonicalize, vposition
from .interp import isoslice
from .utilities import _check_grid, ddeta, ddxi, ddz, order
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


@with_grid_coords
def density(temp, salt, z=None, *, grid=None, zeta=None, eos="roms", lon=None, lat=None):
    """Calculate the in-situ density [kg/m^3], with ROMS' equation of state or TEOS-10.

    Parameters
    ----------
    temp : DataArray, ndarray
        Potential temperature [Celsius], as ROMS carries it
    salt : DataArray, ndarray
        Practical salinity
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
    eos : {"roms", "teos10"}, optional
        Equation of state: ROMS' own (Nonlinear/rho_eos.F, the default), or
        TEOS-10 with gsw (``pip install 'xroms[teos10]'``), which converts
        practical to absolute salinity at each point's pressure and location and
        potential to conservative temperature first.
    lon, lat : float or DataArray, optional
        Location of the points, for ``eos="teos10"`` only. By default they are
        ``temp``'s lon/lat coordinates, else ``grid``'s; pass them for a
        Cartesian grid (constants will do).

    Returns
    -------
    DataArray or ndarray of calculated density, on the points of the inputs
    (rho/rho by default). A DataArray is ordered (time, vertical, eta, xi, then
    any other dimensions), whatever the order of the inputs' dimensions.

    Raises
    ------
    ValueError
        If neither ``z`` nor ``grid`` is given, or ``grid`` is given but ``temp``
        is not a DataArray (there is nothing to locate the points with).

    Notes
    -----
    ``eos="roms"`` is ROMS' equation of state (Nonlinear/rho_eos.F).
    ``eos="teos10"`` uses gsw's ``p_from_z``, ``SA_from_SP``, ``CT_from_pt`` and
    ``rho``.

    Examples
    --------
    >>> xroms.density(ds.temp, ds.salt, grid=ds)
    >>> xroms.density(ds.temp, ds.salt, z=xroms.z(ds))
    >>> xroms.density(ds.temp, ds.salt, grid=ds, eos="teos10")
    """
    _check_eos(eos)
    if grid is not None:
        grid = _check_grid(grid, "density")
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
        z = z_like(temp, grid, zeta=zeta)

    if eos == "teos10":
        var = _teos10(temp, salt, z, *_lonlat(temp, grid, lon, lat))
    else:
        var = _roms_eos(temp, salt, z)

    if isinstance(var, xr.DataArray):
        var = _with_cf_standard_names(order(var))
        _label(var, "rho", "density" if eos == "roms" else "density (TEOS-10)", "kg/m^3")

    return var


def _roms_eos(temp, salt, z):
    """ROMS' equation of state (Nonlinear/rho_eos.F): density [kg/m^3] at height ``z``."""
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
    return (den1 * bulk) / (bulk + 0.1 * z)


def _teos10(temp, salt, z, lon, lat, z_ref=None):
    """Density [kg/m^3] with TEOS-10 (gsw), from ROMS' potential temperature and practical salinity.

    TEOS-10 works in absolute salinity and conservative temperature, so the chain
    is SP -> SA (at each point's pressure and location) -> CT (from potential
    temperature) -> density, in situ, or at the pressure of height ``z_ref`` for
    potential density.
    """
    try:
        import gsw
    except ImportError as err:
        raise ImportError("eos='teos10' needs gsw: pip install 'xroms[teos10]', or conda install -c conda-forge gsw") from err

    def ufunc(func, *args):
        return xr.apply_ufunc(func, *args, dask="parallelized", output_dtypes=[np.float64])

    p = ufunc(gsw.p_from_z, z, lat)
    sa = ufunc(gsw.SA_from_SP, salt, p, lon, lat)
    ct = ufunc(gsw.CT_from_pt, sa, temp)
    if z_ref is not None:
        p = ufunc(gsw.p_from_z, z_ref, lat)
    return ufunc(gsw.rho, sa, ct, p)


def _lonlat(temp, grid, lon, lat):
    """Longitude and latitude at the points of ``temp``: given, its coords, or the grid's."""
    if lon is not None and lat is not None:
        return lon, lat
    if lon is not None or lat is not None:
        raise ValueError("pass both lon= and lat=, or neither")
    if isinstance(temp, xr.DataArray):
        pos = conventions.hposition(temp) or "rho"
        for source in (temp, grid):
            if source is None:
                continue
            xname, yname = conventions.horizontal_coords(source, pos)
            if xname is not None and xname.startswith("lon"):
                if source is temp:
                    return temp[xname].reset_coords(drop=True), temp[yname].reset_coords(drop=True)
                return tuple(select_like(grid[name], temp, name=name).reset_coords(drop=True) for name in (xname, yname))
    raise ValueError(
        "eos='teos10' needs longitude and latitude to convert practical to absolute salinity, and found "
        "none on the variable or the grid: pass lon= and lat= (constants will do for a Cartesian grid)."
    )


def _check_eos(eos):
    if eos not in ("roms", "teos10"):
        raise ValueError(f"eos must be 'roms' (ROMS' own equation of state) or 'teos10' (TEOS-10, with gsw), not {eos!r}")


@with_grid_coords
def potential_density(temp, salt, z=0, *, eos="roms", grid=None, zeta=None, z_points=None, lon=None, lat=None):
    """Calculate potential density [kg/m^3] with constant depth reference.

    Parameters
    ----------
    temp : DataArray, ndarray
        Potential temperature [Celsius], as ROMS carries it
    salt : DataArray, ndarray
        Practical salinity
    z : int, float, optional
        Reference height [m] (0 is the mean sea level; negative is below it).
    eos : {"roms", "teos10"}, optional
        Equation of state, as in :func:`xroms.density`.
    grid, zeta, z_points : optional
        For ``eos="teos10"`` only: practical salinity is converted to absolute
        salinity at each point's own pressure, so the heights of the points are
        needed, ``z_points`` (m, negative below the surface), or computed from
        ``grid`` (and ``zeta``) as in :func:`xroms.density`.
    lon, lat : float or DataArray, optional
        For ``eos="teos10"`` only, as in :func:`xroms.density`.

    Returns
    -------
    DataArray or ndarray of calculated potential density, on the points of the
    inputs (rho/rho by default). A DataArray is ordered (time, vertical, eta, xi,
    then any other dimensions), whatever the order of the inputs' dimensions.

    Notes
    -----
    ``eos="roms"`` is ROMS' equation of state (Nonlinear/rho_eos.F) evaluated at
    height ``z``. ``eos="teos10"`` is gsw's ``rho`` at the pressure of ``z``.
    Either way this is the full density, not the anomaly: subtract 1000 for
    sigma.

    Examples
    --------
    >>> xroms.potential_density(ds.temp, ds.salt)
    >>> xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
    """
    _check_eos(eos)
    if eos == "roms":
        var = density(temp, salt, z)
    else:
        if grid is not None:
            grid = _check_grid(grid, "potential_density")
        if isinstance(temp, xr.DataArray):
            temp = canonicalize(temp)
        if isinstance(salt, xr.DataArray):
            salt = canonicalize(salt)
        if z_points is None:
            if grid is None or not isinstance(temp, xr.DataArray):
                raise ValueError(
                    "eos='teos10' converts practical to absolute salinity at each point's pressure, so it "
                    "needs the heights of the points: pass grid= (the Dataset with h, zeta and the "
                    "s-coordinate parameters) or z_points= (m, negative below the surface)"
                )
            z_points = z_like(temp, grid, zeta=zeta)
        elif isinstance(z_points, xr.DataArray):
            z_points = canonicalize(z_points)
        var = _teos10(temp, salt, z_points, *_lonlat(temp, grid, lon, lat), z_ref=z)
        if isinstance(var, xr.DataArray):
            var = _with_cf_standard_names(order(var))

    if isinstance(var, xr.DataArray):
        _label(var, "sig0", "potential density" if eos == "roms" else "potential density (TEOS-10)", "kg/m^3")

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
    DataArray or ndarray of calculated buoyancy, on the points of ``sig0``. A
    DataArray is ordered (time, vertical, eta, xi, then any other dimensions).

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
        var = _label(order(var), "buoyancy", "buoyancy", "m/s^2")

    return var


@with_grid_coords
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

    # rho0 may be a DataArray with dimensions of its own
    return _label(order(var), "N2", "buoyancy frequency squared, or vertical buoyancy gradient", "1/s^2")


@with_grid_coords
def M2(
    rho,
    grid,
    rho0=None,
    *,
    z=None,
    zeta=None,
    hboundary="extend",
    hfill_value=np.nan,
    sboundary="extend",
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
        constant-depth gradient: "extend" (default) uses one-sided second-order
        differences at the top and bottom levels, so every level has a value;
        "fill" sets those two edge levels of the derivative to ``sfill_value``,
        which leaves the whole surface and bottom layers of M2 NaN (M2 stays on
        the levels of ``rho``).
    sfill_value : float, optional
        Value used at the vertical edges with ``sboundary="fill"``.
    along_s : bool, optional
        For a single selected s-level (or a one-level vertical dim): accept
        derivatives along the s-surface instead of at constant depth (see
        :func:`xroms.ddxi`).

    Returns
    -------
    DataArray of the horizontal buoyancy gradient, on rho points and on the
    vertical levels of ``rho``. With ``sboundary="fill"`` the top and bottom
    levels are NaN.

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

    # rho0 may be a DataArray with dimensions of its own
    return _label(order(var), "M2", "horizontal buoyancy gradient", "1/s^2")


#: criterion variable of mld: its default threshold and the CF standard_name of the result
_MLD_VARIABLES = {
    "density": (0.03, "ocean_mixed_layer_thickness_defined_by_sigma_theta"),
    "temperature": (0.2, "ocean_mixed_layer_thickness_defined_by_temperature"),
}


def _threshold_alias(threshold, thresh, stacklevel=3):
    """``threshold``, accepting v0.6's name ``thresh`` with a FutureWarning."""
    if thresh is None:
        return threshold
    if threshold is not None:
        raise TypeError("pass threshold= only (thresh= is its old name)")
    warnings.warn("mld's thresh= is now threshold=", FutureWarning, stacklevel=stacklevel)
    return thresh


@with_grid_coords
def mld(
    var, grid=None, *args, z=None, zeta=None, threshold=None, reference_depth=0.0,
    variable="density", fill="bottom", method="interp", dim=None, thresh=None,
):
    """Calculate the mixed layer depth [m, positive], by a threshold criterion.

    The base of the mixed layer is the shallowest depth below ``reference_depth``
    at which ``var`` departs from its value at ``reference_depth`` by more than
    ``threshold`` (de Boyer Montégut et al., 2004), interpolated linearly between
    levels.

    Parameters
    ----------
    var : DataArray
        Potential density [kg/m^3] (``variable="density"``, e.g. from
        :func:`xroms.potential_density`) or temperature [C]
        (``variable="temperature"``), with a vertical dimension.
    grid : Dataset, optional
        Dataset with ``h``, ``zeta`` and the s-coordinate parameters, and
        ``mask_rho`` if the grid has land. Used to compute the heights of
        ``var``'s points when ``z`` is not given, and for ``fill="bottom"``.
    z : DataArray, optional
        Heights [m, negative below the mean sea level] of ``var``'s points,
        instead of computing them from ``grid``. Depths are ``-z``.
    zeta : None, float, "mean" or DataArray, optional
        Free surface used when computing the heights from ``grid`` (see
        :func:`xroms.z`).
    threshold : float, optional
        Departure from the reference value that marks the base of the mixed
        layer: an increase of density (default 0.03 kg/m^3), or a change of
        temperature either way (default 0.2 C).
    reference_depth : float, optional
        Depth [m, positive] of the reference value, interpolated linearly
        between levels. The default, 0, takes the shallowest level, as xroms
        always has; de Boyer Montégut et al. (2004), ocean-skill and roms-tools
        use 10. A reference depth above the shallowest point of a profile takes
        that point, and one below the deepest takes the deepest.
    variable : {"density", "temperature"}, optional
        What ``var`` is, which sets the criterion and the default ``threshold``.
    fill : {"bottom", "nan"}, optional
        What a water column without a crossing gets: the depth of the bottom,
        or NaN. The bottom is ``h`` from ``grid`` over water (``mask_rho == 1``,
        or where the profile has data if ``grid`` has no ``mask_rho``) or,
        without ``grid`` or its ``h``, the depth of the deepest point with data.
        Columns without data (land) are NaN either way.
    method : {"interp", "transform"}, optional
        ``"interp"`` searches each profile for the shallowest level past the
        threshold and interpolates between it and the level above; profiles
        need not be monotonic and may have missing values. ``"transform"``
        interpolates the depth onto the threshold with xgcm's ``transform``, as
        :func:`xroms.isoslice` does (and as xroms did before 1.0, and roms-tools
        does): each profile must be monotonic (temperature decreasing with
        depth), without missing values. The two agree on monotonic profiles.
    dim : str, optional
        Vertical dimension of ``var``: its s-coordinate dimension by default.
        Name it for profiles on other levels (``z`` is then required and is
        used as given, e.g. ``z=-woa.depth``).

    Returns
    -------
    DataArray of mixed layer depth [m, positive] on the points of ``var``,
    without the vertical dimension. Its attrs record the criterion
    (``standard_name``, ``mld_variable``, ``mld_threshold``,
    ``mld_reference_depth``).

    Notes
    -----
    Depths are ``-z``, measured from the mean sea level like ``h`` (pass ``z``
    from :func:`xroms.z` with ``reference="surface", positive="up"`` to measure
    from the moving surface instead, and ``fill="nan"`` or no ``grid``, since
    ``h`` is measured from the mean sea level).

    Before xroms 1.0 the call was ``mld(sig0, xgrid, h, mask, thresh=0.03)``;
    ``h`` and the mask are read from ``grid`` now, and passing them (or
    anything else positionally after ``grid``) raises a `TypeError`. ``thresh``
    still works, with a FutureWarning; it is now ``threshold``.

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
    >>> xroms.mld(ds.temp, ds, variable="temperature", reference_depth=10)
    """
    _reject_legacy(
        args,
        "mld",
        "h and mask now come from grid (the Dataset that holds them): use xroms.mld(sig0, ds). "
        "threshold, z and zeta are keyword arguments.",
    )
    threshold = _threshold_alias(threshold, thresh)
    if not isinstance(var, xr.DataArray):
        raise TypeError("var must be a DataArray")
    if variable not in _MLD_VARIABLES:
        raise ValueError(f"variable must be 'density' or 'temperature', not {variable!r}")
    if fill not in ("bottom", "nan"):
        raise ValueError(f"fill must be 'bottom' or 'nan', not {fill!r}")
    if method not in ("interp", "transform"):
        raise ValueError(f"method must be 'interp' or 'transform', not {method!r}")
    default, standard_name = _MLD_VARIABLES[variable]
    threshold = default if threshold is None else float(threshold)
    reference_depth = float(reference_depth)
    if grid is not None:
        grid = _check_grid(grid, "mld")
    var = canonicalize(var)
    vdim = dim or vposition(var)
    if vdim is None or vdim not in var.dims:
        raise ValueError(
            f"{var.name!r} has no vertical dimension{'' if dim is None else ' ' + repr(dim)}; mld needs profiles"
            + (" (name it with dim= if it is not an s-coordinate)" if dim is None else "")
        )
    if vdim in ("s_rho", "s_w"):
        if z is None and grid is None:
            raise ValueError("mld needs the heights of var's points: pass grid= (a Dataset with h, zeta and the s-coordinate parameters) or z=")
        zz = z_like(var, grid, zeta=zeta, z=z)
    else:
        if not isinstance(z, xr.DataArray) or vdim not in z.dims:
            raise ValueError(f"for levels other than s_rho/s_w, pass z= (heights, negative below the surface) along {vdim!r}")
        zz = z
    depth = -zz.reset_coords(drop=True)

    if method == "interp":
        out = _mld_interp(var, depth, vdim, threshold, reference_depth, signed=variable == "density")
    else:
        out = _mld_transform(var, depth, vdim, threshold, reference_depth, signed=variable == "density")

    has_data = (var.notnull() & depth.notnull()).any(vdim)
    if fill == "bottom":
        if grid is not None and "h" in grid.variables:
            bottom = select_like(grid["h"], var, name="h").reset_coords(drop=True)
            if "mask_rho" in grid.variables:
                water = select_like(grid["mask_rho"], var, name="mask_rho").reset_coords(drop=True) == 1
            else:
                water = has_data
        else:
            bottom, water = _nanmax(depth.where(var.notnull()), vdim), has_data
        out = out.fillna(bottom.where(water))
    out = order(out)
    _label(out, "mld", "mixed layer depth", "m")
    out.attrs.update(
        standard_name=standard_name, mld_variable=variable, mld_threshold=threshold, mld_reference_depth=reference_depth
    )
    return out


def _nanmax(x, dim):
    """``x.max(dim)`` skipping NaN, and NaN where all of it is (land), without dask's all-NaN warning."""
    return x.fillna(-np.inf).max(dim).where(x.notnull().any(dim))


def _nanmin(x, dim):
    """``x.min(dim)`` skipping NaN; see :func:`_nanmax`."""
    return x.fillna(np.inf).min(dim).where(x.notnull().any(dim))


def _at(values, depth, where_depth, dim):
    """``values`` at the level of each profile whose depth is ``where_depth`` (NaN if none)."""
    return _nanmax(values.where(depth == where_depth), dim)


def _mld_interp(var, depth, dim, threshold, reference_depth, signed):
    """Shallowest crossing below the reference depth, found level by level (ocean-skill's method).

    Order-free: levels are found by depth with reductions over ``dim``, so
    profiles may run either way, have gaps and need not be monotonic.
    """
    depth = depth.where(var.notnull())
    # reference value: linear in depth between the points either side of reference_depth
    above = _nanmax(depth.where(depth <= reference_depth), dim)
    below = _nanmin(depth.where(depth >= reference_depth), dim)
    v_above, v_below = _at(var, depth, above, dim), _at(var, depth, below, dim)
    span = (below - above).where(below != above)
    ref = xr.where(
        above.isnull(),
        v_below,
        xr.where(below.isnull() | (below == above), v_above, v_above + (reference_depth - above) / span * (v_below - v_above)),
    )
    diff = var - ref
    past = (diff if signed else abs(diff)) > threshold
    d1 = _nanmin(depth.where(past & (depth >= reference_depth)), dim)
    d0 = _nanmax(depth.where(depth < d1), dim)
    v1, v0 = _at(diff, depth, d1, dim), _at(diff, depth, d0, dim)
    target = threshold if signed else xr.where(v1 < 0, -threshold, threshold)
    step = (v1 - v0).where(v1 != v0)
    crossing = d0 + (target - v0) / step * (d1 - d0)
    return crossing.where(d0.notnull() & step.notnull(), d1)


def _mld_transform(var, depth, dim, threshold, reference_depth, signed):
    """Crossing found with xgcm's transform (monotonic profiles), as xroms did before 1.0."""
    depth = depth.broadcast_like(var)
    ref = isoslice(var, [reference_depth], depth, dim=dim, new_dim="iso", mask_edges=False).squeeze("iso", drop=True)
    excess = var - ref if signed else ref - var
    crossing = isoslice(depth, [threshold], excess, dim=dim, new_dim="iso")
    return crossing.squeeze("iso", drop=True)
