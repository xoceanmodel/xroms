"""The ``ds.xroms`` and ``da.xroms`` accessors: thin, stateless conveniences.

``ds.xroms`` holds a reference to the Dataset and nothing else: no cache, no
copies, no writes into the Dataset. Every property and method recomputes from
the Dataset's current contents (lazily under dask), so results always match the
data, even after in-place edits or subsetting. Results come back in the
Dataset's own dim naming (Rutgers ``eta_u`` stays ``eta_u``) with the Dataset's
coordinates for their grid position attached. Assign a result to a variable if
you reuse it; use ``.persist()`` for expensive reuse.

A ``grid=`` Dataset (a grid file kept apart from the output) is completed from
the Dataset before use: whatever the grid lacks of the Dataset's own global
attributes, s-coordinate parameters and ``zeta`` is taken from the Dataset, so
results match those of the merged Dataset.

``da.xroms`` offers only operations that need nothing beyond the DataArray.
"""

import numpy as np
import xarray as xr

from . import _xgcm, derived, interp, metrics, roms_seawater, utilities, vector, vertical
from ._align import GridMismatchError
from .conventions import (
    CANONICAL,
    HCOORDS,
    RUTGERS,
    canonicalize,
    convention,
    free_surface_name,
    horizontal_coords,
    hposition,
    rename_like,
    rho0,
    sgrid_topology,
    vertical_params,
)


def _removed(name, hint):
    def method(self, *args, **kwargs):
        raise AttributeError(f"xroms 1.0 removed {name}. {hint}")

    method.__doc__ = f"Removed in xroms 1.0. {hint}"
    return method


def _removed_property(name, hint):
    def getter(self):
        raise AttributeError(f"xroms 1.0 removed {name}. {hint}")

    return property(getter, doc=f"Removed in xroms 1.0. {hint}")


#: variables ``conventions.vertical_params`` reads by name (it also reads global attributes)
_VERTICAL_VARS = (
    "s_rho", "s_w", "sigma_r", "sigma_w", "sc_r", "sc_w", "Cs_r", "Cs_w",
    "hc", "theta_s", "theta_b", "Vtransform", "Vstretching", "Tcline",
)

#: keyword arguments of v0.6.2 that no longer exist, with what to do instead
_REMOVED_KWARGS = {
    "include_vars_adcp": (
        "Build the set you need from ds.xroms.east, ds.xroms.north, ds.xroms.east_rotated(angle) and "
        "ds.xroms.north_rotated(angle), e.g. ds[['angle']].assign(east=..., north=..., eastrot=..., northrot=...)."
    ),
}


def _reject_kwargs(kwargs, func):
    """Raise ``TypeError`` for unexpected keywords, with a hint for those xroms 1.0 removed."""
    for key in kwargs:
        hint = _REMOVED_KWARGS.get(key)
        detail = f" xroms 1.0 removed it. {hint}" if hint else ""
        raise TypeError(f"{func}() got an unexpected keyword argument {key!r}.{detail}")


def _keyword_only(args, func):
    """Guardrail for v0.6.2 calls that passed ``hcoord`` and ``scoord`` positionally."""
    if args:
        raise TypeError(
            f"xroms 1.0: ds.xroms.{func} takes everything after var (hcoord, scoord, grid, ...) as keyword-only "
            f"arguments; v0.6.2 accepted hcoord and scoord positionally. Write ds.xroms.{func}(var, hcoord='rho', scoord='s_rho')."
        )


def _is_xgcm_grid(obj):
    return type(obj).__module__.startswith("xgcm")


def _check_footprint(obj, grid, what):
    """Raise unless ``obj`` covers the same rho points as ``grid`` (sizes and, if both have them, labels)."""
    for dim in ("eta_rho", "xi_rho"):
        if dim not in obj.dims or dim not in grid.dims:
            continue
        labels_differ = dim in obj.indexes and dim in grid.indexes and not obj.indexes[dim].equals(grid.indexes[dim])
        if obj.sizes[dim] != grid.sizes[dim] or labels_differ:
            raise GridMismatchError(
                f"{what} and grid= cover different {dim} points ({obj.sizes[dim]} vs {grid.sizes[dim]}), so "
                "they cannot be used together. Subset the Dataset and the grid the same way "
                "(e.g. xroms.subset on both), or merge them: xroms.merge_grid(ds, grid)."
            )


def _effective_grid(ds, grid):
    """``grid`` plus what it lacks of ``ds``: global attrs, s-coordinate parameters and ``zeta``.

    A grid kept apart from the output (UCLA ROMS, roms-tools, a Rutgers grid file)
    has ``h``, ``pm`` and ``pn`` but neither the free surface nor, often, the
    vertical parameters, which live with the output. Wherever both have something,
    ``grid`` wins. ``zeta`` joins only if it covers the grid's rho points. Only
    metadata is touched (lazy under dask) and neither input is modified.
    """
    out = grid.assign_attrs({**ds.attrs, **grid.attrs})
    names = [n for n in _VERTICAL_VARS if n in ds.variables and n not in out.variables]
    topology = sgrid_topology(ds)
    if topology is not None and topology["variable"] not in out.variables:
        names.append(topology["variable"])  # vertical_params reads it to settle Vtransform
    out = out.assign_coords({n: ds.variables[n] for n in names if n in ds.dims})
    out = out.assign({n: ds.variables[n] for n in names if n not in ds.dims})
    for dim in ("s_rho", "s_w"):
        if dim in ds.dims and dim not in out.dims:
            # no variable carries the dim (UCLA output keeps Cs_r etc. in attributes), and vertical_params needs its size
            out = out.assign({f"_{dim}_levels": xr.Variable(dim, np.zeros(ds.sizes[dim]))})
    zeta = free_surface_name(ds)
    if zeta is not None and free_surface_name(out) is None:
        _check_footprint(ds[zeta], out, "this Dataset's zeta")
        out = out.assign({zeta: ds[zeta].reset_coords(drop=True)})
    return out


def _with_w_levels(ds):
    """``ds`` with an ``s_w`` dim (integer labels) if only ``s_rho`` exists, as in UCLA output."""
    if "s_rho" in ds.dims and "s_w" not in ds.dims:
        return ds.assign_coords(s_w=np.arange(ds.sizes["s_rho"] + 1))
    return ds


def _grid_metrics(grid, vertical_metrics, zeta):
    """xgcm metrics by axes: dx, dy and dA (when ``pm``/``pn`` exist), dz only if asked."""
    out = {}
    if "pm" in grid.variables and "pn" in grid.variables:
        out[("X",)] = [metrics.dx(grid, h) for h in HCOORDS]
        out[("Y",)] = [metrics.dy(grid, h) for h in HCOORDS]
        out[("X", "Y")] = [metrics.dA(grid, h) for h in HCOORDS]
    if vertical_metrics:
        out[("Z",)] = [vertical.dz(grid, hcoord=h, scoord=s, zeta=zeta) for s in ("s_rho", "s_w") for h in HCOORDS]
    return out


def _alias_like(out, like):
    """``out`` in Rutgers alias dims if ``like`` uses them, else as computed (canonical)."""
    pos = hposition(out)
    if convention(like) != "rutgers" or pos in (None, "rho"):
        return out
    rename = {c: r for c, r in zip(CANONICAL[pos], RUTGERS[pos], strict=True) if c != r and c in out.dims}
    return out.rename(rename) if rename else out


@xr.register_dataset_accessor("xroms")
class xromsDatasetAccessor:
    """ROMS-aware calculations on a Dataset (grid variables included or passed)."""

    def __init__(self, ds):
        self._obj = ds

    # --- plumbing -------------------------------------------------------------------

    def _grid(self, grid=None):
        """Where grid variables are read from: this Dataset, or ``grid`` completed from it."""
        if grid is None:
            return self._obj
        if not isinstance(grid, xr.Dataset):
            return grid  # the function it reaches raises the guided error (xgcm Grid, wrong type)
        return _effective_grid(self._obj, grid)

    def _var(self, var):
        if isinstance(var, str):
            if var not in self._obj.variables:
                raise KeyError(f"{var!r} is not a variable of this Dataset")
            return self._obj[var]
        if isinstance(var, xr.DataArray):
            return var
        raise TypeError("pass a variable name or a DataArray")

    def _out(self, da):
        """Return ``da`` in the Dataset's naming, with its matching coords attached."""
        ds = self._obj
        out = rename_like(da, ds)
        coords = {}
        for name, coord in ds.coords.items():
            if name in out.coords or not coord.dims:
                continue
            if set(coord.dims) <= set(out.dims) and all(ds.sizes[d] == out.sizes[d] for d in coord.dims):
                coords[name] = coord
        pos = hposition(canonicalize(da))
        if pos is not None:
            for name in horizontal_coords(ds, pos):
                # coordinates only: lon/lat kept as data variables (UCLA) would not merge back into ds
                if name is None or name in out.coords or name in coords or name not in ds.coords:
                    continue
                var = ds[name]
                if set(var.dims) <= set(out.dims) and all(ds.sizes[d] == out.sizes[d] for d in var.dims):
                    coords[name] = var.reset_coords(drop=True).variable
        return out.assign_coords(coords) if coords else out

    def find_horizontal_velocities(self):
        """Names of the horizontal velocity pair present: grid-aligned or eastward/northward."""
        for pair in (("u", "v"), ("u_eastward", "v_northward"), ("east", "north")):
            if all(name in self._obj.variables for name in pair):
                return pair
        raise KeyError("cannot identify horizontal velocity variable names")

    # --- grid facts, on demand --------------------------------------------------------

    @property
    def vertical_params(self):
        """The s-coordinate parameters (Vtransform, hc, Cs, sigma) of this Dataset."""
        return vertical_params(self._obj)

    def z(self, hcoord="rho", scoord="s_rho", *, zeta=None, reference="mean_sea_level", positive="up", method="average", grid=None):
        """Vertical position at (``hcoord``, ``scoord``); see :func:`xroms.z`."""
        return self._out(vertical.z(self._grid(grid), hcoord=hcoord, scoord=scoord, zeta=zeta, reference=reference, positive=positive, method=method))

    @property
    def z_rho(self):
        """z (m, positive up, relative to mean sea level) at rho points and levels."""
        return self.z()

    @property
    def z_w(self):
        """z at rho points on w levels (layer interfaces)."""
        return self.z(scoord="s_w")

    def dz(self, hcoord="rho", scoord="s_rho", *, zeta=None, grid=None):
        """Layer thickness (m); see :func:`xroms.dz`."""
        return self._out(vertical.dz(self._grid(grid), hcoord=hcoord, scoord=scoord, zeta=zeta))

    def dx(self, hcoord="rho", *, grid=None):
        """Grid spacing along xi (m) at ``hcoord``."""
        return self._out(metrics.dx(self._grid(grid), hcoord))

    def dy(self, hcoord="rho", *, grid=None):
        """Grid spacing along eta (m) at ``hcoord``."""
        return self._out(metrics.dy(self._grid(grid), hcoord))

    def dA(self, hcoord="rho", *, grid=None):
        """Cell area (m²) at ``hcoord``."""
        return self._out(metrics.dA(self._grid(grid), hcoord))

    def dV(self, hcoord="rho", scoord="s_rho", *, zeta=None, grid=None):
        """Cell volume (m³) at (``hcoord``, ``scoord``)."""
        return self._out(metrics.dV(self._grid(grid), hcoord, scoord, zeta=zeta))

    def assign_z(self, *, zeta=None, hcoord="rho", grid=None):
        """A **new** Dataset with lazy ``z_rho``/``z_w`` coordinates attached.

        This is an explicit snapshot: recompute after changing ``zeta`` or ``h``.
        On numpy-backed data the depths are computed immediately. ``h`` comes from
        ``grid`` if given (completed from this Dataset, as in :meth:`z`).
        """
        z_rho = self.z(hcoord, "s_rho", zeta=zeta, grid=grid)
        z_w = self.z(hcoord, "s_w", zeta=zeta, grid=grid)
        return self._obj.assign_coords({z_rho.name: z_rho.variable, z_w.name: z_w.variable})

    def xgcm_grid(self, padding="extend", *, vertical_metrics=False, zeta=None, grid=None):
        """A fresh xgcm Grid for this Dataset's canonical dims (``xroms.canonicalize(ds)``).

        The grid has X, Y and Z axes (UCLA output, which has no ``s_w`` dim, gets a
        synthesized one) and the horizontal metrics ``dx`` (key ``("X",)``), ``dy``
        (``("Y",)``) and the cell area ``dA`` (``("X", "Y")``) at rho, u, v and psi
        points, from :func:`xroms.dx`, :func:`xroms.dy` and :func:`xroms.dA`. So
        ``g.derivative(da, "X")`` and ``g.average(da, ["X", "Y"])`` work. The metrics
        need ``pm`` and ``pn`` (here or in ``grid``); without them the grid has none.

        Parameters
        ----------
        padding : str, optional
            xgcm padding for all axes (``"extend"``, ``"fill"``, ...).
        vertical_metrics : bool, optional
            Also attach the layer thickness ``dz`` (key ``("Z",)``) at s_rho and s_w
            for each horizontal position, for ``g.integrate(da, "Z")`` and vertical
            averages. They are 4-D (they follow ``zeta`` in time), so they match only
            arrays with the same dims: subset the Dataset first, then build the grid.
            They are computed now, eagerly for numpy-backed data and lazily under dask.
        zeta : None, float, "mean" or DataArray, optional
            Free surface for those thicknesses, as in :meth:`z`.
        grid : Dataset, optional
            Grid kept apart from this Dataset, completed from it as in :meth:`z`.

        Examples
        --------
        Arrays passed to the grid use the canonical dims:

        >>> g = ds.xroms.xgcm_grid(vertical_metrics=True)
        >>> c = xroms.canonicalize(ds)
        >>> g.average(c.temp, ["X", "Y"])
        >>> g.integrate(c.temp, "Z")
        """
        ds = canonicalize(self._obj)
        if grid is not None:
            if not isinstance(grid, xr.Dataset):
                raise TypeError(f"grid must be an xarray Dataset holding pm and pn, not {type(grid).__name__}")
            _check_footprint(ds, canonicalize(grid), "this Dataset")
        arrays = _grid_metrics(self._grid(grid), vertical_metrics, zeta)
        return _xgcm.grid_for(_with_w_levels(ds), padding=padding, metrics=arrays)

    set_grid = _removed("ds.xroms.set_grid", "xroms no longer stores an xgcm grid; just call the methods, or use ds.xroms.xgcm_grid() for your own xgcm work.")
    xgrid = _removed_property("ds.xroms.xgrid", "Use ds.xroms.xgcm_grid() for a fresh xgcm Grid.")
    w = _removed_property("ds.xroms.w", "In 0.6.2 it was an unfinished placeholder that returned nothing. If your output has a 'w' variable, read ds['w'] directly.")
    omega = _removed_property("ds.xroms.omega", "In 0.6.2 it was an unfinished placeholder that returned nothing. If your output has an 'omega' variable, read ds['omega'] directly.")

    # --- grid moves and calculus -------------------------------------------------------

    def to_grid(self, var, hcoord=None, scoord=None, **kwargs):
        """Move ``var`` (name or DataArray) to ``hcoord``/``scoord``."""
        return self._out(utilities.to_grid(self._var(var), hcoord, scoord, **kwargs))

    def ddxi(self, var, *args, grid=None, **kwargs):
        """d/dxi at constant depth; see :func:`xroms.ddxi`."""
        _keyword_only(args, "ddxi")
        return self._out(utilities.ddxi(self._var(var), self._grid(grid), **kwargs))

    def ddeta(self, var, *args, grid=None, **kwargs):
        """d/deta at constant depth; see :func:`xroms.ddeta`."""
        _keyword_only(args, "ddeta")
        return self._out(utilities.ddeta(self._var(var), self._grid(grid), **kwargs))

    def ddz(self, var, *args, grid=None, **kwargs):
        """d/dz; see :func:`xroms.ddz`."""
        _keyword_only(args, "ddz")
        return self._out(utilities.ddz(self._var(var), self._grid(grid), **kwargs))

    def hgrad(self, var, *, grid=None, **kwargs):
        """Both horizontal derivatives at constant depth."""
        return self.ddxi(var, grid=grid, **kwargs), self.ddeta(var, grid=grid, **kwargs)

    def _out_reduced(self, out, var):
        """``_out`` for a reduction of ``var``: the dims it kept are named as ``var`` names them.

        With one horizontal dim gone, the result's position can't be told from its dims (a v
        variable's ``xi_v`` and a rho variable's ``xi_rho`` are both canonical ``xi_rho``).
        """
        own = dict(zip(canonicalize(var).dims, var.dims))
        return self._out(out.rename({c: o for c, o in own.items() if c != o and c in out.dims}))

    def gridsum(self, var, dims, *, grid=None, **kwargs):
        """Grid-weighted sum over ``dims``; see :func:`xroms.gridsum`."""
        var = self._var(var)
        return self._out_reduced(utilities.gridsum(var, self._grid(grid), dims, **kwargs), var)

    def gridmean(self, var, dims, *, grid=None, **kwargs):
        """Grid-weighted mean over ``dims``; see :func:`xroms.gridmean`."""
        var = self._var(var)
        return self._out_reduced(utilities.gridmean(var, self._grid(grid), dims, **kwargs), var)

    def depth_average(self, var, *, grid=None, **kwargs):
        """Thickness-weighted vertical mean; see :func:`xroms.depth_average`."""
        return self._out(vertical.depth_average(self._var(var), self._grid(grid), **kwargs))

    def surface(self, var):
        """Top layer of ``var``."""
        return self._out(vertical.surface(self._var(var)))

    def bottom(self, var):
        """Bottom layer of ``var``."""
        return self._out(vertical.bottom(self._var(var)))

    def zslice(self, var, depths, *, grid=None, **kwargs):
        """Interpolate ``var`` to fixed depths; see :func:`xroms.zslice`."""
        return self._out(interp.zslice(self._var(var), depths, self._grid(grid), **kwargs))

    def isoslice(self, var, iso_values, iso_array, **kwargs):
        """Interpolate ``var`` onto values of ``iso_array`` (names or DataArrays)."""
        return self._out(interp.isoslice(self._var(var), iso_values, self._var(iso_array), **kwargs))

    def subset(self, X=None, Y=None, *, halo=0):
        """Horizontal subset keeping staggers consistent; see :func:`xroms.subset`."""
        return utilities.subset(self._obj, X=X, Y=Y, halo=halo)

    def argsel2d(self, lon0, lat0, hcoord="rho", **kwargs):
        """Indices of the ``hcoord`` point nearest to ``(lon0, lat0)`` (or x/y)."""
        xname, yname = horizontal_coords(self._obj, hcoord)
        if xname is None:
            raise KeyError(f"no lon/lat or x/y coordinates at {hcoord} points")
        if xname.startswith("x_"):
            kwargs.setdefault("method", "cartesian")
        return utilities.argsel2d(self._obj[xname], self._obj[yname], lon0, lat0, **kwargs)

    def sel2d(self, var, lon0, lat0, **kwargs):
        """``var`` at the grid point nearest to ``(lon0, lat0)``."""
        da = self._var(var)
        pos = hposition(canonicalize(da)) or "rho"
        xname, yname = horizontal_coords(self._obj, pos)
        if xname is None:
            raise KeyError(f"no lon/lat or x/y coordinates at {pos} points")
        if xname.startswith("x_"):
            kwargs.setdefault("method", "cartesian")
        return utilities.sel2d(da, self._obj[xname], self._obj[yname], lon0, lat0, **kwargs)

    # --- velocities ------------------------------------------------------------------------

    def _uv(self):
        ds = self._obj
        if "u" in ds.variables and "v" in ds.variables:
            return ds["u"], ds["v"]
        return self.u, self.v

    @property
    def east(self):
        """Eastward velocity on rho points (the file's own ``u_eastward`` if present)."""
        if "u_eastward" in self._obj.variables:
            return self._obj["u_eastward"]
        return self.eastnorth[0]

    @property
    def north(self):
        """Northward velocity on rho points (the file's own ``v_northward`` if present)."""
        if "v_northward" in self._obj.variables:
            return self._obj["v_northward"]
        return self.eastnorth[1]

    @property
    def eastnorth(self):
        """``(east, north)`` velocities on rho points."""
        ds = self._obj
        if "u_eastward" in ds.variables and "v_northward" in ds.variables:
            return ds["u_eastward"], ds["v_northward"]
        east, north = vector.grid_to_earth(ds["u"], ds["v"], ds["angle"])
        return self._out(east), self._out(north)

    def _grid_uv_from_earth(self):
        ds = self._obj
        east = next((name for name in ("u_eastward", "east") if name in ds.variables), None)
        north = next((name for name in ("v_northward", "north") if name in ds.variables), None)
        if east is None or north is None:
            missing = " or ".join(repr(name) for name in ("u", "v") if name not in ds.variables)
            raise KeyError(
                f"this Dataset has no velocity {missing}, nor both eastward and northward velocities "
                "(u_eastward and v_northward, or east and north) to rotate onto the grid"
            )
        u, v = vector.earth_to_grid(ds[east], ds[north], ds["angle"], hcoord="native")
        return self._out(u), self._out(v)

    @property
    def u(self):
        """Grid-aligned u (the Dataset's own, or rotated from east/north onto u points)."""
        if "u" in self._obj.variables:
            return self._obj["u"]
        return self._grid_uv_from_earth()[0]

    @property
    def v(self):
        """Grid-aligned v (the Dataset's own, or rotated from east/north onto v points)."""
        if "v" in self._obj.variables:
            return self._obj["v"]
        return self._grid_uv_from_earth()[1]

    def east_rotated(self, angle, *, reference="xaxis", isradians=True, name=None, **removed):
        """x component of (east, north) rotated by ``angle`` (e.g. along-channel)."""
        _reject_kwargs(removed, "east_rotated")
        return self._rotated(angle, reference, isradians, name, 0)

    def north_rotated(self, angle, *, reference="xaxis", isradians=True, name=None, **removed):
        """y component of (east, north) rotated by ``angle`` (e.g. across-channel)."""
        _reject_kwargs(removed, "north_rotated")
        return self._rotated(angle, reference, isradians, name, 1)

    def _rotated(self, angle, reference, isradians, name, which):
        east, north = self.eastnorth
        attrs = {
            "x": {"name": "eastrot", "standard_name": "sea_water_x_velocity", "long_name": "eastward velocity rotated by angle", "units": "m/s"},
            "y": {"name": "northrot", "standard_name": "sea_water_y_velocity", "long_name": "northward velocity rotated by angle", "units": "m/s"},
        }
        out = vector.rotate_vectors(east, north, angle, isradians=isradians, reference=reference, attrs=attrs)[which]
        if isinstance(angle, (int, float)):
            out.attrs["long_name"] += f" {angle}"
        if name is not None:
            out = out.rename(name)
            out.attrs["name"] = name
        return self._out(out)

    # --- derived physics ---------------------------------------------------------------------

    @property
    def speed(self):
        """Horizontal speed (m/s) on rho points."""
        ds = self._obj
        if "u" not in ds.variables and "u_eastward" in ds.variables:
            out = np.sqrt(ds["u_eastward"] ** 2 + ds["v_northward"] ** 2)
            out.attrs = {"name": "speed", "long_name": "horizontal speed", "units": "m/s"}
            return self._out(out.rename("speed"))
        u, v = self._uv()
        return self._out(derived.speed(u, v))

    @property
    def KE(self):
        """Kinetic energy (kg/(m s²)) on rho points, using the Dataset's rho0."""
        return self._out(derived.KE(rho0(self._obj), self.speed))

    @property
    def ug(self):
        """Geostrophic u (m/s) from zeta, on u points."""
        return self._out(derived.uv_geostrophic(self._obj[free_surface_name(self._obj) or "zeta"], self._obj["f"], self._obj, which="xi"))

    @property
    def vg(self):
        """Geostrophic v (m/s) from zeta, on v points."""
        return self._out(derived.uv_geostrophic(self._obj[free_surface_name(self._obj) or "zeta"], self._obj["f"], self._obj, which="eta"))

    @property
    def EKE(self):
        """Eddy kinetic energy of the geostrophic velocities (m²/s²) on rho points."""
        return self._out(derived.EKE(self.ug, self.vg))

    @property
    def dudz(self):
        """du/dz (1/s) on u points, w levels."""
        return self._out(derived.dudz(self.u, self._obj))

    @property
    def dvdz(self):
        """dv/dz (1/s) on v points, w levels."""
        return self._out(derived.dvdz(self.v, self._obj))

    @property
    def vertical_shear(self):
        """Magnitude of the vertical shear (1/s) on rho points, w levels."""
        return self._out(derived.vertical_shear(self.dudz, self.dvdz))

    @property
    def vort(self):
        """Vertical relative vorticity (1/s) on psi points."""
        u, v = self._uv()
        return self._out(derived.relative_vorticity(u, v, self._obj))

    @property
    def convergence(self):
        """Horizontal convergence du/dx + dv/dy (1/s) on rho points."""
        u, v = self._uv()
        return self._out(derived.convergence(u, v, self._obj))

    @property
    def convergence_norm(self):
        """Surface convergence normalized by f (dimensionless), on rho points."""
        conv = canonicalize(self.convergence)
        out = conv.isel(s_rho=-1) / canonicalize(self._obj["f"])
        out.attrs = {"name": "convergence_norm", "long_name": "normalized surface horizontal convergence", "units": ""}
        return self._out(out.rename("convergence_norm"))

    @property
    def rho(self):
        """In situ density (kg/m³): the Dataset's ``rho`` if present, else ROMS EOS."""
        if "rho" in self._obj.variables:
            return self._obj["rho"]
        return self._out(roms_seawater.density(self._obj["temp"], self._obj["salt"], grid=self._obj))

    @property
    def sig0(self):
        """Potential density referenced to the surface (kg/m³)."""
        return self._out(roms_seawater.potential_density(self._obj["temp"], self._obj["salt"], 0))

    @property
    def buoyancy(self):
        """Buoyancy (m/s²) from potential density and the Dataset's rho0."""
        return self._out(roms_seawater.buoyancy(self.sig0, rho0(self._obj)))

    @property
    def N2(self):
        """Buoyancy frequency squared (1/s²) on w levels."""
        return self._out(roms_seawater.N2(self.rho, self._obj, rho0(self._obj)))

    @property
    def M2(self):
        """Horizontal buoyancy gradient (1/s²) on rho points."""
        return self._out(roms_seawater.M2(self.rho, self._obj, rho0(self._obj)))

    @property
    def ertel(self):
        """Ertel potential vorticity of buoyancy on rho points and levels."""
        u, v = self._uv()
        return self._out(derived.ertel(self.buoyancy, u, v, self._obj["f"], self._obj))

    def mld(self, threshold=None, *, thresh=None, grid=None, **kwargs):
        """Mixed layer depth (m, positive) on rho points, from ``sig0`` (or ``temp`` with ``variable="temperature"``); see :func:`xroms.mld`."""
        threshold = roms_seawater._threshold_alias(threshold, thresh)
        var = self._obj["temp"] if kwargs.get("variable") == "temperature" else self.sig0
        return self._out(roms_seawater.mld(var, self._grid(grid), threshold=threshold, **kwargs))


@xr.register_dataarray_accessor("xroms")
class xromsDataArrayAccessor:
    """Operations that need nothing but the DataArray (no grid variables).

    Results use the Rutgers alias dims (``eta_u``, ``xi_v``, ``eta_psi``, ``xi_psi``)
    when this DataArray has any of them, and canonical dims (``xroms.canonicalize``)
    otherwise. A rho-point DataArray has no way to tell which naming its Dataset
    uses, so moving it to u, v or psi points gives canonical names, e.g.
    ``(eta_rho, xi_u)``; on a Rutgers Dataset that does not line up with ``ds.u``
    (``(eta_u, xi_u)``). To get the Dataset's own naming use
    ``ds.xroms.to_grid("temp", "u")``, or work on ``xroms.canonicalize(ds)``.
    """

    def __init__(self, da):
        self._obj = da

    def _named(self, out):
        """``out`` with alias dims if this DataArray has them, else canonical."""
        return _alias_like(out, self._obj)

    def to_grid(self, hcoord=None, scoord=None, **kwargs):
        """Move to ``hcoord``/``scoord`` by averaging neighbours; see :func:`xroms.to_grid`."""
        if _is_xgcm_grid(hcoord) or _is_xgcm_grid(kwargs.get("xgrid")):
            raise TypeError(
                "xroms 1.0: da.xroms.to_grid no longer takes an xgcm Grid; moving between grid positions needs "
                "no grid, e.g. da.xroms.to_grid('u') or da.xroms.to_grid(hcoord='u', scoord='w'). "
                "xroms no longer builds or stores xgcm grids."
            )
        return self._named(utilities.to_grid(self._obj, hcoord, scoord, **kwargs))

    def to_rho(self, **kwargs):
        """Move to rho points horizontally."""
        return self._named(utilities.to_rho(self._obj, **kwargs))

    def to_u(self, **kwargs):
        """Move to u points horizontally."""
        return self._named(utilities.to_u(self._obj, **kwargs))

    def to_v(self, **kwargs):
        """Move to v points horizontally."""
        return self._named(utilities.to_v(self._obj, **kwargs))

    def to_psi(self, **kwargs):
        """Move to psi points horizontally."""
        return self._named(utilities.to_psi(self._obj, **kwargs))

    def to_s_rho(self, **kwargs):
        """Move to rho (layer-centre) levels."""
        return self._named(utilities.to_s_rho(self._obj, **kwargs))

    def to_s_w(self, **kwargs):
        """Move to w (interface) levels."""
        return self._named(utilities.to_s_w(self._obj, **kwargs))

    def order(self):
        """Transpose to (time, vertical, eta, xi, ...)."""
        return utilities.order(self._obj)

    def _xy(self):
        pos = hposition(self._obj) or "rho"
        xname, yname = horizontal_coords(self._obj, pos)
        if xname is None:
            raise KeyError(f"{self._obj.name!r} has no lon/lat or x/y coordinates at {pos} points")
        return self._obj[xname], self._obj[yname], xname.startswith("x_")

    def argsel2d(self, lon0, lat0, **kwargs):
        """Indices of the point nearest to ``(lon0, lat0)``, using this array's coords."""
        x, y, cartesian = self._xy()
        if cartesian:
            kwargs.setdefault("method", "cartesian")
        return utilities.argsel2d(x, y, lon0, lat0, **kwargs)

    def sel2d(self, lon0, lat0, **kwargs):
        """Value(s) at the point nearest to ``(lon0, lat0)``, using this array's coords."""
        x, y, cartesian = self._xy()
        if cartesian:
            kwargs.setdefault("method", "cartesian")
        return utilities.sel2d(self._obj, x, y, lon0, lat0, **kwargs)

    def isoslice(self, iso_values, iso_array, **kwargs):
        """Interpolate onto values of ``iso_array``; see :func:`xroms.isoslice`."""
        return self._named(interp.isoslice(self._obj, iso_values, iso_array, **kwargs))

    def interpll(self, lons=None, lats=None, **kwargs):
        """Interpolate to lon/lat points with xESMF; see :func:`xroms.interpll`."""
        return interp.interpll(self._obj, lons, lats, **kwargs)

    _moved = "It needs grid variables: use ds.xroms.{name}(da) or xroms.{name}(da, ds)."
    ddxi = _removed("da.xroms.ddxi", _moved.format(name="ddxi"))
    ddeta = _removed("da.xroms.ddeta", _moved.format(name="ddeta"))
    ddz = _removed("da.xroms.ddz", _moved.format(name="ddz"))
    zslice = _removed("da.xroms.zslice", _moved.format(name="zslice"))
    gridmean = _removed("da.xroms.gridmean", _moved.format(name="gridmean"))
    gridsum = _removed("da.xroms.gridsum", _moved.format(name="gridsum"))
