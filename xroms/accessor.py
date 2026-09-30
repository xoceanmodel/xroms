"""The ``ds.xroms`` and ``da.xroms`` accessors: thin, stateless conveniences.

``ds.xroms`` holds a reference to the Dataset and nothing else: no cache, no
copies, no writes into the Dataset. Every property and method recomputes from
the Dataset's current contents (lazily under dask), so results always match the
data, even after in-place edits or subsetting. Results come back in the
Dataset's own dim naming (Rutgers ``eta_u`` stays ``eta_u``) with the Dataset's
coordinates for their grid position attached. Assign a result to a variable if
you reuse it; use ``.persist()`` for expensive reuse.

``da.xroms`` offers only operations that need nothing beyond the DataArray.
"""

import numpy as np
import xarray as xr

from . import _xgcm, derived, interp, metrics, roms_seawater, utilities, vector, vertical
from .conventions import canonicalize, horizontal_coords, hposition, rename_like, rho0, vertical_params


def _removed(name, hint):
    def method(self, *args, **kwargs):
        raise AttributeError(f"xroms 1.0 removed {name}. {hint}")

    method.__doc__ = f"Removed in xroms 1.0. {hint}"
    return method


@xr.register_dataset_accessor("xroms")
class xromsDatasetAccessor:
    """ROMS-aware calculations on a Dataset (grid variables included or passed)."""

    def __init__(self, ds):
        self._obj = ds

    # --- plumbing -------------------------------------------------------------------

    def _grid(self, grid=None):
        return self._obj if grid is None else grid

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
                if name is None or name in out.coords or name in coords:
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

    def assign_z(self, *, zeta=None, hcoord="rho"):
        """A **new** Dataset with lazy ``z_rho``/``z_w`` coordinates attached.

        This is an explicit snapshot: recompute after changing ``zeta`` or ``h``.
        On numpy-backed data the depths are computed immediately.
        """
        z_rho = self.z(hcoord, "s_rho", zeta=zeta)
        z_w = self.z(hcoord, "s_w", zeta=zeta)
        return self._obj.assign_coords({z_rho.name: z_rho.variable, z_w.name: z_w.variable})

    def xgcm_grid(self, padding="extend"):
        """A fresh xgcm Grid for this Dataset's canonical dims (``xroms.canonicalize(ds)``)."""
        return _xgcm.grid_for(canonicalize(self._obj), padding=padding)

    set_grid = _removed("ds.xroms.set_grid", "xroms no longer stores an xgcm grid; just call the methods, or use ds.xroms.xgcm_grid() for your own xgcm work.")

    @property
    def xgrid(self):
        """Removed in xroms 1.0; see :meth:`xgcm_grid`."""
        raise AttributeError("xroms 1.0 removed ds.xroms.xgrid; use ds.xroms.xgcm_grid() for a fresh xgcm Grid.")

    # --- grid moves and calculus -------------------------------------------------------

    def to_grid(self, var, hcoord=None, scoord=None, **kwargs):
        """Move ``var`` (name or DataArray) to ``hcoord``/``scoord``."""
        return self._out(utilities.to_grid(self._var(var), hcoord, scoord, **kwargs))

    def ddxi(self, var, *, grid=None, **kwargs):
        """d/dxi at constant depth; see :func:`xroms.ddxi`."""
        return self._out(utilities.ddxi(self._var(var), self._grid(grid), **kwargs))

    def ddeta(self, var, *, grid=None, **kwargs):
        """d/deta at constant depth; see :func:`xroms.ddeta`."""
        return self._out(utilities.ddeta(self._var(var), self._grid(grid), **kwargs))

    def ddz(self, var, *, grid=None, **kwargs):
        """d/dz; see :func:`xroms.ddz`."""
        return self._out(utilities.ddz(self._var(var), self._grid(grid), **kwargs))

    def hgrad(self, var, *, grid=None, **kwargs):
        """Both horizontal derivatives at constant depth."""
        return self.ddxi(var, grid=grid, **kwargs), self.ddeta(var, grid=grid, **kwargs)

    def gridsum(self, var, dims, *, grid=None, **kwargs):
        """Grid-weighted sum over ``dims``; see :func:`xroms.gridsum`."""
        return self._out(utilities.gridsum(self._var(var), self._grid(grid), dims, **kwargs))

    def gridmean(self, var, dims, *, grid=None, **kwargs):
        """Grid-weighted mean over ``dims``; see :func:`xroms.gridmean`."""
        return self._out(utilities.gridmean(self._var(var), self._grid(grid), dims, **kwargs))

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
        east = ds["u_eastward"] if "u_eastward" in ds.variables else ds["east"]
        north = ds["v_northward"] if "v_northward" in ds.variables else ds["north"]
        u, v = vector.earth_to_grid(east, north, ds["angle"], hcoord="native")
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

    def east_rotated(self, angle, *, reference="xaxis", isradians=True, name=None):
        """x component of (east, north) rotated by ``angle`` (e.g. along-channel)."""
        return self._rotated(angle, reference, isradians, name, 0)

    def north_rotated(self, angle, *, reference="xaxis", isradians=True, name=None):
        """y component of (east, north) rotated by ``angle`` (e.g. across-channel)."""
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
        return self._out(derived.uv_geostrophic(self._obj["zeta"], self._obj["f"], self._obj, which="xi"))

    @property
    def vg(self):
        """Geostrophic v (m/s) from zeta, on v points."""
        return self._out(derived.uv_geostrophic(self._obj["zeta"], self._obj["f"], self._obj, which="eta"))

    @property
    def EKE(self):
        """Eddy kinetic energy of the geostrophic velocities (m²/s²) on rho points."""
        return self._out(derived.EKE(self.ug, self.vg))

    @property
    def dudz(self):
        """du/dz (1/s) on u points, w levels."""
        return self._out(derived.dudz(self._uv()[0], self._obj))

    @property
    def dvdz(self):
        """dv/dz (1/s) on v points, w levels."""
        return self._out(derived.dvdz(self._uv()[1], self._obj))

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

    def mld(self, thresh=0.03, **kwargs):
        """Mixed layer depth (m, positive) on rho points; see :func:`xroms.mld`."""
        return self._out(roms_seawater.mld(self.sig0, self._obj, thresh=thresh, **kwargs))


@xr.register_dataarray_accessor("xroms")
class xromsDataArrayAccessor:
    """Operations that need nothing but the DataArray (no grid variables)."""

    def __init__(self, da):
        self._obj = da

    def to_grid(self, hcoord=None, scoord=None, **kwargs):
        """Move to ``hcoord``/``scoord`` by averaging neighbours; see :func:`xroms.to_grid`."""
        return utilities.to_grid(self._obj, hcoord, scoord, **kwargs)

    def to_rho(self, **kwargs):
        """Move to rho points horizontally."""
        return utilities.to_rho(self._obj, **kwargs)

    def to_u(self, **kwargs):
        """Move to u points horizontally."""
        return utilities.to_u(self._obj, **kwargs)

    def to_v(self, **kwargs):
        """Move to v points horizontally."""
        return utilities.to_v(self._obj, **kwargs)

    def to_psi(self, **kwargs):
        """Move to psi points horizontally."""
        return utilities.to_psi(self._obj, **kwargs)

    def to_s_rho(self, **kwargs):
        """Move to rho (layer-centre) levels."""
        return utilities.to_s_rho(self._obj, **kwargs)

    def to_s_w(self, **kwargs):
        """Move to w (interface) levels."""
        return utilities.to_s_w(self._obj, **kwargs)

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
        return interp.isoslice(self._obj, iso_values, iso_array, **kwargs)

    def interpll(self, lons, lats, **kwargs):
        """Interpolate to lon/lat points with xESMF; see :func:`xroms.interpll`."""
        return interp.interpll(self._obj, lons, lats, **kwargs)

    _moved = "It needs grid variables: use ds.xroms.{name}(da) or xroms.{name}(da, ds)."
    ddxi = _removed("da.xroms.ddxi", _moved.format(name="ddxi"))
    ddeta = _removed("da.xroms.ddeta", _moved.format(name="ddeta"))
    ddz = _removed("da.xroms.ddz", _moved.format(name="ddz"))
    zslice = _removed("da.xroms.zslice", _moved.format(name="zslice"))
    gridmean = _removed("da.xroms.gridmean", _moved.format(name="gridmean"))
    gridsum = _removed("da.xroms.gridsum", _moved.format(name="gridsum"))
