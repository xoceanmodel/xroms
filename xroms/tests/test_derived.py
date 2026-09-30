"""Derived variables (speed, KE, geostrophy, EKE, shear, vorticity, convergence, Ertel PV).

Expected values are analytic or computed in plain numpy, independently of xgcm.
The ``uniform`` fixture has constant grid spacing and ``u = U_0 + U_A * x_u``,
``v = V_0 + V_A * y_v`` (nothing depends on depth); ``with_land`` has a land patch
with NaN over it. Flows that do depend on depth are built from ``xroms.z``.
"""

from collections import namedtuple

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms import derived
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


G = 9.81
F0 = syn.F0

RHO = ("eta_rho", "xi_rho")
U = ("eta_rho", "xi_u")
V = ("eta_v", "xi_rho")
PSI = ("eta_v", "xi_u")


# --- numpy helpers (axis -1 is xi, axis -2 is eta) -----------------------------------


def to_center(a, axis, fill=None):
    """Staggered -> center: n - 1 points -> n, with the edge value (or ``fill``) outside."""
    a = np.moveaxis(np.asarray(a), axis, -1)
    pad = [(0, 0)] * (a.ndim - 1) + [(1, 1)]
    padded = np.pad(a, pad, mode="edge") if fill is None else np.pad(a, pad, constant_values=fill)
    return np.moveaxis(0.5 * (padded[..., :-1] + padded[..., 1:]), -1, axis)


def to_inner(a, axis):
    """Center -> staggered: n points -> n - 1 (neighbours averaged)."""
    a = np.moveaxis(np.asarray(a), axis, -1)
    return np.moveaxis(0.5 * (a[..., :-1] + a[..., 1:]), -1, axis)


def xy(ds):
    """Distance along xi and eta at the rho points of a uniform grid, and the spacings."""
    can = C.canonicalize(ds)
    dx, dy = 1 / float(can.pm[0, 0]), 1 / float(can.pn[0, 0])
    x = xr.DataArray(np.arange(can.sizes["xi_rho"]) * dx, dims="xi_rho")
    y = xr.DataArray(np.arange(can.sizes["eta_rho"]) * dy, dims="eta_rho")
    return x, y, dx, dy


def depth_only_flow(ds, a_u, a_v):
    """``u = a_u * z`` and ``v = a_v * z`` at their own points: only depth matters."""
    return a_u * xroms.z(ds, hcoord="u"), a_v * xroms.z(ds, hcoord="v")


# --- every derived variable, for the properties they all share -----------------------

Case = namedtuple("Case", "make name long_name units dims")


def _geostrophic(ds):
    return xroms.uv_geostrophic(ds.zeta, ds.f, ds)


CASES = [
    Case(lambda d: xroms.speed(d.u, d.v), "speed", "horizontal speed", "m/s", ("s_rho",) + RHO),
    Case(
        lambda d: xroms.KE(xroms.rho0(d), xroms.speed(d.u, d.v)),
        "KE", "kinetic energy", "kg/(m*s^2)", ("s_rho",) + RHO,
    ),
    Case(lambda d: _geostrophic(d)[0], "ug", "geostrophic u velocity", "m/s", U),
    Case(lambda d: _geostrophic(d)[1], "vg", "geostrophic v velocity", "m/s", V),
    Case(lambda d: xroms.EKE(*_geostrophic(d)), "EKE", "eddy kinetic energy", "m^2/s^2", RHO),
    Case(lambda d: xroms.dudz(d.u, d), "dudz", "u component of vertical shear", "1/s", ("s_w",) + U),
    Case(lambda d: xroms.dvdz(d.v, d), "dvdz", "v component of vertical shear", "1/s", ("s_w",) + V),
    Case(
        lambda d: xroms.vertical_shear(xroms.dudz(d.u, d), xroms.dvdz(d.v, d)),
        "shear", "vertical shear", "1/s", ("s_w",) + RHO,
    ),
    Case(
        lambda d: xroms.relative_vorticity(d.u, d.v, d),
        "vort", "vertical component of vorticity", "1/s", ("s_rho",) + PSI,
    ),
    Case(
        lambda d: xroms.convergence(d.u, d.v, d),
        "convergence", "horizontal convergence", "1/s", ("s_rho",) + RHO,
    ),
    Case(
        lambda d: xroms.ertel(d.temp, d.u, d.v, d.f, d),
        "ertel", "ertel potential vorticity", "tracer/(m*s)", ("s_rho",) + RHO,
    ),
]
EACH = pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])


class TestEveryFunction:
    @EACH
    def test_dims_every_layout(self, layout, case):
        ds = merged(layout)
        out = case.make(ds)
        assert out.dims == (C.time_dim(C.canonicalize(ds).u),) + case.dims
        assert not set(out.dims) & set(C.ALIASES)  # canonical names whatever the layout
        assert np.isfinite(out.values).all()

    @EACH
    def test_name_and_attrs(self, uniform, case):
        out = case.make(uniform)
        assert out.name == case.name
        # exactly these: nothing is inherited from the inputs (u has units "meter second-1")
        assert out.attrs == {"name": case.name, "long_name": case.long_name, "units": case.units}

    @EACH
    def test_inputs_not_modified(self, uniform, case):
        before = uniform.copy(deep=True)
        case.make(uniform)
        xr.testing.assert_identical(uniform, before)

    @EACH
    def test_grid_angle_is_irrelevant(self, case):
        # everything is along xi and eta: nothing is rotated towards east and north
        a = case.make(syn.make_dataset("rutgers"))
        b = case.make(syn.make_dataset("rutgers", angle=0.7))
        xr.testing.assert_identical(a, b)

    @EACH
    def test_chunked_equals_numpy(self, rutgers, case):
        expected = case.make(rutgers)
        out = case.make(chunked(rutgers))
        assert out.chunks is not None  # stays lazy
        assert out.dims == expected.dims
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-18)

    @EACH
    def test_open_water_unaffected_by_land(self, rutgers, with_land, case):
        a, b = case.make(rutgers), case.make(with_land)
        # rows 0-1 and columns 0-2 are land; look well away from it and its stencils
        away = {d: slice(5 if d.startswith("eta") else 6, None) for d in a.dims if d.startswith(("eta", "xi"))}
        a, b = a.isel(away), b.isel(away)
        assert np.isfinite(b.values).all()
        np.testing.assert_allclose(b.values, a.values, rtol=1e-12, atol=1e-18)


class TestSpeed:
    def test_values_every_layout(self, layout):
        ds = merged(layout)
        can = C.canonicalize(ds)
        out = xroms.speed(ds.u, ds.v)
        assert out.dims == can.temp.dims
        u_rho = to_center(can.u.fillna(0).values, -1)
        v_rho = to_center(can.v.fillna(0).values, -2)
        np.testing.assert_allclose(out.values, np.sqrt(u_rho**2 + v_rho**2), rtol=1e-12)

    def test_matches_manual_to_rho(self, rutgers):
        u, v = xroms.to_rho(rutgers.u), xroms.to_rho(rutgers.v)
        out = xroms.speed(rutgers.u, rutgers.v)
        np.testing.assert_allclose(out.values, np.sqrt(u**2 + v**2).values, rtol=1e-12)

    def test_uniform_current(self, rutgers):
        u, v = xr.full_like(rutgers.u, 0.3), xr.full_like(rutgers.v, 0.4)
        np.testing.assert_allclose(xroms.speed(u, v).values, 0.5, rtol=1e-12)  # edges too

    def test_land_does_not_spread_into_water(self, with_land):
        can = C.canonicalize(with_land)
        water = can.mask_rho.values == 1
        out = xroms.speed(with_land.u, with_land.v)
        assert np.isfinite(out.values[..., water]).all()
        # this is the fillna(0): moved to rho as they are, the masked velocities put NaN in the water
        naive = np.sqrt(xroms.to_rho(with_land.u) ** 2 + xroms.to_rho(with_land.v) ** 2)
        assert np.isnan(naive.values[..., water]).any()
        # rho (0, 3) is water; u is land-masked on its west side, so only half of the u east of it counts
        u03, v03 = float(can.u[0, 0, 0, 3]), float(can.v[0, 0, 0, 3])
        assert out.values[0, 0, 0, 3] == pytest.approx(np.hypot(0.5 * u03, v03), rel=1e-12)
        u_rho = to_center(can.u.fillna(0).values, -1)
        v_rho = to_center(can.v.fillna(0).values, -2)
        np.testing.assert_allclose(out.values, np.sqrt(u_rho**2 + v_rho**2), rtol=1e-12)

    def test_boundary_options(self, rutgers):
        can = C.canonicalize(rutgers)
        zero = xroms.speed(rutgers.u, rutgers.v, hboundary="fill", hfill_value=0.0)
        u_rho = to_center(can.u.values, -1, fill=0.0)
        v_rho = to_center(can.v.values, -2, fill=0.0)
        np.testing.assert_allclose(zero.values, np.sqrt(u_rho**2 + v_rho**2), rtol=1e-12)
        nan = xroms.speed(rutgers.u, rutgers.v, hboundary="fill").values
        assert np.isfinite(nan[..., 1:-1, 1:-1]).all()
        for edge in (nan[..., 0, :], nan[..., -1, :], nan[..., :, 0], nan[..., :, -1]):
            assert np.isnan(edge).all()

    def test_selected_level_and_time(self, rutgers):
        out = xroms.speed(rutgers.u.isel(ocean_time=1, s_rho=-1), rutgers.v.isel(ocean_time=1, s_rho=-1))
        assert out.dims == RHO
        full = xroms.speed(rutgers.u, rutgers.v)
        np.testing.assert_allclose(out.values, full.isel(ocean_time=1, s_rho=-1).values)


class TestKE:
    def test_values_every_layout(self, layout):
        ds = merged(layout)
        sp = xroms.speed(ds.u, ds.v)
        rho0 = xroms.rho0(ds)
        out = xroms.KE(rho0, sp)
        assert out.dims == sp.dims
        np.testing.assert_allclose(out.values, 0.5 * rho0 * sp.values**2, rtol=1e-12)

    def test_physical_value(self, rutgers):
        sp = xroms.speed(xr.full_like(rutgers.u, 0.3), xr.full_like(rutgers.v, 0.4))
        np.testing.assert_allclose(xroms.KE(1025.0, sp).values, 0.5 * 1025.0 * 0.25, rtol=1e-12)

    def test_rho0_can_be_a_field(self, rutgers):
        sp = xroms.speed(rutgers.u, rutgers.v)
        out = xroms.KE(xr.full_like(sp, 1030.0), sp)
        np.testing.assert_allclose(out.values, 0.5 * 1030.0 * sp.values**2, rtol=1e-12)


class TestGeostrophic:
    def test_dims_and_values_against_numpy_finite_differences(self, uniform):
        _, _, dx, dy = xy(uniform)
        zeta = C.canonicalize(uniform).zeta.values  # (time, eta_rho, xi_rho)
        ug, vg = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform)
        assert ug.dims == ("ocean_time", "eta_rho", "xi_u")
        assert vg.dims == ("ocean_time", "eta_v", "xi_rho")
        # zeta_eta sits at v points: moved to u points by averaging along xi, then along eta
        zeta_eta_u = to_center(to_inner(np.diff(zeta, axis=-2) / dy, -1), -2)
        # zeta_xi sits at u points: moved to v points by averaging along eta, then along xi
        zeta_xi_v = to_center(to_inner(np.diff(zeta, axis=-1) / dx, -2), -1)
        np.testing.assert_allclose(ug.values, -G * zeta_eta_u / F0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vg.values, G * zeta_xi_v / F0, rtol=1e-12, atol=1e-12)

    def test_tilted_surface_gives_uniform_flow(self, uniform):
        x, y, _, _ = xy(uniform)
        a, b = 2e-6, -3e-6  # surface slopes along xi and eta
        zeta = xr.zeros_like(C.canonicalize(uniform).zeta) + a * x + b * y
        ug, vg = xroms.uv_geostrophic(zeta, uniform.f, uniform)
        np.testing.assert_allclose(ug.values, -G * b / F0, rtol=1e-12)  # edges included
        np.testing.assert_allclose(vg.values, G * a / F0, rtol=1e-12)

    def test_f_is_taken_at_the_velocity_points(self, uniform):
        x, y, _, _ = xy(uniform)
        a, b = 2e-6, -3e-6
        zeta = xr.zeros_like(C.canonicalize(uniform).zeta) + a * x + b * y
        fx, fy = 1e-11, 2e-11
        f = (F0 + fx * x + fy * y).transpose("eta_rho", "xi_rho")  # linear, so averaging is exact
        ug, vg = xroms.uv_geostrophic(zeta, f, uniform)
        x_u, y_v = to_inner(x.values, -1), to_inner(y.values, -1)
        f_u = F0 + fx * x_u[None, :] + fy * y.values[:, None]
        f_v = F0 + fx * x.values[None, :] + fy * y_v[:, None]
        np.testing.assert_allclose(ug.values, np.broadcast_to(-G * b / f_u, ug.shape), rtol=1e-12)
        np.testing.assert_allclose(vg.values, np.broadcast_to(G * a / f_v, vg.shape), rtol=1e-12)

    def test_which(self, uniform):
        both = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform)
        xi = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform, which="xi")
        eta = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform, which="eta")
        assert isinstance(both, tuple) and len(both) == 2
        xr.testing.assert_identical(xi, both[0])
        xr.testing.assert_identical(eta, both[1])
        assert xi.name == "ug" and eta.name == "vg"
        with pytest.raises(ValueError, match="which"):
            xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform, which="north")

    def test_boundary_options(self, uniform):
        _, _, dx, dy = xy(uniform)
        zeta = C.canonicalize(uniform).zeta.values
        ug, vg = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform, hboundary="fill", hfill_value=0.0)
        # the move of zeta_eta from v to u points pads along eta: fill=0 halves the edge rows
        zeta_eta_u = to_center(to_inner(np.diff(zeta, axis=-2) / dy, -1), -2, fill=0.0)
        zeta_xi_v = to_center(to_inner(np.diff(zeta, axis=-1) / dx, -2), -1, fill=0.0)
        np.testing.assert_allclose(ug.values, -G * zeta_eta_u / F0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vg.values, G * zeta_xi_v / F0, rtol=1e-12, atol=1e-12)
        ug, vg = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform, hboundary="fill")
        assert np.isnan(ug.isel(eta_rho=[0, -1]).values).all() and np.isfinite(ug.isel(eta_rho=slice(1, -1)).values).all()
        assert np.isnan(vg.isel(xi_rho=[0, -1]).values).all() and np.isfinite(vg.isel(xi_rho=slice(1, -1)).values).all()

    def test_selected_time(self, uniform):
        ug, vg = xroms.uv_geostrophic(uniform.zeta.isel(ocean_time=1), uniform.f, uniform)
        full_ug, full_vg = xroms.uv_geostrophic(uniform.zeta, uniform.f, uniform)
        np.testing.assert_allclose(ug.values, full_ug.isel(ocean_time=1).values)
        np.testing.assert_allclose(vg.values, full_vg.isel(ocean_time=1).values)


class TestEKE:
    def test_dims_and_values(self, uniform):
        ug, vg = _geostrophic(uniform)
        out = xroms.EKE(ug, vg)
        assert out.dims == ("ocean_time", "eta_rho", "xi_rho")
        expected = 0.5 * (to_center(ug.values, -1) ** 2 + to_center(vg.values, -2) ** 2)
        np.testing.assert_allclose(out.values, expected, rtol=1e-12)

    def test_inputs_already_on_rho(self, uniform):
        ug, vg = (xroms.to_rho(a) for a in _geostrophic(uniform))
        np.testing.assert_allclose(xroms.EKE(ug, vg).values, (0.5 * (ug**2 + vg**2)).values, rtol=1e-12)

    def test_boundary_options(self, uniform):
        ug, vg = _geostrophic(uniform)
        out = xroms.EKE(ug, vg, hboundary="fill", hfill_value=0.0)
        expected = 0.5 * (to_center(ug.values, -1, fill=0.0) ** 2 + to_center(vg.values, -2, fill=0.0) ** 2)
        np.testing.assert_allclose(out.values, expected, rtol=1e-12)
        assert np.isnan(xroms.EKE(ug, vg, hboundary="fill").isel(xi_rho=0)).all()


class TestVerticalShear:
    @pytest.mark.parametrize("which", ["u", "v"])
    def test_zero_without_depth_dependence_every_layout(self, layout, which):
        ds = merged(layout)
        fn, var, dims = (xroms.dudz, ds.u, U) if which == "u" else (xroms.dvdz, ds.v, V)
        out = fn(var, ds)
        assert out.dims == (C.time_dim(C.canonicalize(ds).u), "s_w") + dims  # on the w levels
        np.testing.assert_allclose(out.values, 0.0, atol=1e-15)
        for edge in (out.isel(s_w=0), out.isel(s_w=-1)):  # bottom and top w levels too
            assert np.isfinite(edge.values).all()
            np.testing.assert_allclose(edge.values, 0.0, atol=1e-15)

    def test_recovers_linear_shear_including_top_and_bottom(self, uniform):
        u, v = depth_only_flow(uniform, 3e-3, -4e-3)
        u, v = C.canonicalize(uniform).u + u, C.canonicalize(uniform).v + v
        du, dv = xroms.dudz(u, uniform), xroms.dvdz(v, uniform)
        assert du.dims == ("ocean_time", "s_w", "eta_rho", "xi_u")
        assert dv.dims == ("ocean_time", "s_w", "eta_v", "xi_rho")
        np.testing.assert_allclose(du.values, 3e-3, rtol=1e-9)
        np.testing.assert_allclose(dv.values, -4e-3, rtol=1e-9)

    def test_boundary_options(self, uniform):
        u, _ = depth_only_flow(uniform, 3e-3, 0.0)
        nan = xroms.dudz(u, uniform, sboundary="fill")
        assert np.isnan(nan.isel(s_w=0)).all() and np.isnan(nan.isel(s_w=-1)).all()
        np.testing.assert_allclose(nan.isel(s_w=slice(1, -1)).values, 3e-3, rtol=1e-9)
        zero = xroms.dudz(u, uniform, sboundary="fill", sfill_value=0.0)
        assert (zero.isel(s_w=0) == 0).all() and (zero.isel(s_w=-1) == 0).all()

    def test_zeta_and_z_arguments(self, uniform):
        can = C.canonicalize(uniform)
        # built from static depths: only zeta=0 recovers the shear, the moving surface does not
        u = (3e-3 * xroms.z(uniform, hcoord="u", zeta=0)).broadcast_like(can.u).transpose(*can.u.dims)
        np.testing.assert_allclose(xroms.dudz(u, uniform, zeta=0).values, 3e-3, rtol=1e-9)
        assert np.abs(xroms.dudz(u, uniform).values - 3e-3).max() > 1e-5
        # explicit depths: at the u points, or at rho points (averaged onto the u points)
        u, v = depth_only_flow(uniform, 3e-3, -4e-3)
        expected = xroms.dudz(u, uniform)
        np.testing.assert_allclose(xroms.dudz(u, z=xroms.z(uniform, hcoord="u")).values, expected.values)
        np.testing.assert_allclose(xroms.dudz(u, z=xroms.z(uniform)).values, expected.values)
        np.testing.assert_allclose(xroms.dvdz(v, z=xroms.z(uniform)).values, xroms.dvdz(v, uniform).values)
        with pytest.raises(ValueError, match="points"):
            xroms.dudz(u, uniform, z=xroms.z(uniform, hcoord="v"))
        with pytest.raises(ValueError, match="grid"):
            xroms.dudz(can.u)  # neither grid nor z

    def test_vertical_shear_dims_and_values(self, uniform):
        u, v = depth_only_flow(uniform, 3e-3, -4e-3)
        out = xroms.vertical_shear(xroms.dudz(u, uniform), xroms.dvdz(v, uniform))
        assert out.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")  # rho points, w levels
        np.testing.assert_allclose(out.values, 5e-3, rtol=1e-9)

    def test_vertical_shear_combines_components(self, uniform):
        du, dv = xroms.dudz(uniform.u, uniform), xroms.dvdz(uniform.v, uniform)
        out = xroms.vertical_shear(xr.full_like(du, 3.0), xr.full_like(dv, 4.0))
        np.testing.assert_allclose(out.values, 5.0, rtol=1e-12)  # edges too

    def test_vertical_shear_boundary_options(self, uniform):
        du, dv = xroms.dudz(uniform.u, uniform), xroms.dvdz(uniform.v, uniform)
        nan = xroms.vertical_shear(du + 3.0, dv + 4.0, hboundary="fill")
        assert np.isfinite(nan.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1)).values).all()
        assert np.isnan(nan.isel(xi_rho=0).values).all()
        zero = xroms.vertical_shear(du + 3.0, dv + 4.0, hboundary="fill", hfill_value=0.0)
        np.testing.assert_allclose(zero.isel(eta_rho=slice(1, -1), xi_rho=0).values, np.hypot(1.5, 4.0), rtol=1e-12)


class TestRelativeVorticity:
    def test_zero_on_uniform_grid(self, uniform):
        out = xroms.relative_vorticity(uniform.u, uniform.v, uniform)
        assert out.dims == ("ocean_time", "s_rho", "eta_v", "xi_u")  # psi points
        # v has no x dependence and u no y dependence
        np.testing.assert_allclose(out.values, 0.0, atol=1e-18)

    def test_solid_body_rotation(self, uniform):
        can = C.canonicalize(uniform)
        x, y, _, _ = xy(uniform)
        omega = 1e-5
        u = (xr.zeros_like(can.u) - omega * y).transpose(*can.u.dims)
        v = (xr.zeros_like(can.v) + omega * x).transpose(*can.v.dims)
        out = xroms.relative_vorticity(u, v, uniform)
        np.testing.assert_allclose(out.values, 2 * omega, rtol=1e-9)
        # the same flow has no convergence
        np.testing.assert_allclose(xroms.convergence(u, v, uniform).values, 0.0, atol=1e-18)

    def test_shear_flow(self, uniform):
        can = C.canonicalize(uniform)
        x, y, _, _ = xy(uniform)
        u = (xr.zeros_like(can.u) - 2e-5 * y).transpose(*can.u.dims)  # u_y = -2e-5
        v = xr.zeros_like(can.v)
        np.testing.assert_allclose(xroms.relative_vorticity(u, v, uniform).values, 2e-5, rtol=1e-9)
        v = (xr.zeros_like(can.v) + 3e-5 * x).transpose(*can.v.dims)  # v_x = 3e-5
        np.testing.assert_allclose(xroms.relative_vorticity(u, v, uniform).values, 5e-5, rtol=1e-9)

    def test_flow_that_only_changes_with_depth_has_no_vorticity(self, uniform):
        # u and v vary along the sloping s-surfaces but not at constant depth: the
        # derivative must be taken at constant depth
        u, v = depth_only_flow(uniform, 1e-2, -2e-2)
        np.testing.assert_allclose(xroms.relative_vorticity(u, v, uniform).values, 0.0, atol=1e-15)
        along_s = xroms.ddeta(u.isel(s_rho=3), uniform, along_s=True)
        assert np.abs(along_s.values).max() > 1e-6  # so this test has something to catch

    def test_boundary_options_and_zeta(self, uniform):
        u, v = depth_only_flow(uniform, 1e-2, -2e-2)
        opts = dict(hboundary="fill", sboundary="fill")
        out = xroms.relative_vorticity(u, v, uniform, **opts)
        np.testing.assert_allclose(out.isel(s_rho=slice(1, -1)).values, 0.0, atol=1e-15)
        assert np.isnan(out.isel(s_rho=[0, -1])).all()  # no one-sided vertical estimate with "fill"
        # zeta: static depths cannot see the moving surface, so along-s terms do not cancel
        assert np.abs(xroms.relative_vorticity(u, v, uniform, zeta=0).values).max() > 1e-9

    def test_z_argument(self, uniform):
        u, v = depth_only_flow(uniform, 1e-2, -2e-2)
        expected = xroms.relative_vorticity(u, v, uniform)
        # one z serves u and v, which sit at different points: it is averaged onto each
        out = xroms.relative_vorticity(u, v, uniform, z=xroms.z(uniform))
        xr.testing.assert_allclose(out, expected)
        with pytest.raises(ValueError, match="points"):
            xroms.relative_vorticity(u, v, uniform, z=xroms.z(uniform, hcoord="psi"))


class TestConvergence:
    def test_uniform_grid_gives_ua_plus_va(self, uniform):
        out = xroms.convergence(uniform.u, uniform.v, uniform)
        assert out.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")  # rho points
        np.testing.assert_allclose(out.values, syn.U_A + syn.V_A, rtol=1e-12, atol=0)  # edges too

    def test_components_are_not_mixed_up(self, uniform):
        can = C.canonicalize(uniform)
        zero_v = xr.zeros_like(can.v)
        zero_u = xr.zeros_like(can.u)
        np.testing.assert_allclose(xroms.convergence(uniform.u, zero_v, uniform).values, syn.U_A, rtol=1e-12)
        np.testing.assert_allclose(xroms.convergence(zero_u, uniform.v, uniform).values, syn.V_A, rtol=1e-12)

    def test_flow_that_only_changes_with_depth_has_no_convergence(self, uniform):
        u, v = depth_only_flow(uniform, 1e-2, -2e-2)
        np.testing.assert_allclose(xroms.convergence(u, v, uniform).values, 0.0, atol=1e-15)

    def test_keeps_the_vertical_levels_of_the_inputs(self, uniform):
        u_w, v_w = (xroms.to_s_w(a) for a in (uniform.u, uniform.v))
        out = xroms.convergence(u_w, v_w, uniform)
        assert out.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(out.values, syn.U_A + syn.V_A, rtol=1e-12)

    def test_boundary_options(self, uniform):
        nan = xroms.convergence(uniform.u, uniform.v, uniform, hboundary="fill")
        interior = nan.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))
        np.testing.assert_allclose(interior.values, syn.U_A + syn.V_A, rtol=1e-12)
        assert np.isnan(nan.isel(xi_rho=0)).all() and np.isnan(nan.isel(eta_rho=-1)).all()
        zero = xroms.convergence(uniform.u, uniform.v, uniform, hboundary="fill", hfill_value=0.0)
        # no u_x on the western edge, so only v_y is left (and at its corners, neither term)
        np.testing.assert_allclose(zero.isel(xi_rho=0, eta_rho=slice(1, -1)).values, syn.V_A, rtol=1e-12)
        assert (zero.isel(xi_rho=0, eta_rho=[0, -1]) == 0).all()

    def test_z_argument(self, uniform):
        u, v = depth_only_flow(uniform, 1e-2, -2e-2)
        out = xroms.convergence(u, v, uniform, z=xroms.z(uniform))
        np.testing.assert_allclose(out.values, 0.0, atol=1e-15)

    def test_chunked_equals_numpy(self, rutgers, uniform):
        out = xroms.convergence(*(getattr(chunked(rutgers), n) for n in ("u", "v")), chunked(rutgers))
        expected = xroms.convergence(rutgers.u, rutgers.v, rutgers)
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-18)
        c = chunked(uniform)
        np.testing.assert_allclose(xroms.convergence(c.u, c.v, c).values, syn.U_A + syn.V_A, rtol=1e-12)


def analytic_inputs(ds, phi_scoord="s_rho"):
    """phi, u, v with known gradients, and the Ertel PV that follows at every point.

    ``phi`` is built on ``phi_scoord`` levels from the depths there.

    On a uniform grid, ``phi = A x + B y + C z``, ``u = S_u z - Om y`` and
    ``v = S_v z + Om x``. At constant depth phi_x = A, phi_y = B, phi_z = C,
    u_z = S_u, v_z = S_v and the relative vorticity is 2 Om, so
    ``epv = -S_v A + S_u B + (f + 2 Om) C``.
    """
    x, y, _, _ = xy(ds)
    a, b, c = 1e-4, 3e-5, 0.05
    s_u, s_v, omega = 2e-3, -1.5e-3, 1e-5
    z_rho, z_u, z_v = xroms.z(ds), xroms.z(ds, hcoord="u"), xroms.z(ds, hcoord="v")
    z_phi = xroms.z(ds, scoord=phi_scoord)
    phi = (a * x + b * y + c * z_phi).transpose(*z_phi.dims)
    u = (s_u * z_u - omega * y).transpose(*z_u.dims)
    v = (s_v * z_v + omega * x).transpose(*z_v.dims)
    return phi, u, v, -s_v * a + s_u * b + (F0 + 2 * omega) * c


class TestErtel:
    @pytest.mark.parametrize("hcoord", C.HCOORDS)
    @pytest.mark.parametrize("scoord", ["s_rho", "s_w", "rho", "w"])
    def test_dims(self, uniform, hcoord, scoord):
        out = xroms.ertel(uniform.temp, uniform.u, uniform.v, uniform.f, uniform, hcoord=hcoord, scoord=scoord)
        vertical = "s_w" if scoord in ("s_w", "w") else "s_rho"
        assert out.dims == ("ocean_time", vertical) + C.CANONICAL[hcoord]
        assert np.isfinite(out.values).all()

    def test_default_dims(self, uniform):
        out = xroms.ertel(uniform.temp, uniform.u, uniform.v, uniform.f, uniform)
        assert out.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        psi = xroms.ertel(uniform.temp, uniform.u, uniform.v, uniform.f, uniform, hcoord="psi")
        assert psi.dims == ("ocean_time", "s_rho", "eta_v", "xi_u")

    @pytest.mark.parametrize("phi_name", ["temp", "salt"])
    def test_zero_velocity_is_f_times_dphi_dz(self, uniform, phi_name):
        phi = uniform[phi_name]
        u, v = xr.zeros_like(uniform.u), xr.zeros_like(uniform.v)
        out = xroms.ertel(phi, u, v, uniform.f, uniform)
        expected = uniform.f * xroms.ddz(phi, uniform, hcoord="rho", scoord="s_rho")
        np.testing.assert_allclose(out.values, expected.transpose(*out.dims).values, rtol=1e-12)

    def test_zero_velocity_analytic(self, uniform):
        u, v = xr.zeros_like(uniform.u), xr.zeros_like(uniform.v)
        # temp is linear in z: a constant at every position
        for hcoord in C.HCOORDS:
            for scoord in ("s_rho", "s_w"):
                out = xroms.ertel(uniform.temp, u, v, uniform.f, uniform, hcoord=hcoord, scoord=scoord)
                np.testing.assert_allclose(out.values, F0 * syn.TEMP_B, rtol=1e-9)
        # salt is quadratic in z: f * 2 c z, which follows the depths to the output position
        for hcoord in C.HCOORDS:
            out = xroms.ertel(uniform.salt, u, v, uniform.f, uniform, hcoord=hcoord)
            z = xroms.z(uniform, hcoord=hcoord)
            np.testing.assert_allclose(out.values, (F0 * 2 * syn.SALT_C * z).transpose(*out.dims).values, rtol=1e-9)

    @pytest.mark.parametrize("hcoord", C.HCOORDS)
    @pytest.mark.parametrize("scoord", ["s_rho", "s_w"])
    def test_analytic_all_terms(self, uniform, hcoord, scoord):
        phi, u, v, expected = analytic_inputs(uniform)
        out = xroms.ertel(phi, u, v, uniform.f, uniform, hcoord=hcoord, scoord=scoord)
        np.testing.assert_allclose(out.values, expected, rtol=1e-9)

    def test_each_term_matters(self, uniform):
        phi, u, v, expected = analytic_inputs(uniform)
        f = uniform.f
        # dropping any ingredient moves the answer away from the analytic value
        still = xr.zeros_like(u), xr.zeros_like(v)
        assert not np.allclose(xroms.ertel(phi, still[0], v, f, uniform).values, expected, rtol=1e-3)  # u_z phi_y
        assert not np.allclose(xroms.ertel(phi, u, still[1], f, uniform).values, expected, rtol=1e-3)  # v_z phi_x
        assert not np.allclose(xroms.ertel(phi, u, v, xr.zeros_like(f), uniform).values, expected, rtol=1e-3)  # f

    def test_phi_on_w_levels(self, uniform):
        phi_w, u, v, expected = analytic_inputs(uniform, phi_scoord="s_w")
        for scoord in ("s_rho", "s_w"):
            out = xroms.ertel(phi_w, u, v, uniform.f, uniform, scoord=scoord)
            np.testing.assert_allclose(out.values, expected, rtol=1e-9)

    def test_f_at_any_position(self, uniform):
        phi, u, v, expected = analytic_inputs(uniform)
        for f in (uniform.f, xroms.to_u(uniform.f), xroms.to_psi(uniform.f)):
            out = xroms.ertel(phi, u, v, f, uniform, hcoord="v")
            assert out.dims == ("ocean_time", "s_rho", "eta_v", "xi_rho")
            np.testing.assert_allclose(out.values, expected, rtol=1e-9)

    def test_zeta(self, uniform):
        u, v = xr.zeros_like(uniform.u), xr.zeros_like(uniform.v)
        z0 = xroms.z(uniform, zeta=0)
        phi = (syn.SALT_C * z0**2).broadcast_like(uniform.temp).transpose(*uniform.temp.dims)
        static = xroms.ertel(phi, u, v, uniform.f, uniform, zeta=0)
        expected = (F0 * 2 * syn.SALT_C * z0).broadcast_like(static).transpose(*static.dims)
        np.testing.assert_allclose(static.values, expected.values, rtol=1e-9)
        assert np.abs(xroms.ertel(phi, u, v, uniform.f, uniform).values - static.values).max() > 1e-10

    def test_boundary_options(self, uniform):
        phi, u, v, expected = analytic_inputs(uniform)
        nan = xroms.ertel(phi, u, v, uniform.f, uniform, hboundary="fill", sboundary="fill")
        interior = nan.isel(s_rho=slice(1, -1), eta_rho=slice(1, -1), xi_rho=slice(1, -1))
        np.testing.assert_allclose(interior.values, expected, rtol=1e-9)
        assert np.isnan(nan.isel(s_rho=0)).any()

    def test_hcoord_and_scoord_are_required(self, uniform):
        args = (uniform.temp, uniform.u, uniform.v, uniform.f, uniform)
        with pytest.raises(ValueError, match="hcoord and scoord"):
            xroms.ertel(*args, hcoord=None)
        with pytest.raises(ValueError, match="hcoord and scoord"):
            xroms.ertel(*args, scoord=None)
        with pytest.raises(ValueError, match="hcoord"):
            xroms.ertel(*args, hcoord="north")


class FakeXgcmGrid:
    """Stands in for the xgcm Grid that xroms used to take."""


FakeXgcmGrid.__module__ = "xgcm.grid"


class TestLegacyAndBadCalls:
    def test_positional_grid_rejected_where_none_is_needed(self, rutgers):
        ug, vg = _geostrophic(rutgers)
        du, dv = xroms.dudz(rutgers.u, rutgers), xroms.dvdz(rutgers.v, rutgers)
        for call in (
            lambda: xroms.speed(rutgers.u, rutgers.v, object()),
            lambda: xroms.EKE(ug, vg, object()),
            lambda: xroms.vertical_shear(du, dv, object()),
        ):
            with pytest.raises(TypeError, match="1.0") as err:
                call()
            assert "no longer takes an xgcm grid" in str(err.value)

    def test_xgcm_grid_rejected_where_a_grid_is_needed(self, rutgers):
        grid = FakeXgcmGrid()
        for call in (
            lambda: xroms.uv_geostrophic(rutgers.zeta, rutgers.f, grid),
            lambda: xroms.dudz(rutgers.u, grid),
            lambda: xroms.dvdz(rutgers.v, grid),
            lambda: xroms.relative_vorticity(rutgers.u, rutgers.v, grid),
            lambda: xroms.convergence(rutgers.u, rutgers.v, grid),
            lambda: xroms.ertel(rutgers.temp, rutgers.u, rutgers.v, rutgers.f, grid),
        ):
            with pytest.raises(TypeError, match="xgcm"):
                call()

    def test_grid_must_be_a_dataset(self, rutgers):
        with pytest.raises(TypeError, match="Dataset"):
            xroms.convergence(rutgers.u, rutgers.v, rutgers.u)
        with pytest.raises(ValueError, match="grid"):
            xroms.convergence(rutgers.u, rutgers.v, None)

    def test_arguments_must_be_dataarrays(self, rutgers):
        for call in (
            lambda: xroms.speed(rutgers.u.values, rutgers.v),
            lambda: xroms.KE(1025.0, np.ones(3)),
            lambda: xroms.uv_geostrophic(rutgers.zeta, np.ones((9, 12)), rutgers),
            lambda: xroms.EKE(rutgers.u, rutgers.v.values),
            lambda: xroms.dudz(rutgers.u.values, rutgers),
            lambda: xroms.vertical_shear(rutgers.u, np.ones(3)),
            lambda: xroms.relative_vorticity(rutgers.u, rutgers.v.values, rutgers),
            lambda: xroms.ertel(rutgers.temp.values, rutgers.u, rutgers.v, rutgers.f, rutgers),
        ):
            with pytest.raises(TypeError, match="DataArray"):
                call()


class TestModule:
    def test_unimplemented_placeholders_are_gone(self):
        assert not hasattr(derived, "w") and not hasattr(derived, "omega")

    def test_package_exposes_the_same_functions(self):
        for name in ("speed", "KE", "uv_geostrophic", "EKE", "dudz", "dvdz", "vertical_shear", "relative_vorticity", "convergence", "ertel"):
            assert getattr(xroms, name) is getattr(derived, name)


@pytest.mark.parametrize("func", [xroms.relative_vorticity, xroms.convergence])
def test_single_s_level_needs_along_s(rutgers, func):
    can = C.canonicalize(rutgers)
    u, v = can.u.isel(s_rho=-1), can.v.isel(s_rho=-1)
    with pytest.raises(ValueError, match="along_s"):
        func(u, v, rutgers)
    out = func(u, v, rutgers, along_s=True)
    assert "s_rho" not in out.dims and np.isfinite(out.values).any()
