"""Seawater density, N2, M2 and the mixed layer depth, on reference values and analytic fields."""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms._align import GridMismatchError
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import INPUT, chunked, merged


G = 9.81
RHO0 = 1025.0

# The example output in tests/input is analytic: every s-level is horizontally uniform,
# with these values from the bottom to the top level. Z_COL is the height of the levels
# in the column at eta_rho=0, xi_rho=0 (where the free surface is -0.1 m).
TEMP_COL = np.linspace(15, 20, 3)
SALT_COL = np.linspace(25, 15, 3)
Z_COL = np.array([-97.5025, -50.05, -2.5975])

# (temp, salt, z, density) as calculated by the equation of state before the 1.0 rewrite
PINNED = [
    (10.0, 35.0, 0.0, 1026.9524116011753),
    (5.0, 35.0, 0.0, 1027.6754652782759),
    (25.0, 35.0, 0.0, 1023.3430584772268),
    (5.0, 35.0, -1000.0, 1032.2467605084607),
    (2.0, 34.7, -4000.0, 1045.7240191074106),
    (20.0, 10.0, -50.0, 1006.0161357249141),
]
RHO_COL = [1018.714588392324, 1014.155847861404, 1009.5873935328003]  # density at Z_COL
SIG0_COL = [1018.2789884611748, 1013.9328915891306, 1009.5758533623364]  # density at z=0


class FakeXgcmGrid:
    """Stands in for the xgcm Grid that pre-1.0 calls passed."""


FakeXgcmGrid.__module__ = "xgcm.grid"


@pytest.fixture(scope="module")
def real():
    """The small example model output with its grid, as in the pre-1.0 tests."""
    grid = xr.open_dataset(INPUT / "grid.nc")
    ds = xr.open_dataset(INPUT / "ocean_his_0001.nc")
    return ds.merge(grid, overwrite_vars=True, compat="override")


def _rho0(layout):
    """rho0 of the synthetic datasets: UCLA output carries it as an attribute, the others don't."""
    return 1027.4 if layout == "ucla" else RHO0


def _x(ds):
    """Position along xi [m] of the rho points, as the grid's pm puts them (pm varies only along xi)."""
    pm = ds.pm.isel(eta_rho=0).values
    return xr.DataArray(np.concatenate([[0.0], np.cumsum(1.0 / (0.5 * (pm[:-1] + pm[1:])))]), dims="xi_rho")


def _y(ds):
    """Position along eta [m] of the rho points, as the grid's pn puts them (pn varies only along eta)."""
    pn = ds.pn.isel(xi_rho=0).values
    return xr.DataArray(np.concatenate([[0.0], np.cumsum(1.0 / (0.5 * (pn[:-1] + pn[1:])))]), dims="eta_rho")


def _linear_rho(ds, a=RHO0, b=0.0, ax=0.0, ay=0.0):
    """rho = a + b z + ax x + ay y on the rho points and levels of ``ds``: analytic gradients."""
    rho = a + b * xroms.z(ds)
    if ax:
        rho = rho + ax * _x(ds)
    if ay:
        rho = rho + ay * _y(ds)
    rho.attrs = {}
    return rho.rename("rho")


def _linear_sig0(ds, slope=0.01):
    """sig0 = 1025 - slope z: stable, increasing linearly with depth."""
    sig0 = 1025.0 - slope * xroms.z(ds)
    sig0.attrs = {}
    return sig0.rename("sig0")


# --- density ------------------------------------------------------------------------


class TestDensity:
    @pytest.mark.parametrize("temp, salt, z, expected", PINNED)
    def test_reference_values_unchanged(self, temp, salt, z, expected):
        np.testing.assert_allclose(xroms.density(temp, salt, z), expected, rtol=1e-13)

    def test_matches_eos80_check_values(self):
        # Fofonoff & Millard (1983), UNESCO technical papers 44: S=35, p=0
        np.testing.assert_allclose(xroms.density(5.0, 35.0, 0), 1027.67547, rtol=0, atol=1e-4)
        np.testing.assert_allclose(xroms.density(25.0, 35.0, 0), 1023.34306, rtol=0, atol=1e-4)
        # pure water at 0 Celsius
        np.testing.assert_allclose(xroms.density(0.0, 0.0, 0), 999.842594, rtol=0, atol=1e-6)

    def test_example_columns(self):
        np.testing.assert_allclose(xroms.density(TEMP_COL, SALT_COL, Z_COL), RHO_COL, rtol=1e-13)
        np.testing.assert_allclose(xroms.density(TEMP_COL, SALT_COL, 0), SIG0_COL, rtol=1e-13)

    def test_numpy_in_numpy_out(self):
        out = xroms.density(TEMP_COL, SALT_COL, Z_COL)
        assert isinstance(out, np.ndarray) and out.shape == (3,)

    def test_dataarray_with_explicit_z(self, real):
        rho = xroms.density(real.temp, real.salt, xroms.z(real))
        assert rho.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        np.testing.assert_allclose(rho[0, :, 0, 0], xroms.density(TEMP_COL, SALT_COL, Z_COL))
        np.testing.assert_allclose(rho[0, :, 0, 0], RHO_COL, rtol=1e-13)

    def test_dataarray_with_grid_equals_explicit_z(self, layout):
        ds = merged(layout)
        from_grid = xroms.density(ds.temp, ds.salt, grid=ds)
        explicit = xroms.density(ds.temp, ds.salt, xroms.z(ds))
        xr.testing.assert_identical(from_grid, explicit)
        assert from_grid.dims[1:] == ("s_rho", "eta_rho", "xi_rho")

    def test_zeta_chooses_the_free_surface(self, rutgers):
        static = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers, zeta=0)
        expected = xroms.density(rutgers.temp, rutgers.salt, xroms.z(rutgers, zeta=0))
        xr.testing.assert_identical(static, expected)
        moving = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        assert not np.allclose(static.values, moving.values, rtol=0, atol=1e-12)

    def test_constant_reference_depth(self, rutgers):
        for z in (0, 0.0, -100.0):
            out = xroms.density(rutgers.temp, rutgers.salt, z)
            assert out.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
            np.testing.assert_allclose(out.isel(ocean_time=0, s_rho=2, eta_rho=1, xi_rho=3), xroms.density(
                float(rutgers.temp.isel(ocean_time=0, s_rho=2, eta_rho=1, xi_rho=3)),
                float(rutgers.salt.isel(ocean_time=0, s_rho=2, eta_rho=1, xi_rho=3)),
                z,
            ), rtol=1e-13)

    def test_needs_z_or_grid(self, rutgers):
        with pytest.raises(ValueError, match="z=.*grid="):
            xroms.density(rutgers.temp, rutgers.salt)
        with pytest.raises(ValueError, match="z=.*grid="):
            xroms.density(rutgers.temp, rutgers.salt, zeta=0)
        with pytest.raises(ValueError, match="z=.*grid="):
            xroms.density(TEMP_COL, SALT_COL)

    def test_grid_needs_a_dataarray_temp(self, rutgers):
        with pytest.raises(ValueError, match="DataArray"):
            xroms.density(TEMP_COL, SALT_COL, grid=rutgers)

    def test_z_is_not_looked_up_by_coordinate_name(self, rutgers):
        # pre-1.0 picked any coordinate starting with "z_"; a stale one must be ignored now
        stale = rutgers.temp.assign_coords(z_rho=xroms.z(rutgers) * 0 - 1.0)
        out = xroms.density(stale, rutgers.salt, grid=rutgers)
        np.testing.assert_allclose(out.values, xroms.density(rutgers.temp, rutgers.salt, grid=rutgers).values)
        with pytest.raises(ValueError, match="z="):
            xroms.density(stale, rutgers.salt)

    def test_attrs_and_name(self, rutgers):
        rho = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        assert rho.name == "rho"
        assert rho.attrs == {"name": "rho", "long_name": "density", "units": "kg/m^3"}
        assert rho.lon_rho.attrs["standard_name"] == "longitude"
        assert rho.lat_rho.attrs["standard_name"] == "latitude"

    def test_point_types_other_than_rho_use_canonical_dims(self, rutgers):
        # Rutgers names u points (eta_u, xi_u); z is computed in canonical names, and the
        # two must not broadcast against each other into an extra dimension
        t_u = xr.full_like(rutgers.u, 10.0)
        s_u = xr.full_like(rutgers.u, 35.0)
        rho = xroms.density(t_u, s_u, grid=rutgers)
        assert rho.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")

    def test_inputs_not_modified(self, rutgers):
        snapshot = rutgers.copy(deep=True)
        xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        xroms.density(rutgers.temp, rutgers.salt, 0)
        xroms.potential_density(rutgers.temp, rutgers.salt)
        xr.testing.assert_identical(rutgers, snapshot)
        assert "standard_name" not in rutgers.lon_rho.attrs
        assert "standard_name" not in rutgers.temp.lon_rho.attrs

    def test_chunked_equals_numpy(self, rutgers):
        c = chunked(rutgers)
        lazy = xroms.density(c.temp, c.salt, grid=c)
        assert lazy.chunks is not None
        np.testing.assert_allclose(lazy.values, xroms.density(rutgers.temp, rutgers.salt, grid=rutgers).values, rtol=1e-13)

    def test_legacy_xgcm_grid_rejected(self, rutgers):
        with pytest.raises(TypeError, match="xgcm"):
            xroms.density(rutgers.temp, rutgers.salt, grid=FakeXgcmGrid())


class TestPotentialDensityAndBuoyancy:
    def test_potential_density_is_density_at_zero(self, real):
        sig0 = xroms.potential_density(real.temp, real.salt)
        np.testing.assert_allclose(sig0[0, :, 0, 0], xroms.density(TEMP_COL, SALT_COL, 0))
        np.testing.assert_allclose(sig0[0, :, 0, 0], SIG0_COL, rtol=1e-13)
        assert sig0.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")

    def test_potential_density_reference_depth(self, rutgers):
        deep = xroms.potential_density(rutgers.temp, rutgers.salt, z=-1000.0)
        np.testing.assert_allclose(deep.values, xroms.density(rutgers.temp, rutgers.salt, -1000.0).values)
        assert (deep.values > xroms.potential_density(rutgers.temp, rutgers.salt).values).all()

    def test_potential_density_attrs(self, rutgers):
        sig0 = xroms.potential_density(rutgers.temp, rutgers.salt)
        assert sig0.name == "sig0"
        assert sig0.attrs == {"name": "sig0", "long_name": "potential density", "units": "kg/m^3"}

    def test_potential_density_of_numpy(self):
        np.testing.assert_allclose(xroms.potential_density(TEMP_COL, SALT_COL), SIG0_COL, rtol=1e-13)

    def test_buoyancy(self, real):
        sig0 = xroms.potential_density(real.temp, real.salt)
        buoy = xroms.buoyancy(sig0)
        np.testing.assert_allclose(buoy[0, :, 0, 0], -G * xroms.density(TEMP_COL, SALT_COL, 0) / RHO0)
        assert buoy.name == "buoyancy"
        assert buoy.attrs == {"name": "buoyancy", "long_name": "buoyancy", "units": "m/s^2"}

    def test_buoyancy_rho0_and_numpy(self):
        np.testing.assert_allclose(xroms.buoyancy(np.array([1000.0, 1030.0]), rho0=1000.0), [-G, -G * 1.03])


# --- N2 -----------------------------------------------------------------------------


class TestN2:
    def test_linear_density_gives_constant_n2(self, layout):
        ds = merged(layout)
        b = -0.02  # density increases downward
        n2 = xroms.N2(_linear_rho(ds, b=b), ds)
        assert n2.dims[1:] == ("s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(n2.isel(s_w=slice(1, -1)).values, -G * b / _rho0(layout), rtol=1e-8)
        # with the default fill, the top and bottom w levels have nothing to difference against
        assert n2.isel(s_w=[0, -1]).isnull().all()
        assert n2.isel(s_w=slice(1, -1)).notnull().all()

    def test_sign_follows_stratification(self, rutgers):
        stable = xroms.N2(_linear_rho(rutgers, b=-0.02), rutgers).isel(s_w=slice(1, -1))
        unstable = xroms.N2(_linear_rho(rutgers, b=+0.02), rutgers).isel(s_w=slice(1, -1))
        assert (stable > 0).all() and (unstable < 0).all()

    def test_boundary_options(self, rutgers):
        rho = _linear_rho(rutgers, b=-0.02)
        extended = xroms.N2(rho, rutgers, sboundary="extend")
        np.testing.assert_allclose(extended.values, G * 0.02 / RHO0, rtol=1e-8)
        zero = xroms.N2(rho, rutgers, sboundary="fill", sfill_value=0.0)
        assert (zero.isel(s_w=[0, -1]) == 0).all()
        np.testing.assert_allclose(zero.isel(s_w=slice(1, -1)).values, G * 0.02 / RHO0, rtol=1e-8)

    def test_equals_differenced_density_profile(self, rutgers):
        rho = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        n2 = xroms.N2(rho, rutgers)
        col = dict(ocean_time=1, eta_rho=3, xi_rho=4)
        r = rho.isel(col).values
        z = xroms.z(rutgers).isel(col).values
        np.testing.assert_allclose(n2.isel(col).values[1:-1], -G / RHO0 * np.diff(r) / np.diff(z), rtol=1e-10)
        # the synthetic profile is stable
        assert (n2.isel(s_w=slice(1, -1)) > 0).all()

    def test_example_column(self, real):
        # the pre-1.0 check: the mean of the two interior w levels is the centred difference
        rho = xroms.density(real.temp, real.salt, grid=real)
        r = xroms.density(TEMP_COL, SALT_COL, Z_COL)
        expected = -G * (r[2] - r[0]) / (Z_COL[2] - Z_COL[0]) / RHO0
        np.testing.assert_allclose(xroms.N2(rho, real)[0, 1:3, 0, 0].mean(), expected)

    def test_rho0_defaults_to_1025(self, rutgers):
        rho = _linear_rho(rutgers, b=-0.02)
        np.testing.assert_allclose(xroms.N2(rho, rutgers).isel(s_w=1).values, -G * -0.02 / 1025.0, rtol=1e-8)

    def test_rho0_found_in_the_grid(self):
        # UCLA output has rho0 as a global attribute
        ds = merged("ucla")
        assert ds.attrs["rho0"] == 1027.4
        rho = _linear_rho(ds, b=-0.02)
        n2 = xroms.N2(rho, ds).isel(s_w=slice(1, -1))
        np.testing.assert_allclose(n2.values, G * 0.02 / 1027.4, rtol=1e-8)
        assert not np.allclose(n2.values, G * 0.02 / 1025.0, rtol=1e-8)

    def test_rho0_variable_beats_attribute(self):
        ds = merged("ucla")
        ds["rho0"] = 1030.0
        n2 = xroms.N2(_linear_rho(ds, b=-0.02), ds).isel(s_w=slice(1, -1))
        np.testing.assert_allclose(n2.values, G * 0.02 / 1030.0, rtol=1e-8)

    def test_explicit_rho0_overrides_the_grid(self):
        ds = merged("ucla")
        n2 = xroms.N2(_linear_rho(ds, b=-0.02), ds, rho0=1000.0).isel(s_w=slice(1, -1))
        np.testing.assert_allclose(n2.values, G * 0.02 / 1000.0, rtol=1e-8)
        n2 = xroms.N2(_linear_rho(ds, b=-0.02), ds, 1000.0).isel(s_w=slice(1, -1))
        np.testing.assert_allclose(n2.values, G * 0.02 / 1000.0, rtol=1e-8)

    def test_explicit_z_needs_no_grid(self, rutgers):
        rho = _linear_rho(rutgers, b=-0.02)
        n2 = xroms.N2(rho, None, rho0=1025.0, z=xroms.z(rutgers))
        np.testing.assert_allclose(n2.isel(s_w=slice(1, -1)).values, G * 0.02 / 1025.0, rtol=1e-8)
        # no grid and nothing to find rho0 in: the default
        n2 = xroms.N2(rho, None, z=xroms.z(rutgers))
        np.testing.assert_allclose(n2.isel(s_w=slice(1, -1)).values, G * 0.02 / 1025.0, rtol=1e-8)

    def test_zeta_is_passed_on(self, rutgers):
        # rho linear in the moving z: the static z gives a different (wrong) gradient
        rho = _linear_rho(rutgers, b=-0.02)
        static = xroms.N2(rho, rutgers, zeta=0).isel(s_w=slice(1, -1))
        assert not np.allclose(static.values, G * 0.02 / RHO0, rtol=1e-3)
        np.testing.assert_allclose(static.values, xroms.N2(rho, rutgers, z=xroms.z(rutgers, zeta=0)).isel(s_w=slice(1, -1)).values)

    def test_w_level_density_lands_on_rho_levels(self, rutgers):
        rho_w = xroms.to_s_w(_linear_rho(rutgers, b=-0.02))
        n2 = xroms.N2(rho_w, rutgers)
        assert n2.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")

    def test_attrs_and_name(self, rutgers):
        n2 = xroms.N2(_linear_rho(rutgers, b=-0.02), rutgers)
        assert n2.name == "N2"
        assert n2.attrs == {
            "name": "N2",
            "long_name": "buoyancy frequency squared, or vertical buoyancy gradient",
            "units": "1/s^2",
        }

    def test_inputs_not_modified(self, rutgers):
        rho = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        snapshot, ds_snapshot = rho.copy(deep=True), rutgers.copy(deep=True)
        xroms.N2(rho, rutgers)
        xr.testing.assert_identical(rho, snapshot)
        xr.testing.assert_identical(rutgers, ds_snapshot)

    def test_chunked_equals_numpy(self, rutgers):
        c = chunked(rutgers)
        lazy = xroms.N2(xroms.density(c.temp, c.salt, grid=c), c)
        assert lazy.chunks is not None
        eager = xroms.N2(xroms.density(rutgers.temp, rutgers.salt, grid=rutgers), rutgers)
        np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-10)

    def test_argument_errors(self, rutgers):
        with pytest.raises(TypeError, match="DataArray"):
            xroms.N2(np.ones(3), rutgers)
        with pytest.raises(TypeError, match="xgcm"):
            xroms.N2(_linear_rho(rutgers, b=-0.02), FakeXgcmGrid())
        with pytest.raises(ValueError, match="grid"):
            xroms.N2(_linear_rho(rutgers, b=-0.02), None)


# --- M2 -----------------------------------------------------------------------------


class TestM2:
    def test_linear_in_x_gives_constant_m2(self, layout):
        ds = merged(layout)
        a = 2e-3
        rho = _linear_rho(ds, ax=a)
        m2 = xroms.M2(rho, ds)
        assert m2.dims == rho.dims
        np.testing.assert_allclose(m2.isel(s_rho=slice(1, -1)).values, G * a / _rho0(layout), rtol=1e-8)
        # with the default fill the top and bottom levels of the depth correction are missing
        assert m2.isel(s_rho=[0, -1]).isnull().all()

    def test_gradient_is_at_constant_depth(self, rutgers):
        # rho also depends on z, and the s-surfaces slope (h and zeta vary): the along-s
        # gradient of b*z is far from zero, the constant-depth one is exactly ax
        a, b = 2e-3, -0.02
        m2 = xroms.M2(_linear_rho(rutgers, b=b, ax=a), rutgers)
        np.testing.assert_allclose(m2.isel(s_rho=slice(1, -1)).values, G * a / RHO0, rtol=1e-7)
        along_s = xroms.ddxi(_linear_rho(rutgers, b=b), rutgers, hcoord="rho", sboundary="extend")
        assert np.abs(along_s.values).max() < 1e-12  # rho = b z: no gradient at constant depth
        assert np.abs(np.diff(xroms.z(rutgers).values, axis=-1)).max() > 0.1  # but z does vary along xi

    def test_xi_and_eta_gradients_are_combined(self, rutgers):
        a, c, b = 2e-3, 1.5e-3, -0.02
        m2 = xroms.M2(_linear_rho(rutgers, b=b, ax=a, ay=c), rutgers)
        np.testing.assert_allclose(m2.isel(s_rho=slice(1, -1)).values, G / RHO0 * np.hypot(a, c), rtol=1e-7)
        only_eta = xroms.M2(_linear_rho(rutgers, ay=c), rutgers)
        np.testing.assert_allclose(only_eta.isel(s_rho=slice(1, -1)).values, G * c / RHO0, rtol=1e-7)

    def test_sign_of_the_gradient_does_not_matter(self, rutgers):
        up = xroms.M2(_linear_rho(rutgers, ax=2e-3), rutgers)
        down = xroms.M2(_linear_rho(rutgers, ax=-2e-3), rutgers)
        np.testing.assert_allclose(up.values, down.values, rtol=1e-7)

    def test_sboundary_extend_reaches_every_level_and_point(self, rutgers):
        a = 2e-3
        m2 = xroms.M2(_linear_rho(rutgers, b=-0.02, ax=a), rutgers, sboundary="extend")
        np.testing.assert_allclose(m2.values, G * a / RHO0, rtol=1e-7)

    def test_hboundary_fill_leaves_the_domain_edges_empty(self, rutgers):
        a = 2e-3
        m2 = xroms.M2(_linear_rho(rutgers, ax=a), rutgers, hboundary="fill", sboundary="extend")
        for edge in ({"xi_rho": 0}, {"xi_rho": -1}, {"eta_rho": 0}, {"eta_rho": -1}):
            assert m2.isel(edge).isnull().all()
        np.testing.assert_allclose(m2.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1)).values, G * a / RHO0, rtol=1e-7)

    def test_rho0_found_in_the_grid(self):
        ds = merged("ucla")
        a = 2e-3
        rho = _linear_rho(ds, ax=a)
        inner = slice(1, -1)
        np.testing.assert_allclose(xroms.M2(rho, ds).isel(s_rho=inner).values, G * a / 1027.4, rtol=1e-8)
        np.testing.assert_allclose(xroms.M2(rho, ds, 1000.0).isel(s_rho=inner).values, G * a / 1000.0, rtol=1e-8)
        np.testing.assert_allclose(xroms.M2(rho, ds, rho0=1000.0).isel(s_rho=inner).values, G * a / 1000.0, rtol=1e-8)

    def test_explicit_z_and_zeta(self, rutgers):
        rho = _linear_rho(rutgers, b=-0.02, ax=2e-3)
        default = xroms.M2(rho, rutgers)
        xr.testing.assert_identical(xroms.M2(rho, rutgers, z=xroms.z(rutgers)), default)
        static = xroms.M2(rho, rutgers, zeta=0)
        xr.testing.assert_identical(static, xroms.M2(rho, rutgers, z=xroms.z(rutgers, zeta=0)))
        assert not np.allclose(static.values, default.values, rtol=1e-3, equal_nan=True)

    def test_lands_on_rho_points_from_other_points(self, rutgers):
        # density on u points: the result is moved to rho horizontally, levels are kept
        rho_u = xroms.to_u(_linear_rho(rutgers, ax=2e-3))
        m2 = xroms.M2(rho_u, rutgers)
        assert m2.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")

    def test_w_levels_are_kept(self, rutgers):
        m2 = xroms.M2(xroms.to_s_w(_linear_rho(rutgers, ax=2e-3)), rutgers)
        assert m2.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(m2.isel(s_w=slice(1, -1)).values, G * 2e-3 / RHO0, rtol=1e-7)

    def test_attrs_and_name(self, rutgers):
        m2 = xroms.M2(_linear_rho(rutgers, ax=2e-3), rutgers)
        assert m2.name == "M2"
        assert m2.attrs == {"name": "M2", "long_name": "horizontal buoyancy gradient", "units": "1/s^2"}

    def test_inputs_not_modified(self, rutgers):
        rho = xroms.density(rutgers.temp, rutgers.salt, grid=rutgers)
        snapshot, ds_snapshot = rho.copy(deep=True), rutgers.copy(deep=True)
        xroms.M2(rho, rutgers)
        xr.testing.assert_identical(rho, snapshot)
        xr.testing.assert_identical(rutgers, ds_snapshot)

    def test_chunked_equals_numpy(self, rutgers):
        c = chunked(rutgers)
        lazy = xroms.M2(xroms.density(c.temp, c.salt, grid=c), c)
        assert lazy.chunks is not None
        eager = xroms.M2(xroms.density(rutgers.temp, rutgers.salt, grid=rutgers), rutgers)
        np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-10)

    def test_argument_errors(self, rutgers):
        with pytest.raises(TypeError, match="DataArray"):
            xroms.M2(np.ones(3), rutgers)
        with pytest.raises(TypeError, match="xgcm"):
            xroms.M2(_linear_rho(rutgers, ax=2e-3), FakeXgcmGrid())
        with pytest.raises(ValueError, match="grid"):
            xroms.M2(_linear_rho(rutgers, ax=2e-3), None)


# --- mixed layer depth --------------------------------------------------------------


class TestMLD:
    def test_linear_profile_depth_is_where_the_threshold_is_crossed(self, layout):
        # sig0 = 1025 - s z: exceeds its top value by thresh a distance thresh/s below the top level
        ds = merged(layout)
        s, thresh = 0.01, 0.03
        md = xroms.mld(_linear_sig0(ds, s), ds, thresh=thresh)
        z_top = xroms.z(ds).isel(s_rho=-1)
        np.testing.assert_allclose(md.values, (-z_top + thresh / s).values, rtol=1e-9)
        assert md.dims == z_top.dims
        assert md.name == "mld"
        assert md.attrs == {"name": "mld", "long_name": "mixed layer depth", "units": "m"}

    def test_positive_and_not_deeper_than_the_water(self, layout):
        ds = merged(layout)
        md = xroms.mld(xroms.potential_density(ds.temp, ds.salt), ds)
        h = ds.h
        assert (md > 0).all()
        assert (md <= h + 1e-9).all()
        assert md.dims[1:] == ("eta_rho", "xi_rho")

    def test_fills_with_h_where_there_is_no_crossing(self, rutgers):
        # only the deeper columns are stratified enough to reach the threshold
        s, thresh = 0.01, 0.5
        z = xroms.z(rutgers)
        md = xroms.mld(_linear_sig0(rutgers, s), rutgers, thresh=thresh)
        column_range = s * (z.isel(s_rho=-1) - z.isel(s_rho=0))
        crossing = column_range > thresh
        assert crossing.any() and (~crossing).any()
        assert float(abs(column_range - thresh).min()) > 1e-6
        expected = xr.where(crossing, -z.isel(s_rho=-1) + thresh / s, rutgers.h)
        np.testing.assert_allclose(md.values, expected.transpose(*md.dims).values, rtol=1e-9)
        filled = md.where(~crossing)
        np.testing.assert_allclose(filled.values[~np.isnan(filled.values)], np.broadcast_to(rutgers.h.values, md.shape)[~np.isnan(filled.values)])

    def test_unstratified_water_is_the_full_depth_in_every_layout(self, layout):
        ds = merged(layout)
        md = xroms.mld(_linear_sig0(ds), ds, thresh=1e6)
        np.testing.assert_allclose(md.values, np.broadcast_to(ds.h.values, md.shape))

    def test_land_stays_nan_and_water_is_filled(self, with_land):
        mask = with_land.mask_rho.values == 1
        sig0 = xroms.potential_density(with_land.temp, with_land.salt)
        for thresh in (0.03, 1e6):
            md = xroms.mld(sig0, with_land, thresh=thresh)
            assert np.isnan(md.values[:, ~mask]).all()
            assert np.isfinite(md.values[:, mask]).all()
        np.testing.assert_allclose(md.values[:, mask], np.broadcast_to(with_land.h.values[mask], (2, mask.sum())))

    def test_without_mask_rho_the_surface_values_decide_what_is_water(self, with_land):
        sig0 = xroms.potential_density(with_land.temp, with_land.salt)
        for thresh in (0.03, 1e6):
            with_mask = xroms.mld(sig0, with_land, thresh=thresh)
            no_mask = xroms.mld(sig0, with_land.drop_vars("mask_rho"), thresh=thresh)
            xr.testing.assert_identical(no_mask, with_mask)

    def test_only_the_new_dimension_is_squeezed(self, rutgers):
        one = rutgers.isel(ocean_time=slice(0, 1))
        md = xroms.mld(xroms.potential_density(one.temp, one.salt), one)
        assert md.dims == ("ocean_time", "eta_rho", "xi_rho") and md.shape == (1, 9, 12)
        assert "iso" not in md.coords and "iso" not in md.dims

    def test_selected_time_against_the_full_grid(self, rutgers):
        sig0 = xroms.potential_density(rutgers.temp, rutgers.salt)
        full = xroms.mld(sig0, rutgers)
        at_t1 = xroms.mld(sig0.isel(ocean_time=1), rutgers)
        assert at_t1.dims == ("eta_rho", "xi_rho")
        np.testing.assert_allclose(at_t1.values, full.isel(ocean_time=1).values)

    def test_explicit_z_and_zeta(self, rutgers):
        sig0 = _linear_sig0(rutgers)
        default = xroms.mld(sig0, rutgers)
        xr.testing.assert_identical(xroms.mld(sig0, rutgers, z=xroms.z(rutgers)), default)
        static = xroms.mld(sig0, rutgers, zeta=0)
        xr.testing.assert_identical(static, xroms.mld(sig0, rutgers, z=xroms.z(rutgers, zeta=0)))
        assert not np.allclose(static.values, default.values, rtol=0, atol=1e-6)

    def test_example_output_threshold_on_a_level(self, real):
        # the pre-1.0 check: a threshold equal to the density step between the top two levels
        # puts the base of the mixed layer exactly on the second level from the top
        sig0 = xroms.density(real.temp, real.salt, 0)
        thresh = float(sig0[0, -2, 0, 0] - sig0[0, -1, 0, 0])
        md = xroms.mld(sig0, real, thresh=thresh)
        np.testing.assert_allclose(md[0, 0, 0], abs(Z_COL[-2]))
        np.testing.assert_allclose(md.values, np.abs(xroms.z(real).isel(s_rho=-2).values))

    def test_inputs_not_modified(self, rutgers):
        sig0 = xroms.potential_density(rutgers.temp, rutgers.salt)
        snapshot, ds_snapshot = sig0.copy(deep=True), rutgers.copy(deep=True)
        xroms.mld(sig0, rutgers)
        xr.testing.assert_identical(sig0, snapshot)
        xr.testing.assert_identical(rutgers, ds_snapshot)

    def test_chunked_equals_numpy(self, with_land):
        c = chunked(with_land)
        lazy = xroms.mld(xroms.potential_density(c.temp, c.salt), c)
        assert lazy.chunks is not None
        eager = xroms.mld(xroms.potential_density(with_land.temp, with_land.salt), with_land)
        np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-12)

    def test_argument_errors(self, rutgers):
        sig0 = _linear_sig0(rutgers)
        with pytest.raises(TypeError, match="DataArray"):
            xroms.mld(np.ones(3), rutgers)
        with pytest.raises(TypeError, match="xgcm"):
            xroms.mld(sig0, FakeXgcmGrid())
        with pytest.raises(ValueError, match="grid"):
            xroms.mld(sig0, None)
        with pytest.raises(GridMismatchError, match="'h'"):
            xroms.mld(sig0, rutgers.drop_vars("h"))
        with pytest.raises(ValueError, match="vertical"):
            xroms.mld(rutgers.zeta, rutgers)
        # the pre-1.0 signature took h and mask as arguments
        with pytest.raises(TypeError):
            xroms.mld(sig0, rutgers, rutgers.h, rutgers.mask_rho)
