"""Vector rotation: ``rotate_vectors``, ``grid_to_earth`` and ``earth_to_grid``."""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


RHO = ("ocean_time", "s_rho", "eta_rho", "xi_rho")


class TestRotateVectors:
    def test_numbers(self):
        u, v = 1, 0
        assert (u, v) == xroms.rotate_vectors(u, v, 0, isradians=True, reference="xaxis")
        np.testing.assert_allclose((0, 1), xroms.rotate_vectors(u, v, 90, isradians=False, reference="xaxis"), atol=1e-15)
        np.testing.assert_allclose(
            xroms.rotate_vectors(u, v, np.pi / 2), xroms.rotate_vectors(u, v, 90, isradians=False)
        )

    def test_compass_reference(self):
        u, v = 1, 0
        assert (u, v) == xroms.rotate_vectors(u, v, 0, reference="compass")
        np.testing.assert_allclose((0, -1), xroms.rotate_vectors(u, v, 90, isradians=False, reference="compass"), atol=1e-15)
        np.testing.assert_allclose(
            xroms.rotate_vectors(u, v, 180, isradians=False, reference="compass"),
            xroms.rotate_vectors(u, v, 180, isradians=False, reference="xaxis"),
            atol=1e-15,
        )

    def test_none_reference_is_xaxis_and_typos_raise(self):
        np.testing.assert_allclose(xroms.rotate_vectors(1, 0, 0.3, reference=None), xroms.rotate_vectors(1, 0, 0.3))
        with pytest.raises(ValueError, match="compass"):
            xroms.rotate_vectors(1, 0, 0.3, reference="Compass")

    def test_dataarrays_are_moved_to_rho_first(self, rutgers):
        can = C.canonicalize(rutgers)
        x, y = xroms.rotate_vectors(can.u, can.v, 0.0)
        assert x.dims == y.dims == RHO
        np.testing.assert_allclose(x.values, xroms.to_rho(can.u).values)
        assert x.name == "u_rot" and y.name == "v_rot"

    def test_hcoord_none_leaves_positions_alone(self, rutgers):
        can = C.canonicalize(rutgers)
        x, y = xroms.rotate_vectors(can.temp, can.salt, np.pi / 2, hcoord=None)
        assert x.dims == can.temp.dims
        np.testing.assert_allclose(x.values, -can.salt.values, atol=1e-12)
        np.testing.assert_allclose(y.values, can.temp.values, atol=1e-12)

    def test_inputs_are_not_modified(self, rutgers):
        angle = rutgers.angle.copy() + 0.2
        before = angle.copy()
        xroms.rotate_vectors(rutgers.temp, rutgers.salt, angle, reference="compass", isradians=False)
        xr.testing.assert_identical(angle, before)

    def test_attrs(self, rutgers):
        attrs = {"x": {"name": "along", "units": "m/s"}, "y": {"name": "across", "units": "m/s"}}
        x, y = xroms.rotate_vectors(rutgers.temp, rutgers.salt, 0.1, attrs=attrs)
        assert (x.name, y.name) == ("along", "across") and x.attrs["units"] == "m/s"
        attrs["x"]["units"] = "changed"
        assert x.attrs["units"] == "m/s"  # the caller's dict is copied
        with pytest.raises(KeyError):
            xroms.rotate_vectors(rutgers.temp, rutgers.salt, 0.1, attrs={"x": {}})
        with pytest.raises(KeyError, match="name"):
            xroms.rotate_vectors(rutgers.temp, rutgers.salt, 0.1, attrs={"x": {}, "y": {"name": "b"}})

    def test_xgrid_is_rejected(self, rutgers):
        with pytest.raises(TypeError, match="xgrid"):
            xroms.rotate_vectors(rutgers.u, rutgers.v, 0.0, xgrid=object())
        # for numbers too, which never need a grid
        with pytest.raises(TypeError, match="rotate_vectors no longer needs xgrid; remove it"):
            xroms.rotate_vectors(1.0, 0.0, 0.3, xgrid=object())

    @pytest.mark.parametrize("keyword", ["isradian", "hbounary", "hcoords", "sboundary", "reference_frame"])
    def test_unknown_keywords_raise(self, rutgers, keyword):
        # they used to be swallowed (for numbers and arrays, which never reach to_grid):
        # rotate_vectors(1., 0., 90, isradian=False) rotated by 90 radians
        can = C.canonicalize(rutgers)
        for x, y, angle in (
            (1.0, 0.0, 90),
            (np.ones(3), np.zeros(3), np.full(3, 90.0)),
            (can.temp, can.salt, 90),
        ):
            with pytest.raises(TypeError, match=keyword):
                xroms.rotate_vectors(x, y, angle, **{keyword: False})

    def test_typo_does_not_rotate_by_the_wrong_unit(self):
        with pytest.raises(TypeError, match="isradian"):
            xroms.rotate_vectors(1.0, 0.0, 90, isradian=False)
        np.testing.assert_allclose(xroms.rotate_vectors(1.0, 0.0, 90, isradians=False), (0, 1), atol=1e-15)

    def test_boundary_keywords_go_to_the_move(self, rutgers):
        can = C.canonicalize(rutgers)
        extend = xroms.rotate_vectors(can.u, can.v, 0.0)[0]
        np.testing.assert_allclose(extend.values, xroms.to_rho(can.u).values)
        # u is averaged along xi and v along eta, and both feed each rotated component
        nan = xroms.rotate_vectors(can.u, can.v, 0.0, hboundary="fill")[0]
        for edge in ({"xi_rho": 0}, {"xi_rho": -1}, {"eta_rho": 0}, {"eta_rho": -1}):
            assert np.isnan(nan.isel(edge).values).all()
        assert np.isfinite(nan.isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1)).values).all()
        zero = xroms.rotate_vectors(can.u, can.v, 0.0, hboundary="fill", hfill_value=0.0)[0]
        np.testing.assert_allclose(zero.values, xroms.to_rho(can.u, hboundary="fill", hfill_value=0.0).values)
        with pytest.raises(ValueError, match="boundary"):
            xroms.rotate_vectors(can.u, can.v, 0.0, hboundary="wrap")

    def test_results_are_ordered(self):
        # a mean flow rotated by an angle that varies in time: the time dimension comes from the angle
        x = xr.DataArray(np.ones((9, 12)), dims=("eta_rho", "xi_rho"))
        y = xr.DataArray(np.zeros((9, 12)), dims=("eta_rho", "xi_rho"))
        angle = xr.DataArray([0.0, np.pi / 2], dims="ocean_time")
        xrot, yrot = xroms.rotate_vectors(x, y, angle)
        assert xrot.dims == yrot.dims == ("ocean_time", "eta_rho", "xi_rho")
        np.testing.assert_allclose(xrot.isel(ocean_time=1).values, 0.0, atol=1e-15)
        np.testing.assert_allclose(yrot.isel(ocean_time=1).values, 1.0)
        # other dimensions follow the four
        member = xr.DataArray([0.0, 1.0], dims="member")
        assert xroms.rotate_vectors(x, y, member)[0].dims == ("eta_rho", "xi_rho", "member")


class TestGridToEarth:
    @pytest.mark.parametrize("angle", [0.0, np.pi / 2, 0.4])
    def test_values(self, angle):
        ds = syn.make_dataset("rutgers", angle=angle)
        can = C.canonicalize(ds)
        east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        u, v = xroms.to_rho(can.u.fillna(0)), xroms.to_rho(can.v.fillna(0))
        np.testing.assert_allclose(east.values, (u * np.cos(angle) - v * np.sin(angle)).values, atol=1e-14)
        np.testing.assert_allclose(north.values, (u * np.sin(angle) + v * np.cos(angle)).values, atol=1e-14)

    def test_names_dims_and_positions(self, rutgers):
        east, north = xroms.grid_to_earth(rutgers.u, rutgers.v, rutgers.angle)
        assert east.dims == north.dims == RHO
        assert (east.name, north.name) == ("east", "north")
        assert east.attrs["standard_name"] == "eastward_sea_water_velocity"
        assert north.attrs["standard_name"] == "northward_sea_water_velocity"
        east_psi, _ = xroms.grid_to_earth(rutgers.u, rutgers.v, rutgers.angle, hcoord="psi")
        assert east_psi.dims == ("ocean_time", "s_rho", "eta_v", "xi_u")

    def test_land_is_zero_not_spread(self, with_land):
        east, _ = xroms.grid_to_earth(with_land.u, with_land.v, with_land.angle)
        assert np.isfinite(east.values).all()
        land = C.canonicalize(with_land).mask_rho == 0
        assert (east.where(land) .fillna(0) == 0).all()

    def test_requires_dataarrays(self):
        with pytest.raises(TypeError):
            xroms.grid_to_earth(1.0, 0.0, 0.0)

    def test_swapped_components_raise_naming_the_positions(self, layout):
        # u on v points and v on u points: rotated and averaged to rho without a word before
        ds = merged(layout)
        with pytest.raises(ValueError, match="u is on v points and v is on u points") as err:
            xroms.grid_to_earth(ds.v, ds.u, ds.angle)
        assert "not swapped" in str(err.value)

    def test_each_component_is_checked(self, rutgers):
        with pytest.raises(ValueError, match="u is on v points and v is on v points"):
            xroms.grid_to_earth(rutgers.v, rutgers.v, rutgers.angle)
        with pytest.raises(ValueError, match="u is on u points and v is on u points"):
            xroms.grid_to_earth(rutgers.u, rutgers.u, rutgers.angle)
        with pytest.raises(ValueError, match="u is on v points and v is on rho points"):
            xroms.grid_to_earth(rutgers.v, xroms.to_rho(rutgers.v), rutgers.angle)
        with pytest.raises(ValueError, match="u is on rho points and v is on u points"):
            xroms.grid_to_earth(xroms.to_rho(rutgers.u), rutgers.u, rutgers.angle)

    @pytest.mark.parametrize("angle", [0.0, 0.4])
    def test_velocities_already_on_rho_points_are_accepted(self, angle):
        ds = syn.make_dataset("rutgers", angle=angle)
        east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        east_rho, north_rho = xroms.grid_to_earth(xroms.to_rho(ds.u), xroms.to_rho(ds.v), ds.angle)
        np.testing.assert_allclose(east_rho.values, east.values, atol=1e-14)
        np.testing.assert_allclose(north_rho.values, north.values, atol=1e-14)

    def test_a_position_the_dims_do_not_give_is_accepted(self, rutgers):
        # a section at one xi has no xi dim to place u with
        can = C.canonicalize(rutgers)
        east, _ = xroms.grid_to_earth(can.u.isel(xi_u=3), can.v.isel(xi_rho=3), rutgers.angle.isel(xi_rho=3))
        assert np.isfinite(east.values).all()

    def test_chunked_stays_lazy(self, rutgers):
        c = chunked(rutgers)
        east, north = xroms.grid_to_earth(c.u, c.v, c.angle)
        assert east.chunks is not None
        np.testing.assert_allclose(east.values, xroms.grid_to_earth(rutgers.u, rutgers.v, rutgers.angle)[0].values)


class TestEarthToGrid:
    def test_round_trip_interior_is_exact_for_linear_fields(self):
        ds = syn.make_dataset("rutgers", angle=0.3, uniform=True)
        can = C.canonicalize(ds)
        east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        u, v = xroms.earth_to_grid(east, north, ds.angle)
        assert u.dims == can.u.dims and v.dims == can.v.dims
        assert (u.name, v.name) == ("u", "v")
        inner = dict(xi_u=slice(1, -1), eta_v=slice(1, -1))
        np.testing.assert_allclose(u.isel(xi_u=inner["xi_u"]).values, can.u.isel(xi_u=inner["xi_u"]).values, rtol=1e-12)
        np.testing.assert_allclose(v.isel(eta_v=inner["eta_v"]).values, can.v.isel(eta_v=inner["eta_v"]).values, rtol=1e-12)

    def test_hcoord(self, rutgers):
        east, north = xroms.grid_to_earth(rutgers.u, rutgers.v, rutgers.angle)
        u, v = xroms.earth_to_grid(east, north, rutgers.angle, hcoord="rho")
        assert u.dims == v.dims == RHO
        with pytest.raises(ValueError, match="native"):
            xroms.earth_to_grid(east, north, rutgers.angle, hcoord="w")

    def test_requires_dataarrays(self):
        with pytest.raises(TypeError):
            xroms.earth_to_grid(1.0, 0.0, 0.0)
