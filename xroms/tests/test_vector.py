"""Vector rotation: ``rotate_vectors``, ``grid_to_earth`` and ``earth_to_grid``."""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked


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
