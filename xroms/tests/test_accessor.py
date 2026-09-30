"""The stateless ``ds.xroms`` / ``da.xroms`` accessors."""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


class TestStateless:
    def test_holds_only_a_reference(self, rutgers):
        assert rutgers.xroms._obj is rutgers
        assert set(vars(rutgers.xroms)) == {"_obj"}

    def test_never_writes_into_the_dataset(self, rutgers):
        before = rutgers.copy(deep=True)
        rutgers.xroms.speed, rutgers.xroms.ddxi("temp"), rutgers.xroms.z_rho, rutgers.xroms.vort
        xr.testing.assert_identical(rutgers, before)

    def test_sees_in_place_edits(self, rutgers):
        s1 = rutgers.xroms.speed.values.copy()
        rutgers["u"] = rutgers.u * 10
        rutgers["v"] = rutgers.v * 10
        np.testing.assert_allclose(rutgers.xroms.speed.values, 10 * s1)


class TestNamingAndCoords:
    def test_results_use_dataset_naming(self, rutgers):
        u_like = rutgers.xroms.to_grid("temp", hcoord="u")
        assert u_like.dims == rutgers.u.dims
        assert (rutgers.u + u_like).dims == rutgers.u.dims
        assert rutgers.xroms.ddxi("temp").dims == rutgers.u.dims
        assert rutgers.xroms.vort.dims == ("ocean_time", "s_rho", "eta_psi", "xi_psi")

    def test_pure_functions_are_canonical(self, rutgers):
        assert xroms.to_u(rutgers.temp).dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
        can = xroms.canonicalize(rutgers)
        assert (can.u + xroms.to_u(can.temp)).dims == can.u.dims

    def test_position_coords_attached(self, rutgers):
        out = rutgers.xroms.ddxi("temp")
        assert {"lon_u", "lat_u", "ocean_time", "s_rho"} <= set(out.coords)
        assert "lon_rho" not in out.coords

    def test_ucla_lonlat_data_vars_become_coords(self, ucla):
        out, grid = ucla
        ds = xr.merge([out, grid.drop_vars("spherical")])
        assert {"lon_rho", "lat_rho"} <= set(ds.xroms.speed.coords)


class TestGridFacts:
    def test_z_and_metrics(self, rutgers):
        np.testing.assert_allclose(rutgers.xroms.z_rho.values, xroms.z(rutgers).values)
        assert rutgers.xroms.z_w.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        assert rutgers.xroms.z(hcoord="u").dims == rutgers.u.dims
        assert rutgers.xroms.dz(scoord="w").dims[1] == "s_w"
        assert rutgers.xroms.dx("u").dims == ("eta_u", "xi_u")
        assert rutgers.xroms.dA("psi").dims == ("eta_psi", "xi_psi")
        assert rutgers.xroms.dV().dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        assert rutgers.xroms.vertical_params.Vtransform == 2

    def test_assign_z_is_a_new_dataset(self, rutgers):
        withz = rutgers.xroms.assign_z()
        assert {"z_rho", "z_w"} <= set(withz.coords) and "z_rho" not in rutgers.coords
        assert "z_rho" in withz.temp.coords

    def test_xgcm_grid(self, rutgers):
        import xgcm

        assert isinstance(rutgers.xroms.xgcm_grid(), xgcm.Grid)


class TestCalculations:
    def test_methods_accept_names_or_arrays(self, rutgers):
        xr.testing.assert_allclose(rutgers.xroms.ddxi("temp"), rutgers.xroms.ddxi(rutgers.temp))

    def test_zslice_gridmean_depth_average(self, rutgers):
        assert rutgers.xroms.zslice("temp", [-5.0]).dims == ("ocean_time", "z", "eta_rho", "xi_rho")
        assert rutgers.xroms.gridmean("temp", ("X", "Y")).dims == ("ocean_time", "s_rho")
        assert rutgers.xroms.depth_average("temp").dims == ("ocean_time", "eta_rho", "xi_rho")
        assert rutgers.xroms.surface("temp").dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_separate_grid(self, ucla):
        out, grid = ucla
        merged_grid = xr.merge([out, grid.drop_vars("spherical")])
        a = out.xroms.ddxi("temp", grid=merged_grid)
        np.testing.assert_allclose(a.values, syn.TEMP_A, rtol=1e-9)

    def test_subset_and_selection(self, rutgers, remora):
        sub = rutgers.xroms.subset(X=slice(2, 8), halo=1)
        assert sub.attrs["xroms_halo"] == [1, 1, 0, 0]
        lon, lat = float(rutgers.lon_rho[3, 4]), float(rutgers.lat_rho[3, 4])
        assert rutgers.xroms.argsel2d(lon, lat) == (3, 4)
        assert rutgers.xroms.sel2d("temp", lon, lat).dims == ("ocean_time", "s_rho")
        x0, y0 = float(remora.x_rho[2, 5]), float(remora.y_rho[2, 5])
        assert remora.xroms.argsel2d(x0, y0) == (2, 5)

    def test_every_layout(self, layout):
        ds = merged(layout)
        np.testing.assert_allclose(ds.xroms.ddxi("temp").values, syn.TEMP_A, rtol=1e-9)
        assert ds.xroms.speed.dims[-2:] == ("eta_rho", "xi_rho")

    def test_chunked(self, rutgers):
        c = chunked(rutgers)
        out = c.xroms.convergence
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, rutgers.xroms.convergence.values)


class TestPhysics:
    def test_speed_ke_eke(self, rutgers):
        s = rutgers.xroms.speed
        np.testing.assert_allclose(s.values, xroms.speed(rutgers.u, rutgers.v).values)
        np.testing.assert_allclose(rutgers.xroms.KE.values, 0.5 * 1025.0 * s.values**2)
        assert rutgers.xroms.ug.dims == ("ocean_time", "eta_u", "xi_u")
        assert rutgers.xroms.vg.dims == ("ocean_time", "eta_v", "xi_v")
        assert rutgers.xroms.EKE.dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_rotation(self):
        ds = syn.make_dataset("rutgers", angle=np.pi / 2)
        east, north = ds.xroms.eastnorth
        np.testing.assert_allclose(north.values, xroms.to_rho(ds.u.fillna(0)).values, atol=1e-12)
        np.testing.assert_allclose(east.values, -xroms.to_rho(ds.v.fillna(0)).values, atol=1e-12)
        rot = ds.xroms.east_rotated(90, reference="compass", isradians=False, name="along")
        assert rot.name == "along"

    def test_u_v_from_earth_components(self, rutgers):
        east, north = rutgers.xroms.eastnorth
        only_earth = rutgers.drop_vars(["u", "v"]).assign(u_eastward=east, v_northward=north)
        assert only_earth.xroms.u.dims == rutgers.u.dims
        assert only_earth.xroms.find_horizontal_velocities() == ("u_eastward", "v_northward")

    def test_shear_vorticity_convergence(self, uniform):
        assert uniform.xroms.dudz.dims == ("ocean_time", "s_w", "eta_u", "xi_u")
        assert uniform.xroms.vertical_shear.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(uniform.xroms.vort.values, 0.0, atol=1e-12)
        np.testing.assert_allclose(uniform.xroms.convergence.values, syn.U_A + syn.V_A, rtol=1e-9)
        assert uniform.xroms.convergence_norm.dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_density_family(self, rutgers):
        assert rutgers.xroms.rho.dims == rutgers.temp.dims
        assert rutgers.xroms.sig0.dims == rutgers.temp.dims
        assert rutgers.xroms.N2.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        assert rutgers.xroms.M2.dims == rutgers.temp.dims
        assert rutgers.xroms.ertel.dims == rutgers.temp.dims
        mld = rutgers.xroms.mld()
        assert mld.dims == ("ocean_time", "eta_rho", "xi_rho") and (mld >= 0).all()


class TestDataArrayAccessor:
    def test_grid_free_operations(self, rutgers):
        assert rutgers.u.xroms.to_rho().dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        assert rutgers.temp.xroms.to_grid("psi", "w").dims == ("ocean_time", "s_w", "eta_v", "xi_u")
        assert rutgers.temp.transpose("xi_rho", ...).xroms.order().dims[0] == "ocean_time"

    def test_selection_uses_own_coords(self, rutgers):
        lon, lat = float(rutgers.lon_u[2, 3]), float(rutgers.lat_u[2, 3])
        assert rutgers.u.xroms.argsel2d(lon, lat) == (2, 3)
        assert rutgers.u.xroms.sel2d(lon, lat).dims == ("ocean_time", "s_rho")

    def test_isoslice(self, rutgers):
        z = xroms.z(rutgers)
        out = C.canonicalize(rutgers).temp.xroms.isoslice([-5.0], z)
        assert "z_rho" in out.dims
