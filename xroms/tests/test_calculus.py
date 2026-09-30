"""Derivatives, grid moves, weighted sums, selection and interpolation on analytic fields."""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


ATOL = 1e-12


def _layout_ds(layout, **kw):
    ds = merged(layout, **kw)
    if layout in ("ucla", "remora") and kw.get("vtransform", 2) == 1:
        pytest.skip("UCLA ROMS and REMORA use Vtransform 2 only")
    return ds


class TestHorizontalDerivatives:
    @pytest.mark.parametrize("vt", [1, 2])
    def test_ddxi_analytic_every_layout(self, layout, vt):
        ds = _layout_ds(layout, vtransform=vt)
        out = xroms.ddxi(ds.temp, ds)
        assert out.dims[1:] == ("s_rho", "eta_rho", "xi_u")
        np.testing.assert_allclose(out.values, syn.TEMP_A, rtol=1e-9, atol=ATOL)

    def test_ddeta_at_constant_depth_is_zero(self, rutgers):
        # temp does not depend on y, but z (and so temp along s-surfaces) does:
        # only the sigma-slope correction gets this right
        out = xroms.ddeta(rutgers.temp, rutgers)
        assert out.dims == ("ocean_time", "s_rho", "eta_v", "xi_rho")
        np.testing.assert_allclose(out.values, 0.0, atol=1e-12)
        along_s = xroms.ddeta(rutgers.temp.isel(s_rho=3), rutgers, along_s=True)
        assert np.abs(along_s.values).max() > 1e-6

    @pytest.mark.parametrize("vt", [1, 2])
    def test_exact_for_fields_quadratic_in_depth(self, vt):
        # salt = S0 + C z**2 has no horizontal dependence at constant depth, while its
        # along-s gradient does not vanish; the second-order vertical stencil removes
        # it exactly (v0.6.2's w-level scheme did not)
        ds = syn.make_dataset("rutgers", vtransform=vt)
        assert float(abs(xroms.ddxi(ds.salt.isel(s_rho=2), ds, along_s=True)).max()) > 1e-6
        np.testing.assert_allclose(xroms.ddxi(ds.salt, ds).values, 0.0, atol=1e-15)
        np.testing.assert_allclose(xroms.ddeta(ds.salt, ds).values, 0.0, atol=1e-15)

    def test_top_and_bottom_layers_not_halved(self, rutgers):
        rho = xroms.ddxi(rutgers.temp, rutgers, hcoord="rho")
        np.testing.assert_allclose(rho.values, syn.TEMP_A, rtol=1e-9)

    def test_hcoord_scoord_overrides(self, rutgers):
        out = xroms.ddxi(rutgers.temp, rutgers, hcoord="rho", scoord="w")
        assert out.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")

    def test_u_point_derivative_uniform_grid(self, uniform):
        out = xroms.ddxi(uniform.u, uniform)
        assert out.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        np.testing.assert_allclose(out.values, syn.U_A, rtol=1e-9)

    def test_2d_field(self, rutgers):
        out = xroms.ddxi(rutgers.zeta, rutgers)
        assert out.dims == ("ocean_time", "eta_rho", "xi_u")

    def test_single_level_requires_along_s(self, rutgers):
        with pytest.raises(ValueError, match="along_s"):
            xroms.ddxi(rutgers.temp.isel(s_rho=-1), rutgers)

    def test_needs_grid(self, rutgers):
        with pytest.raises(ValueError, match="grid"):
            xroms.ddxi(rutgers.temp)

    def test_hgrad(self, rutgers):
        dx, dy = xroms.hgrad(rutgers.temp, rutgers)
        assert dx.dims[-1] == "xi_u" and dy.dims[-2] == "eta_v"

    def test_attrs(self, rutgers):
        out = xroms.ddxi(rutgers.temp, rutgers)
        assert out.name == "dtempdxi" and "xi derivative" in out.attrs["long_name"]

    def test_chunked_equals_numpy(self, rutgers):
        c = chunked(rutgers)
        out = xroms.ddxi(c.temp, c)
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, xroms.ddxi(rutgers.temp, rutgers).values)

    def test_legacy_grid_rejected(self, rutgers):
        class FakeGrid:
            pass

        FakeGrid.__module__ = "xgcm.grid"
        with pytest.raises(TypeError, match="xgcm"):
            xroms.ddxi(rutgers.temp, FakeGrid())


class TestVerticalDerivative:
    def test_natural_on_w_including_edges(self, rutgers):
        out = xroms.ddz(rutgers.temp, rutgers)
        assert out.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(out.values, syn.TEMP_B, rtol=1e-9)

    def test_same_levels_second_order(self, rutgers):
        can = C.canonicalize(rutgers)
        out = xroms.ddz(rutgers.salt, rutgers, scoord="s_rho")
        expected = 2 * syn.SALT_C * xroms.z(rutgers)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-9, atol=1e-12)
        assert out.dims == can.salt.dims

    def test_fill_boundary(self, rutgers):
        out = xroms.ddz(rutgers.temp, rutgers, sboundary="fill")
        assert np.isnan(out.isel(s_w=0)).all() and np.isnan(out.isel(s_w=-1)).all()
        zero = xroms.ddz(rutgers.temp, rutgers, sboundary="fill", sfill_value=0.0)
        assert (zero.isel(s_w=0) == 0).all()

    def test_explicit_z(self, rutgers):
        z = xroms.z(rutgers)
        np.testing.assert_allclose(xroms.ddz(rutgers.temp, z=z).values, syn.TEMP_B, rtol=1e-9)


class TestGridMoves:
    def test_no_grid_needed(self, rutgers):
        assert xroms.to_rho(rutgers.u).dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        assert xroms.to_psi(rutgers.temp).dims == ("ocean_time", "s_rho", "eta_v", "xi_u")
        assert xroms.to_s_w(rutgers.temp).dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        assert xroms.to_grid(rutgers.temp, "u", "w").dims == ("ocean_time", "s_w", "eta_rho", "xi_u")

    def test_legacy_positional_grid_rejected(self, rutgers):
        with pytest.raises(TypeError, match="xroms 1.0"):
            xroms.to_rho(rutgers.u, object())

    def test_order(self, rutgers):
        t = C.canonicalize(rutgers).temp.transpose("xi_rho", "s_rho", "ocean_time", "eta_rho")
        assert xroms.order(t).dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")


class TestGridSums:
    def test_gridsum_z_is_depth(self, rutgers):
        can = C.canonicalize(rutgers)
        ones = xr.ones_like(can.temp)
        total = (can.h + can.zeta).transpose("ocean_time", "eta_rho", "xi_rho")
        np.testing.assert_allclose(xroms.gridsum(ones, rutgers, "Z").values, total.values)

    def test_gridmean_multiple_dims(self, rutgers):
        can = C.canonicalize(rutgers)
        out = xroms.gridmean(can.temp, rutgers, ("X", "Y"))
        w = xroms.dA(rutgers)
        expected = (can.temp * w).sum(("eta_rho", "xi_rho")) / w.sum()
        np.testing.assert_allclose(out.values, expected.values)

    def test_gridmean_ignores_nan(self, with_land):
        can = C.canonicalize(with_land)
        out = xroms.gridmean(can.temp, with_land, ("X", "Y"))
        assert np.isfinite(out.values).all()


class TestSubset:
    def test_staggers_consistent(self, rutgers):
        sub = xroms.subset(rutgers, X=slice(2, 9), Y=slice(1, 7))
        assert sub.sizes["xi_rho"] == 7 and sub.sizes["xi_u"] == 6 and sub.sizes["xi_v"] == 7 and sub.sizes["xi_psi"] == 6
        assert sub.sizes["eta_rho"] == 6 and sub.sizes["eta_v"] == 5 and sub.sizes["eta_u"] == 6

    def test_open_ended_and_negative(self, rutgers):
        assert xroms.subset(rutgers, X=slice(2, None)).sizes["xi_rho"] == 10
        assert xroms.subset(rutgers, X=slice(-5, -1)).sizes["xi_rho"] == 4

    def test_strided_rejected(self, rutgers):
        with pytest.raises(ValueError, match="strided"):
            xroms.subset(rutgers, X=slice(2, 9, 2))

    def test_subset_derivative_matches_full(self, rutgers):
        sub = xroms.subset(rutgers, X=slice(2, 9), Y=slice(1, 7))
        full = xroms.ddxi(rutgers.temp, rutgers)
        np.testing.assert_allclose(xroms.ddxi(sub.temp, sub).values, full.isel(eta_rho=slice(1, 7), xi_u=slice(2, 8)).values)

    def test_halo_and_trim_exact(self, rutgers):
        can = C.canonicalize(rutgers)
        halo = xroms.subset(rutgers, X=slice(3, 8), halo=1)
        assert halo.attrs["xroms_halo"] == [1, 1, 0, 0]
        u_rho = xroms.trim(xroms.to_rho(C.canonicalize(halo).u).to_dataset(name="u_rho").assign_attrs(halo.attrs))
        np.testing.assert_allclose(u_rho.u_rho.values, xroms.to_rho(can.u).isel(xi_rho=slice(3, 8)).values)

    def test_halo_clipped_at_edge(self, rutgers):
        halo = xroms.subset(rutgers, X=slice(0, 5), halo=2)
        assert halo.attrs["xroms_halo"][:2] == [0, 2]


class TestSelection:
    def test_argsel2d_methods(self, rutgers):
        lon, lat = float(rutgers.lon_rho[4, 7]), float(rutgers.lat_rho[4, 7])
        assert xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, lon, lat) == (4, 7)
        assert xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, lon, lat, method="geodesic") == (4, 7)
        many = xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, [lon, float(rutgers.lon_rho[1, 1])], [lat, float(rutgers.lat_rho[1, 1])])
        assert list(many[0]) == [4, 1] and list(many[1]) == [7, 1]

    def test_cartesian(self, remora):
        x0, y0 = float(remora.x_rho[2, 3]), float(remora.y_rho[2, 3])
        assert xroms.argsel2d(remora.x_rho, remora.y_rho, x0, y0, method="cartesian") == (2, 3)

    def test_sel2d(self, rutgers):
        out = xroms.sel2d(rutgers.temp, rutgers.lon_rho, rutgers.lat_rho, float(rutgers.lon_rho[4, 7]), float(rutgers.lat_rho[4, 7]))
        assert out.dims == ("ocean_time", "s_rho")


class TestSlices:
    def test_zslice_mean_sea_level(self, rutgers):
        out = xroms.zslice(rutgers.temp, [-10.0, -5.0], rutgers)
        assert out.dims == ("ocean_time", "z", "eta_rho", "xi_rho")
        can = C.canonicalize(rutgers)
        x = xr.DataArray(syn._canonical(2, 6, 9, 12, 2, 20.0, 5.0, 2.0, False, 0.0, False)[1]["x_rho"], dims=("eta_rho", "xi_rho"))
        expected = syn.TEMP_A * x + syn.TEMP_B * (-10.0) + syn.TEMP_0
        np.testing.assert_allclose(out.sel(z=-10.0).isel(ocean_time=0).values, expected.values)
        assert out.z.attrs["vertical_reference"] == "mean_sea_level"

    def test_zslice_below_surface(self, rutgers):
        out = xroms.zslice(rutgers.temp, [5.0], rutgers, reference="surface", positive="down")
        can = C.canonicalize(rutgers)
        x = xr.DataArray(syn._canonical(2, 6, 9, 12, 2, 20.0, 5.0, 2.0, False, 0.0, False)[1]["x_rho"], dims=("eta_rho", "xi_rho"))
        expected = syn.TEMP_A * x + syn.TEMP_B * (can.zeta - 5.0) + syn.TEMP_0
        np.testing.assert_allclose(out.isel(z=0).values, expected.transpose("ocean_time", "eta_rho", "xi_rho").values)
        assert out.z.attrs["standard_name"] == "depth"

    def test_zslice_infers_reference_from_metadata(self, rutgers):
        depths = xr.DataArray([5.0], dims="z", attrs={"standard_name": "depth", "units": "m"})
        a = xroms.zslice(rutgers.temp, depths, rutgers)
        b = xroms.zslice(rutgers.temp, [5.0], rutgers, reference="surface", positive="down")
        np.testing.assert_allclose(a.values, b.values)

    def test_isoslice_on_other_variable_and_nearest(self, rutgers):
        can = C.canonicalize(rutgers)
        on_temp = xroms.isoslice(can.salt, [float(can.temp.mean())], can.temp)
        assert "temp" in on_temp.dims
        near = xroms.isoslice(can.temp, [-10.0], xroms.z(rutgers), method="nearest")
        assert near.dims == ("ocean_time", "z_rho", "eta_rho", "xi_rho")

    def test_isoslice_mask_edges(self, rutgers):
        can = C.canonicalize(rutgers)
        assert np.isnan(xroms.isoslice(can.temp, [-500.0], xroms.z(rutgers)).values).all()

    def test_xisoslice_kept(self, rutgers):
        can = C.canonicalize(rutgers)
        out = xroms.xisoslice(xroms.z(rutgers), -10.0, can.temp, "s_rho")
        np.testing.assert_allclose(out.values, xroms.zslice(rutgers.temp, [-10.0], rutgers).isel(z=0).values, atol=1e-9)
