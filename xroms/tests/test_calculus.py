"""Derivatives, grid moves, weighted sums, selection and interpolation on analytic fields."""

import sys
import warnings

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

    def test_gridsum_units_gain_a_metre_per_dim(self, rutgers):
        units = rutgers.u.attrs["units"]
        assert xroms.gridsum(rutgers.u, rutgers, "Z").attrs["units"] == f"{units} m"
        assert xroms.gridsum(rutgers.u, rutgers, ("Z", "Y")).attrs["units"] == f"{units} m2"
        assert xroms.gridsum(rutgers.temp, rutgers, ("X", "Y", "Z")).attrs["units"] == f"{rutgers.temp.attrs['units']} m3"
        bare = rutgers.u.copy()
        bare.attrs = {}
        assert "units" not in xroms.gridsum(bare, rutgers, "Z").attrs
        assert xroms.gridmean(rutgers.u, rutgers, "Z").attrs["units"] == units

    def test_dims_in_the_datasets_own_alias_naming(self, rutgers):
        """A Rutgers u variable's own dims (eta_u) name the same axes as the canonical ones."""
        xr.testing.assert_identical(xroms.gridsum(rutgers.u, rutgers, "eta_u"), xroms.gridsum(rutgers.u, rutgers, "Y"))
        xr.testing.assert_identical(xroms.gridmean(rutgers.v, rutgers, ("s_rho", "xi_v")), xroms.gridmean(rutgers.v, rutgers, ("Z", "X")))
        xr.testing.assert_identical(rutgers.xroms.gridsum("u", "eta_u"), rutgers.xroms.gridsum("u", "Y"))


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
        many = xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, [lon, float(rutgers.lon_rho[1, 1])], [lat, float(rutgers.lat_rho[1, 1])])
        assert list(many[0]) == [4, 1] and list(many[1]) == [7, 1]

    def test_argsel2d_either_longitude_convention(self, rutgers):
        # great-circle distance is periodic in longitude: a target in the other convention
        # (a -144 station against a 0-360 grid, ocean-skill's _nearest_indices case) finds the same cell
        lon, lat = float(rutgers.lon_rho[4, 7]), float(rutgers.lat_rho[4, 7])
        for shift in (360.0, -360.0):
            assert xroms.argsel2d(rutgers.lon_rho + shift, rutgers.lat_rho, lon, lat) == (4, 7)
            assert xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, lon + shift, lat) == (4, 7)

    def test_argsel2d_geodesic(self, rutgers):
        pytest.importorskip("pyproj")  # optional: pip install "xroms[geodesic]"
        lon, lat = float(rutgers.lon_rho[4, 7]), float(rutgers.lat_rho[4, 7])
        assert xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, lon, lat, method="geodesic") == (4, 7)

    def test_geodesic_without_pyproj_says_how_to_install_it(self, rutgers, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyproj", None)  # makes `import pyproj` fail
        with pytest.raises(ModuleNotFoundError, match=r"xroms\[geodesic\].*conda install pyproj"):
            xroms.argsel2d(rutgers.lon_rho, rutgers.lat_rho, -89.9, 28.0, method="geodesic")

    def test_cartesian(self, remora):
        x0, y0 = float(remora.x_rho[2, 3]), float(remora.y_rho[2, 3])
        assert xroms.argsel2d(remora.x_rho, remora.y_rho, x0, y0, method="cartesian") == (2, 3)

    def test_sel2d(self, rutgers):
        out = xroms.sel2d(rutgers.temp, rutgers.lon_rho, rutgers.lat_rho, float(rutgers.lon_rho[4, 7]), float(rutgers.lat_rho[4, 7]))
        assert out.dims == ("ocean_time", "s_rho")

    @pytest.mark.parametrize(
        "make, lon, lat, point",
        [
            (lambda ds: xroms.ddxi(ds.temp, ds), "lon_u", "lat_u", {"eta_rho": 4, "xi_u": 7}),
            (lambda ds: xroms.ddeta(ds.temp, ds), "lon_v", "lat_v", {"eta_v": 3, "xi_rho": 6}),
            (lambda ds: xroms.to_psi(ds.temp), "lon_psi", "lat_psi", {"eta_v": 3, "xi_u": 5}),
        ],
        ids=["u", "v", "psi"],
    )
    def test_sel2d_takes_canonical_results_with_rutgers_coords(self, rutgers, make, lon, lat, point):
        # results come back with canonical dims (eta_rho, xi_u, ...) while Rutgers
        # files name the dims of their coordinates eta_u, xi_v, eta_psi, xi_psi
        out = make(rutgers)
        lons, lats = rutgers[lon], rutgers[lat]
        assert set(lons.dims) - set(out.dims)
        can_lons, can_lats = C.canonicalize(lons), C.canonicalize(lats)
        lon0, lat0 = float(can_lons.isel(point)), float(can_lats.isel(point))
        got = xroms.sel2d(out, lons, lats, lon0, lat0)
        xr.testing.assert_identical(got, out.isel(point))
        xr.testing.assert_identical(rutgers.xroms.sel2d(out, lon0, lat0), got)
        assert xroms.argsel2d(lons, lats, lon0, lat0) == tuple(point[d] for d in can_lons.dims)
        # several points at once
        first = {d: 1 for d in point}
        lon1, lat1 = float(can_lons.isel(first)), float(can_lats.isel(first))
        many = xroms.sel2d(out, lons, lats, [lon0, lon1], [lat0, lat1])
        assert many.dims[-1] == "points" and many.sizes["points"] == 2
        np.testing.assert_array_equal(many.isel(points=0).values, out.isel(point).values)
        np.testing.assert_array_equal(many.isel(points=1).values, out.isel(first).values)

    def test_sel2d_takes_rutgers_variables_with_canonical_coords(self, rutgers):
        can = C.canonicalize(rutgers)
        lon0, lat0 = float(can.lon_u[4, 7]), float(can.lat_u[4, 7])
        got = xroms.sel2d(rutgers.u, can.lon_u, can.lat_u, lon0, lat0)
        np.testing.assert_array_equal(got.values, can.u.isel(eta_rho=4, xi_u=7).values)

    def test_sel2d_cartesian_remora_results(self, remora):
        out = xroms.ddxi(remora.temp, remora)
        x0, y0 = float(remora.x_u[2, 3]), float(remora.y_u[2, 3])
        got = xroms.sel2d(out, remora.x_u, remora.y_u, x0, y0, method="cartesian")
        xr.testing.assert_identical(got, out.isel(eta_rho=2, xi_u=3))


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

    def test_xisoslice_without_a_crossing_is_nan_without_warnings(self, rutgers):
        c = chunked(C.canonicalize(rutgers))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = xroms.xisoslice(xroms.z(c), -1e4, c.temp, "s_rho").compute()
        assert out.isnull().all()


class TestInputsAreMatchedToTheGrid:
    """Regression tests for the input-matching gaps found while porting (2026-09-30)."""

    @pytest.mark.parametrize("times", [[0], slice(1, 2), [1, 0]])
    def test_time_subsets_that_keep_the_time_dim(self, rutgers, times):
        can = C.canonicalize(rutgers)
        full = xroms.ddxi(can.temp, rutgers)
        part = xroms.ddxi(can.temp.isel(ocean_time=times), rutgers)
        xr.testing.assert_allclose(part, full.isel(ocean_time=times))

    def test_time_subsets_without_time_labels_need_decoding(self):
        ds = merged("ucla")
        with pytest.raises(xroms._align.GridMismatchError, match="decode_time"):
            xroms.ddxi(ds.temp.isel(time=[0]), ds)
        decoded = xroms.decode_time(ds)
        part = xroms.ddxi(decoded.temp.isel(time=[0]), decoded)
        np.testing.assert_allclose(part.values, xroms.ddxi(decoded.temp, decoded).isel(time=[0]).values)

    def test_z_at_rho_points_is_averaged_onto_the_variable(self, rutgers):
        can = C.canonicalize(rutgers)
        xr.testing.assert_allclose(xroms.ddeta(can.u, rutgers, z=xroms.z(rutgers)), xroms.ddeta(can.u, rutgers))

    def test_z_at_the_wrong_position_or_levels_is_explained(self, rutgers):
        can = C.canonicalize(rutgers)
        with pytest.raises(ValueError, match="v points but 'u' is at u points"):
            xroms.ddeta(can.u, rutgers, z=xroms.z(rutgers, hcoord="v"))
        with pytest.raises(ValueError, match="s_w levels"):
            xroms.ddz(can.temp, rutgers, z=xroms.z(rutgers, scoord="s_w"))

    def test_isoslice_broadcasts_the_variable_over_the_iso_arrays_dims(self, rutgers):
        # static depths (no time) sliced on a time-varying field
        can = C.canonicalize(rutgers)
        z0 = xroms.z(rutgers, zeta=0)
        out = xroms.isoslice(z0, [can.temp.mean().item()], can.temp, new_dim="temp")
        assert out.dims[0] == "ocean_time" and "ocean_time" in out.coords
        for t in range(can.sizes["ocean_time"]):
            one = xroms.isoslice(z0, [can.temp.mean().item()], can.temp.isel(ocean_time=t), new_dim="temp")
            np.testing.assert_allclose(out.isel(ocean_time=t).values, one.values)


# --- regression tests for the code-review findings on the calculus module (2026-09-30) ------------

HORIZONTAL_DERIVATIVES = {
    "ddxi": xroms.ddxi,
    "ddeta": xroms.ddeta,
    "hgrad": xroms.hgrad,
    "hgrad-xi": lambda var, grid, **kw: xroms.hgrad(var, grid, which="xi", **kw),
    "hgrad-eta": lambda var, grid, **kw: xroms.hgrad(var, grid, which="eta", **kw),
}


def _metric_along_xi(ds):
    """``1 / dx`` between neighbouring rho points, i.e. at u points."""
    pm = C.canonicalize(ds).pm.values
    return 0.5 * (pm[:, :-1] + pm[:, 1:])


class TestSingleSelectedLevel:
    """A level cut out of a 3-D field has no vertical dim left to correct the slope with.

    Rutgers and REMORA files keep a scalar s coordinate behind ``isel``; UCLA and CROCO
    files have no s labels, so nothing but the name gives the level away there.
    """

    @pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
    @pytest.mark.parametrize("func", HORIZONTAL_DERIVATIVES.values(), ids=HORIZONTAL_DERIVATIVES)
    def test_raises_unless_along_s_on_every_layout(self, layout, func, lazy):
        ds = chunked(merged(layout)) if lazy else merged(layout)
        with pytest.raises(ValueError, match="along_s"):
            func(ds.temp.isel(s_rho=-1), ds)

    @pytest.mark.parametrize("select", [xroms.surface, xroms.bottom])
    def test_surface_and_bottom_are_single_levels(self, layout, select):
        ds = merged(layout)
        for func in (xroms.ddxi, xroms.ddeta):
            with pytest.raises(ValueError, match="along_s"):
                func(select(ds.temp), ds)

    def test_along_s_is_the_plain_difference_over_the_metric(self, layout):
        ds = merged(layout)
        bottom = ds.temp.isel(s_rho=0)
        got = xroms.ddxi(bottom, ds, along_s=True)
        assert got.dims[-2:] == ("eta_rho", "xi_u")
        np.testing.assert_allclose(got.values, np.diff(bottom.values, axis=-1) * _metric_along_xi(ds), rtol=1e-10, atol=1e-14)
        # which is not the derivative at constant depth that a silent return passed it off as
        assert np.abs(got.values - syn.TEMP_A).max() > 1e-4

    def test_reductions_over_s_keep_the_name_and_take_along_s(self, layout):
        # a depth mean is named like the 3-D variable it came from but has no levels to correct
        ds = merged(layout)
        mean = ds.temp.mean("s_rho")
        got = xroms.ddxi(mean, ds, along_s=True)
        np.testing.assert_allclose(got.values, np.diff(mean.values, axis=-1) * _metric_along_xi(ds), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("name", ["zeta", "h"])
    def test_variables_that_never_had_levels_still_work(self, layout, name):
        ds = merged(layout)
        out = xroms.ddxi(ds[name], ds)
        assert out.dims[-2:] == ("eta_rho", "xi_u")

    def test_a_z_slice_is_at_constant_depth_already(self, layout):
        # same name as the 3-D variable, no s dim, but a z dim of its own: not a selected level
        ds = merged(layout)
        zs = xroms.zslice(ds.temp, [-10.0, -5.0], ds)
        np.testing.assert_allclose(xroms.ddxi(zs, ds).values, syn.TEMP_A, rtol=1e-9, atol=ATOL)
        np.testing.assert_allclose(xroms.ddeta(zs, ds).values, 0.0, atol=1e-12)


class TestNeedsTwoLevels:
    """The chain rule and the vertical derivative need two levels; one used to give all NaN."""

    CUTS = ["one-level dataset", "isel list", "isel slice"]

    @staticmethod
    def thin(layout, cut):
        """``(ds, temp)`` with a single s_rho level, cut the way ``cut`` says."""
        if cut == "one-level dataset":
            ds = merged(layout, N=1)
            return ds, ds.temp
        ds = merged(layout)
        return ds, ds.temp.isel(s_rho=[5] if cut == "isel list" else slice(5, 6))

    @pytest.mark.parametrize("cut", CUTS)
    @pytest.mark.parametrize("func", HORIZONTAL_DERIVATIVES.values(), ids=HORIZONTAL_DERIVATIVES)
    def test_chain_rule_raises(self, layout, cut, func):
        ds, var = self.thin(layout, cut)
        with pytest.raises(ValueError, match=r"chain rule needs at least 2 vertical levels; pass along_s=True"):
            func(var, ds)

    @pytest.mark.parametrize("cut", CUTS)
    def test_along_s_takes_the_derivative_along_the_single_layer(self, layout, cut):
        ds, var = self.thin(layout, cut)
        out = xroms.ddxi(var, ds, along_s=True)
        assert out.sizes["s_rho"] == 1 and out.dims[-3:] == ("s_rho", "eta_rho", "xi_u")
        np.testing.assert_allclose(out.values, np.diff(var.values, axis=-1) * _metric_along_xi(ds), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("cut", CUTS)
    @pytest.mark.parametrize("scoord", [None, "s_rho"])
    def test_ddz_raises(self, layout, cut, scoord):
        ds, var = self.thin(layout, cut)
        with pytest.raises(ValueError, match="ddz needs at least 2 levels"):
            xroms.ddz(var, ds, scoord=scoord)

    def test_ddz_raises_on_a_single_w_level(self, layout):
        ds = merged(layout)
        with pytest.raises(ValueError, match="ddz needs at least 2 levels"):
            xroms.ddz(xroms.to_s_w(ds.temp).isel(s_w=[3]), ds)

    def test_functions_built_on_ddz_and_the_chain_rule_raise_too(self, layout):
        ds = merged(layout, N=1)
        with pytest.raises(ValueError, match="at least 2"):
            xroms.N2(ds.salt, ds)
        with pytest.raises(ValueError, match="at least 2"):
            xroms.relative_vorticity(ds.u, ds.v, ds)

    def test_two_levels_are_enough(self, layout):
        ds = merged(layout, N=2)
        np.testing.assert_allclose(xroms.ddxi(ds.temp, ds).values, syn.TEMP_A, rtol=1e-9, atol=ATOL)
        np.testing.assert_allclose(xroms.ddz(ds.temp, ds).values, syn.TEMP_B, rtol=1e-9)
        np.testing.assert_allclose(xroms.ddz(ds.temp, ds, scoord="s_rho").values, syn.TEMP_B, rtol=1e-9)


def _linear_at_staggered_points(ds, axis, slope):
    """A field on the u (``axis="X"``) or v points that is exactly linear in the grid's own position.

    The two staggered points on either side of a rho point are ``1/pm`` (``1/pn``) apart
    *at that rho point*, so the difference over that spacing is ``slope`` at every rho
    point. (Positions averaged from the synthetic dataset's x and y give the slope only
    to about 2e-4 on stretched grids: fine for judging the interior, but it leaves too
    little room to tell an edge that is off by the 2-3 % spacing ratio from one that is not.)
    """
    can = C.canonicalize(ds)
    if axis == "X":
        step = 1.0 / can.pm.values
        position = np.concatenate([np.zeros_like(step[:, :1]), np.cumsum(step[:, 1:-1], axis=1)], axis=1)
        return xr.DataArray(slope * position, dims=("eta_rho", "xi_u"))
    step = 1.0 / can.pn.values
    position = np.concatenate([np.zeros_like(step[:1]), np.cumsum(step[1:-1], axis=0)], axis=0)
    return xr.DataArray(slope * position, dims=("eta_v", "xi_rho"))


class TestEdgesAreOneSidedDerivatives:
    """``extend`` copies the nearest derivative to the edge, not the nearest difference.

    Moving from u (v) to rho points leaves the first and last rho point without a
    neighbour pair. On a stretched grid the spacing there is not the neighbour's, so a
    copied difference gave the edge a slope off by the ratio of the two spacings.
    """

    AXES = {"X": (xroms.ddxi, "u", "xi_rho"), "Y": (xroms.ddeta, "v", "eta_rho")}

    @pytest.mark.parametrize("vertical", [False, True], ids=["2d", "3d"])
    @pytest.mark.parametrize("axis", ["X", "Y"])
    def test_linear_fields_have_the_analytic_slope_at_the_edges_too(self, layout, axis, vertical):
        func, hcoord, dim = self.AXES[axis]
        ds = merged(layout, uniform=False)
        slope = syn.TEMP_A
        field = _linear_at_staggered_points(ds, axis, slope)
        if vertical:
            # linear in z as well: dq/ds and dz/ds must both be one-sided slopes or the
            # chain rule no longer cancels them at the edges
            field = syn.TEMP_B * xroms.z(ds, hcoord=hcoord) + field
        out = func(field, ds)
        assert out.dims[-2:] == ("eta_rho", "xi_rho")
        np.testing.assert_allclose(out.isel({dim: slice(1, -1)}).values, slope, rtol=1e-9, atol=ATOL)
        np.testing.assert_allclose(out.isel({dim: [0, -1]}).values, slope, rtol=1e-9, atol=ATOL)

    @pytest.mark.parametrize("axis", ["X", "Y"])
    def test_2d_edges_equal_the_neighbour_for_any_field(self, layout, axis):
        func, hcoord, dim = self.AXES[axis]
        ds = merged(layout, uniform=False)
        out = func(xroms.to_grid(ds.h, hcoord), ds)  # the sloping bathymetry: not linear in anything
        np.testing.assert_array_equal(out.isel({dim: 0}).values, out.isel({dim: 1}).values)
        np.testing.assert_array_equal(out.isel({dim: -1}).values, out.isel({dim: -2}).values)

    @pytest.mark.parametrize("axis", ["X", "Y"])
    def test_lazy_input_gets_the_same_edges(self, layout, axis):
        func, hcoord, dim = self.AXES[axis]
        ds = chunked(merged(layout, uniform=False))
        field = syn.TEMP_B * xroms.z(ds, hcoord=hcoord) + _linear_at_staggered_points(ds, axis, syn.TEMP_A)
        out = func(field.chunk({d: 4 for d in field.dims}), ds)
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, syn.TEMP_A, rtol=1e-9, atol=ATOL)

    def test_fill_value_is_the_derivative_at_the_edges(self, layout):
        ds = merged(layout, uniform=False)
        field = _linear_at_staggered_points(ds, "X", syn.TEMP_A)
        out = xroms.ddxi(field, ds, hboundary="fill", hfill_value=5.0)
        np.testing.assert_array_equal(out.isel(xi_rho=[0, -1]).values, 5.0)
        np.testing.assert_allclose(out.isel(xi_rho=slice(1, -1)).values, syn.TEMP_A, rtol=1e-9, atol=ATOL)
        assert np.isnan(xroms.ddxi(field, ds, hboundary="fill").isel(xi_rho=[0, -1]).values).all()


class TestInputsAreNotModified:
    """xarray up to 2025.7 stripped the attrs of merged coordinates in place (``apply_ufunc``)."""

    CALLS = {
        "ddxi": lambda ds: xroms.ddxi(ds.temp, ds),
        "ddeta": lambda ds: xroms.ddeta(ds.temp, ds),
        "ddxi-u": lambda ds: xroms.ddxi(ds.u, ds),
        "ddeta-v": lambda ds: xroms.ddeta(ds.v, ds),
        "hgrad": lambda ds: xroms.hgrad(ds.temp, ds),
        "ddz": lambda ds: xroms.ddz(ds.temp, ds),
        "ddz-same-levels": lambda ds: xroms.ddz(ds.temp, ds, scoord="rho"),
        "accessor": lambda ds: ds.xroms.ddxi("temp"),
    }

    @pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
    @pytest.mark.parametrize("call", CALLS)
    def test_coordinates_and_attrs_are_left_alone(self, layout, call, lazy):
        ds = chunked(merged(layout)) if lazy else merged(layout)
        before = ds.copy(deep=True)
        self.CALLS[call](ds)
        xr.testing.assert_identical(ds, before)


def test_ddz_fill_value_is_the_derivative_at_the_edges(rutgers):
    # sfill_value is the value of the derivative there, not a difference divided by dz
    out = xroms.ddz(rutgers.temp, rutgers, sboundary="fill", sfill_value=7.0)
    assert (out.isel(s_w=[0, -1]) == 7.0).all()
    assert np.isfinite(out.isel(s_w=slice(1, -1))).all()
