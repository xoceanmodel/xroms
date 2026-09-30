"""Unit tests for the stateless core: xgcm engine, conventions, vertical, metrics, alignment."""

import numpy as np
import pytest
import xarray as xr

from xroms import _xgcm, conventions as C, metrics as M, vertical as V
from xroms._align import GridMismatchError, select_like
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


@pytest.fixture
def canon():
    ds, ex = syn._canonical(2, 6, 9, 12, 2, 20.0, 5.0, 2.0, False, 0.0, False)
    return ds, ex


def _edge_pad(a, dim):
    p = a.variable.pad({dim: (1, 1)}, mode="edge")
    return 0.5 * (p.isel({dim: slice(None, -1)}).data + p.isel({dim: slice(1, None)}).data)


# --- engine -------------------------------------------------------------------


class TestEngine:
    def test_interp_inner_to_center_extend(self, canon):
        ds, _ = canon
        out = _xgcm.interp(ds.u, "X")
        assert out.dims == ("time", "s_rho", "eta_rho", "xi_rho")
        np.testing.assert_allclose(out.values, _edge_pad(ds.u, "xi_u"))

    def test_interp_center_to_inner(self, canon):
        ds, _ = canon
        out = _xgcm.interp(ds.temp, "X")
        np.testing.assert_allclose(out.values, 0.5 * (ds.temp.values[..., :-1] + ds.temp.values[..., 1:]))

    def test_interp_fill_boundary(self, canon):
        ds, _ = canon
        out = _xgcm.interp(ds.u, "X", boundary="fill")
        assert np.isnan(out.isel(xi_rho=0)).all() and np.isnan(out.isel(xi_rho=-1)).all()

    def test_labels_carried(self, canon):
        ds, _ = canon
        u = ds.u.assign_coords(xi_u=np.arange(11) + 4)
        assert (_xgcm.interp(u, "X").xi_rho.values == np.arange(12) + 4).all()
        t = ds.temp.assign_coords(xi_rho=np.arange(12) + 4)
        assert (_xgcm.interp(t, "X").xi_u.values == np.arange(11) + 4).all()

    def test_coords_spanning_operated_dim_dropped(self, canon):
        ds, _ = canon
        t = ds.temp.assign_coords(lon_rho=ds.lon_rho, h=ds.h)
        out = _xgcm.interp(t, "X")
        assert "lon_rho" not in out.coords and "h" not in out.coords

    def test_diff_edges_are_one_sided_not_zero(self, canon):
        ds, _ = canon
        d = _xgcm.diff(ds.temp, "Z")
        interior = np.diff(ds.temp.values, axis=1)
        np.testing.assert_allclose(d.values[:, 1:-1], interior)
        np.testing.assert_allclose(d.values[:, 0], interior[:, 0])
        np.testing.assert_allclose(d.values[:, -1], interior[:, -1])
        assert (np.abs(d.values[:, 0]) > 0).all()

    def test_diff_fill_value_imposes_condition(self, canon):
        ds, _ = canon
        d = _xgcm.diff(ds.temp, "Z", boundary="fill", fill_value=0.0)
        assert (d.values[:, 0] == 0).all() and (d.values[:, -1] == 0).all()
        assert np.isnan(_xgcm.diff(ds.temp, "Z", boundary="fill").values[:, 0]).all()

    def test_bad_boundary(self, canon):
        with pytest.raises(ValueError, match="boundary"):
            _xgcm.interp(canon[0].u, "X", boundary="periodic")

    def test_missing_axis(self, canon):
        with pytest.raises(ValueError, match="no 'X' dimension"):
            _xgcm.interp(canon[0].zeta.isel(xi_rho=0), "X")

    @pytest.mark.parametrize("axis,var", [("X", "u"), ("X", "temp"), ("Y", "v"), ("Z", "temp")])
    def test_chunked_matches_numpy_and_restores_chunks(self, canon, axis, var):
        ds, _ = canon
        da = ds[var]
        c = chunked(da.to_dataset())[var]
        out = _xgcm.interp(c, axis)
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, _xgcm.interp(da, axis).values)
        dim, from_center = _xgcm.axis_dim(da, axis)
        for other in da.dims:
            if other != dim:
                assert out.chunksizes[other] == c.chunksizes[other]
        new_dim = _xgcm.target_dim(axis, from_center)
        assert len(out.chunksizes[new_dim]) == len(c.chunksizes[dim])

    def test_chained_moves_keep_chunks(self, canon):
        ds, _ = canon
        c = chunked(ds[["temp"]]).temp
        psi = _xgcm.interp(_xgcm.interp(c, "X"), "Y")

        def shrunk(chunks):  # last chunk loses one point; an emptied chunk merges away
            out = list(chunks)
            out[-1] -= 1
            if out[-1] == 0:
                out.pop()
            return tuple(out)

        assert psi.chunksizes["xi_u"] == shrunk(c.chunksizes["xi_rho"])
        assert psi.chunksizes["eta_v"] == shrunk(c.chunksizes["eta_rho"])
        assert psi.chunksizes["s_rho"] == c.chunksizes["s_rho"]

    def test_transform_1d_targets(self, canon):
        ds, ex = canon
        zr = xr.DataArray(ex["z_rho"], dims=ds.temp.dims)
        out = _xgcm.transform(ds.temp, [-10.0, -5.0], zr, "s_rho", new_dim="depth")
        assert "depth" in out.dims and (out.depth.values == [-10.0, -5.0]).all()
        expected = syn.TEMP_A * ex["x_rho"] + syn.TEMP_B * (-10.0) + syn.TEMP_0
        np.testing.assert_allclose(out.sel(depth=-10.0).isel(time=0).values, expected)

    def test_transform_mask_edges(self, canon):
        ds, ex = canon
        zr = xr.DataArray(ex["z_rho"], dims=ds.temp.dims)
        masked = _xgcm.transform(ds.temp, [-1000.0], zr, "s_rho")
        assert np.isnan(masked.values).all()
        held = _xgcm.transform(ds.temp, [-1000.0], zr, "s_rho", mask_edges=False)
        np.testing.assert_allclose(held.squeeze().values, ds.temp.isel(s_rho=0).values)

    def test_transform_nd_targets(self, canon):
        ds, ex = canon
        zr = xr.DataArray(ex["z_rho"], dims=ds.temp.dims)
        target = (zr.isel(time=0) * 0.5).rename({"s_rho": "lev"})
        out = _xgcm.transform(ds.temp, target, zr, "s_rho", new_dim="lev")
        assert out.sizes["lev"] == 6

    def test_grid_for(self, canon):
        g = _xgcm.grid_for(canon[0])
        assert set(g.axes) == {"X", "Y"}


# --- conventions ------------------------------------------------------------------


class TestConventions:
    def test_vertical_params_all_layouts(self, layout, canon):
        _, ex = canon
        p = C.vertical_params(merged(layout))
        assert p.Vtransform == 2 and p.hc == 20.0
        np.testing.assert_allclose(p.Cs_r.values, ex["cs_r"])
        np.testing.assert_allclose(p.sigma_w.values, ex["s_w"])
        assert p.Cs_r.dims == ("s_rho",) and p.Cs_w.dims == ("s_w",)

    def test_vertical_params_vtransform1(self):
        assert C.vertical_params(syn.make_dataset("rutgers", vtransform=1)).Vtransform == 1
        assert C.vertical_params(syn.make_dataset("croco", vtransform=1)).Vtransform == 1

    def test_vertical_params_missing_vtransform_raises(self, rutgers):
        with pytest.raises(ValueError, match="Vtransform"):
            C.vertical_params(rutgers.drop_vars("Vtransform"))
        assert C.vertical_params(rutgers.drop_vars("Vtransform"), Vtransform=2).Vtransform == 2

    def test_vertical_params_computes_cs_from_theta(self, rutgers, canon):
        p = C.vertical_params(rutgers.drop_vars(["Cs_r", "Cs_w"]))
        np.testing.assert_allclose(p.Cs_r.values, canon[1]["cs_r"])

    def test_vertical_params_length_check(self, ucla):
        out, grid = ucla
        with pytest.raises(ValueError, match="levels"):
            C.vertical_params(out.isel(s_rho=slice(0, 3)))

    def test_vertical_params_romstools_grid(self, ucla_romstools, canon):
        out, grid = ucla_romstools
        p = C.vertical_params(out.drop_attrs() if hasattr(out, "drop_attrs") else out.assign_attrs({}), grid)
        np.testing.assert_allclose(p.sigma_r.values, canon[1]["s_rho"])

    def test_stretching_matches(self, canon):
        _, ex = canon
        np.testing.assert_allclose(C.stretching(ex["s_rho"], 5.0, 2.0).values, ex["cs_r"])
        with pytest.raises(ValueError):
            C.stretching(ex["s_rho"], 5.0, 2.0, Vstretching=3)

    def test_sigma_levels(self):
        np.testing.assert_allclose(C.sigma_levels(4, "rho").values, [-0.875, -0.625, -0.375, -0.125])
        np.testing.assert_allclose(C.sigma_levels(4, "w").values, [-1, -0.75, -0.5, -0.25, 0])

    def test_canonicalize_and_rename_like(self, rutgers):
        can = C.canonicalize(rutgers)
        assert {"eta_u", "xi_v", "eta_psi", "xi_psi"}.isdisjoint(can.dims)
        assert can.u.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
        back = C.rename_like(can.u, rutgers)
        assert back.dims == rutgers.u.dims
        assert (rutgers.u + back).dims == rutgers.u.dims
        assert C.rename_like(can.temp, rutgers).dims == can.temp.dims
        assert C.convention(rutgers) == "rutgers" and C.convention(can) == "canonical"

    def test_psi_sized_as_corners(self):
        ds = xr.Dataset({"lon_psi": (("eta_psi", "xi_psi"), np.zeros((4, 5))), "h": (("eta_rho", "xi_rho"), np.zeros((3, 4)))})
        assert set(C.canonicalize(ds).lon_psi.dims) == {"eta_vert", "xi_vert"}

    def test_positions(self, rutgers):
        can = C.canonicalize(rutgers)
        assert [C.hposition(can[v]) for v in ("temp", "u", "v", "lon_psi")] == ["rho", "u", "v", "psi"]
        assert C.hposition(rutgers.u) == "u"
        assert C.vposition(can.temp) == "s_rho" and C.vposition(can.zeta) is None
        lab = C.canonicalize(C.add_cf_attrs(rutgers, index_coords=True))
        assert C.hposition(lab.u.isel(xi_u=3)) == "u"

    def test_decode_time(self, ucla, remora):
        out, _ = ucla
        dec = C.decode_time(out)
        assert np.issubdtype(dec.time.dtype, np.datetime64) and "ocean_time" in dec.data_vars
        assert dec.time.values[1] - dec.time.values[0] == np.timedelta64(1, "D")
        assert dec.temp.isel(time=0).time.ndim == 0
        with pytest.raises(ValueError, match="disagrees"):
            C.decode_time(out, reference_date="1999-01-01")
        assert C.decode_time(remora) is remora  # already decoded

    def test_decode_time_needs_epoch(self, croco):
        with pytest.raises(ValueError, match="reference_date"):
            C.decode_time(croco)
        assert np.issubdtype(C.decode_time(croco, reference_date="2010-01-01").time.dtype, np.datetime64)

    def test_rho0(self, rutgers, ucla):
        assert C.rho0(ucla[0]) == 1027.4 and C.rho0(rutgers) == 1025.0

    def test_sgrid(self, remora, rutgers):
        assert C.sgrid_topology(remora)["variable"] == "grid"
        assert C.sgrid_topology(rutgers) is None
        dec = C.add_cf_attrs(rutgers)
        assert C.sgrid_topology(dec) is not None and "grid" not in rutgers

    def test_add_cf_attrs_index_coords_only_when_asked(self, rutgers):
        assert "xi_rho" not in C.add_cf_attrs(rutgers).coords
        lab = C.add_cf_attrs(rutgers, index_coords=True)
        assert (lab.xi_rho.values == np.arange(12)).all()
        relabelled = C.add_cf_attrs(lab.isel(xi_rho=slice(3, 6)), index_coords=True)
        assert (relabelled.xi_rho.values == [3, 4, 5]).all()  # never renumbered

    def test_horizontal_coords(self, rutgers, remora):
        assert C.horizontal_coords(rutgers, "u") == ("lon_u", "lat_u")
        assert C.horizontal_coords(remora, "rho") == ("x_rho", "y_rho")
        assert C.is_spherical(rutgers) and not C.is_spherical(remora)


# --- vertical -----------------------------------------------------------------------


class TestVertical:
    @pytest.mark.parametrize("vt", [1, 2])
    def test_z_exact_every_layout(self, layout, vt):
        ds = merged(layout, vtransform=vt)
        _, ex = syn._canonical(2, 6, 9, 12, vt, 20.0, 5.0, 2.0, False, 0.0, False)
        # UCLA ROMS and REMORA only use Vtransform 2, which xroms assumes for them;
        # the synthetic Vtransform-1 variants must say so explicitly
        explicit = vt if layout in ("ucla", "remora") else None
        np.testing.assert_allclose(V.z(ds, Vtransform=explicit).values, ex["z_rho"], rtol=0, atol=1e-10)

    def test_z_w_bounds(self, rutgers):
        can = C.canonicalize(rutgers)
        zw = V.z(rutgers, scoord="w")
        np.testing.assert_allclose(zw.isel(s_w=-1).values, can.zeta.values)
        np.testing.assert_allclose(zw.isel(s_w=0).values, np.broadcast_to(-can.h.values, zw.isel(s_w=0).shape))

    def test_references_and_labels(self, rutgers):
        can = C.canonicalize(rutgers)
        zr = V.z(rutgers)
        below = V.z(rutgers, reference="surface", positive="down")
        np.testing.assert_allclose(below.values, (can.zeta - zr).transpose(*below.dims).values)
        above = V.z(rutgers, reference="bottom")
        np.testing.assert_allclose(above.values, (zr + can.h).transpose(*above.dims).values)
        assert below.attrs["standard_name"] == "depth" and below.attrs["positive"] == "down"
        assert above.attrs["standard_name"] == "height_above_sea_floor"
        assert zr.attrs["vertical_reference"] == "mean_sea_level" and zr.attrs["units"] == "m"
        with pytest.raises(ValueError):
            V.z(rutgers, reference="geoid")

    def test_infer_reference(self):
        assert V.infer_reference({"standard_name": "depth"}) == ("surface", "down")
        assert V.infer_reference({"standard_name": "height_above_sea_floor"}) == ("bottom", "up")
        assert V.infer_reference({"positive": "down"}) == ("mean_sea_level", "down")
        assert V.infer_reference({}) == ("mean_sea_level", "up")
        with pytest.raises(ValueError, match="contradicts"):
            V.infer_reference({"standard_name": "depth", "positive": "up"})

    def test_zeta_options(self, rutgers):
        static = V.z(rutgers, zeta=0)
        assert "ocean_time" not in static.dims
        mean = V.z(rutgers, zeta="mean")
        assert "ocean_time" not in mean.dims
        explicit = V.z(rutgers, zeta=C.canonicalize(rutgers).zeta.isel(ocean_time=1))
        np.testing.assert_allclose(explicit.values, V.z(rutgers).isel(ocean_time=1).values)

    def test_methods_at_u(self, rutgers):
        avg = V.z(rutgers, hcoord="u")
        inp = V.z(rutgers, hcoord="u", method="interp_inputs")
        assert avg.dims == inp.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
        assert not np.allclose(avg.values, inp.values)  # nonlinear in h for Vtransform 2
        np.testing.assert_allclose(avg.values, inp.values, rtol=1e-2)

    def test_dz(self, rutgers):
        can = C.canonicalize(rutgers)
        total = (can.h + can.zeta).transpose("ocean_time", "eta_rho", "xi_rho").values
        np.testing.assert_allclose(V.dz(rutgers).sum("s_rho").values, total)
        dzw = V.dz(rutgers, scoord="w")
        np.testing.assert_allclose(dzw.sum("s_w").values, total)
        assert (dzw > 0).all()
        zr, zw = V.z(rutgers), V.z(rutgers, scoord="w")
        np.testing.assert_allclose(dzw.isel(s_w=0).values, (zr.isel(s_rho=0) - zw.isel(s_w=0)).values)

    def test_depth_band_and_average(self, rutgers):
        below_surface = V.z(rutgers, scoord="w", reference="surface")
        np.testing.assert_allclose(V.depth_band_weights(below_surface, 0, 10).sum("s_rho").values, 10.0)
        # relative to mean sea level the band holds less water where zeta < 0
        msl = V.depth_band_weights(V.z(rutgers, scoord="w"), 0, 10).sum("s_rho")
        zeta = C.canonicalize(rutgers).zeta.transpose(*msl.dims)
        np.testing.assert_allclose(msl.values, np.minimum(10.0, 10.0 + zeta.values))
        can = C.canonicalize(rutgers)
        full = V.depth_average(can.temp, rutgers)
        expected = (can.temp * V.dz(rutgers)).sum("s_rho") / V.dz(rutgers).sum("s_rho")
        np.testing.assert_allclose(full.values, expected.transpose(*full.dims).values)
        top10 = V.depth_average(can.temp, rutgers, shallow=0, deep=10, reference="surface")
        assert top10.dims == full.dims

    def test_surface_bottom(self, rutgers):
        can = C.canonicalize(rutgers)
        assert (V.surface(can.temp).values == can.temp.isel(s_rho=-1).values).all()
        assert (V.bottom(can.temp).values == can.temp.isel(s_rho=0).values).all()

    def test_z_like_single_time_and_subset(self, rutgers):
        lab = C.canonicalize(C.add_cf_attrs(rutgers, index_coords=True))
        t1 = lab.temp.isel(ocean_time=1)
        np.testing.assert_allclose(V.z_like(t1, lab).values, V.z(lab).isel(ocean_time=1).values)
        usub = lab.u.isel(xi_u=slice(3, 7), eta_rho=slice(2, 5))
        np.testing.assert_allclose(V.z_like(usub, lab).values, V.z(lab, hcoord="u").isel(xi_u=slice(3, 7), eta_rho=slice(2, 5)).values)

    def test_z_like_time_mean_guidance(self, rutgers):
        can = C.canonicalize(rutgers)
        with pytest.raises(GridMismatchError, match="zeta='mean'"):
            V.z_like(can.temp.mean("ocean_time"), rutgers)
        assert "ocean_time" not in V.z_like(can.temp.mean("ocean_time"), rutgers, zeta="mean").dims

    def test_lazy_on_dask(self, rutgers):
        c = chunked(rutgers)
        assert V.z(c).chunks is not None and V.dz(c, scoord="w").chunks is not None


# --- metrics ------------------------------------------------------------------------


class TestMetrics:
    def test_dx_dy(self, rutgers):
        can = C.canonicalize(rutgers)
        np.testing.assert_allclose(M.dx(rutgers).values, 1 / can.pm.values)
        np.testing.assert_allclose(M.dx(rutgers, "u").values, 1 / (0.5 * (can.pm.values[:, :-1] + can.pm.values[:, 1:])))
        assert M.dy(rutgers, "v").dims == ("eta_v", "xi_rho") and M.dA(rutgers, "psi").dims == ("eta_v", "xi_u")

    def test_dV_sums_to_volume(self, rutgers):
        can = C.canonicalize(rutgers)
        vol = (M.dA(rutgers) * (can.h + can.zeta)).sum(("eta_rho", "xi_rho"))
        np.testing.assert_allclose(M.dV(rutgers).sum(("s_rho", "eta_rho", "xi_rho")).values, vol.values)

    def test_mask_at(self, with_land):
        for pos in ("u", "v", "psi"):
            assert (M.mask_at(with_land.mask_rho, pos).values == with_land[f"mask_{pos}"].values).all()
        tv = M.mask_at(syn.make_dataset("remora").mask_rho, "u")
        assert tv.dims[0] == "ocean_time"

    def test_nominal_resolution(self, rutgers):
        assert M.nominal_resolution(rutgers) == pytest.approx(1400.0, rel=1e-6)
        assert 0 < M.nominal_resolution(rutgers, "degrees") < 0.1

    def test_missing_grid_vars_message(self, ucla):
        out, _ = ucla
        with pytest.raises(GridMismatchError, match="pm"):
            M.dx(out)


# --- alignment ------------------------------------------------------------------------


class TestAlign:
    def test_same_shape_noop(self, rutgers):
        can = C.canonicalize(rutgers)
        assert select_like(can.h, can.temp).identical(can.h)

    def test_positional_mismatch_raises(self, rutgers):
        can = C.canonicalize(rutgers)
        with pytest.raises(GridMismatchError, match="(?i)subset the Dataset"):
            select_like(can.h, can.temp.isel(xi_rho=slice(2, 6)))

    def test_label_alignment(self, rutgers):
        lab = C.canonicalize(C.add_cf_attrs(rutgers, index_coords=True))
        sub = lab.temp.isel(xi_rho=slice(2, 6))
        assert (select_like(lab.h, sub).xi_rho.values == [2, 3, 4, 5]).all()
        usub = lab.u.isel(xi_u=slice(2, 6))
        assert (select_like(lab.h, usub).xi_rho.values == [2, 3, 4, 5, 6]).all()

    def test_indexed_away_without_labels(self, rutgers):
        can = C.canonicalize(rutgers)
        with pytest.raises(GridMismatchError, match="xi_rho"):
            select_like(can.h, can.temp.isel(xi_rho=3))
