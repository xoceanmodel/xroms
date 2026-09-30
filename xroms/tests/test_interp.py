"""``isoslice``, ``zslice`` and ``interpll``: staggers, dask, labels, dtypes, guardrails.

Regression tests for gaps found reviewing the interpolation port:

* slicing never broadcasts one stagger of an axis against another;
* ``method="nearest"`` is one lazy kernel, so it works on dask input and computes
  nothing while the result is built;
* ``zslice(z=...)`` honours the vertical labels of the z it is given, and explicit
  ``reference=``/``positive=`` win over the metadata of the targets;
* the lazy dtype of a result is the dtype it computes to;
* an xgcm ``Grid`` passed the pre-1.0 way fails with the 1.0 call to use instead;
* ``interpll``/``make_regridder`` return the field's own values at grid points.

Expected values come from the analytic fields of ``_synthetic.py`` or from a brute
force search written independently of the implementation.
"""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests.conftest import chunked, merged
from xroms.tests.test_modify_then_compute import tapped


# ---------------------------------------------------------------------- helpers
def assert_same(got, want):
    """Same dims in the same order, coordinates, values (NaNs alike), name and attrs."""
    xr.testing.assert_allclose(got, want)
    assert got.dims == want.dims
    assert got.name == want.name
    assert got.attrs == want.attrs


def columns(da, dim="s_rho"):
    """Values with ``dim`` moved last (what a column-wise search works along)."""
    return da.transpose(..., dim).values


def brute_force_nearest(var, iso, values, mask_edges):
    """Nearest-level pick along the last axis, written as plain loops.

    ``var`` and ``iso`` broadcast to (..., n); the result is (..., len(values)).
    NaN distances are never the nearest, and a column with no valid ``iso`` point
    falls back to its first level (and to NaN when ``mask_edges``).
    """
    var, iso = np.broadcast_arrays(var, iso)
    out = np.full(var.shape[:-1] + (len(values),), np.nan)
    for idx in np.ndindex(*var.shape[:-1]):
        col, key = var[idx], iso[idx]
        valid = key[~np.isnan(key)]
        for k, value in enumerate(values):
            dist = np.abs(key - value)
            dist[np.isnan(dist)] = np.inf
            out[idx + (k,)] = col[int(np.argmin(dist))]
            if mask_edges and not (valid.size and valid.min() <= value <= valid.max()):
                out[idx + (k,)] = np.nan
    return out


def as_float32(ds):
    return ds.assign(temp=ds.temp.astype("float32"), salt=ds.salt.astype("float32"))


# ------------------------------------------------------------- stagger matching
class TestSlicingNeverBroadcastsAcrossStaggers:
    # a u-point variable against rho-point depths used to come back 5-D, with both
    # xi_rho and xi_u, and no error
    PAIRS = [("u", "rho"), ("v", "rho"), ("temp", "u"), ("temp", "v"), ("u", "v"), ("temp", "psi")]

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    @pytest.mark.parametrize("name, hcoord", PAIRS)
    def test_variable_and_iso_array_on_different_points_are_rejected(self, layout, method, name, hcoord):
        ds = merged(layout)
        with pytest.raises(ValueError, match="same points") as err:
            xroms.isoslice(ds[name], [-5.0], xroms.z(ds, hcoord=hcoord), method=method)
        message = str(err.value)
        assert "xroms.to_grid" in message and "xroms.zslice" in message

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_targets_on_different_points_are_rejected_too(self, rutgers, method):
        can = C.canonicalize(rutgers)
        targets = xr.DataArray(np.full((1, 9, 11), -5.0), dims=("z", "eta_rho", "xi_u"))
        with pytest.raises(ValueError, match="same points"):
            xroms.isoslice(can.temp, targets, xroms.z(rutgers), new_dim="z", method=method)
        # alias names are the same points as the canonical ones
        ok = xr.DataArray(np.full((1, 9, 12), -5.0), dims=("z", "eta_u", "xi_v"))
        out = xroms.isoslice(can.temp, ok, xroms.z(rutgers), new_dim="z", method=method)
        assert out.dims == ("ocean_time", "z", "eta_rho", "xi_rho")

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_iso_array_with_a_horizontal_dim_the_variable_lacks_is_rejected(self, rutgers, method):
        can = C.canonicalize(rutgers)
        column = can.temp.isel(xi_rho=3)  # no index coordinate: the position is unknown
        assert C.hposition(column) is None
        z = xroms.z(rutgers)
        with pytest.raises(ValueError, match="xi_rho") as err:
            xroms.isoslice(column, [-5.0], z, method=method)
        assert "xr.broadcast" in str(err.value)
        # the way out the message gives: say explicitly that it is the same everywhere
        out = xroms.isoslice(xr.broadcast(column, z)[0], [-5.0], z, method=method)
        assert out.dims == ("ocean_time", "z_rho", "eta_rho", "xi_rho")
        one = xroms.isoslice(column, [-5.0], z.isel(xi_rho=3), method=method)
        np.testing.assert_allclose(out.isel(xi_rho=3).values, one.values)

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_levels_of_different_length_are_rejected(self, method):
        # nearest used to answer from whichever level number the shorter array reached
        var = xr.DataArray([10.0, 20.0, 30.0], dims="s_rho", name="v")
        with pytest.raises(ValueError, match="3 points along 's_rho' but the iso array has 2"):
            xroms.isoslice(var, [1.0], xr.DataArray([0.0, 1.0], dims="s_rho"), new_dim="iso", method=method)

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_iso_array_without_the_searched_dim_is_rejected(self, method):
        var = xr.DataArray([10.0, 20.0, 30.0], dims="s_rho", name="v")
        with pytest.raises(ValueError, match="both the variable and the iso array need dim 's_rho'"):
            xroms.isoslice(var, [1.0], xr.DataArray([0.0, 1.0, 2.0], dims="s_w"), new_dim="iso", method=method)

    def test_the_transform_itself_does_not_broadcast_over_grid_dims(self):
        ds = merged("ucla")
        with pytest.raises(ValueError):
            xroms._xgcm.transform(ds.u, [-5.0], xroms.z(ds), "s_rho", new_dim="z")

    @pytest.mark.parametrize("hcoord", ["u", "v", "psi"])
    def test_variable_and_iso_array_on_the_same_points_still_slice(self, layout, hcoord):
        ds = merged(layout)
        var = xroms.to_grid(ds.temp, hcoord=hcoord)
        eta, xi = C.CANONICAL[hcoord]
        out = xroms.isoslice(var, [-5.0], xroms.z(ds, hcoord=hcoord), new_dim="z")
        assert out.dims == (C.time_dim(ds), "z", eta, xi)
        np.testing.assert_allclose(out.values, xroms.zslice(var, [-5.0], ds).values, atol=1e-12)

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    @pytest.mark.parametrize("dask", [False, True])
    def test_dims_that_are_not_grid_dims_are_still_broadcast(self, rutgers, method, dask):
        # static depths (no time) sliced on a time-varying field
        can = C.canonicalize(chunked(rutgers) if dask else rutgers)
        z0 = xroms.z(rutgers, zeta=0)
        target = [float(can.temp.mean())]
        out = xroms.isoslice(z0, target, can.temp, new_dim="temp", method=method)
        assert out.dims == ("ocean_time", "temp", "eta_rho", "xi_rho")
        assert (out.chunks is not None) == dask
        for t in range(can.sizes["ocean_time"]):
            one = xroms.isoslice(z0, target, can.temp.isel(ocean_time=t), new_dim="temp", method=method)
            np.testing.assert_allclose(out.isel(ocean_time=t).values, one.values)


# -------------------------------------------------------------- nearest on dask
class TestNearestIsOneLazyKernel:
    def test_zslice_on_chunked_input_equals_numpy(self, layout):
        ds = merged(layout)
        c = chunked(ds)
        depths = [-5.0, -40.0, -500.0]
        want = xroms.zslice(ds.temp, depths, ds, method="nearest")
        got = xroms.zslice(c.temp, depths, c, method="nearest")
        assert got.chunks is not None
        assert_same(got.compute(), want)

    def test_isoslice_on_chunked_input_equals_numpy(self, layout):
        ds = merged(layout)
        c = chunked(ds)
        values = [34.99, 35.05, 35.5, 99.0]
        want = xroms.isoslice(C.canonicalize(ds).temp, values, C.canonicalize(ds).salt, method="nearest")
        got = xroms.isoslice(C.canonicalize(c).temp, values, C.canonicalize(c).salt, method="nearest")
        assert got.chunks is not None
        assert_same(got.compute(), want)

    @pytest.mark.parametrize("which", ["zslice", "isoslice"])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_building_the_chunked_result_computes_nothing(self, layout, which, method):
        c, loads = tapped(chunked(merged(layout)))
        if which == "zslice":
            out = xroms.zslice(c.temp, [-5.0, -40.0], c, method=method)
        else:
            out = xroms.isoslice(C.canonicalize(c).temp, [35.05], C.canonicalize(c).salt, method=method)
        assert out.chunks is not None
        assert not loads, "model fields were evaluated while building the result"
        out.compute()
        assert loads, "the counter never fired, so the check above proves nothing"

    def test_only_the_searched_dim_is_rechunked(self, rutgers):
        c = chunked(rutgers)
        out = xroms.zslice(c.temp, [-5.0], c, method="nearest")
        assert out.chunksizes["eta_rho"] == c.temp.chunksizes["eta_rho"]
        assert out.chunksizes["xi_rho"] == c.temp.chunksizes["xi_rho"]
        assert out.chunksizes["ocean_time"] == c.temp.chunksizes["ocean_time"]
        assert out.chunksizes["z"] == (1,)

    def test_dask_targets_split_along_the_new_dim_work(self, rutgers):
        ds, c = C.canonicalize(rutgers), C.canonicalize(chunked(rutgers))
        z = xroms.z(rutgers)
        targets = xr.DataArray(
            np.stack([np.full((9, 12), -3.0), np.full((9, 12), -7.0), np.full((9, 12), -700.0)]),
            dims=("d", "eta_rho", "xi_rho"),
        )
        want = xroms.isoslice(ds.temp, targets, z, new_dim="d", method="nearest")
        split = targets.chunk({"d": 1, "xi_rho": 5})
        got = xroms.isoslice(c.temp, split, xroms.z(chunked(rutgers)), new_dim="d", method="nearest")
        assert got.chunks is not None
        assert_same(got.compute(), want)

    @pytest.mark.parametrize("dask", [False, True])
    @pytest.mark.parametrize("mask_edges", [True, False])
    def test_matches_a_brute_force_search_over_land(self, with_land, mask_edges, dask):
        # land columns have NaN depths (NaN zeta) and NaN temperature
        ds = chunked(with_land) if dask else with_land
        depths = [-5.0, -10.0, -500.0, 0.5]
        out = xroms.zslice(ds.temp, depths, ds, method="nearest", mask_edges=mask_edges)
        z = C.canonicalize(xroms.z(with_land))
        want = brute_force_nearest(columns(C.canonicalize(with_land).temp), columns(z), depths, mask_edges)
        got = out.compute().transpose("ocean_time", "eta_rho", "xi_rho", "z").values
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("dask", [False, True])
    @pytest.mark.parametrize("mask_edges", [True, False])
    def test_columns_without_valid_iso_values_and_ties(self, mask_edges, dask):
        var = xr.DataArray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dims=("s_rho", "xi_rho"), name="v")
        # column 0 rises; column 1 has no valid iso values at all
        iso = xr.DataArray([[0.0, np.nan], [2.0, np.nan], [4.0, np.nan]], dims=("s_rho", "xi_rho"))
        if dask:
            var, iso = var.chunk({"s_rho": 1, "xi_rho": 1}), iso.chunk({"s_rho": 1, "xi_rho": 1})
        out = xroms.isoslice(var, [1.0, 3.0, 9.0], iso, new_dim="iso", method="nearest", mask_edges=mask_edges)
        assert out.dims == ("iso", "xi_rho")
        nan = np.nan
        if mask_edges:
            # 1 and 3 are exact ties (first level wins); 9 lies outside [0, 4]; column 1 is masked
            want = [[10.0, nan], [20.0, nan], [nan, nan]]
        else:
            # edges hold the nearest level; a column without valid iso values reads level 0
            want = [[10.0, 11.0], [20.0, 11.0], [30.0, 11.0]]
        np.testing.assert_array_equal(out.compute().values, want)

    def test_ties_go_to_the_first_level_when_iso_decreases(self):
        var = xr.DataArray([10.0, 20.0, 30.0], dims="s_rho")
        iso = xr.DataArray([4.0, 2.0, 0.0], dims="s_rho")  # depths positive down
        out = xroms.isoslice(var, [1.0, 3.0], iso, new_dim="iso", method="nearest")
        np.testing.assert_array_equal(out.values, [20.0, 10.0])

    def test_a_nan_target_is_masked_like_any_target_outside_the_range(self):
        var = xr.DataArray([10.0, 20.0], dims="s_rho")
        iso = xr.DataArray([0.0, 1.0], dims="s_rho")
        targets = xr.DataArray([np.nan, 0.9], dims="iso")
        masked = xroms.isoslice(var, targets, iso, new_dim="iso", method="nearest")
        np.testing.assert_array_equal(masked.values, [np.nan, 20.0])
        held = xroms.isoslice(var, targets, iso, new_dim="iso", method="nearest", mask_edges=False)
        np.testing.assert_array_equal(held.values, [10.0, 20.0])

    def test_integer_levels_and_targets(self):
        var = xr.DataArray([10.0, 20.0, 30.0], dims="s_rho")
        iso = xr.DataArray([0, 2, 4], dims="s_rho")
        out = xroms.isoslice(var, xr.DataArray([1, 3, 9], dims="iso"), iso, method="nearest")
        np.testing.assert_array_equal(out.values, [10.0, 20.0, np.nan])

    def test_times_can_be_picked(self):
        times = xr.DataArray(np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]"), dims="s_rho")
        out = xroms.isoslice(times, [1.1, 5.0], xr.DataArray([0.0, 1.0, 2.0], dims="s_rho"), method="nearest")
        assert out.dtype == times.dtype
        assert out.values[0] == times.values[1] and np.isnat(out.values[1])

    def test_variable_values_are_passed_through_not_searched(self):
        # NaN in the variable at the nearest level stays NaN; it does not move the pick
        var = xr.DataArray([np.nan, 20.0, 30.0], dims="s_rho")
        iso = xr.DataArray([0.0, 1.0, 2.0], dims="s_rho")
        out = xroms.isoslice(var, [0.1], iso, new_dim="iso", method="nearest")
        assert np.isnan(out.values).all()

    @pytest.mark.parametrize("dask", [False, True])
    def test_spatially_varying_targets(self, rutgers, dask):
        ds = chunked(rutgers) if dask else rutgers
        depth = np.linspace(-60.0, -2.0, 9 * 12).reshape(9, 12)  # deeper than the shallow columns
        targets = xr.DataArray(np.stack([depth, depth / 2]), dims=("d", "eta_rho", "xi_rho"))
        out = xroms.isoslice(C.canonicalize(ds).temp, targets, xroms.z(ds), new_dim="d", method="nearest")
        assert out.dims == ("ocean_time", "d", "eta_rho", "xi_rho")
        temp, z = columns(C.canonicalize(rutgers).temp), columns(xroms.z(rutgers))
        got = out.compute()
        for k in range(2):
            want = np.empty(temp.shape[:-1])
            for t, j, i in np.ndindex(*want.shape):
                col, target = z[t, j, i], targets.values[k, j, i]
                inside = col.min() <= target <= col.max()
                want[t, j, i] = temp[t, j, i][np.argmin(np.abs(col - target))] if inside else np.nan
            np.testing.assert_array_equal(got.isel(d=k).values, want)
        assert np.isnan(got.values).any() and np.isfinite(got.values).any()

    @pytest.mark.parametrize("dask", [False, True])
    def test_a_horizontal_dim_can_be_searched(self, rutgers, dask):
        can = C.canonicalize(chunked(rutgers) if dask else rutgers)
        lon = can.lon_rho  # rises along xi_rho only
        target = [float(lon.isel(eta_rho=0, xi_rho=4))]
        out = xroms.isoslice(can.temp, target, lon, dim="X", new_dim="lon", method="nearest")
        assert out.dims == ("ocean_time", "s_rho", "eta_rho", "lon")
        assert (out.chunks is not None) == dask
        np.testing.assert_array_equal(out.compute().isel(lon=0).values, can.temp.isel(xi_rho=4).compute().values)

    def test_inputs_are_not_modified(self, rutgers):
        c = chunked(rutgers)
        z = xroms.z(c)
        before = (c.temp.copy(deep=True), z.copy(deep=True))
        xroms.zslice(c.temp, [-5.0], c, z=z, method="nearest").compute()
        xr.testing.assert_identical(c.temp, before[0])
        xr.testing.assert_identical(z, before[1])

    @pytest.mark.parametrize("dask", [False, True])
    def test_attrs_name_and_coords_follow_the_variable(self, rutgers, dask):
        ds = chunked(rutgers) if dask else rutgers
        out = xroms.zslice(ds.temp, [-5.0, -10.0], ds, method="nearest").compute()
        assert out.name == "temp" and out.attrs == rutgers.temp.attrs
        assert {"lon_rho", "lat_rho", "ocean_time", "z"} <= set(out.coords)
        assert "s_rho" not in out.coords
        np.testing.assert_array_equal(out.z.values, [-5.0, -10.0])

    @pytest.mark.parametrize(
        "dtype, mask_edges, expected",
        [
            ("float32", True, "float32"),
            ("float32", False, "float32"),
            ("float64", True, "float64"),
            ("int32", False, "int32"),
            ("int32", True, "float64"),  # NaN has to fit
            ("int16", True, "float32"),
        ],
    )
    @pytest.mark.parametrize("dask", [False, True])
    def test_dtype_follows_the_variable(self, rutgers, dtype, mask_edges, expected, dask):
        ds = chunked(rutgers) if dask else rutgers
        var = (C.canonicalize(ds).temp * 10).astype(dtype)
        out = xroms.isoslice(var, [-5.0], xroms.z(ds), method="nearest", mask_edges=mask_edges)
        assert out.dtype == np.dtype(expected)
        assert out.compute().dtype == out.dtype


# ---------------------------------------------------------------- labels of z=
class TestZsliceWithAGivenZ:
    def test_z_in_the_requested_reference_is_used_as_is(self, layout):
        ds = merged(layout)
        zs = xroms.z(ds, reference="surface", positive="down")
        got = xroms.zslice(ds.temp, [10.0], z=zs, reference="surface", positive="down")
        want = xroms.zslice(ds.temp, [10.0], ds, reference="surface", positive="down")
        assert not np.isnan(got.values).any()
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        assert got.z.attrs["vertical_reference"] == "surface" and got.z.attrs["positive"] == "down"

    def test_z_with_the_opposite_sign_is_negated(self, layout):
        ds = merged(layout)
        got = xroms.zslice(ds.temp, [5.0], z=xroms.z(ds), reference="mean_sea_level", positive="down")
        want = xroms.zslice(ds.temp, [5.0], ds, reference="mean_sea_level", positive="down")
        assert not np.isnan(got.values).any()
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        assert got.z.attrs["positive"] == "down" and got.z.attrs["standard_name"] == "depth_below_geoid"

    @pytest.mark.parametrize("reference, depth", [("mean_sea_level", 5.0), ("surface", 5.0), ("bottom", -5.0)])
    def test_the_sign_flip_holds_in_every_reference(self, rutgers, reference, depth):
        z = xroms.z(rutgers, reference=reference)
        got = xroms.zslice(rutgers.temp, [depth], z=z, reference=reference, positive="down")
        want = xroms.zslice(rutgers.temp, [depth], rutgers, reference=reference, positive="down")
        assert not np.isnan(got.values).all()
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        assert got.z.attrs["vertical_reference"] == reference

    def test_labels_given_as_a_cf_standard_name_are_read_too(self, rutgers):
        z = xroms.z(rutgers, positive="down")
        z.attrs = {"standard_name": "depth_below_geoid"}  # mean sea level, positive down
        got = xroms.zslice(rutgers.temp, [-5.0], z=z)
        want = xroms.zslice(rutgers.temp, [-5.0], rutgers)
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        z.attrs = {"standard_name": "depth"}  # relative to the surface: another reference
        with pytest.raises(ValueError, match="'surface'"):
            xroms.zslice(rutgers.temp, [-5.0], z=z)

    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_negating_z_also_works_for_nearest_and_for_u_points(self, rutgers, method):
        got = xroms.zslice(rutgers.u, [5.0], z=xroms.z(rutgers), positive="down", method=method)
        want = xroms.zslice(rutgers.u, [5.0], rutgers, positive="down", method=method)
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)

    @pytest.mark.parametrize(
        "z_reference, request_",
        [
            ("mean_sea_level", {"reference": "surface", "positive": "down"}),
            ("surface", {}),
            ("bottom", {"reference": "surface"}),
        ],
    )
    def test_z_in_another_reference_is_rejected_with_a_way_out(self, rutgers, z_reference, request_):
        z = xroms.z(rutgers, reference=z_reference)
        with pytest.raises(ValueError, match="xroms.z") as err:
            xroms.zslice(rutgers.temp, [10.0], z=z, **request_)
        message = str(err.value)
        assert z_reference in message and request_.get("reference", "mean_sea_level") in message

    def test_building_z_in_the_requested_reference_is_the_way_out(self, rutgers):
        with pytest.raises(ValueError):
            xroms.zslice(rutgers.temp, [10.0], z=xroms.z(rutgers), reference="surface", positive="down")
        z = xroms.z(rutgers, reference="surface", positive="down")
        got = xroms.zslice(rutgers.temp, [10.0], z=z, reference="surface", positive="down")
        want = xroms.zslice(rutgers.temp, [10.0], rutgers, reference="surface", positive="down")
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)

    def test_a_z_without_labels_is_taken_at_its_word(self, layout):
        ds = merged(layout)
        z = xroms.z(ds, reference="surface", positive="down")
        z.attrs = {}
        got = xroms.zslice(ds.temp, [10.0], z=z, reference="surface", positive="down")
        want = xroms.zslice(ds.temp, [10.0], ds, reference="surface", positive="down")
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        assert got.z.attrs["vertical_reference"] == "surface"

    def test_only_positive_is_compared_when_z_names_no_reference(self, rutgers):
        z = xroms.z(rutgers, positive="down")
        z.attrs = {"positive": "down"}
        got = xroms.zslice(rutgers.temp, [-5.0], z=z)  # heights relative to mean sea level
        want = xroms.zslice(rutgers.temp, [-5.0], rutgers)
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)

    def test_the_callers_z_is_left_alone(self, rutgers):
        z = xroms.z(rutgers)
        before = z.copy(deep=True)
        xroms.zslice(rutgers.temp, [5.0], z=z, positive="down")
        xr.testing.assert_identical(z, before)

    def test_z_with_contradictory_metadata_is_explained(self, rutgers):
        z = xroms.z(rutgers)
        z.attrs = {"standard_name": "depth", "positive": "up"}
        with pytest.raises(ValueError, match="contradicts") as err:
            xroms.zslice(rutgers.temp, [5.0], z=z)
        assert "attrs of z" in str(err.value)


class TestExplicitFlagsOverrideTheTargetsMetadata:
    CONTRADICTORY = {"standard_name": "depth", "positive": "up"}

    @pytest.mark.parametrize("use_z", [False, True])
    def test_both_flags_ignore_the_metadata_of_depths(self, rutgers, use_z):
        depths = xr.DataArray([10.0], dims="z", attrs=self.CONTRADICTORY)
        kwargs = {"z": xroms.z(rutgers, reference="surface", positive="down")} if use_z else {}
        grid = None if use_z else rutgers
        got = xroms.zslice(rutgers.temp, depths, grid, reference="surface", positive="down", **kwargs)
        want = xroms.zslice(rutgers.temp, [10.0], rutgers, reference="surface", positive="down")
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)
        assert got.z.attrs["vertical_reference"] == "surface" and got.z.attrs["positive"] == "down"

    def test_a_missing_flag_is_taken_from_the_metadata(self, rutgers):
        depths = xr.DataArray([10.0], dims="z", attrs={"positive": "down"})
        got = xroms.zslice(rutgers.temp, depths, rutgers, reference="surface")
        want = xroms.zslice(rutgers.temp, [10.0], rutgers, reference="surface", positive="down")
        np.testing.assert_allclose(got.values, want.values, rtol=0, atol=1e-12)

    def test_contradictory_metadata_still_needs_both_flags(self, rutgers):
        depths = xr.DataArray([10.0], dims="z", attrs=self.CONTRADICTORY)
        with pytest.raises(ValueError, match="contradicts") as err:
            xroms.zslice(rutgers.temp, depths, rutgers, reference="surface")
        assert "both reference= and positive=" in str(err.value)
        with pytest.raises(ValueError, match="contradicts"):
            xroms.zslice(rutgers.temp, depths, rutgers)

    def test_invalid_flags_are_named(self, rutgers):
        with pytest.raises(ValueError, match="reference must be one of"):
            xroms.zslice(rutgers.temp, [1.0], z=xroms.z(rutgers), reference="sea", positive="down")
        with pytest.raises(ValueError, match="positive must be"):
            xroms.zslice(rutgers.temp, [1.0], rutgers, positive="sideways")


# ------------------------------------------------------------------- dtypes
class TestLazyDtypeIsTheComputedDtype:
    @pytest.mark.parametrize("method", ["linear", "nearest"])
    def test_float32_chunked_input(self, layout, method):
        c = chunked(as_float32(merged(layout)))
        can = C.canonicalize(c)
        outs = {
            "zslice": xroms.zslice(c.temp, [-5.0, -40.0], c, method=method),
            "isoslice": xroms.isoslice(can.temp, [35.05], can.salt, method=method),
        }
        for name, out in outs.items():
            assert out.chunks is not None
            assert out.dtype == out.compute().dtype, name

    @pytest.mark.parametrize("dask", [False, True])
    @pytest.mark.parametrize("iso_dtype", ["float32", "float64"])
    @pytest.mark.parametrize("var_dtype", ["float32", "float64"])
    def test_linear_result_has_the_promoted_dtype_of_its_inputs(self, rutgers, var_dtype, iso_dtype, dask):
        can = C.canonicalize(chunked(rutgers) if dask else rutgers)
        var, iso = can.temp.astype(var_dtype), can.salt.astype(iso_dtype)
        out = xroms.isoslice(var, [35.05, 35.5], iso)
        assert out.dtype == np.result_type(var.dtype, iso.dtype)
        assert out.compute().dtype == out.dtype
        if iso_dtype == "float64":
            # interpolated in float64 and rounded once: float32 results agree to ~1e-7
            ref = xroms.isoslice(can.temp.astype("float64"), [35.05, 35.5], iso)
            np.testing.assert_allclose(out.compute().values, ref.compute().values, rtol=2e-6)

    def test_float32_targets_do_not_change_the_dtype(self, rutgers):
        can = C.canonicalize(chunked(as_float32(rutgers)))
        targets = xr.DataArray(np.array([35.05, 35.5], dtype="float32"), dims="iso")
        out = xroms.isoslice(can.temp, targets, can.salt, new_dim="iso")
        assert out.dtype == np.float32 and out.compute().dtype == np.float32

    def test_integer_input_gives_floats(self, rutgers):
        c = chunked(rutgers)
        var = (C.canonicalize(c).temp * 10).astype("int32")
        out = xroms.isoslice(var, [-5.0], xroms.z(c).astype("float32"))
        assert out.dtype.kind == "f"
        assert out.dtype == out.compute().dtype


# ------------------------------------------ pre-1.0 xgcm Grid and other bad arguments
class TestArgumentChecks:
    def test_isoslice_explains_the_1_0_call(self, rutgers):
        with pytest.raises(TypeError, match="xroms 1.0") as err:
            xroms.isoslice(rutgers.salt, [-5.0], rutgers.xroms.xgcm_grid())
        message = str(err.value)
        assert "xroms.isoslice(var, values, xroms.z(ds))" in message
        assert "xroms.zslice(var, depths, ds)" in message
        assert "xgcm" in message

    @pytest.mark.parametrize("bad", [None, "salt", np.zeros(3)], ids=["None", "str", "ndarray"])
    def test_isoslice_needs_a_dataarray_as_iso_array(self, rutgers, bad):
        with pytest.raises(TypeError, match="iso_array"):
            xroms.isoslice(rutgers.salt, [-5.0], bad)

    @pytest.mark.parametrize("with_z", [False, True])
    def test_zslice_rejects_an_xgcm_grid(self, rutgers, with_z):
        kwargs = {"z": xroms.z(rutgers)} if with_z else {}
        with pytest.raises(TypeError, match="xroms 1.0") as err:
            xroms.zslice(rutgers.temp, [-5.0], rutgers.xroms.xgcm_grid(), **kwargs)
        assert "Dataset" in str(err.value)

    def test_unknown_method_is_rejected(self, rutgers):
        with pytest.raises(ValueError, match="method must be 'linear' or 'nearest'"):
            xroms.zslice(rutgers.temp, [-5.0], rutgers, method="cubic")

    def test_zslice_still_takes_a_dataset_or_nothing(self, rutgers):
        z = xroms.z(rutgers)
        a = xroms.zslice(rutgers.temp, [-5.0], rutgers)
        b = xroms.zslice(rutgers.temp, [-5.0], z=z)
        np.testing.assert_allclose(a.values, b.values)
        with pytest.raises(ValueError, match="grid= .*or z="):
            xroms.zslice(rutgers.temp, [-5.0])
        with pytest.raises(TypeError, match="xarray Dataset"):
            xroms.zslice(rutgers.temp, [-5.0], [1, 2, 3])


# ---------------------------------------------------------- xESMF interpolation
class TestInterpll:
    """Lon/lat interpolation hands back the field's own values at the grid's points."""

    EETA, EXI = np.array([2, 4, 6]), np.array([3, 5, 9])  # interior rho points

    @pytest.fixture(autouse=True)
    def xesmf(self):
        return pytest.importorskip("xesmf")

    @pytest.fixture(params=["rutgers", "ucla", "croco"])
    def field(self, request):
        ds = merged(request.param)
        if request.param == "ucla":  # UCLA output keeps lon/lat as plain data variables
            ds = ds.set_coords(["lon_rho", "lat_rho"])
        return C.canonicalize(ds)

    @pytest.mark.parametrize("dask", [False, True])
    def test_pairs_returns_the_values_at_the_points(self, field, dask):
        lons, lats = field.lon_rho.values[self.EETA, self.EXI], field.lat_rho.values[self.EETA, self.EXI]
        src = chunked(field) if dask else field
        out = xroms.interpll(src.temp, lons, lats, which="pairs")
        assert (out.chunks is not None) == dask
        want = field.temp.isel(
            eta_rho=xr.DataArray(self.EETA, dims="locations"), xi_rho=xr.DataArray(self.EXI, dims="locations")
        )
        assert out.dims == want.dims
        np.testing.assert_allclose(out.compute().values, want.values, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(out["locations"].values, [0, 1, 2])
        assert out["locations"].attrs == {"axis": "X"}
        assert field.temp.attrs.items() <= out.attrs.items()  # xESMF adds regrid_method

    @pytest.mark.parametrize("dask", [False, True])
    def test_grid_returns_the_values_on_the_lat_lon_grid(self, field, dask):
        lons = field.lon_rho.isel(eta_rho=0).values[2:-2]
        lats = field.lat_rho.isel(xi_rho=0).values[2:-2]
        src = chunked(field) if dask else field
        out = xroms.interpll(src.temp, lons, lats, which="grid")
        assert (out.chunks is not None) == dask
        want = field.temp.isel(eta_rho=slice(2, -2), xi_rho=slice(2, -2))
        assert out.dims == want.dims[:-2] + ("lat", "lon")
        np.testing.assert_allclose(out.compute().values, want.values, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(out["lon"].values, lons)
        np.testing.assert_allclose(out["lat"].values, lats)

    @pytest.mark.parametrize("dask", [False, True])
    @pytest.mark.parametrize("name", ["u", "v"])
    @pytest.mark.parametrize("layout_name", ["rutgers", "croco"])
    def test_velocity_points_use_their_own_lon_lat(self, layout_name, name, dask):
        ds = C.canonicalize(merged(layout_name))
        var = chunked(ds)[name] if dask else ds[name]
        eta_dim, xi_dim = [d for d in ds[name].dims if d.startswith(("eta_", "xi_"))]
        eta, xi = np.array([2, 3]), np.array([3, 5])
        lons, lats = ds[f"lon_{name}"].values[eta, xi], ds[f"lat_{name}"].values[eta, xi]
        out = xroms.interpll(var, lons, lats, which="pairs")
        want = ds[name].isel({eta_dim: xr.DataArray(eta, dims="locations"), xi_dim: xr.DataArray(xi, dims="locations")})
        np.testing.assert_allclose(out.compute().values, want.values, rtol=1e-9, atol=1e-9)

    def test_a_regridder_is_reused_across_variables(self, field):
        lons, lats = field.lon_rho.values[self.EETA, self.EXI], field.lat_rho.values[self.EETA, self.EXI]
        regridder = xroms.make_regridder(field.temp, lons, lats, which="pairs")
        for name in ("temp", "salt"):
            reused = xroms.interpll(field[name], lons, lats, regridder=regridder)
            fresh = xroms.interpll(field[name], lons, lats)
            xr.testing.assert_allclose(reused, fresh)
            want = field[name].isel(
                eta_rho=xr.DataArray(self.EETA, dims="locations"), xi_rho=xr.DataArray(self.EXI, dims="locations")
            )
            np.testing.assert_allclose(reused.values, want.values, rtol=1e-9, atol=1e-9)

    def test_the_input_is_not_modified(self, field):
        before = field.temp.copy(deep=True)
        xroms.interpll(field.temp, [field.lon_rho.values[4, 5]], [field.lat_rho.values[4, 5]])
        xr.testing.assert_identical(field.temp, before)

    def test_unknown_mode_is_explained(self, field):
        with pytest.raises(ValueError, match="'pairs' or 'grid'"):
            xroms.make_regridder(field.temp, [1.0], [2.0], which="nope")

    def test_cartesian_output_without_lonlat_is_explained(self, remora):
        with pytest.raises(ValueError, match="lon_rho/lat_rho"):
            xroms.interpll(C.canonicalize(remora).temp, [1.0], [2.0])


def test_linear_slicing_needs_two_levels():
    ds = merged("rutgers", N=1)
    with pytest.raises(ValueError, match="at least 2 points"):
        xroms.zslice(ds.temp, [-5.0], ds)
