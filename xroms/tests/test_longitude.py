"""Longitude helpers: ``wrap_longitude``, ``straddles`` and ``lonlat_at``.

Sections: wrapping numbers, arrays, DataArrays and Datasets (edge values, kinds, attrs,
laziness, index coordinates), reading the convention off the data, ``straddles``, the
longitude and latitude of staggered points on every layout (stored, averaged from rho,
Greenwich- and dateline-crossing grids, Cartesian grids), and cases ported as plain
expected values from ocean-skill's tests of ``harmonize_longitude``/``natural_convention``.
"""

import dask
import numpy as np
import pytest
import xarray as xr

import xroms

from xroms import conventions as C
from xroms.tests.conftest import chunked, merged


CONVENTIONS = ("-180-180", "0-360")
POSITIONS = ("rho", "u", "v", "psi")


def ref_wrap(lon, convention):
    """Wrapped values by the plain definition (modulo 360, then shifted), independent of xroms."""
    wrapped = np.asarray(lon, dtype=float) % 360.0
    return wrapped if convention == "0-360" else np.where(wrapped > 180, wrapped - 360, wrapped)


def averaged(a, hcoord):
    """A rho-point numpy array averaged onto ``hcoord`` (the last two axes are eta, xi)."""
    if hcoord in ("u", "psi"):
        a = 0.5 * (a[..., :-1] + a[..., 1:])
    if hcoord in ("v", "psi"):
        a = 0.5 * (a[..., :-1, :] + a[..., 1:, :])
    return a


def no_compute(*args, **kwargs):
    raise AssertionError("dask data was computed")


def lonlat_dataset(lon1d, *, store, neta=5, tilt=0.0, velocity=False, as_coords=True):
    """Grid with ``lon_rho``/``lat_rho`` (and u, v, psi pairs if ``velocity``) written as a file would.

    ``lon1d`` are the longitudes along xi *unwrapped* (358, 359, 360, 361 for a grid crossing
    Greenwich), ``tilt`` skews them with eta to make the grid curvilinear, and ``store`` is the
    convention they are stored in. Returns ``(dataset, unwrapped rho longitudes)``.
    """
    lon1d = np.asarray(lon1d, dtype=float)
    eta = np.linspace(0.0, 1.0, neta)[:, None]
    xi = np.linspace(0.0, 1.0, lon1d.size)[None, :]
    lon = lon1d[None, :] + tilt * eta
    lat = 10.0 + 4.0 * eta + 1.5 * xi**2
    variables = {
        "lon_rho": (C.CANONICAL["rho"], ref_wrap(lon, store), {"units": "degrees_east"}),
        "lat_rho": (C.CANONICAL["rho"], lat, {"units": "degrees_north"}),
    }
    if velocity:
        for pos in ("u", "v", "psi"):
            variables[f"lon_{pos}"] = (C.CANONICAL[pos], ref_wrap(averaged(lon, pos), store), {"units": "degrees_east"})
            variables[f"lat_{pos}"] = (C.CANONICAL[pos], averaged(lat, pos), {"units": "degrees_north"})
    ds = xr.Dataset(coords=variables) if as_coords else xr.Dataset(variables)
    return ds, lon


GREENWICH = np.linspace(-2.0, 2.0, 9)  # crosses Greenwich: 358..360 and 0..2 when stored in 0-360
DATELINE = np.linspace(178.0, 182.0, 9)  # crosses the dateline: 180 and -180 when stored in -180-180
REGIONAL = np.linspace(10.0, 20.0, 9)


# ----------------------------------------------------------------- wrapping numbers and arrays
def test_the_helpers_are_exported_at_the_top_level():
    from xroms import longitude

    for name in ("wrap_longitude", "straddles", "lonlat_at"):
        assert getattr(xroms, name) is getattr(longitude, name)


class TestWrapNumbers:
    @pytest.mark.parametrize(
        "lon, to_180, to_360",
        [
            (180, 180, 180),
            (-180, 180, 180),  # the seam is 180 in (-180, 180]
            (360, 0, 0),
            (540, 180, 180),
            (-540, 180, 180),
            (0, 0, 0),
            (190, -170, 190),
            (-190, 170, 170),
            (-0.5, -0.5, 359.5),
            (720.5, 0.5, 0.5),
            (-359.5, 0.5, 0.5),
        ],
    )
    def test_documented_values(self, lon, to_180, to_360):
        assert xroms.wrap_longitude(lon, "-180-180") == to_180
        assert xroms.wrap_longitude(lon, "0-360") == to_360

    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_edge_values_match_the_plain_definition(self, convention):
        edges = np.array([180, -180, 360, 540, -540, -0.0, np.nan, 0.0, 190.0, -190.0, 719.5, -359.5])
        np.testing.assert_array_equal(xroms.wrap_longitude(edges, convention), ref_wrap(edges, convention))

    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_nan_stays_nan_and_negative_zero_is_zero(self, convention):
        assert np.isnan(xroms.wrap_longitude(np.nan, convention))
        out = xroms.wrap_longitude(np.array([np.nan, -0.0, 1.0]), convention)
        assert np.isnan(out[0]) and out[1] == 0.0 and out[2] == 1.0
        assert not np.signbit(out[1])  # the same zero in both conventions

    @pytest.mark.parametrize("convention, low, high", [("-180-180", -180, 180), ("0-360", 0, 360)])
    def test_results_land_in_the_documented_range(self, convention, low, high):
        rng = np.random.default_rng(0)
        awkward = [0.0, 180.0, -180.0, 360.0, 1e-20, -1e-20, -1e-15, np.nextafter(180, 200), np.nextafter(-180, -200)]
        lon = np.concatenate([rng.uniform(-1500, 1500, 2000), awkward])
        out = xroms.wrap_longitude(lon, convention)
        if convention == "-180-180":
            assert ((out > low) & (out <= high)).all()
        else:
            assert ((out >= low) & (out < high)).all()  # a tiny negative must not round up to 360
        turns = (lon - out) / 360.0
        np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)  # moved by whole turns only

    def test_values_in_range_come_back_bit_for_bit(self):
        rng = np.random.default_rng(1)
        in180, in360 = rng.uniform(-179.999, 180, 500), rng.uniform(0, 359.999, 500)
        np.testing.assert_array_equal(xroms.wrap_longitude(in180, "-180-180"), in180)
        np.testing.assert_array_equal(xroms.wrap_longitude(in360, "0-360"), in360)

    def test_numbers_and_arrays_come_back_as_the_kind_they_were(self):
        out = xroms.wrap_longitude(190.0, "-180-180")
        assert out == -170.0 and type(out) is float
        out = xroms.wrap_longitude(190, "-180-180")
        assert out == -170 and type(out) is int
        out = xroms.wrap_longitude(np.float32(190), "-180-180")
        assert out == -170 and isinstance(out, np.float32)
        out = xroms.wrap_longitude(np.array([190.0, 10.0]), "-180-180")
        assert isinstance(out, np.ndarray) and out.tolist() == [-170.0, 10.0]
        out = xroms.wrap_longitude(np.array(190.0), "-180-180")  # 0-d
        assert out == -170.0
        assert isinstance(xroms.wrap_longitude([190.0, 10.0], "-180-180"), np.ndarray)

    def test_dtypes_are_kept(self):
        out = xroms.wrap_longitude(np.array([190, -190, 10], dtype="float32"), "-180-180")
        assert out.dtype == np.float32 and out.tolist() == [-170.0, 170.0, 10.0]
        out = xroms.wrap_longitude(np.array([190, -190, 10]), "0-360")
        assert np.issubdtype(out.dtype, np.integer) and out.tolist() == [190, 170, 10]

    def test_inputs_are_not_modified(self):
        for convention in (*CONVENTIONS, None):
            lon = np.array([190.0, -190.0, 10.0])
            xroms.wrap_longitude(lon, convention)
            assert lon.tolist() == [190.0, -190.0, 10.0]
        unchanged = np.array([10.0, 20.0])  # nothing to do for this one
        out = xroms.wrap_longitude(unchanged)
        out[0] = 99.0
        assert unchanged.tolist() == [10.0, 20.0]

    @pytest.mark.parametrize("convention", ["180", "0-359", "0-360 ", "-180..180", "", 360, "degrees"])
    def test_an_unknown_convention_raises_naming_the_options(self, convention):
        for obj in (190.0, np.array([190.0]), xr.DataArray([190.0]), xr.Dataset({"lon": ("x", [190.0])})):
            with pytest.raises(ValueError, match="-180-180.*0-360"):
                xroms.wrap_longitude(obj, convention)

    def test_non_numbers_raise(self):
        with pytest.raises(TypeError, match="numbers"):
            xroms.wrap_longitude(np.array(["a", "b"]), "0-360")
        with pytest.raises(TypeError, match="numbers"):
            xroms.wrap_longitude(xr.DataArray(np.array(["a", "b"])), "0-360")
        with pytest.raises(TypeError, match="numbers"):
            xroms.straddles(np.array(["a", "b"]))


# ------------------------------------------------------------------------------- DataArrays
def lon_array(**kwargs):
    return xr.DataArray(
        np.array([[170.0, 190.0, -190.0], [-10.0, 360.0, 540.0]], **kwargs),
        dims=("eta_rho", "xi_rho"),
        name="lon_rho",
        attrs={"units": "degrees_east", "long_name": "longitude of rho-points"},
        coords={"eta_rho": [10, 11], "mask": (("eta_rho", "xi_rho"), np.ones((2, 3)))},
    )


class TestWrapDataArray:
    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_longitudes_keep_name_attrs_coords_and_dtype(self, convention):
        da = lon_array()
        out = xroms.wrap_longitude(da, convention)
        np.testing.assert_array_equal(out.values, ref_wrap(da.values, convention))
        assert out.name == "lon_rho" and out.dims == da.dims and out.dtype == da.dtype
        assert out.attrs == da.attrs
        xr.testing.assert_identical(out.coords.to_dataset(), da.coords.to_dataset())  # the mask and labels are untouched
        out.attrs["changed"] = True
        assert "changed" not in da.attrs

    def test_single_precision_stays_single_precision(self):
        out = xroms.wrap_longitude(lon_array(dtype="float32"), "-180-180")
        assert out.dtype == np.float32

    def test_inputs_are_not_modified(self):
        da = lon_array()
        before = da.copy(deep=True)
        for convention in (*CONVENTIONS, None):
            xroms.wrap_longitude(da, convention)
        xr.testing.assert_identical(da, before)

    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_dask_input_stays_lazy_and_equals_numpy(self, convention):
        da = lon_array()
        lazy = da.chunk({"xi_rho": 2})
        with dask.config.set(scheduler=no_compute):
            out = xroms.wrap_longitude(lazy, convention)
            assert out.chunks is not None
        xr.testing.assert_identical(out.compute(), xroms.wrap_longitude(da, convention))
        assert out.name == "lon_rho" and out.attrs == da.attrs

    def test_reading_the_convention_computes_only_the_extent_and_stays_lazy(self):
        lon = xr.DataArray(np.linspace(-2.0, 2.0, 9) % 360, dims="xi_rho", name="lon_rho").chunk({"xi_rho": 4})
        out = xroms.wrap_longitude(lon)
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, np.linspace(-2.0, 2.0, 9))

    def test_a_field_with_longitude_coordinates_has_its_coordinates_wrapped_not_its_values(self):
        lon = np.array([170.0, 175.0, 180.0, 185.0, 190.0])
        field = xr.DataArray(
            np.arange(5.0), dims="lon", coords={"lon": ("lon", lon, {"units": "degrees_east"})}, name="chl", attrs={"units": "mg m-3"}
        )
        out = xroms.wrap_longitude(field, "-180-180")
        np.testing.assert_array_equal(out.lon.values, [-175.0, -170.0, 170.0, 175.0, 180.0])  # sorted again
        np.testing.assert_array_equal(out.values, [3.0, 4.0, 0.0, 1.0, 2.0])  # and the data went with it
        assert out.name == "chl" and out.attrs == {"units": "mg m-3"} and out.lon.attrs == {"units": "degrees_east"}
        np.testing.assert_array_equal(field.lon.values, lon)  # the input is as it was

    def test_two_dimensional_coordinates_are_wrapped_and_never_reordered(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360")
        field = xr.DataArray(np.arange(45.0).reshape(5, 9), dims=C.CANONICAL["rho"], coords=ds.coords, name="temp")
        out = xroms.wrap_longitude(field)
        np.testing.assert_array_equal(out.values, field.values)
        np.testing.assert_allclose(out.lon_rho.values, lon)
        np.testing.assert_array_equal(out.lat_rho.values, field.lat_rho.values)

    def test_a_lazy_field_along_a_re_sorted_index_stays_lazy(self):
        field = xr.DataArray(
            np.arange(10.0).reshape(2, 5), dims=("lat", "lon"), name="chl", coords={"lon": [170.0, 175.0, 180.0, 185.0, 190.0], "lat": [0, 1]}
        )
        with dask.config.set(scheduler=no_compute):
            out = xroms.wrap_longitude(field.chunk({"lon": 2}), "-180-180")
            assert out.chunks is not None
        xr.testing.assert_identical(out.compute(), xroms.wrap_longitude(field, "-180-180"))
        np.testing.assert_array_equal(out.values[0], [3.0, 4.0, 0.0, 1.0, 2.0])

    def test_a_longitude_taken_out_of_a_dataset_is_wrapped_with_its_own_coordinate(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360")
        out = xroms.wrap_longitude(ds.lon_rho)
        np.testing.assert_allclose(out.values, lon)
        np.testing.assert_array_equal(out.lon_rho.values, out.values)

    def test_a_longitude_index_coordinate_taken_from_a_dataset_is_sorted(self):
        ds = xr.Dataset(coords={"lon": ("lon", [170.0, 175.0, 180.0, 185.0, 190.0])})
        out = xroms.wrap_longitude(ds.lon, "-180-180")
        np.testing.assert_array_equal(out.values, [-175.0, -170.0, 170.0, 175.0, 180.0])
        np.testing.assert_array_equal(out.lon.values, out.values)

    def test_a_dataarray_with_no_longitude_coordinates_is_longitudes(self):
        da = xr.DataArray([170.0, 190.0, np.nan], dims="x", attrs={"units": "degrees_east"})
        out = xroms.wrap_longitude(da, "-180-180")
        np.testing.assert_array_equal(out.values, [170.0, -170.0, np.nan])
        assert out.attrs == da.attrs
        # a standard_name marks it too
        named = xr.DataArray([190.0], dims="x", name="xlong", attrs={"standard_name": "longitude"})
        assert xroms.wrap_longitude(named, "-180-180").item() == -170.0


# -------------------------------------------------------------------------------- Datasets
class TestWrapDataset:
    @pytest.mark.parametrize("convention", CONVENTIONS)
    def test_every_longitude_is_wrapped_and_nothing_else(self, convention):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360", velocity=True)
        ds["temp"] = (C.CANONICAL["rho"], 100.0 + np.arange(45.0).reshape(5, 9), {"units": "Celsius"})
        ds.attrs["title"] = "grid"
        before = ds.copy(deep=True)
        out = xroms.wrap_longitude(ds, convention)
        for pos in POSITIONS:
            expected = ref_wrap(averaged(lon, pos), convention)
            np.testing.assert_allclose(out[f"lon_{pos}"].values, expected, atol=1e-12)
            assert out[f"lon_{pos}"].attrs == ds[f"lon_{pos}"].attrs
            assert f"lon_{pos}" in out.coords  # still coordinates
            np.testing.assert_array_equal(out[f"lat_{pos}"].values, ds[f"lat_{pos}"].values)
            assert out[f"lat_{pos}"].attrs == ds[f"lat_{pos}"].attrs
        np.testing.assert_array_equal(out["temp"].values, ds["temp"].values)  # a field is not a longitude
        assert out["temp"].attrs == {"units": "Celsius"} and out.attrs == {"title": "grid"}
        xr.testing.assert_identical(ds, before)  # the input is as it was

    def test_longitude_data_variables_are_wrapped_too(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360", as_coords=False)
        out = xroms.wrap_longitude(ds)
        assert "lon_rho" in out.data_vars
        np.testing.assert_allclose(out["lon_rho"].values, lon)

    def test_variables_are_longitudes_by_name_or_standard_name(self):
        ds = xr.Dataset(
            {
                "lon": ("x", [190.0]),
                "longitude": ("x", [190.0]),
                "lon_anything": ("x", [190.0]),
                "xlong": ("x", [190.0], {"standard_name": "longitude"}),
                "long_term_mean": ("x", [190.0]),  # not a longitude: lon_ needs the underscore
                "lonely": ("x", [190.0]),
                "lat_rho": ("x", [190.0]),
                "lon_label": ("x", ["east"]),  # not a number
            }
        )
        out = xroms.wrap_longitude(ds, "-180-180")
        for name in ("lon", "longitude", "lon_anything", "xlong"):
            assert out[name].item() == -170.0, name
        for name in ("long_term_mean", "lonely", "lat_rho"):
            assert out[name].item() == 190.0, name
        assert out["lon_label"].item() == "east"

    def test_a_dataset_without_longitudes_is_returned_unchanged(self):
        ds = xr.Dataset({"a": ("x", [1.0, 400.0])}, coords={"x": [0, 1]})
        for convention in (*CONVENTIONS, None):
            xr.testing.assert_identical(xroms.wrap_longitude(ds, convention), ds)

    def test_one_convention_is_shared_and_lon_rho_decides_it(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360")
        near = np.array([[359.0, 359.5, 359.75]] * 5)  # on its own, contiguous in 0-360 and left alone
        ds = ds.assign_coords(lon_near=(("eta_rho", "xi_near"), near))
        np.testing.assert_array_equal(xroms.wrap_longitude(ds["lon_near"]), near)
        out = xroms.wrap_longitude(ds)  # but the grid crosses Greenwich, so it follows lon_rho into -180-180
        np.testing.assert_allclose(out["lon_rho"].values, lon)
        np.testing.assert_allclose(out["lon_near"].values, [[-1.0, -0.5, -0.25]] * 5)

    def test_without_lon_rho_the_first_longitude_found_decides(self):
        straddler = ("x", [358.0, 359.0, 0.5, 1.0])  # crosses Greenwich
        other = ("y", [190.0])  # alone, it has nothing to decide
        crossing_first = xroms.wrap_longitude(xr.Dataset({"lon_a": straddler, "lon_b": other}))
        np.testing.assert_array_equal(crossing_first["lon_a"].values, [-2.0, -1.0, 0.5, 1.0])
        assert crossing_first["lon_b"].item() == -170.0  # wrapped with lon_a's convention
        other_first = xroms.wrap_longitude(xr.Dataset({"lon_b": other, "lon_a": straddler}))
        np.testing.assert_array_equal(other_first["lon_a"].values, [358.0, 359.0, 0.5, 1.0])  # lon_b ties: nothing moves
        assert other_first["lon_b"].item() == 190.0

    def test_a_longitude_index_coordinate_is_sorted_again_with_its_data(self):
        ds = xr.Dataset(
            {"t": ("lon", np.arange(5.0)), "grid": (("lat", "lon"), np.arange(10.0).reshape(2, 5))},
            coords={"lon": ("lon", [170.0, 175.0, 180.0, 185.0, 190.0], {"units": "degrees_east"}), "lat": [0.0, 1.0]},
        )
        out = xroms.wrap_longitude(ds, "-180-180")
        np.testing.assert_array_equal(out.lon.values, [-175.0, -170.0, 170.0, 175.0, 180.0])
        np.testing.assert_array_equal(out["t"].values, [3.0, 4.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out["grid"].values[1], [8.0, 9.0, 5.0, 6.0, 7.0])
        assert out.lon.attrs == {"units": "degrees_east"}
        assert out.indexes["lon"].is_monotonic_increasing
        np.testing.assert_array_equal(ds.lon.values, [170.0, 175.0, 180.0, 185.0, 190.0])

    def test_an_index_coordinate_that_stays_monotonic_is_left_as_it_is(self):
        ascending = xr.Dataset({"t": ("lon", [1.0, 2.0, 3.0])}, coords={"lon": [10.0, 20.0, 30.0]})
        descending = xr.Dataset({"t": ("lon", [1.0, 2.0, 3.0])}, coords={"lon": [30.0, 20.0, 10.0]})
        for ds in (ascending, descending):
            out = xroms.wrap_longitude(ds, "-180-180")
            xr.testing.assert_identical(out, ds)

    def test_two_dimensional_longitudes_are_never_reordered(self):
        ds, lon = lonlat_dataset(DATELINE, store="0-360", tilt=0.5)
        out = xroms.wrap_longitude(ds, "-180-180")
        np.testing.assert_allclose(out["lon_rho"].values, ref_wrap(lon, "-180-180"))
        assert out["lon_rho"].dims == ds["lon_rho"].dims  # the xi axis was not sorted: the longitudes jump where they cross
        assert (np.diff(out["lon_rho"].values, axis=1) < 0).any()

    def test_nan_stays_nan(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360")
        holes = ds["lon_rho"].values.copy()
        holes[2, 3] = np.nan
        out = xroms.wrap_longitude(ds.assign_coords(lon_rho=(C.CANONICAL["rho"], holes)))
        expected = ref_wrap(lon, "-180-180")
        expected[2, 3] = np.nan
        np.testing.assert_allclose(out["lon_rho"].values, expected)
        assert np.isnan(out["lon_rho"].values).sum() == 1

    @pytest.mark.parametrize("convention", [*CONVENTIONS, None])
    def test_dask_datasets_stay_lazy_and_equal_the_numpy_result(self, convention):
        ds, _ = lonlat_dataset(GREENWICH, store="0-360", velocity=True)
        lazy = ds.chunk({"eta_rho": 2, "eta_v": 2, "xi_rho": 3, "xi_u": 3})
        if convention is not None:  # naming the convention needs nothing computed
            with dask.config.set(scheduler=no_compute):
                out = xroms.wrap_longitude(lazy, convention)
        else:
            out = xroms.wrap_longitude(lazy)  # reads the extent of lon_rho, no more
        for pos in POSITIONS:
            assert out[f"lon_{pos}"].chunks is not None
        xr.testing.assert_allclose(out.compute(), xroms.wrap_longitude(ds, convention))


# ------------------------------------------------- reading the convention (convention=None)
class TestConventionFromTheData:
    def test_a_greenwich_crossing_domain_stored_in_0_360_becomes_plus_minus_180(self):
        stored = ref_wrap(GREENWICH, "0-360")
        assert stored.min() == 0.0 and stored.max() == 359.5  # 358..359.5 and 0..2
        np.testing.assert_allclose(xroms.wrap_longitude(stored), GREENWICH)
        ds, lon = lonlat_dataset(GREENWICH, store="0-360", tilt=0.3)
        np.testing.assert_allclose(xroms.wrap_longitude(ds)["lon_rho"].values, lon)

    def test_a_dateline_crossing_domain_stored_in_plus_minus_180_becomes_0_360(self):
        # the Pacific domain 77E..316E: stored as 77..180 and -180..-44
        unwrapped = np.linspace(77.0, 316.0, 40)
        stored = ref_wrap(unwrapped, "-180-180")
        assert stored.min() < -100 and stored.max() > 100
        np.testing.assert_allclose(xroms.wrap_longitude(stored), unwrapped)
        ds, lon = lonlat_dataset(DATELINE, store="-180-180", tilt=0.3)
        np.testing.assert_allclose(xroms.wrap_longitude(ds)["lon_rho"].values, lon)

    def test_a_domain_contiguous_in_both_is_returned_unchanged(self):
        for lon in (REGIONAL, REGIONAL + 360.0, np.arange(0.0, 360.0), np.arange(-180.0, 180.0), np.linspace(-100.0, -80.0, 9)):
            np.testing.assert_array_equal(xroms.wrap_longitude(lon), lon)
        turns = xr.Dataset(coords={"lon_rho": (C.CANONICAL["rho"], np.tile(REGIONAL + 720.0, (5, 1))), "lat_rho": (C.CANONICAL["rho"], np.zeros((5, 9)))})
        xr.testing.assert_identical(xroms.wrap_longitude(turns), turns)  # two turns out of range, but nothing to decide

    def test_one_value_and_nothing_finite_have_nothing_to_decide(self):
        assert xroms.wrap_longitude(190.0) == 190.0
        assert xroms.wrap_longitude(-190) == -190
        out = xroms.wrap_longitude(np.array([np.nan, np.nan]))
        assert np.isnan(out).all()
        assert xroms.wrap_longitude(np.array([])).size == 0

    def test_wrapping_twice_changes_nothing_more(self):
        for unwrapped in (GREENWICH, DATELINE, REGIONAL, np.linspace(77.0, 316.0, 40), np.arange(0.0, 360.0), np.linspace(-170.0, 120.0, 30)):
            for store in CONVENTIONS:
                once = xroms.wrap_longitude(ref_wrap(unwrapped, store))
                np.testing.assert_array_equal(xroms.wrap_longitude(once), once)

    def test_nan_does_not_change_the_decision(self):
        lon = np.array([358.0, np.nan, 359.0, 0.5, np.nan, 2.0])
        out = xroms.wrap_longitude(lon)
        np.testing.assert_allclose(out, [-2.0, np.nan, -1.0, 0.5, np.nan, 2.0])


# ------------------------------------------------------------------------------- straddles
class TestStraddles:
    def test_true_when_the_domain_crosses_the_prime_meridian(self):
        assert xroms.straddles([350.0, 355.0, 5.0, 10.0]) is True
        assert xroms.straddles(ref_wrap(GREENWICH, "0-360")) is True
        assert xroms.straddles(GREENWICH) is True  # stored in -180-180 it is the same domain
        assert xroms.straddles(np.linspace(-40.0, 20.0, 31)) is True  # the Atlantic
        assert xroms.straddles(np.array([359.9, 0.1])) is True

    def test_false_when_contiguous_in_0_360_or_in_both(self):
        assert xroms.straddles(REGIONAL) is False  # a tie
        assert xroms.straddles(np.linspace(-100.0, -80.0, 9)) is False
        assert xroms.straddles(np.arange(0.0, 360.0)) is False  # global
        assert xroms.straddles(np.arange(-180.0, 180.0)) is False
        assert xroms.straddles(np.linspace(150.0, 250.0, 40)) is False  # crosses the dateline instead
        assert xroms.straddles(ref_wrap(np.linspace(150.0, 250.0, 40), "-180-180")) is False
        assert xroms.straddles(190.0) is False
        assert xroms.straddles(np.array([])) is False

    def test_a_tie_is_not_broken_by_rounding(self):
        rng = np.random.default_rng(3)
        for lon0 in (-160.0, 10.0, 100.0):
            lon = np.linspace(lon0, lon0 + 35.0, 24) + rng.normal(0, 1e-9, 24)
            assert xroms.straddles(lon) is False

    def test_the_flag_agrees_with_the_convention_wrapping_picks(self):
        for unwrapped in (GREENWICH, DATELINE, REGIONAL, np.linspace(-170.0, 120.0, 30), np.linspace(100.0, 350.0, 30)):
            for store in CONVENTIONS:
                stored = ref_wrap(unwrapped, store)
                wrapped = xroms.wrap_longitude(stored)
                spans = {c: np.ptp(ref_wrap(stored, c)) for c in CONVENTIONS}
                assert np.ptp(wrapped) <= min(spans.values()) + 1e-9  # the most compact form
                if xroms.straddles(stored):
                    np.testing.assert_allclose(wrapped, ref_wrap(stored, "-180-180"))
                    assert spans["-180-180"] < spans["0-360"]

    def test_nan_is_ignored(self):
        assert xroms.straddles(np.array([350.0, np.nan, 10.0])) is True
        assert xroms.straddles(np.array([np.nan, np.nan])) is False

    @pytest.mark.parametrize("store", CONVENTIONS)
    def test_datasets_and_dataarrays_are_judged_by_lon_rho(self, store):
        ds, _ = lonlat_dataset(GREENWICH, store=store, velocity=True)
        assert xroms.straddles(ds) is True
        assert xroms.straddles(ds["lon_rho"]) is True
        field = xr.DataArray(np.zeros((5, 9)), dims=C.CANONICAL["rho"], coords=ds[["lon_rho", "lat_rho"]].coords)  # a field carrying lon_rho
        assert xroms.straddles(field) is True
        regional, _ = lonlat_dataset(REGIONAL, store=store, velocity=True)
        assert xroms.straddles(regional) is False

    def test_lon_rho_wins_over_other_longitudes_and_else_the_first_found(self):
        ds = xr.Dataset(
            coords={
                "lon_v": (C.CANONICAL["v"], np.tile(REGIONAL, (4, 1))),
                "lon_u": (C.CANONICAL["u"], np.tile(np.linspace(-2.0, 2.0, 8) % 360, (5, 1))),  # alone, it straddles
                "lon_rho": (C.CANONICAL["rho"], np.tile(REGIONAL, (5, 1))),
            }
        )
        assert xroms.straddles(ds["lon_u"]) is True
        assert xroms.straddles(ds) is False  # lon_rho is regional
        assert xroms.straddles(ds.drop_vars("lon_rho")) is False  # the first one found is lon_v, regional too
        assert xroms.straddles(ds.drop_vars(["lon_rho", "lon_v"])) is True  # now it is lon_u

    def test_a_dataset_without_longitudes_raises(self):
        with pytest.raises(ValueError, match="no longitude"):
            xroms.straddles(xr.Dataset({"a": ("x", [1.0])}))

    def test_dask_input_gives_a_bool(self):
        ds, _ = lonlat_dataset(GREENWICH, store="0-360")
        lazy = ds.chunk({"eta_rho": 2, "xi_rho": 3})
        assert xroms.straddles(lazy) is True
        assert xroms.straddles(lazy["lon_rho"]) is True
        assert xroms.straddles(lonlat_dataset(REGIONAL, store="0-360")[0].chunk({"xi_rho": 3})) is False


# ------------------------------------------------------------------------------ lonlat_at
def without_pairs(ds, positions=("u", "v", "psi")):
    """``ds`` without the stored lon/lat or x/y pairs at ``positions``."""
    drop = [n for pos in positions for n in (f"lon_{pos}", f"lat_{pos}", f"x_{pos}", f"y_{pos}") if n in ds.variables]
    return ds.drop_vars(drop)


class TestLonLatAt:
    def test_a_stored_pair_is_returned_as_it_is(self, layout):
        ds = merged(layout)
        can = C.canonicalize(ds)
        checked = 0
        for pos in POSITIONS:
            xname, yname = C.horizontal_coords(ds, pos)
            if xname is None:
                continue
            x, y = xroms.lonlat_at(ds, pos)
            assert (x.name, y.name) == (xname, yname)
            assert x.dims == y.dims == C.CANONICAL[pos]
            for got, name in ((x, xname), (y, yname)):
                np.testing.assert_array_equal(got.values, can[name].values)
                assert got.attrs == can[name].attrs
                assert set(got.coords) <= set(got.dims)  # nothing stale rides along
            checked += 1
        assert checked == {"rutgers": 4, "ucla": 1, "croco": 3, "remora": 3}[layout]  # which pairs each layout stores

    def test_averaging_the_rho_points_matches_a_plain_numpy_average(self, layout):
        ds = without_pairs(merged(layout))
        xname, yname = C.horizontal_coords(ds, "rho")
        assert xname is not None
        for pos in ("u", "v", "psi"):
            x, y = xroms.lonlat_at(ds, pos)
            assert x.name == f"{xname[:-3]}{pos}" and y.name == f"{yname[:-3]}{pos}"
            assert x.dims == y.dims == C.CANONICAL[pos]
            assert C.hposition(x) == pos
            np.testing.assert_allclose(x.values, averaged(ds[xname].values, pos), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(y.values, averaged(ds[yname].values, pos), rtol=1e-13, atol=1e-13)
            assert set(x.coords) <= set(x.dims) and set(y.coords) <= set(y.dims)

    def test_averaged_positions_equal_the_ones_the_grid_stores(self, layout):
        full = merged(layout)
        stripped = without_pairs(full)
        for pos in ("u", "v", "psi"):
            if C.horizontal_coords(full, pos)[0] is None:
                continue
            for got, want in zip(xroms.lonlat_at(stripped, pos), xroms.lonlat_at(full, pos), strict=True):
                np.testing.assert_allclose(got.values, want.values, rtol=1e-12)
                assert got.dims == want.dims and got.name == want.name

    def test_rho_points_are_returned_from_the_grid(self, layout):
        ds = merged(layout)
        x, y = xroms.lonlat_at(ds, "rho")
        xname, yname = C.horizontal_coords(ds, "rho")
        assert (x.name, y.name) == (xname, yname) and x.dims == C.CANONICAL["rho"]

    def test_rutgers_alias_dims_come_back_canonical_and_rename_back(self, rutgers):
        assert "eta_u" in rutgers.dims and "xi_psi" in rutgers.dims
        stripped = without_pairs(rutgers)
        for pos, rutgers_dims in (("u", ("eta_u", "xi_u")), ("v", ("eta_v", "xi_v")), ("psi", ("eta_psi", "xi_psi"))):
            for ds in (rutgers, stripped):  # stored, and averaged from rho
                x, y = xroms.lonlat_at(ds, pos)
                assert x.dims == y.dims == C.CANONICAL[pos]
                assert x.sizes[C.CANONICAL[pos][0]] == rutgers.sizes[rutgers_dims[0]]
                assert x.sizes[C.CANONICAL[pos][1]] == rutgers.sizes[rutgers_dims[1]]
                assert C.rename_like(x, rutgers).dims == rutgers_dims

    def test_ucla_grid_in_a_separate_file(self, ucla, ucla_romstools):
        _, grid = ucla
        neta, nxi = grid.sizes["eta_rho"], grid.sizes["xi_rho"]
        shapes = {"rho": (neta, nxi), "u": (neta, nxi - 1), "v": (neta - 1, nxi), "psi": (neta - 1, nxi - 1)}
        for pos, shape in shapes.items():
            x, y = xroms.lonlat_at(grid, pos)
            assert x.shape == y.shape == shape and x.dims == C.CANONICAL[pos]
        # roms-tools' grid file has u and v stored (not psi): those come back as stored, psi is averaged
        _, grid_rt = ucla_romstools
        stored = xroms.lonlat_at(grid_rt, "u")[0]
        np.testing.assert_array_equal(stored.values, grid_rt["lon_u"].values)
        psi = xroms.lonlat_at(grid_rt, "psi")[1]
        np.testing.assert_allclose(psi.values, averaged(grid_rt["lat_rho"].values, "psi"))

    def test_the_output_of_a_ucla_run_without_a_grid_says_what_is_missing(self, ucla):
        output, _ = ucla
        with pytest.raises(ValueError, match="neither lon_rho/lat_rho nor x_rho/y_rho"):
            xroms.lonlat_at(output, "u")
        with pytest.raises(ValueError, match="separate file"):
            xroms.lonlat_at(output, "rho")

    def test_names_and_attrs_of_averaged_points(self):
        ds, _ = lonlat_dataset(REGIONAL, store="0-360")
        for pos in ("u", "v", "psi"):
            lon, lat = xroms.lonlat_at(ds, pos)
            assert (lon.name, lat.name) == (f"lon_{pos}", f"lat_{pos}")
            assert lon.attrs == {"long_name": f"longitude of {pos}-points", "units": "degrees_east", "standard_name": "longitude"}
            assert lat.attrs == {"long_name": f"latitude of {pos}-points", "units": "degrees_north", "standard_name": "latitude"}
            assert not (set(lon.coords) | set(lat.coords)) - {*lon.dims, *lat.dims}  # lon_rho/lat_rho are not carried along

    def test_a_greenwich_crossing_grid_in_0_360_averages_across_the_seam(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360", tilt=0.4, velocity=True)
        assert ds["lon_rho"].min() == 0.0 and ds["lon_rho"].max() > 359.0
        stripped = without_pairs(ds)
        for pos in ("u", "v", "psi"):
            got, _ = xroms.lonlat_at(stripped, pos)
            np.testing.assert_allclose(got.values, ref_wrap(averaged(lon, pos), "0-360"), atol=1e-12)
            assert float(got.min()) >= 0.0 and float(got.max()) < 360.0
            assert 180.0 not in got.values  # 359.5 and 0.5 are 0.0, never 180.0
        u = xroms.lonlat_at(stripped, "u")[0]
        assert abs(u.values[0, 3] - 359.75) < 1e-12 and abs(u.values[0, 4] - 0.25) < 1e-12  # next to the seam
        np.testing.assert_allclose(u.values, xroms.lonlat_at(ds, "u")[0].values, atol=1e-12)  # as the grid stores them

    def test_the_seam_midpoint_of_a_0_360_grid_is_zero_not_180(self):
        ds = xr.Dataset(
            coords={
                "lon_rho": (C.CANONICAL["rho"], np.array([[359.5, 0.5], [359.5, 0.5]])),
                "lat_rho": (C.CANONICAL["rho"], np.array([[10.0, 10.0], [11.0, 11.0]])),
            }
        )
        lon_u, lat_u = xroms.lonlat_at(ds, "u")
        np.testing.assert_array_equal(lon_u.values, [[0.0], [0.0]])
        np.testing.assert_allclose(lat_u.values, [[10.0], [11.0]])
        lon_psi, _ = xroms.lonlat_at(ds, "psi")
        assert lon_psi.item() == 0.0

    def test_a_greenwich_crossing_grid_in_plus_minus_180_stays_in_plus_minus_180(self):
        ds, lon = lonlat_dataset(GREENWICH, store="-180-180", tilt=0.4)
        for pos in ("u", "v", "psi"):
            got, _ = xroms.lonlat_at(ds, pos)
            np.testing.assert_allclose(got.values, averaged(lon, pos), atol=1e-12)
            assert float(got.min()) < 0

    def test_a_dateline_crossing_grid_in_plus_minus_180_averages_across_the_seam(self):
        ds, lon = lonlat_dataset(DATELINE, store="-180-180", tilt=0.4, velocity=True)
        assert ds["lon_rho"].min() < -179 and ds["lon_rho"].max() == 180.0
        stripped = without_pairs(ds)
        for pos in ("u", "v", "psi"):
            got, _ = xroms.lonlat_at(stripped, pos)
            np.testing.assert_allclose(got.values, ref_wrap(averaged(lon, pos), "-180-180"), atol=1e-12)
            assert float(got.min()) > -180.0 and float(got.max()) <= 180.0
            assert 0.0 not in got.values  # 179.5 and -179.5 are 180, never 0

    def test_a_dateline_crossing_grid_in_0_360_is_left_in_0_360(self):
        ds, lon = lonlat_dataset(DATELINE, store="0-360", tilt=0.4)
        u, _ = xroms.lonlat_at(ds, "u")
        np.testing.assert_allclose(u.values, averaged(lon, "u"), atol=1e-12)
        assert float(u.min()) > 178 and float(u.max()) < 183  # still counting up through 180, not folded

    def test_a_regional_grid_is_a_plain_average(self):
        ds, lon = lonlat_dataset(REGIONAL, store="0-360", tilt=0.4)
        for pos in ("u", "v", "psi"):
            np.testing.assert_allclose(xroms.lonlat_at(ds, pos)[0].values, averaged(lon, pos), atol=1e-12)

    def test_single_precision_grids_stay_single_precision(self):
        ds, lon = lonlat_dataset(GREENWICH, store="0-360")
        single = ds.assign_coords({n: (ds[n].dims, ds[n].values.astype("float32"), ds[n].attrs) for n in ("lon_rho", "lat_rho")})
        assert single["lon_rho"].dtype == np.float32
        for grid in (single, single.chunk({"eta_rho": 2, "xi_rho": 4})):
            lon_psi, lat_psi = xroms.lonlat_at(grid, "psi")
            assert lon_psi.dtype == lat_psi.dtype == np.float32
            np.testing.assert_allclose(lon_psi.values, ref_wrap(averaged(lon, "psi"), "0-360"), atol=1e-4)

    def test_a_global_grid_is_a_plain_average(self):
        ds, lon = lonlat_dataset(np.arange(0.0, 360.0, 30.0), store="0-360")
        np.testing.assert_allclose(xroms.lonlat_at(ds, "u")[0].values, averaged(lon, "u"), atol=1e-12)

    def test_cartesian_grids_are_averaged_as_they_are(self):
        ds = merged("remora")
        assert "lon_rho" not in ds.variables
        ds["x_rho"].attrs["units"] = "m"
        stripped = without_pairs(ds)
        for pos in ("u", "v", "psi"):
            x, y = xroms.lonlat_at(stripped, pos)
            assert (x.name, y.name) == (f"x_{pos}", f"y_{pos}") and x.dims == C.CANONICAL[pos]
            np.testing.assert_allclose(x.values, averaged(ds["x_rho"].values, pos), rtol=1e-13)
            np.testing.assert_allclose(y.values, averaged(ds["y_rho"].values, pos), rtol=1e-13)
            assert x.attrs == {"long_name": f"x-location of {pos}-points", "units": "m"}
            assert y.attrs == {"long_name": f"y-location of {pos}-points"}  # no units to carry over, and no standard_name
        # Cartesian values are never wrapped, whatever they are
        big = stripped.assign_coords(x_rho=stripped["x_rho"] + 5000.0)
        np.testing.assert_allclose(xroms.lonlat_at(big, "u")[0].values, averaged(big["x_rho"].values, "u"), rtol=1e-13)

    def test_horizontal_index_coordinates_move_with_the_stagger(self):
        ds, _ = lonlat_dataset(REGIONAL, store="0-360")
        ds = ds.assign_coords(eta_rho=np.arange(5), xi_rho=np.arange(9))
        u, _ = xroms.lonlat_at(ds, "u")
        np.testing.assert_array_equal(u["xi_u"].values, np.arange(8))
        np.testing.assert_array_equal(u["eta_rho"].values, np.arange(5))
        _, lat_psi = xroms.lonlat_at(ds, "psi")
        assert lat_psi.dims == C.CANONICAL["psi"] and set(lat_psi.coords) == {"eta_v", "xi_u"}

    def test_one_dimensional_rectilinear_positions(self):
        ds = xr.Dataset(coords={"lon_rho": ("xi_rho", np.linspace(-2.0, 2.0, 9) % 360), "lat_rho": ("eta_rho", np.linspace(0.0, 4.0, 5))})
        lon_u, lat_u = xroms.lonlat_at(ds, "u")
        assert lon_u.dims == ("xi_u",) and lat_u.dims == ("eta_rho",)
        np.testing.assert_allclose(lon_u.values, averaged(np.linspace(-2.0, 2.0, 9), "u") % 360)
        lon_psi, lat_psi = xroms.lonlat_at(ds, "psi")
        assert lon_psi.dims == ("xi_u",) and lat_psi.dims == ("eta_v",)

    def test_lazy_positions_stay_lazy_and_equal_the_numpy_ones(self, layout):
        ds = without_pairs(merged(layout))
        lazy = chunked(ds)
        for pos in ("u", "v", "psi"):
            for got, want in zip(xroms.lonlat_at(lazy, pos), xroms.lonlat_at(ds, pos), strict=True):
                assert got.chunks is not None
                assert got.dims == want.dims and got.name == want.name and got.attrs == want.attrs
                np.testing.assert_allclose(got.compute().values, want.values, rtol=1e-13, atol=1e-13)

    def test_stored_positions_need_nothing_computed(self, rutgers):
        lazy = chunked(rutgers)
        with dask.config.set(scheduler=no_compute):
            for pos in POSITIONS:
                for got in xroms.lonlat_at(lazy, pos):
                    assert got.chunks is not None

    def test_inputs_are_not_modified(self, layout):
        ds = merged(layout)
        for stripped in (ds, without_pairs(ds)):
            before = stripped.copy(deep=True)
            for pos in POSITIONS:
                x, y = xroms.lonlat_at(stripped, pos)
                x.attrs["changed"] = True
                y.attrs["changed"] = True
            xr.testing.assert_identical(stripped, before)  # attrs included

    def test_bad_arguments(self, rutgers):
        for hcoord in (None, "w", "U", ("u", "v")):
            with pytest.raises(ValueError, match="hcoord must be one of"):
                xroms.lonlat_at(rutgers, hcoord)
        with pytest.raises(TypeError, match="Dataset"):
            xroms.lonlat_at(rutgers["temp"], "u")
        empty = xr.Dataset({"h": (C.CANONICAL["rho"], np.ones((3, 4)))})
        with pytest.raises(ValueError, match="neither lon_rho/lat_rho nor x_rho/y_rho"):
            xroms.lonlat_at(empty, "psi")
        # a lone lon_u without lat_u is not a pair: both are averaged from rho
        half = without_pairs(rutgers).assign_coords(lon_u=xroms.lonlat_at(rutgers, "u")[0] + 1.0)
        assert xroms.lonlat_at(half, "u")[0].attrs["standard_name"] == "longitude"


# ------------------------------------------------- cases ported from ocean-skill's tests
# ocean-skill's natural_convention names the convention a domain is contiguous in and
# harmonize_longitude moves data into one. wrap_longitude(obj) without a convention does both:
# it moves a domain into the convention it is contiguous in and leaves a tie alone, and
# straddles(obj) is True when that convention is -180..180 by a real margin rather than a tie.
# The cases below are ocean-skill's own, restated as plain expected values (ocean_skill is
# not imported).
class TestPortedFromOceanSkill:
    def pacific(self, ny=24, nx=40):
        """A curvilinear 2-D lane over 150..250E stored in 0-360, as a ROMS model would."""
        lon = np.linspace(150.0, 250.0, nx)[None, :] * np.ones((ny, 1))
        lat = np.linspace(-20.0, 20.0, ny)[:, None] * np.ones((1, nx))
        return xr.DataArray(
            np.cos(np.deg2rad(lat)) * 10 + 5,
            dims=("eta", "xi"),
            coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
            attrs={"units": "mmol/m^3"},
        )

    def global_reference(self):
        """A 2-degree global grid in -180..180, World Ocean Atlas style."""
        lat, lon = np.arange(-89.0, 90.0, 2.0), np.arange(-179.0, 180.0, 2.0)
        return xr.DataArray(np.full((lat.size, lon.size), 5.0), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "mmol/m^3"})

    def test_convention_follows_contiguity(self):
        pacific = self.pacific()
        assert xroms.straddles(pacific) is False
        xr.testing.assert_identical(xroms.wrap_longitude(pacific), pacific)  # contiguous in 0-360: nothing moves
        # the same lane stored in -180..180 comes back to 0-360 (150..250), not forced to -180..180
        in180 = pacific.assign_coords(lon=(("eta", "xi"), xroms.wrap_longitude(pacific.lon.values, "-180-180")))
        assert float(in180.lon.min()) < 0
        np.testing.assert_allclose(xroms.wrap_longitude(in180).lon.values, pacific.lon.values)
        assert xroms.wrap_longitude(in180).attrs == {"units": "mmol/m^3"}
        # a global reference ties, and ties are left alone
        reference = self.global_reference()
        assert xroms.straddles(reference) is False
        xr.testing.assert_identical(xroms.wrap_longitude(reference), reference)
        # the Atlantic (-40..20) crosses Greenwich: contiguous in -180..180 only
        atlantic = reference.sel(lon=slice(-40, 20))
        assert xroms.straddles(atlantic) is True
        xr.testing.assert_identical(xroms.wrap_longitude(atlantic), atlantic)
        stored_in_0360 = atlantic.assign_coords(lon=atlantic.lon.values % 360)
        assert float(stored_in_0360.lon.min()) == 1.0 and float(stored_in_0360.lon.max()) == 359.0
        out = xroms.wrap_longitude(stored_in_0360)
        np.testing.assert_array_equal(out.lon.values, atlantic.lon.values)  # back to -39..19, west to east

    def test_convention_is_not_flipped_by_float_noise_on_a_tied_span(self):
        # -160..-125 is an exact tie; the two wraps round differently at the 1e-14 level
        rng = np.random.default_rng(7)
        lon = np.linspace(-160.0, -125.0, 24) + rng.normal(0, 1e-9, 24)
        field = xr.DataArray(np.full((10, 24), 5.0), dims=("lat", "lon"), coords={"lat": np.linspace(-15.0, 15.0, 10), "lon": lon})
        assert xroms.straddles(field) is False
        out = xroms.wrap_longitude(field)
        np.testing.assert_array_equal(out.lon.values, lon)  # untouched, bit for bit
        # and the same for many irregular grids
        for seed in range(25):
            jitter = np.random.default_rng(seed).normal(0, 1e-9, 24)
            for lon0 in (-160.0, -90.0, 20.0):
                lon = np.linspace(lon0, lon0 + 35.0, 24) + jitter
                np.testing.assert_array_equal(xroms.wrap_longitude(lon), lon)

    def test_180_is_a_seam_not_a_wrap(self):
        # a domain that reaches, but does not cross, the dateline
        edge = np.linspace(120.0, 180.0, 41)
        assert xroms.straddles(edge) is False
        np.testing.assert_array_equal(xroms.wrap_longitude(edge), edge)  # not wrapped to -180
        overshoot = np.linspace(120.0, 180.0000001, 41)
        assert xroms.straddles(overshoot) is False
        np.testing.assert_array_equal(xroms.wrap_longitude(overshoot), overshoot)
        # and the seam value does not inflate a domain that crosses Greenwich and ends at 180: it is not
        # a tie (wrapping 180 to -180 would make it one), the domain is contiguous in -180..180 only
        from_greenwich = np.arange(-90.0, 181.0, 10.0)
        assert xroms.straddles(from_greenwich) is True
        np.testing.assert_array_equal(xroms.wrap_longitude(from_greenwich), from_greenwich)
        np.testing.assert_allclose(xroms.wrap_longitude(from_greenwich % 360), from_greenwich)  # 270..350 and 0..180 stored in 0-360
        # a genuine straddler through 180 is contiguous in 0-360
        straddler = np.linspace(170.0, 190.0, 41)
        assert xroms.straddles(straddler) is False
        np.testing.assert_array_equal(xroms.wrap_longitude(straddler), straddler)
        np.testing.assert_allclose(xroms.wrap_longitude(ref_wrap(straddler, "-180-180")), straddler)  # stored in -180..180

    def test_a_dateline_straddler_stored_in_180_with_a_sorted_index(self):
        """The real-data case: 80..180 and -180..-44 in -180..180, with the Atlantic empty."""
        lon = np.unique(np.concatenate([np.linspace(80.0, 180.0, 20), np.linspace(-180.0, -44.0, 15)]))
        field = xr.DataArray(
            np.stack([lon, lon * 2]), dims=("band", "lon"), coords={"lon": lon, "band": [0, 1]}, attrs={"units": "mmol/m^3"}
        )
        assert float(field.lon.max()) <= 180.0  # nothing exceeds 180, yet it crosses the dateline
        assert xroms.straddles(field) is False
        out = xroms.wrap_longitude(field)
        assert out.indexes["lon"].is_monotonic_increasing
        assert float(out.lon.min()) == 80.0 and abs(float(out.lon.max()) - 316.0) < 1e-12
        # the values moved with their longitudes: row 0 held the longitude itself (in -180..180)
        np.testing.assert_allclose(ref_wrap(out.values[0], "0-360"), out.lon.values)
        np.testing.assert_allclose(out.values[1], out.values[0] * 2)
        assert out.attrs == field.attrs
