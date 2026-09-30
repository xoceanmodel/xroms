"""Unit tests for the stateless core: xgcm engine, conventions, vertical, metrics, alignment."""

import re

import dask
import dask.array
import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import _xgcm, conventions as C, metrics as M, vertical as V
from xroms._align import GridMismatchError, select_like
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import INPUT, chunked, merged


@pytest.fixture
def canon():
    ds, ex = syn._canonical(2, 6, 9, 12, 2, 20.0, 5.0, 2.0, False, 0.0, False)
    return ds, ex


def _edge_pad(a, dim):
    p = a.variable.pad({dim: (1, 1)}, mode="edge")
    return 0.5 * (p.isel({dim: slice(None, -1)}).data + p.isel({dim: slice(1, None)}).data)


def _time_copies(arr, nt=2):
    """``arr`` with a leading ``ocean_time`` dim, as xr.open_mfdataset(data_vars="all") makes it.

    Real copies are identical; here later ones are altered, so that a function
    that compared the copies, or took the wrong one, would show it.
    """
    return xr.concat([arr + k for k in range(nt)], dim="ocean_time")


def _first_record_only(arr, nt=2):
    """Like :func:`_time_copies`, lazily, and computing any record but the first raises."""

    def record(k):
        if k:
            raise AssertionError(f"record {k} was read")
        return np.asarray(arr.values)[None]

    shape = (1,) + arr.shape
    blocks = [dask.array.from_delayed(dask.delayed(record)(k), shape=shape, dtype=arr.dtype) for k in range(nt)]
    return xr.DataArray(dask.array.concatenate(blocks), dims=("ocean_time",) + arr.dims)


def _ucla_time(values, **attrs):
    """UCLA-style time: an ``ocean_time`` variable on a ``time`` dim without coordinate."""
    return xr.Dataset({"ocean_time": ("time", np.asarray(values, dtype=float), attrs)})


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

    @pytest.mark.filterwarnings("ignore:In a future version of xarray the default value:FutureWarning")
    @pytest.mark.parametrize(
        "func",
        [xroms.z, lambda ds: ds.xroms.z_rho, lambda ds: xroms.ddxi(ds.temp, ds), lambda ds: xroms.ddz(ds.temp, ds)],
        ids=["z", "z_rho", "ddxi", "ddz"],
    )
    def test_open_mfdataset_defaults_match_minimal(self, func):
        """Plain xr.open_mfdataset (data_vars="all") gives every parameter a time dim; results stay the same."""
        files = [INPUT / "ocean_his_0001.nc", INPUT / "ocean_his_0002.nc"]
        minimal_kw = dict(data_vars="minimal", coords="minimal", compat="override")
        with xr.open_mfdataset(files) as plain, xr.open_mfdataset(files, **minimal_kw) as minimal:
            if "ocean_time" not in plain.Cs_r.dims:
                pytest.skip("this xarray no longer gives parameters a time dim by default")
            xr.testing.assert_allclose(func(plain), func(minimal))

    @pytest.mark.parametrize("time_last", [False, True])
    def test_vertical_params_parameters_with_a_time_dim(self, rutgers, time_last):
        """The level dim is found by name and the first record is used, wherever the time dim sits."""
        names = ["Cs_r", "Cs_w", "hc", "theta_s", "theta_b", "Vtransform", "Vstretching"]
        timed = rutgers.assign({name: _time_copies(rutgers[name]) for name in names})
        if time_last:
            timed = timed.assign({name: timed[name].transpose(..., "ocean_time") for name in ("Cs_r", "Cs_w")})
        assert timed.Cs_r.dims == (("s_rho", "ocean_time") if time_last else ("ocean_time", "s_rho"))
        got, want = C.vertical_params(timed), C.vertical_params(rutgers)
        assert (got.Vtransform, got.hc) == (want.Vtransform, want.hc)
        assert got.Cs_r.dims == ("s_rho",) and got.Cs_w.dims == ("s_w",)
        for name in ("Cs_r", "Cs_w", "sigma_r", "sigma_w"):
            xr.testing.assert_allclose(getattr(got, name), getattr(want, name))
        xr.testing.assert_allclose(V.z(timed), V.z(rutgers))
        assert V.z(chunked(timed)).chunks is not None

    def test_vertical_params_reads_only_the_first_record(self, rutgers):
        """The copies are identical by construction, and reading all of them would read every file."""
        names = ["Cs_r", "Cs_w", "hc", "theta_s", "theta_b", "Vtransform", "Vstretching"]
        lazy = rutgers.assign({name: _first_record_only(rutgers[name]) for name in names})
        got, want = C.vertical_params(lazy), C.vertical_params(rutgers)  # computing another record raises
        assert (got.Vtransform, got.hc) == (want.Vtransform, want.hc)
        np.testing.assert_allclose(got.Cs_r.values, want.Cs_r.values)
        np.testing.assert_allclose(V.z(lazy).values, V.z(rutgers).values)

    def test_vertical_params_level_dim_with_another_name(self, rutgers):
        """A 1-D profile on a differently named dim is taken as the profile; a 2-D one is ambiguous."""
        one_d = rutgers.assign(Cs_r=("lev", rutgers.Cs_r.values))
        got = C.vertical_params(one_d).Cs_r
        assert got.dims == ("s_rho",)
        np.testing.assert_allclose(got.values, rutgers.Cs_r.values)
        two_d = rutgers.assign(Cs_r=(("ocean_time", "lev"), np.tile(rutgers.Cs_r.values, (2, 1))))
        with pytest.raises(ValueError, match="rename its vertical dim"):
            C.vertical_params(two_d)

    def test_vertical_params_s_w_must_follow_s_rho(self, rutgers):
        """Selecting only s_rho leaves s_w at full length, and dz used to return too many levels."""
        sub = rutgers.isel(s_rho=slice(2, 5))
        with pytest.raises(ValueError, match="together"):
            C.vertical_params(sub)
        with pytest.raises(ValueError, match="together"):
            V.dz(sub)
        both = rutgers.isel(s_rho=slice(2, 5), s_w=slice(2, 6))
        assert V.dz(both).sizes["s_rho"] == 3
        np.testing.assert_allclose(V.dz(both).values, V.dz(rutgers).isel(s_rho=slice(2, 5)).values)

    def test_vertical_params_missing_vtransform_says_how_to_set_it(self, rutgers):
        bare = rutgers.drop_vars("Vtransform")
        with pytest.raises(ValueError, match=r"ds\['Vtransform'\] = 1"):
            C.vertical_params(bare)
        bare["Vtransform"] = 1  # what the message says to do
        assert C.vertical_params(bare).Vtransform == 1
        xr.testing.assert_allclose(V.z(bare), V.z(rutgers.drop_vars("Vtransform"), Vtransform=1))

    @pytest.mark.parametrize("bad", [2.5, 0, 3, float("nan")])
    def test_vertical_params_rejects_bad_vtransform(self, rutgers, bad):
        """Vtransform=2.5 used to become 2 silently."""
        bare = rutgers.drop_vars("Vtransform")
        croco = syn.make_dataset("croco")
        croco.attrs["Vtransform"] = bad
        for call in (
            lambda: C.vertical_params(bare, Vtransform=bad),
            lambda: C.vertical_params(bare.assign(Vtransform=bad)),
            lambda: C.vertical_params(croco),
        ):
            with pytest.raises(ValueError, match="Vtransform must be 1 or 2"):
                call()
        with pytest.raises(ValueError, match="Vtransform must be 1 or 2"):
            C.vertical_params(bare, Vtransform="x")

    def test_vertical_params_accepts_integer_valued_vtransform(self, rutgers):
        bare = rutgers.drop_vars("Vtransform")
        assert [C.vertical_params(bare, Vtransform=v).Vtransform for v in (1, 2.0, np.int64(2))] == [1, 2, 2]

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

    def test_rename_like_canonicalizes_for_a_canonical_dataset(self, rutgers):
        """rename_like returns ds's naming; for canonical ds that was left as the Rutgers aliases."""
        can = C.canonicalize(rutgers)
        for name in ("u", "v", "mask_psi", "temp"):
            assert C.rename_like(rutgers[name], can).dims == can[name].dims
        assert (C.rename_like(rutgers.u, can) + can.u).dims == can.u.dims
        u = can.u
        assert C.rename_like(u, can) is u
        # the other way round still gives the aliases
        assert C.rename_like(rutgers.u, rutgers).dims == rutgers.u.dims

    def test_canonicalize_renames_the_sgrid_topology(self, remora, rutgers):
        """The topology attrs named the alias dims after canonicalize, no longer describing the dims."""
        can = C.canonicalize(remora)
        dim_attrs = {key: value for key, value in can["grid"].attrs.items() if key.endswith("_dimensions")}
        assert dim_attrs
        for key, value in dim_attrs.items():
            assert not set(C.ALIASES) & set(re.findall(r"\w+", value)), key
        # the text xroms writes for a canonically named Dataset
        assert dim_attrs == {key: C.sgrid_attrs(can)[key] for key in dim_attrs}
        assert "xi_psi" in remora["grid"].attrs["face_dimensions"]  # the input is left alone
        assert C.canonicalize(can) is can
        # decorated before or after renaming, xroms's own topology reads the same
        before, after = C.canonicalize(C.add_cf_attrs(rutgers)), C.add_cf_attrs(C.canonicalize(rutgers))
        assert before["grid"].attrs == after["grid"].attrs

    def test_canonicalize_renames_sgrid_corner_dims(self):
        topology = {
            "cf_role": "grid_topology",
            "node_dimensions": "xi_psi eta_psi",
            "face_dimensions": "xi_rho: xi_psi (padding: both)",
        }
        ds = xr.Dataset(
            {
                "lon_psi": (("eta_psi", "xi_psi"), np.zeros((4, 5))),
                "h": (("eta_rho", "xi_rho"), np.zeros((3, 4))),
                "grid": ((), 0, topology),
            }
        )
        attrs = C.canonicalize(ds)["grid"].attrs
        assert attrs["node_dimensions"] == "xi_vert eta_vert"
        assert attrs["face_dimensions"] == "xi_rho: xi_vert (padding: both)"

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

    def test_decode_time_cf_units_with_decode_times_false(self):
        """Data opened without decoding still carries CF units ("hours since ..."); decode_time reads them."""
        path = INPUT / "ocean_his_0001.nc"
        with xr.open_dataset(path, decode_times=False) as raw, xr.open_dataset(path) as reference:
            assert not np.issubdtype(raw.ocean_time.dtype, np.datetime64)
            decoded = C.decode_time(raw)
            assert np.issubdtype(decoded.ocean_time.dtype, np.datetime64)
            np.testing.assert_array_equal(decoded.ocean_time.values, reference.ocean_time.values)
            assert not np.issubdtype(raw.ocean_time.dtype, np.datetime64)  # the input is not modified

    @pytest.mark.parametrize(
        "unit,seconds",
        [
            ("second", 1), ("seconds", 1), ("s", 1),
            ("minutes", 60),
            ("hour", 3600), ("hours", 3600),
            ("day", 86400), ("days", 86400),
        ],
    )
    def test_decode_time_units_without_an_epoch(self, unit, seconds):
        """UCLA-style: the epoch is in the long_name and the units only give the step."""
        ds = _ucla_time([0, 1, 2.5], units=unit, long_name="Time since 2000/01/01")
        steps = (np.array([0, 1, 2.5]) * seconds * 1e9).astype("timedelta64[ns]")
        want = np.datetime64("2000-01-01T00:00:00", "ns") + steps
        np.testing.assert_array_equal(C.decode_time(ds).time.values, want)

    def test_decode_time_unsupported_units(self):
        with pytest.raises(ValueError, match="unsupported time units"):
            C.decode_time(_ucla_time([0, 1], units="weeks", long_name="Time since 2000/01/01"))

    def test_decode_time_epoch_outside_the_datetime64_range(self):
        """A "since 0001/01/01" epoch raised OverflowError; now it decodes if the dates fit, else explains."""
        late = _ucla_time([6.3e10, 6.3e10 + 86400], units="second", long_name="Time since 0001/01/01")
        assert C.decode_time(late).time.values[0] == np.datetime64("1997-05-23T16:00:00")
        early = _ucla_time([0, 86400], units="second", long_name="Time since 0001/01/01")
        far = _ucla_time([0, 1e30], units="second", long_name="Time since 2000/01/01")
        for ds in (early, far):
            with pytest.raises(ValueError, match="cftime") as err:
                C.decode_time(ds)
            assert "reference_date" in str(err.value)

    def test_decode_time_epoch_at_1970(self):
        """np.datetime64 of the Unix epoch is falsy, which once made it look like no epoch at all."""
        ds = _ucla_time([0, 86400], units="second", long_name="Time since 1970/01/01")
        assert C.decode_time(ds).time.values[1] == np.datetime64("1970-01-02")

    def test_decode_time_units_and_long_name_epochs(self):
        """A long_name epoch used to override the CF units silently."""
        agree = _ucla_time([0, 1], units="days since 2000-01-01", long_name="Time since 2000/01/01")
        assert C.decode_time(agree).time.values[1] == np.datetime64("2000-01-02")
        clash = _ucla_time([0, 1], units="days since 2000-01-01", long_name="Time since 1995/01/01")
        with pytest.raises(ValueError, match="different epochs"):
            C.decode_time(clash)
        undated = _ucla_time([0, 1], units="days since 2000-01-01", long_name="time since initialization")
        assert C.decode_time(undated).time.values[1] == np.datetime64("2000-01-02")

    def test_decode_time_honours_the_calendar(self):
        """calendar was ignored: day 365 of a noleap year count is 1 January, not 31 December."""

        def dates(ds):  # cftime dates for these calendars; the first ten characters are the date either way
            return [str(t)[:10] for t in C.decode_time(ds).time.values]

        noleap = _ucla_time([0, 365, 730], units="days since 2000-01-01", calendar="noleap")
        assert dates(noleap) == ["2000-01-01", "2001-01-01", "2002-01-01"]
        day360 = _ucla_time([0, 360], units="days since 2000-01-01", calendar="360_day")
        assert dates(day360) == ["2000-01-01", "2001-01-01"]
        # an epoch from the long_name goes through the same calendar
        no_cf_units = _ucla_time([0, 365], units="day", long_name="Time since 2000/01/01", calendar="noleap")
        assert dates(no_cf_units) == ["2000-01-01", "2001-01-01"]
        # calendars datetime64 can hold stay datetime64
        proleptic = _ucla_time([0, 1], units="days since 2000-01-01", calendar="proleptic_gregorian")
        assert np.issubdtype(C.decode_time(proleptic).time.dtype, np.datetime64)

    def test_decode_time_leaves_decoded_times_alone(self):
        """datetime64 and cftime ocean_time both count as decoded (cftime raised "cannot find a reference date")."""
        cftime = pytest.importorskip("cftime")
        dates = np.array([cftime.DatetimeNoLeap(2000, 1, 1 + k) for k in range(3)], dtype=object)
        as_coord = xr.Dataset({"x": ("ocean_time", np.arange(3))}, coords={"ocean_time": dates})
        assert C.decode_time(as_coord) is as_coord
        on_time = xr.Dataset({"x": ("time", np.arange(3))}, coords={"time": dates})
        assert C.decode_time(on_time) is on_time
        # in a variable on a dim without coordinate, they become that dim's index
        for values in (np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"), dates[:2]):
            ds = xr.Dataset({"ocean_time": ("time", values)})
            np.testing.assert_array_equal(C.decode_time(ds).time.values, values)
            assert "time" not in ds.indexes  # the input is not modified

    def test_decode_time_reference_date_supplies_an_unreadable_epoch(self):
        ds = _ucla_time([0, 3600], units="seconds since initialization")
        with pytest.raises(ValueError, match="reference_date"):
            C.decode_time(ds)
        assert C.decode_time(ds, reference_date="2000-01-01").time.values[1] == np.datetime64("2000-01-01T01:00:00")

    def test_rho0(self, rutgers, ucla):
        assert C.rho0(ucla[0]) == 1027.4 and C.rho0(rutgers) == 1025.0

    def test_sgrid(self, remora, rutgers):
        assert C.sgrid_topology(remora)["variable"] == "grid"
        assert C.sgrid_topology(rutgers) is None
        dec = C.add_cf_attrs(rutgers)
        assert C.sgrid_topology(dec) is not None and "grid" not in rutgers

    def test_add_cf_attrs_topology_is_not_remora(self):
        """The topology add_cf_attrs writes made any Dataset look like REMORA, which means Vtransform 2."""
        ds = syn.make_dataset("rutgers", vtransform=1).drop_vars("Vtransform")
        decorated = C.add_cf_attrs(ds)
        assert C.sgrid_topology(decorated) is not None
        for each in (ds, decorated, C.canonicalize(decorated)):
            with pytest.raises(ValueError, match="cannot determine Vtransform"):
                xroms.z(each)
        assert C.vertical_params(decorated, Vtransform=1).Vtransform == 1
        # a REMORA file's own topology still says what it is, decorated or not
        assert C.vertical_params(merged("remora")).Vtransform == 2
        assert C.vertical_params(C.add_cf_attrs(merged("remora"))).Vtransform == 2

    def test_add_cf_attrs_is_not_an_xgcm_preparation(self, rutgers):
        """xgcm's autoparse rejects the decorated Dataset, so the docs point to the accessor."""
        doc = C.add_cf_attrs.__doc__
        assert "xgcm_grid" in doc and "cf-xarray and xgcm" not in doc
        assert {"X", "Y"} <= set(rutgers.xroms.xgcm_grid().axes)

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
