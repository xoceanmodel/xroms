"""Dev-only parity with ocean-skill's ROMS helpers: where they agree, so do the numbers.

Skipped unless ocean-skill is installed (it is not a dependency of xroms). ocean-skill is replacing its own copies of
these helpers (``ocean_skill.roms``, ``mld`` and ``align``) with xroms calls. Every test builds one synthetic ROMS
output (or reads one of ``xroms/tests/input``), runs ocean-skill's own function on it, through ``roms.standardize`` as
its catalog does, and the xroms replacement on the raw output, and compares the two:

* identical by construction (indexing, averaging, the same arithmetic in the same order): bit for bit;
* the same formula in another order of operations: ``rtol=1e-12``, or the looser tolerance the test states and explains;
* different on purpose: the test pins the difference and says which side does what, and why;
* a real discrepancy: ``xfail(strict=True)`` with the reason (it goes red once the discrepancy is gone).

Conventions to know when swapping a call:

* ocean-skill's ``z_rho``/``z_w`` and its ``to_depth`` axis are heights from the mean sea level, negative down: the
  xroms defaults (``positive="up"``, ``reference="mean_sea_level"``). Depths positive down are ``positive="down"``;
* ocean-skill takes Vtransform and hc from its catalog block, else from the file's variables; xroms reads the Dataset
  (Vtransform also as a keyword), so a catalog value has to be put on the Dataset or passed;
* a depth without a free surface is ``zeta=0`` in xroms (ocean-skill falls back to zeta=0 when it has none, which its
  compare pipeline relies on); a time step cut from a UCLA Dataset has no time label left, so its ``zeta=`` is passed;
* ocean-skill's mixed layer depth is NaN where nothing crosses the threshold, from 10 m down: ``xroms.mld`` needs
  ``reference_depth=10`` (its default is 0) and ``fill="nan"`` (its default is the depth of the bottom);
* a vertical interpolation puts the new dimension last in ocean-skill (xgcm's habit), where the vertical was in xroms.

Sections: vertical coordinate, surface, depth bands, interpolation to depths and densities, time decoding, mixed layer
depth, longitude, velocities, dask-backed inputs, the calls on the standardized Dataset itself, and the horizontal
grid (areas, resolution, nearest cell, windows).
"""

import pytest


pytest.importorskip("ocean_skill")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import xroms  # noqa: E402

from ocean_skill import align, mld, roms  # noqa: E402
from ocean_skill.build import ROMS_STANDARD_NAMES, _decode_times, _roms_metadata  # noqa: E402
from xroms.tests import _synthetic as syn  # noqa: E402
from xroms.tests.conftest import INPUT, chunked, merged  # noqa: E402


RTOL = 1e-12

#: ocean-skill's CF names for the model variables the tests slice and average
TEMP, SALT = ROMS_STANDARD_NAMES["temp"], ROMS_STANDARD_NAMES["salt"]
U, V = ROMS_STANDARD_NAMES["u"], ROMS_STANDARD_NAMES["v"]
FIELDS = {"temp": TEMP, "salt": SALT}


# --- inputs and comparisons -------------------------------------------------------------------------


def osk_inputs(*, land=False, vtransform=2, zeta=True, zeta_scale=1.0, **kwargs):
    """One synthetic output and what ocean-skill's catalog builder makes of it: ``(ds, meta)``.

    ``ds`` is what xroms is given: output and grid merged (UCLA-style, with roms-tools' sigma and Cs variables) and
    Vtransform in the file, so both libraries read the same one. ``roms.standardize(ds, meta)`` is what ocean-skill
    works on. ``zeta=False`` drops the free surface and ``zeta_scale`` scales it (``kwargs`` go to the generator).
    """
    ds = merged("ucla", romstools_grid=True, land=land, vtransform=vtransform, **kwargs)
    ds["Vtransform"] = ((), vtransform)
    if zeta_scale != 1.0:
        ds["zeta"] = ds["zeta"] * zeta_scale
    if not zeta:
        ds = ds.drop_vars("zeta")
    meta = _roms_metadata(ds)
    meta["standard_names"] = {name: cf for name, cf in ROMS_STANDARD_NAMES.items() if name in ds.variables}
    return ds, meta


def real(name, **kwargs):
    """A file of ``xroms/tests/input``, loaded."""
    return xr.load_dataset(INPUT / f"{name}.nc", **kwargs)


def with_varying_angle(ds):
    """``ds`` with a grid angle that changes from point to point, as on a rotated curvilinear grid."""
    eta, xi = (xr.DataArray(np.arange(ds.sizes[dim]), dims=dim) for dim in ("eta_rho", "xi_rho"))
    return ds.assign(angle=(0.3 + 0.05 * xi - 0.04 * eta).transpose("eta_rho", "xi_rho"))


def with_mixed_layer(ds, depth=12.0, inverted=False):
    """``ds`` with temperature uniform down to ``depth`` m and a thermocline below, at constant salinity.

    ``inverted`` makes the water below the mixed layer warmer, so that density decreases with depth.
    """
    z = xroms.z(ds)
    slope = -0.05 if inverted else 0.05
    ocean = ds.temp.notnull()
    temp = xr.where(z > -depth, 10.0, 10.0 + slope * (z + depth)).where(ocean).transpose(*ds.temp.dims)
    salt = xr.where(ocean, 35.0, np.nan).transpose(*ds.salt.dims)
    return ds.assign(temp=temp.assign_attrs(ds.temp.attrs), salt=salt.assign_attrs(ds.salt.attrs))


def _aligned(actual, expected, ordered=False):
    """Both as arrays, ``actual``'s dimensions put in the order of ``expected``'s (they must be the same set).

    With ``ordered`` the dimensions must come in the same order to begin with.
    """
    if isinstance(actual, xr.Variable):
        actual = xr.DataArray(actual)
    if isinstance(actual, xr.DataArray) and isinstance(expected, xr.DataArray):
        assert set(actual.dims) == set(expected.dims), (actual.dims, expected.dims)
        assert not ordered or actual.dims == expected.dims, (actual.dims, expected.dims)
        actual = actual.transpose(*expected.dims)
    return np.asarray(actual), np.asarray(expected)


def same(actual, expected, ordered=False):
    """Bit for bit, NaN where it is NaN (and the dimensions in the same order, if ``ordered``)."""
    np.testing.assert_array_equal(*_aligned(actual, expected, ordered))


def close(actual, expected, rtol=RTOL, atol=0.0, ordered=False):
    """Within ``rtol`` (and ``atol``), NaN where it is NaN (and the dimensions in the same order, if ``ordered``)."""
    np.testing.assert_allclose(*_aligned(actual, expected, ordered), rtol=rtol, atol=atol)


# --- 1. vertical coordinate ---------------------------------------------------------------------------


class TestVerticalCoordinate:
    """add_depth_coord, add_interface_coord, _s_to_z and _vertical_params against xroms.z, compute_depth and
    vertical_params."""

    @pytest.mark.parametrize("land", [False, True])
    def test_z_rho_and_z_w_are_xroms_z(self, vtransform, land):
        # Heights from the mean sea level, negative down, as xroms' defaults; the arithmetic is the same in the same
        # order, so the two are bit for bit equal, NaN over land where zeta is.
        ds, meta = osk_inputs(land=land, vtransform=vtransform)
        std = roms.standardize(ds, meta)
        same(std.z_rho, xroms.z(ds), ordered=True)
        same(roms.add_interface_coord(std, meta).z_w, xroms.z(ds, scoord="s_w"), ordered=True)

    def test_zero_zeta_is_zeta_zero(self, vtransform):
        # the zeta-free mesh ocean-skill draws a section's depth axis on: finite over land, as h is
        ds, meta = osk_inputs(land=True, vtransform=vtransform)
        std = roms.standardize(ds, meta)
        same(roms.add_depth_coord(std, meta, zero_zeta=True).z_rho, xroms.z(ds, zeta=0))
        same(roms.add_interface_coord(std, meta, zero_zeta=True).z_w, xroms.z(ds, scoord="s_w", zeta=0))

    def test_classic_rutgers_layout(self, vtransform):
        # A labelled s_rho coordinate, hc and Vtransform as 0-d variables and time on a dimension called ocean_time:
        # ocean-skill moves the sigma values to sigma_r (a bare s_rho dim) first, xroms reads them where they are.
        ds = xroms.canonicalize(syn.make_dataset("rutgers", vtransform=vtransform))
        meta = {"vertical": {"s_dim": "s_rho"}}
        std = roms._normalize_classic_layout(ds)
        same(roms.add_depth_coord(std, meta).z_rho, xroms.z(ds))
        same(roms.add_interface_coord(std, meta).z_w, xroms.z(ds, scoord="s_w"))

    def test_s_to_z_is_compute_depth(self, vtransform):
        rng = np.random.default_rng(0)
        h, zeta = rng.uniform(25.0, 500.0, (4, 5)), rng.uniform(-1.0, 1.0, (4, 5))
        sigma = np.linspace(-0.95, -0.05, 7)[:, None, None]
        cs = syn.stretching(sigma, 5.0, 2.0)
        expected = roms._s_to_z(sigma, cs, h, zeta, 20.0, vtransform)
        kwargs = {"hc": 20.0, "Cs": cs, "sigma": sigma, "Vtransform": vtransform}
        same(xroms.compute_depth(h, zeta, **kwargs), expected)
        same(xroms.compute_depth(h, zeta, positive="down", **kwargs), -expected)

    def test_sign_and_reference_conventions(self):
        # ocean-skill has the one convention; xroms labels it and offers the others
        ds, meta = osk_inputs()
        std = roms.standardize(ds, meta)
        z = xroms.z(ds)
        assert (z.attrs["positive"], z.attrs["vertical_reference"]) == ("up", "mean_sea_level")
        same(std.z_rho, z)
        same(xroms.z(ds, positive="down"), -std.z_rho)
        close(xroms.z(ds, reference="surface"), std.z_rho - std[ROMS_STANDARD_NAMES["zeta"]])
        close(xroms.z(ds, reference="bottom"), std.z_rho + std.h)

    def test_vertical_params(self, vtransform):
        ds, meta = osk_inputs(vtransform=vtransform)
        params = xroms.vertical_params(ds)
        assert roms._vertical_params(ds, meta) == (params.hc, params.Vtransform) == (20.0, vtransform)

    def test_catalog_values_versus_the_dataset(self):
        # ocean-skill's catalog block beats the file; xroms reads the Dataset (Vtransform is also a keyword)
        ds, meta = osk_inputs(vtransform=2)
        meta["vertical"].update(hc=30.0, Vtransform=1)
        assert roms._vertical_params(ds, meta) == (30.0, 1)
        params = xroms.vertical_params(ds)
        assert (params.hc, params.Vtransform) == (20.0, 2)
        params = xroms.vertical_params(ds.assign(hc=30.0), Vtransform=1)
        assert (params.hc, params.Vtransform) == (30.0, 1)

    def test_a_missing_vtransform_is_2_for_ocean_skill_only(self):
        # a classic file without Vtransform: ocean-skill takes 2 (UCLA's and roms-tools'), xroms will not guess
        ds = xroms.canonicalize(syn.make_dataset("rutgers")).drop_vars("Vtransform")
        assert roms._vertical_params(roms._normalize_classic_layout(ds), {})[1] == 2
        with pytest.raises(ValueError, match="Vtransform"):
            xroms.vertical_params(ds)
        assert xroms.vertical_params(ds, Vtransform=2).Vtransform == 2


# --- 2. surface ------------------------------------------------------------------------------------


class TestSurface:
    """surface against xroms.surface."""

    def test_surface_is_the_top_level(self):
        ds, meta = osk_inputs(land=True)
        top = roms.surface(roms.standardize(ds, meta), meta)
        for name in ("temp", "salt", "u", "v"):
            same(top[ROMS_STANDARD_NAMES[name]], xroms.surface(ds[name]), ordered=True)

    def test_the_level_stays_marked_in_xroms(self):
        # ocean-skill drops s_rho and z_rho with the level; xroms keeps a scalar s_rho: the label, or the index
        ds, meta = osk_inputs()
        top = roms.surface(roms.standardize(ds, meta), meta)
        assert "s_rho" not in top.coords and "z_rho" not in top.coords
        marker = xroms.surface(ds.temp).coords["s_rho"]
        assert marker.ndim == 0 and int(marker) == ds.sizes["s_rho"] - 1
        rutgers = syn.make_dataset("rutgers")
        assert float(xroms.surface(rutgers.temp).coords["s_rho"]) == float(rutgers.s_rho[-1])

    def test_a_field_without_levels(self):
        # ocean-skill passes a 2-D field through (its compare pipeline relies on that for grid constants); xroms refuses
        ds, meta = osk_inputs()
        same(roms.surface(ds[["h"]], meta)["h"], ds.h)
        with pytest.raises(ValueError, match="no vertical dimension"):
            xroms.surface(ds.h)


# --- 3. depth bands --------------------------------------------------------------------------------

BANDS = [(0.0, 10.0), (10.0, 40.0), (30.0, 60.0), (5.5, 12.25), (0.0, 1000.0), (0.0, float("inf"))]


class TestDepthBand:
    """depth_band and depth_average against xroms.depth_band_weights and xroms.depth_average.

    Without a free surface (z_w then has no time dimension: what ocean-skill's compare pipeline hands them) the two are
    bit for bit the same.
    """

    @pytest.mark.parametrize("low, high", BANDS)
    def test_band_weights_are_dz(self, low, high):
        ds, meta = osk_inputs(zeta=False)
        band = roms.depth_band(roms.standardize(ds, meta), meta, low, high)
        weights = xroms.depth_band_weights(xroms.z(ds, scoord="s_w", zeta=0), low, high)
        # ocean-skill keeps the layers the band touches anywhere; xroms keeps every layer, with zero weight outside it
        touched = (weights > 0).any([dim for dim in weights.dims if dim != "s_rho"]).values.nonzero()[0]
        assert band.sizes["s_rho"] == touched.size > 0
        same(band["dz"], weights.isel(s_rho=touched), ordered=True)
        assert float(weights.drop_isel(s_rho=touched).sum()) == 0.0

    @pytest.mark.parametrize("land", [False, True])
    @pytest.mark.parametrize("low, high", BANDS)
    def test_depth_average(self, low, high, land):
        # a band that misses a shallow column gives NaN in both (30-60 m); over masked land, where z is finite, both
        # give 0.0 rather than NaN, because the weighted sum skips the NaNs
        ds, meta = osk_inputs(zeta=False, land=land)
        averaged = roms.depth_average(roms.standardize(ds, meta), meta, low, high)
        for name, cf in FIELDS.items():
            same(averaged[cf], xroms.depth_average(ds[name], ds, shallow=low, deep=high, zeta=0), ordered=True)

    def test_weights_do_not_depend_on_the_sign_of_z_w(self):
        # ocean-skill negates z_w; xroms reads the sign off the labels
        ds, _ = osk_inputs(zeta=False)
        up = xroms.depth_band_weights(xroms.z(ds, scoord="s_w", zeta=0), 5.0, 30.0)
        down = xroms.depth_band_weights(xroms.z(ds, scoord="s_w", zeta=0, positive="down"), 5.0, 30.0)
        same(up, down)

    def test_one_time_step_with_a_free_surface(self):
        # With zeta a time step at a time works (z_w then has no time dimension). The band is below the mean sea level,
        # which is xroms' default; reference="surface" would move it with the free surface and give other numbers.
        ds, meta = osk_inputs()
        zeta = ds.zeta.isel(time=1)
        averaged = roms.depth_average(roms.standardize(ds, meta).isel(time=1), meta, 0.0, 10.0)
        for name, cf in FIELDS.items():
            same(averaged[cf], xroms.depth_average(ds[name].isel(time=1), ds, shallow=0.0, deep=10.0, zeta=zeta))
        var = ds.temp.isel(time=1)
        followed = xroms.depth_average(var, ds, shallow=0.0, deep=10.0, zeta=zeta, reference="surface")
        assert float(abs(followed - averaged[TEMP].transpose(*followed.dims)).max()) > 1e-4

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="ocean-skill bug: depth_band takes the first dim of z_w that is not s_rho for the interface dim, which "
        "is 'time' once zeta varies in time, so the band comes back empty (s_rho of size 0) and depth_average "
        "returns variables with a spurious s_w dimension. (Its compare pipeline avoids it by dropping zeta.)",
    )
    def test_a_free_surface_that_varies_in_time(self):
        ds, meta = osk_inputs()
        std = roms.standardize(ds, meta)
        assert roms.depth_band(std, meta, 0.0, 10.0).sizes["s_rho"] > 0
        averaged = roms.depth_average(std, meta, 0.0, 10.0)
        close(averaged[TEMP], xroms.depth_average(ds.temp, ds, shallow=0.0, deep=10.0))


# --- 4. interpolation to depths and densities ------------------------------------------------------------

DEPTHS = [0.5, 2.0, 5.0, 10.0, 25.0, 60.0, 150.0]  # the first and the last are outside the water column of some points


@pytest.mark.filterwarnings("ignore:.*entirely NaN:UserWarning")  # ocean-skill says so for a level nothing reaches
class TestVerticalInterpolation:
    """to_depth, nearest_depth_levels and to_sigma0 against xroms.zslice and xroms.isoslice."""

    @pytest.mark.parametrize("land", [False, True])
    def test_to_depth_is_zslice_of_heights(self, vtransform, land):
        # to_depth(d) interpolates on z_rho at -d: xroms.zslice at the heights -d, with the same NaN outside the column
        ds, meta = osk_inputs(land=land, vtransform=vtransform)
        sliced = roms.to_depth(roms.standardize(ds, meta), meta, DEPTHS)
        np.testing.assert_array_equal(sliced.z, -np.array(DEPTHS))
        for name, cf in FIELDS.items():
            expected = xroms.zslice(ds[name], [-d for d in DEPTHS], ds)
            np.testing.assert_array_equal(expected.z, sliced.z)
            close(sliced[cf], expected)
        assert np.isnan(sliced[TEMP]).any() and np.isfinite(sliced[TEMP]).any()
        # the new dimension goes last in ocean-skill and where the levels were in xroms
        assert sliced[TEMP].dims == ("time", "eta_rho", "xi_rho", "z")
        assert expected.dims == ("time", "z", "eta_rho", "xi_rho")

    def test_to_depth_is_zslice_of_depths(self):
        # ask for depths positive down instead: the same interpolation, but the coordinate comes back positive
        ds, meta = osk_inputs(land=True)
        sliced = roms.to_depth(roms.standardize(ds, meta), meta, DEPTHS)
        for name, cf in FIELDS.items():
            expected = xroms.zslice(ds[name], DEPTHS, ds, positive="down")
            np.testing.assert_array_equal(expected.z, -sliced.z)
            close(sliced[cf], expected)
        # ocean-skill's depths are below the mean sea level: below the moving free surface is another slice
        followed = xroms.zslice(ds.temp, DEPTHS, ds, positive="down", reference="surface")
        below_msl, below_surface = _aligned(sliced[TEMP], followed)
        assert np.nanmax(abs(below_msl - below_surface)) > 1e-3

    @pytest.mark.parametrize("land", [False, True])
    def test_nearest_depth_levels_at_the_reference_time(self, land):
        # ocean-skill finds the nearest level on the first time step's z and applies it at every step; that is xroms'
        # nearest slice on a static z built from that step's free surface (ties go to the deepest level in both)
        ds, meta = osk_inputs(land=land)
        std = roms.standardize(ds, meta)
        for ref_time, step in ((None, 0), (std.time.values[1], 1)):
            picked = roms.nearest_depth_levels(std, meta, DEPTHS, ref_time=ref_time)
            for name, cf in FIELDS.items():
                zeta = ds.zeta.isel(time=step)
                expected = xroms.zslice(ds[name], [-d for d in DEPTHS], ds, zeta=zeta, method="nearest")
                same(picked[cf], expected)
                np.testing.assert_array_equal(picked.z, expected.z)
            assert np.isnan(picked[TEMP]).any() and np.isfinite(picked[TEMP]).any()

    def test_nearest_depth_levels_do_not_follow_the_free_surface(self):
        # The one difference: ocean-skill keeps the reference step's levels at every time (it avoids rebuilding z at
        # each step); xroms' z moves with zeta, so where the free surface has carried a level past the half-way point
        # to its neighbour the nearest level changes. A free surface of metres shows it (at the second step: 59 of 756
        # values).
        ds, meta = osk_inputs(zeta_scale=20.0)
        picked = roms.nearest_depth_levels(roms.standardize(ds, meta), meta, DEPTHS)[TEMP]
        heights = [-d for d in DEPTHS]
        following = xroms.zslice(ds.temp, heights, ds, method="nearest")
        static = xroms.zslice(ds.temp, heights, ds, zeta=ds.zeta.isel(time=0), method="nearest")
        same(picked, static)
        same(picked.isel(time=0), following.isel(time=0))
        late_osk, late_xroms = _aligned(picked.isel(time=1), following.isel(time=1))
        assert not np.isclose(late_osk, late_xroms, rtol=0, atol=0, equal_nan=True).all()

    @pytest.mark.parametrize("land", [False, True])
    def test_to_sigma0_is_isoslice_on_sigma0(self, land):
        # ocean-skill's sigma0 is the anomaly: xroms' potential density minus 1000
        ds, meta = osk_inputs(land=land)
        sigma0 = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds) - 1000
        targets = [round(float(q), 3) for q in np.nanquantile(sigma0, [0.1, 0.3, 0.5, 0.7, 0.9])]
        sliced = roms.to_sigma0(roms.standardize(ds, meta), meta, targets)
        for name, cf in FIELDS.items():
            expected = xroms.isoslice(ds[name], targets, sigma0, dim="s_rho", new_dim="sigma0")
            np.testing.assert_array_equal(sliced.sigma0, expected.sigma0)
            close(sliced[cf], expected)
        assert np.isnan(sliced[TEMP]).any() and np.isfinite(sliced[TEMP]).any()
        assert sliced[TEMP].dims == ("time", "eta_rho", "xi_rho", "sigma0")
        assert expected.dims == ("time", "sigma0", "eta_rho", "xi_rho")

    def test_to_sigma0_refuses_a_full_density_and_xroms_does_not(self):
        # the guard against a density where an anomaly was meant stays in ocean-skill: isoslice takes whatever units
        # its iso array is in
        ds, meta = osk_inputs()
        with pytest.raises(ValueError, match="full density"):
            roms.to_sigma0(roms.standardize(ds, meta), meta, [1027.0])
        density = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        assert np.isfinite(xroms.isoslice(ds.temp, [1027.0], density, dim="s_rho", new_dim="sigma0")).any()


# --- 5. time decoding ---------------------------------------------------------------------------------

EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")
ROMS_TIME = {"time_coord": "ocean_time", "time_dim": "time", "reference_date": "2000-01-01", "time_units": "seconds"}


def ucla_times(values, units="second", long_name="Time since 2000/01/01"):
    """UCLA-style times: an ocean_time variable on a time dim, the epoch only in its long_name."""
    attrs = {"long_name": long_name, "units": units}
    return xr.Dataset({"ocean_time": ("time", np.asarray(values, dtype=float), attrs)})


def decoded_times(out, dim="time"):
    """The decoded time index of ``out`` as datetime64[ns] (ocean-skill decodes to seconds)."""
    return out.indexes[dim].values.astype("datetime64[ns]")


class TestTimeDecoding:
    """_decode_time (and the ROMS branch of build._decode_times) against xroms.decode_time."""

    @pytest.mark.parametrize("source", ["real restart file", "synthetic output"])
    def test_ucla_time(self, source):
        if source == "real restart file":
            ds = real("ucla_rst", decode_times=False)  # 'Time since 1995/01/01', units 'second'
            # the builder detects ROMS from the s-coordinate, which the cut-down restart file lacks
            sigma = ("s_rho", np.zeros(ds.sizes["s_rho"]))
            meta = _roms_metadata(ds.assign(sigma_r=sigma, Cs_r=sigma))
            assert (meta["reference_date"], meta["time_units"]) == ("1995-01-01", "seconds")
        else:
            ds, meta = osk_inputs()
        expected = decoded_times(roms._decode_time(ds, meta))
        np.testing.assert_array_equal(xroms.decode_time(ds).indexes["time"].values, expected)
        np.testing.assert_array_equal(_decode_times(ds, ds["ocean_time"]).astype("datetime64[ns]"), expected)
        if source == "real restart file":
            assert expected[-1] == np.datetime64("1998-01-06T00:00:00")

    def test_classic_time_dimension_keeps_its_name_in_xroms(self):
        # Classic output makes ocean_time the time dimension. ocean-skill moves the data onto a dimension called time
        # and keeps the seconds in ocean_time; xroms decodes in place and keeps the dimension's own name.
        ds = xroms.canonicalize(syn.make_dataset("rutgers"))
        attrs = {"long_name": "time since initialization", "units": "seconds since 2000-01-01 00:00:00 GMT"}
        raw = ds.assign_coords(ocean_time=("ocean_time", [0.0, 86400.0], attrs))
        via_osk, via_xroms = roms._decode_time(raw, _roms_metadata(raw)), xroms.decode_time(raw)
        assert via_osk.temp.dims[0] == "time" and via_xroms.temp.dims[0] == "ocean_time"
        np.testing.assert_array_equal(decoded_times(via_xroms, "ocean_time"), decoded_times(via_osk))
        np.testing.assert_array_equal(via_osk.ocean_time, [0.0, 86400.0])

    def test_fractional_seconds(self):
        # whole seconds agree; ocean-skill truncates a fraction toward zero, xroms keeps it (to the microsecond)
        seconds = np.array([0.0, 0.5, 1.5, 3600.7, -0.5, 1.0e9])
        ds = ucla_times(seconds)
        via_osk = decoded_times(roms._decode_time(ds, ROMS_TIME))
        via_xroms = xroms.decode_time(ds).indexes["time"].values
        whole = seconds == np.round(seconds)
        np.testing.assert_array_equal(via_osk[whole], via_xroms[whole])
        np.testing.assert_array_equal(via_osk, EPOCH + np.trunc(seconds).astype("timedelta64[s]"))
        np.testing.assert_array_equal(via_xroms, EPOCH + np.rint(seconds * 1e6).astype("timedelta64[us]"))
        assert (via_osk != via_xroms).sum() == 4

    @pytest.mark.parametrize("units, length", [("days", 86400), ("hours", 3600), ("minutes", 60)])
    def test_units_other_than_seconds(self, units, length):
        # ocean-skill only decodes seconds (and says so); xroms adds days, hours and minutes to the epoch too
        ds = ucla_times([1.0, 2.5], units=units)
        with pytest.raises(ValueError, match="Unsupported time_units"):
            roms._decode_time(ds, {**ROMS_TIME, "time_units": units})
        expected = EPOCH + np.rint(np.array([1.0, 2.5]) * length).astype("timedelta64[s]")
        np.testing.assert_array_equal(xroms.decode_time(ds).indexes["time"].values, expected)

    def test_a_reference_date_that_disagrees_with_the_file(self):
        # ocean-skill decodes with the catalog's reference_date, whatever the file says; xroms will not override the
        # file's epoch (the xroms replacement is to drop reference_date, or correct the file's long_name)
        ds = ucla_times([0.0, 86400.0], long_name="Time since 1995/01/01")
        assert decoded_times(roms._decode_time(ds, ROMS_TIME))[0] == EPOCH
        with pytest.raises(ValueError, match="disagrees"):
            xroms.decode_time(ds, reference_date="2000-01-01")
        assert xroms.decode_time(ds).indexes["time"].values[0] == np.datetime64("1995-01-01", "ns")

    def test_cf_units(self):
        # build._decode_times' other branch: CF units. Units no fixed-length time can decode give None (a climatology)
        # in ocean-skill and an error in xroms.
        attrs = {"units": "hours since 2013-12-17 00:00:00", "calendar": "standard"}
        ds = xr.Dataset({"ocean_time": ("time", np.array([0.0, 6.0]), attrs)})
        expected = _decode_times(ds, ds["ocean_time"]).astype("datetime64[ns]")
        np.testing.assert_array_equal(xroms.decode_time(ds).indexes["time"].values, expected)
        months = ds.assign(ocean_time=ds.ocean_time.assign_attrs(units="months since 1965-01-01"))
        assert _decode_times(months, months["ocean_time"]) is None
        with pytest.raises(ValueError, match="unsupported time units"):
            xroms.decode_time(months)


# --- 6. mixed layer depth and potential density -------------------------------------------------------------


class TestMixedLayerDepth:
    """mld_threshold and potential_density against xroms.mld and xroms.potential_density."""

    @pytest.mark.parametrize("land", [False, True])
    def test_potential_density_is_sigma0(self, land):
        # xroms returns the full density: ocean-skill's sigma0 is that minus 1000 (the same gsw chain; in gsw 3.6
        # sigma0 is rho at 0 dbar less 1000 exactly, but only rtol is asked for)
        ds, meta = osk_inputs(land=land)
        std = roms.standardize(ds, meta)
        sigma0 = mld.potential_density(std[TEMP], std[SALT], std.z_rho, std.lon, std.lat)
        density = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        close(density - 1000, sigma0, ordered=True)
        # heights, lon and lat passed in rather than found on the grid give the same
        explicit = xroms.potential_density(ds.temp, ds.salt, eos="teos10", z_points=std.z_rho, lon=std.lon, lat=std.lat)
        same(explicit, density)

    def test_a_column_worked_by_hand(self):
        # ocean-skill's hand-worked column (tests/test_mld.py): 20 degC to 10 m, cooling below; ref 10 m, 0.2 degC
        depth = np.array([2.0, 6.0, 10.0, 20.0, 40.0, 80.0])
        temp = np.array([21.0, 20.0, 20.0, 19.9, 19.0, 15.0])
        expected = mld._mld_threshold_1d(temp, depth, threshold=0.2, ref_depth=10.0)
        assert expected == pytest.approx(20.0 + 20.0 / 9.0, rel=1e-12)
        actual = xroms.mld(
            xr.DataArray(temp, dims="depth"),
            z=xr.DataArray(-depth, dims="depth"),
            dim="depth",
            variable="temperature",
            threshold=0.2,
            reference_depth=10.0,
            fill="nan",
        )
        np.testing.assert_allclose(float(actual), expected, rtol=RTOL)

    @pytest.mark.parametrize("reference_depth", [5.0, 10.0])
    @pytest.mark.parametrize("threshold", [0.01, 0.03])
    def test_density_threshold(self, threshold, reference_depth):
        # Same crossing; the reference value is interpolated with another order of operations, and the crossing
        # divides sigma0 differences (~1e-2) taken from values near 27, so the rounding is amplified. The largest
        # difference over thresholds 3e-3 to 0.1 and reference depths 3 to 10 m, on 6 and 20 levels, is 1.4e-11
        # relative: rtol=1e-9.
        ds, meta = osk_inputs(land=True, N=20)
        ds = with_mixed_layer(ds)
        std = roms.standardize(ds, meta)
        expected = mld.mld_density_threshold(std, threshold=threshold, ref_depth=reference_depth)
        sigma0 = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        kwargs = {"threshold": threshold, "reference_depth": reference_depth, "variable": "density"}
        actual = xroms.mld(sigma0, ds, method="interp", fill="nan", **kwargs)
        close(actual, expected, rtol=1e-9, ordered=True)
        assert np.isfinite(expected).any() and 12.0 < float(expected.min()) and float(expected.max()) < 30.0
        # the attributes carry the same recipe under other names
        assert actual.attrs["standard_name"] == expected.attrs["standard_name"]
        assert (actual.attrs["mld_threshold"], actual.attrs["mld_reference_depth"]) == (threshold, reference_depth)
        assert (expected.attrs["mld_threshold"], expected.attrs["mld_ref_depth"]) == (threshold, reference_depth)

    @pytest.mark.parametrize("inverted", [False, True])
    def test_temperature_threshold(self, inverted):
        # the temperature criterion is |T - T(ref)| in both, so even a warm layer below the mixed layer is found
        ds, meta = osk_inputs(land=True, N=20)
        ds = with_mixed_layer(ds, inverted=inverted)
        expected = mld.mld_temperature_threshold(roms.standardize(ds, meta), threshold=0.2, ref_depth=10.0)
        actual = xroms.mld(ds.temp, ds, threshold=0.2, reference_depth=10.0, variable="temperature", fill="nan")
        close(actual, expected, rtol=1e-9, ordered=True)
        assert actual.attrs["standard_name"] == expected.attrs["standard_name"]
        assert np.isfinite(expected).any()

    def test_a_column_whose_shallowest_point_is_deeper_than_the_reference(self):
        # Intended difference. Where even the shallowest point is deeper than the reference depth, ocean-skill returns
        # NaN; xroms takes that shallowest point as the reference value (the same crossing as ocean-skill finds with the
        # reference moved to that point).
        ds, meta = osk_inputs(land=True, N=20)
        ds = with_mixed_layer(ds)
        std = roms.standardize(ds, meta)
        top = (-std.z_rho).min("s_rho")  # depth of each column's shallowest point
        reference = 0.5 * float(top.max())
        expected = mld.mld_density_threshold(std, ref_depth=reference)
        density = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        actual = xroms.mld(density, ds, reference_depth=reference, variable="density", method="interp", fill="nan")
        deeper = ((top > reference) & (std.mask_rho == 1)).transpose(*actual.dims).values
        covered = (top <= reference).transpose(*actual.dims).values
        assert deeper.sum() > 0 and covered.sum() > 0
        assert np.isnan(expected.values[deeper]).all() and np.isfinite(actual.values[deeper]).all()
        close(actual.values[covered], expected.values[covered], rtol=1e-9)
        sigma0 = (density - 1000).transpose(*std.z_rho.dims)
        for index in np.argwhere(deeper)[:6]:
            column = dict(zip(actual.dims, index, strict=True))
            args = (sigma0.isel(column), std.z_rho.isel(column))
            from_top = mld.mld_threshold(*args, threshold=0.03, ref_depth=float(top.isel(column)))
            np.testing.assert_allclose(float(actual.isel(column)), float(from_top), rtol=1e-9)

    def test_a_density_decrease_counts_for_ocean_skill_only(self):
        # Intended difference, and an inconsistency in ocean-skill: mld_density_threshold's docstring says sigma0
        # "exceeds sigma0(ref_depth) + threshold", but mld_threshold tests |difference| for density as for
        # temperature. xroms' density criterion is an increase (de Boyer Montegut et al. 2004; temperature is "either
        # way"), so water that gets lighter with depth ends the mixed layer in ocean-skill and never does in xroms.
        ds, meta = osk_inputs(land=True, N=20)
        ds = with_mixed_layer(ds, inverted=True)
        expected = mld.mld_density_threshold(roms.standardize(ds, meta))
        density = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        actual = xroms.mld(density, ds, reference_depth=10.0, variable="density", method="interp", fill="nan")
        ocean = np.broadcast_to(ds.mask_rho.values == 1, expected.shape)
        assert np.isfinite(expected.values[ocean]).all() and np.isnan(actual.values).all()

    def test_a_column_without_a_crossing(self):
        # fully mixed to the bottom: NaN in ocean-skill; xroms fills such a column with the depth of the bottom (h over
        # water) unless it is asked for fill="nan"
        ds, meta = osk_inputs(land=True, N=20)
        ds = with_mixed_layer(ds, depth=1000.0)
        assert mld.mld_temperature_threshold(roms.standardize(ds, meta)).isnull().all()
        kwargs = {"variable": "temperature", "reference_depth": 10.0}
        assert xroms.mld(ds.temp, ds, fill="nan", **kwargs).isnull().all()
        filled = xroms.mld(ds.temp, ds, **kwargs)
        ocean = ds.mask_rho == 1
        same(filled.where(ocean), ds.h.where(ocean).broadcast_like(filled))
        assert filled.where(~ocean).isnull().all()


# --- 7. longitude -----------------------------------------------------------------------------------


def longitudes(values):
    """A Dataset of longitudes along x, which both libraries find by name (lon)."""
    values = np.asarray(values, dtype=float)
    coords = {"lon": ("x", values), "lat": ("x", np.zeros(values.size))}
    return xr.Dataset({"v": ("x", np.arange(values.size, dtype=float))}, coords=coords)


# fmt: off
ANGLES = np.array([-540.0, -360.0, -270.0, -180.0, -179.9999999, -90.0, -0.1, 0.0, 0.1, 90.0, 179.9999999, 180.0,
                   180.0000001, 270.0, 359.9999999, 360.0, 540.0, 720.1])
# fmt: on

#: (case, longitudes, ocean-skill's natural_convention, xroms' straddles)
CONVENTIONS = [
    ("contiguous in both", [-90, -85, -80], "-180-180", False),  # a tie: ocean-skill keeps -180-180, straddles is False
    ("Greenwich, stored -180..180", [-5, -2, 0, 3, 5], "-180-180", True),
    ("Greenwich, stored 0..360", [355, 357, 359, 1, 3, 5], "-180-180", True),
    ("dateline, stored 0..360", [170, 175, 185, 190], "0-360", False),
    ("dateline, stored -180..180", [170, 175, -175, -170], "0-360", False),
    ("Pacific, 77E to 316E", np.linspace(77, 316, 20), "0-360", False),
    ("global, 0..359", np.arange(360), "-180-180", False),  # a tie
    ("reaches +180", [170, 175, 180], "-180-180", False),  # +180 is the seam's own, not -180: a tie
    ("a single value", [12.5], "-180-180", False),
    ("nothing finite", [np.nan, np.nan], "-180-180", False),
]
CASES = pytest.mark.parametrize("case, values, natural, straddles", CONVENTIONS, ids=[c[0] for c in CONVENTIONS])


class TestLongitude:
    """harmonize_longitude and natural_convention against xroms.wrap_longitude and xroms.straddles."""

    def test_0_360_is_the_same(self):
        ds = longitudes(ANGLES)
        same(align.harmonize_longitude(ds, "0-360").lon, xroms.wrap_longitude(ds, "0-360").lon)

    def test_a_single_longitude(self):
        # sample_at wraps one longitude with _wrap_lon: the same two formulas as harmonize_longitude
        for value in ANGLES:
            assert align._wrap_lon(value, "0-360") == xroms.wrap_longitude(float(value), "0-360")
            if value % 360 != 180.0:
                expected = xroms.wrap_longitude(float(value), "-180-180")
                assert align._wrap_lon(value, "-180-180") == pytest.approx(expected, rel=0, abs=1e-12)

    def test_180_frames_differ_on_the_seam_only(self):
        # ocean-skill's frame is [-180, 180), xroms' (-180, 180]: 180 and -180 are -180 in one and 180 in the other.
        # Elsewhere xroms leaves in-range values as they are, while ocean-skill's (lon + 180) % 360 - 180 moves them by
        # rounding (5.7e-14 degrees at most here: atol=1e-12).
        ds = longitudes(ANGLES)
        osk = align.harmonize_longitude(ds, "-180-180").lon.values
        xrm = xroms.wrap_longitude(ds, "-180-180").lon.values
        seam = ANGLES % 360 == 180.0
        assert seam.sum() == 4
        assert (osk[seam] == -180.0).all() and (xrm[seam] == 180.0).all()
        np.testing.assert_allclose(osk[~seam], xrm[~seam], rtol=0, atol=1e-12)
        assert ((osk >= -180) & (osk < 180)).all() and ((xrm > -180) & (xrm <= 180)).all()

    def test_a_longitude_dimension_is_sorted_with_its_data(self):
        lon = np.array([10.0, 100.0, 200.0, 300.0])  # none on the seam
        da = xr.DataArray(np.arange(4.0), dims="lon", coords={"lon": lon}, name="v")
        for convention in ("-180-180", "0-360"):
            osk, xrm = align.harmonize_longitude(da, convention), xroms.wrap_longitude(da, convention)
            np.testing.assert_allclose(osk.lon, xrm.lon, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(osk.values, xrm.values)
        np.testing.assert_array_equal(xroms.wrap_longitude(da, "-180-180").values, [2.0, 3.0, 0.0, 1.0])
        # on the seam the sorted ends differ, as the frames do
        seam = xr.DataArray(np.arange(4.0), dims="lon", coords={"lon": [0.0, 90.0, 180.0, 270.0]})
        np.testing.assert_array_equal(align.harmonize_longitude(seam, "-180-180").lon, [-180.0, -90.0, 0.0, 90.0])
        np.testing.assert_array_equal(xroms.wrap_longitude(seam, "-180-180").lon, [-90.0, 0.0, 90.0, 180.0])

    def test_one_longitude_or_all(self):
        # ocean-skill wraps the one longitude it finds (lon_rho on a standardized ROMS Dataset), leaving lon, lon_u and
        # lon_v in the old convention; xroms wraps every longitude with the one convention
        ds, meta = osk_inputs()
        std = roms.standardize(ds, meta)
        names = ("lon_rho", "lon_u", "lon_v", "lon")
        found = align._lon_name(std)
        osk, xrm = align.harmonize_longitude(std, "0-360"), xroms.wrap_longitude(std, "0-360")

        def changed(out):
            return {name for name in names if bool((out[name] != std[name]).any())}

        assert changed(osk) == {found} and changed(xrm) == set(names)
        close(osk[found], xrm[found], rtol=0, atol=1e-12)

    @CASES
    def test_natural_convention_and_straddles(self, case, values, natural, straddles):
        # straddles is True when the domain crosses the prime meridian (one span only in -180..180). A tie, contiguous
        # in both conventions, is False there, while natural_convention keeps -180-180 for it.
        ds = longitudes(values)
        assert align.natural_convention(ds) == natural
        assert xroms.straddles(ds) is straddles
        if straddles:
            assert natural == "-180-180"
        if natural == "0-360":
            assert not straddles

    @CASES
    def test_wrapping_to_the_natural_convention(self, case, values, natural, straddles):
        # harmonize_longitude(obj, natural_convention(obj)) is wrap_longitude(obj) where one convention makes the domain
        # contiguous. On a tie xroms leaves the values as stored, while ocean-skill puts them in -180..180.
        ds = longitudes(values)
        osk = align.harmonize_longitude(ds, align.natural_convention(ds)).lon.values
        xrm = xroms.wrap_longitude(ds).lon.values
        if natural == "-180-180" and not straddles:
            np.testing.assert_array_equal(xrm, ds.lon.values)
            seam = ds.lon.values % 360 == 180.0
            frame = xroms.wrap_longitude(ds, "-180-180").lon.values
            np.testing.assert_allclose(osk[~seam], frame[~seam], rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose(osk, xrm, rtol=0, atol=1e-12)


# --- 8. velocities ------------------------------------------------------------------------------------


class TestVelocity:
    """_average_to_rho, _add_geographic_velocity and _rotate_and_assign against xroms.to_rho and xroms.grid_to_earth."""

    @pytest.mark.parametrize("axis, stagger, rho", [("u", "xi_u", "xi_rho"), ("v", "eta_v", "eta_rho")])
    def test_average_to_rho_is_to_rho(self, axis, stagger, rho):
        # the 2-point average inside and the nearest value at the two edges (xgcm's boundary="extend"), NaN spreading;
        # a chunked input gives the same values (the chunks differ: ocean-skill makes the staggered dim one chunk)
        ds, _ = osk_inputs(land=True)
        for field in (ds, chunked(ds)):
            same(roms._average_to_rho(field[axis], stagger, rho), xroms.to_rho(field[axis]), ordered=True)

    @pytest.mark.parametrize("varying", [False, True])
    def test_geographic_velocity_is_grid_to_earth(self, varying):
        # Away from land: average to rho points, then rotate by the angle (ocean-skill's formula is rotate_vectors').
        ds, meta = osk_inputs(angle=0.3)
        ds = with_varying_angle(ds) if varying else ds
        derived = roms._add_geographic_velocity(roms.standardize(ds, meta))
        east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        same(derived["eastward_sea_water_velocity"], east, ordered=True)
        same(derived["northward_sea_water_velocity"], north, ordered=True)

    def test_geographic_velocity_next_to_land(self):
        # Intended difference: grid_to_earth sets masked u and v to zero before averaging them, so a rho point next to
        # land keeps part of its neighbour's value (and land itself is 0); ocean-skill's NaN spreads to every rho point
        # beside a masked u or v and standardize then masks land. Everywhere ocean-skill has a value, the two are equal,
        # and the whole difference is that fill.
        ds, meta = osk_inputs(land=True, angle=0.3)
        ds = with_varying_angle(ds)
        std = roms.standardize(ds, meta)
        derived = roms._add_geographic_velocity(std)["eastward_sea_water_velocity"]
        east, north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        found = derived.notnull()
        assert 0 < int(found.sum()) < found.size and east.notnull().all()
        same(east.where(found), derived.where(found))
        filled = std.assign({U: std[U].fillna(0.0), V: std[V].fillna(0.0)})
        same(roms._add_geographic_velocity(filled)["eastward_sea_water_velocity"], east)
        same(roms._add_geographic_velocity(filled)["northward_sea_water_velocity"], north)

    def test_rotate_and_assign_is_rotate_vectors(self):
        ds, meta = osk_inputs(angle=0.3)
        ds = with_varying_angle(ds)
        u_rho, v_rho = xroms.to_rho(ds.u), xroms.to_rho(ds.v)
        rotated = roms._rotate_and_assign(roms.standardize(ds, meta), u_rho.variable, v_rho.variable, {}, {})
        east, north = xroms.rotate_vectors(u_rho, v_rho, ds.angle)
        same(rotated["eastward_sea_water_velocity"], east)
        same(rotated["northward_sea_water_velocity"], north)


# --- 9. dask-backed inputs ----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*entirely NaN:UserWarning")
class TestLazyInputs:
    """The same comparisons on fields chunked along every dimension, s_rho and the horizontal ones included.

    Both libraries stay lazy and agree; where a sum over s_rho is split differently (ocean-skill narrows s_rho to the
    band first) the last bit can differ, which rtol=1e-12 allows.
    """

    def test_to_depth(self):
        ds, meta = osk_inputs(land=True)
        ds = chunked(ds)
        sliced = roms.to_depth(roms.standardize(ds, meta), meta, DEPTHS)[TEMP]
        expected = xroms.zslice(ds.temp, [-d for d in DEPTHS], ds)
        assert sliced.chunks is not None and expected.chunks is not None
        close(sliced, expected)

    def test_depth_average(self):
        ds, meta = osk_inputs(zeta=False)
        ds = chunked(ds)
        averaged = roms.depth_average(roms.standardize(ds, meta), meta, 0.0, 10.0)[TEMP]
        expected = xroms.depth_average(ds.temp, ds, shallow=0.0, deep=10.0, zeta=0)
        assert averaged.chunks is not None and expected.chunks is not None
        close(averaged, expected)

    def test_to_sigma0(self):
        ds, meta = osk_inputs(land=True)
        ds = chunked(ds)
        sigma0 = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds) - 1000
        targets = [round(float(q), 3) for q in np.nanquantile(sigma0.compute(), [0.2, 0.5, 0.8])]
        sliced = roms.to_sigma0(roms.standardize(ds, meta), meta, targets)[TEMP]
        expected = xroms.isoslice(ds.temp, targets, sigma0, dim="s_rho", new_dim="sigma0")
        assert sliced.chunks is not None and expected.chunks is not None
        close(sliced, expected)

    def test_mixed_layer_depth(self):
        ds, meta = osk_inputs(land=True, N=20)
        ds = chunked(with_mixed_layer(ds))
        expected = mld.mld_density_threshold(roms.standardize(ds, meta))
        density = xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
        actual = xroms.mld(density, ds, reference_depth=10.0, variable="density", method="interp", fill="nan")
        assert expected.chunks is not None and actual.chunks is not None
        close(actual, expected, rtol=1e-9)

    def test_geographic_velocity(self):
        ds, meta = osk_inputs(angle=0.3)
        ds = chunked(ds)
        derived = roms._add_geographic_velocity(roms.standardize(ds, meta))["eastward_sea_water_velocity"]
        east, _ = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        assert derived.chunks is not None and east.chunks is not None
        same(derived, east)


# --- 10. the standardized Dataset ---------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*entirely NaN:UserWarning")
class TestOnTheStandardizedDataset:
    """The replacement calls get ocean-skill's standardized Dataset, not the raw one, and xroms reads it directly.

    It differs from the raw output in three ways that matter here: zeta is renamed (sea_surface_height_above_geoid),
    every rho-point data variable is masked (pm and pn among them), and the grid fields (h, sigma_r, Cs_r, angle,
    longitudes) are coordinates, which xroms reads all the same. Each test makes the call on that Dataset.
    """

    def test_z_needs_the_renamed_zeta_passed(self):
        ds, meta = osk_inputs()
        std = roms.standardize(ds, meta)
        zeta = std[ROMS_STANDARD_NAMES["zeta"]]
        same(xroms.z(std, zeta=zeta), std.z_rho, ordered=True)
        same(xroms.z(std, scoord="s_w", zeta=zeta), roms.add_interface_coord(std, meta).z_w, ordered=True)
        # zeta is not found under its new name: z itself then takes a flat surface, without saying so ...
        same(xroms.z(std), roms.add_depth_coord(std, meta, zero_zeta=True).z_rho)
        # ... while a variable that varies in time has nothing to match a missing zeta to, and is refused
        with pytest.raises(ValueError, match="zeta"):
            xroms.zslice(std[TEMP], [-5.0], std)

    def test_zslice_with_zeta_or_with_ocean_skills_own_z(self):
        ds, meta = osk_inputs(land=True)
        std = roms.standardize(ds, meta)
        heights = [-d for d in DEPTHS]
        sliced = roms.to_depth(std, meta, DEPTHS)[TEMP]
        close(xroms.zslice(std[TEMP], heights, std, zeta=std[ROMS_STANDARD_NAMES["zeta"]]), sliced)
        close(xroms.zslice(std[TEMP], heights, z=std.z_rho), sliced)

    def test_depth_average_without_a_free_surface(self):
        ds, meta = osk_inputs(zeta=False)
        std = roms.standardize(ds, meta)
        averaged = roms.depth_average(std, meta, 0.0, 10.0)
        for cf in FIELDS.values():
            same(averaged[cf], xroms.depth_average(std[cf], std, shallow=0.0, deep=10.0, zeta=0), ordered=True)

    def test_density_and_mixed_layer_depth_on_ocean_skills_z(self):
        ds, meta = osk_inputs(land=True, N=20)
        std = roms.standardize(with_mixed_layer(ds), meta)
        sigma0 = xroms.potential_density(
            std[TEMP], std[SALT], eos="teos10", z_points=std.z_rho, lon=std.lon, lat=std.lat
        )
        actual = xroms.mld(sigma0, z=std.z_rho, reference_depth=10.0, variable="density", method="interp", fill="nan")
        close(actual, mld.mld_density_threshold(std), rtol=1e-9, ordered=True)

    def test_geographic_velocity(self):
        ds, meta = osk_inputs(angle=0.3)
        std = roms.standardize(with_varying_angle(ds), meta)
        derived = roms._add_geographic_velocity(std)
        east, north = xroms.grid_to_earth(std[U], std[V], std.angle)
        same(derived["eastward_sea_water_velocity"], east, ordered=True)
        same(derived["northward_sea_water_velocity"], north, ordered=True)

    def test_cell_area_comes_from_the_unmasked_grid(self):
        # standardize masks pm and pn like every rho-point data variable, so dA of the standardized Dataset is NaN over
        # land; the cell_area coordinate, like dA of the raw grid, is not
        ds, meta = osk_inputs(land=True)
        std = roms.standardize(ds, meta)
        masked = xroms.dA(std)
        assert masked.isnull().any() and not std.cell_area.isnull().any()
        close(xroms.dA(ds), std.cell_area, ordered=True)
        ocean = std.mask_rho == 1
        close(masked.where(ocean), std.cell_area.where(ocean))


# --- 11. horizontal grid ------------------------------------------------------------------------------

POINTS = [(4, 5), (0, 0), (0, 11), (8, 0), (8, 11), (1, 6), (7, 3)]  # interior, corners, one cell in from an edge
CELLS = 2


def point_window(ds, std, j, i):
    """ocean-skill's window around rho point ``(j, i)``, xroms' ``subset`` with a halo around it, and its slices."""
    lon0, lat0 = float(ds.lon_rho[j, i]), float(ds.lat_rho[j, i])
    window = align._point_window(std, "lon", "lat", lon0, lat0, CELLS)
    iy, ix = xroms.argsel2d(ds.lon_rho, ds.lat_rho, lon0, lat0)
    assert (iy, ix) == (j, i)
    rows, columns = slice(max(iy - CELLS, 0), iy + CELLS + 1), slice(max(ix - CELLS, 0), ix + CELLS + 1)
    return window, xroms.subset(ds, X=columns, Y=rows, halo=1), (rows, columns)


class TestHorizontalGrid:
    """cell_area (the coordinate standardize attaches), _cell_km, _nearest_indices and _point_window against xroms.

    ocean-skill's rectilinear branch of _point_window (a 1-D lon/lat source) has no xroms counterpart: xroms is for ROMS
    grids, whose lon/lat are 2-D.
    """

    @pytest.mark.parametrize("grid", ["synthetic", "romstools_grid"])
    def test_cell_area_is_dA(self, grid):
        # There is no cell_area function: standardize attaches 1/(pm*pn) as the cell_area coordinate. xroms' dA is
        # (1/pm)*(1/pn): the same product with another rounding (2.6e-16 relative at most, on both grids).
        if grid == "synthetic":
            ds, meta = osk_inputs()
        else:  # a real roms-tools grid (rotated, with land) is self-contained output as far as standardize is concerned
            ds = real("romstools_grid")
            meta = _roms_metadata(ds)
        area = roms.standardize(ds, meta)["cell_area"]
        close(area, xroms.dA(ds), ordered=True)
        assert area.attrs["units"] == xroms.dA(ds).attrs["units"] == "m2"

    def test_cell_km_on_an_aligned_grid(self):
        # _cell_km is a representative cell *diagonal* in km read off the lon/lat differences (111.32 and 110.57 km per
        # degree); the metrics' diagonal is hypot(dx, dy). On the synthetic grid, whose axes are east and north, they
        # are 0.4% apart (the km per degree), hence rtol=1e-2. xroms.nominal_resolution is another quantity, the mean
        # spacing in metres, so it is not a drop-in: it is smaller than the diagonal.
        ds, meta = osk_inputs()
        cell_km = align._cell_km(roms.standardize(ds, meta), "lon", "lat")
        diagonal = float(np.hypot(xroms.dx(ds).median(), xroms.dy(ds).median())) / 1000
        np.testing.assert_allclose(cell_km, diagonal, rtol=1e-2)
        assert xroms.nominal_resolution(ds) / 1000 < cell_km

    @pytest.mark.parametrize("grid", ["romstools_grid", "ucla_grd"])
    def test_cell_km_on_a_rotated_grid(self, grid):
        # Both real grids are rotated (20 and 31 degrees): the lon differences along xi and lat differences along eta
        # see only cos(angle) of the spacing, so _cell_km is that factor short of the metrics' diagonal (to 1%).
        ds = real(grid)
        cell_km = align._cell_km(ds, "lon_rho", "lat_rho")
        diagonal = float(np.hypot((1 / ds.pm).median(), (1 / ds.pn).median())) / 1000
        np.testing.assert_allclose(cell_km, float(np.cos(ds.angle.mean())) * diagonal, rtol=1e-2)
        assert cell_km < 0.95 * diagonal

    @pytest.mark.parametrize("grid", ["synthetic", "romstools_grid", "ucla_grd"])
    def test_nearest_indices_is_argsel2d(self, grid):
        # Both are a great-circle argmin, so they pick the same cell (ocean-skill's sphere is 6371.0088 km, ROMS'
        # 6371315 m: a scale that cannot change an argmin), in either longitude convention, inside and outside.
        ds = osk_inputs()[0] if grid == "synthetic" else real(grid)
        lon, lat = ds.lon_rho, ds.lat_rho
        rng = np.random.default_rng(0)
        x = rng.uniform(float(lon.min()) - 3, float(lon.max()) + 3, 200)
        y = rng.uniform(float(lat.min()) - 3, float(lat.max()) + 3, 200)
        for turn in (0.0, 360.0, -360.0):
            nearest = [align._nearest_indices(lon.values, lat.values, a + turn, b) for a, b in zip(x, y, strict=True)]
            iy, ix = xroms.argsel2d(lon, lat, x + turn, y)
            np.testing.assert_array_equal(np.array(nearest), np.stack([iy, ix], axis=1))
            assert xroms.argsel2d(lon, lat, x[0] + turn, y[0]) == tuple(nearest[0])
        if grid == "romstools_grid":
            # the test has teeth: degrees taken as Cartesian pick another cell for a third of the points of this rotated
            # grid at 64N
            cartesian = (xroms.argsel2d(lon, lat, a, b, method="cartesian") for a, b in zip(x, y, strict=True))
            great_circle = (xroms.argsel2d(lon, lat, a, b) for a, b in zip(x, y, strict=True))
            assert sum(c != g for c, g in zip(cartesian, great_circle, strict=True)) > 20

    @pytest.mark.parametrize("j, i", POINTS)
    def test_point_window_is_subset_with_a_halo(self, j, i):
        # ocean-skill windows rho fields to the nearest cell plus 2 and u/v one cell wider along their staggered dim (a
        # halo), recording in _roms_stagger_trim how much of it to trim again. xroms.subset(halo=1) is that halo, and
        # xroms.trim removes it.
        ds, meta = osk_inputs()
        window, sub, _ = point_window(ds, roms.standardize(ds, meta), j, i)
        left, right, below, above = sub.attrs["xroms_halo"]
        assert window.attrs["_roms_stagger_trim"] == {"xi_rho": (left, right), "eta_rho": (below, above)}
        n_eta, n_xi = sub.sizes["eta_rho"], sub.sizes["xi_rho"]
        same(window[U], sub.u.isel(eta_rho=slice(below, n_eta - above)))
        same(window[V], sub.v.isel(xi_rho=slice(left, n_xi - right)))
        same(window[TEMP], xroms.trim(sub).temp)

    @pytest.mark.parametrize("j, i", POINTS)
    def test_windowed_velocity_is_halo_rotate_trim(self, j, i):
        # add_geographic_velocity_windowed is: subset with a halo, grid_to_earth, trim. All three are the full-domain
        # derivation cropped, which is what the window exists to avoid computing.
        ds, meta = osk_inputs()
        ds = with_varying_angle(ds)
        window, sub, (rows, columns) = point_window(ds, roms.standardize(ds, meta), j, i)
        derived = roms.add_geographic_velocity_windowed(window, meta)
        east, north = xroms.grid_to_earth(sub.u, sub.v, sub.angle)
        trimmed = xroms.trim(xr.Dataset({"east": east, "north": north}, attrs=sub.attrs))
        same(derived["eastward_sea_water_velocity"], trimmed.east)
        same(derived["northward_sea_water_velocity"], trimmed.north)
        full_east, full_north = xroms.grid_to_earth(ds.u, ds.v, ds.angle)
        same(trimmed.east, full_east.isel(eta_rho=rows, xi_rho=columns))
        same(trimmed.north, full_north.isel(eta_rho=rows, xi_rho=columns))
