"""Dev-only parity with roms-tools: where the formulas are the same, so are the numbers.

Skipped unless roms-tools is installed (it is not a dependency of xroms). Every test builds
synthetic inputs, runs roms-tools' own function (never a copy of it) and the xroms one, and
compares them, at rtol=1e-12 wherever the arithmetic is the same. Where xroms differs on
purpose (it measures longitude conventions instead of taking a flag, labels its outputs, and
extends boundaries by default), the test compares what is common and says what is not.

Sections: longitude wrapping and the straddle test, longitude/latitude at velocity points,
staggering and rotation, the vertical coordinate, and grid metrics.
"""

import types

import numpy as np
import pytest
import xarray as xr

import xroms

from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import merged


rt = pytest.importorskip("roms_tools")
rt_utils = pytest.importorskip("roms_tools.utils")
rt_grid = pytest.importorskip("roms_tools.setup.grid")
rt_vertical = pytest.importorskip("roms_tools.vertical_coordinate")

RTOL = 1e-12
ATOL = 1e-12  # degrees: a few ulp at 360, where two spellings of the same wrap round differently


def curvilinear_rho(lon1d, *, tilt=0.0, neta=7, transpose=False):
    """A Dataset of rho ``lon_rho``/``lat_rho``, stored in 0-360 as roms-tools grids are.

    ``lon1d`` are the longitudes along xi, unwrapped (358..362 crosses Greenwich), ``tilt``
    skews them along eta to make the grid curvilinear, and ``transpose`` makes longitude vary
    along eta instead (so a jump shows up along the other axis).
    """
    lon1d = np.asarray(lon1d, dtype=float)
    eta = np.linspace(0.0, 1.0, neta)[:, None]
    xi = np.linspace(0.0, 1.0, lon1d.size)[None, :]
    lon = (lon1d[None, :] + tilt * eta) % 360.0
    lat = -5.0 + 8.0 * eta + 2.0 * xi**2
    if transpose:
        lon, lat = lon.T, lat.T
    dims = ("eta_rho", "xi_rho")
    return xr.Dataset(
        coords={
            "lon_rho": (dims, lon, {"units": "degrees_east"}),
            "lat_rho": (dims, lat, {"units": "degrees_north"}),
        }
    )


# (unwrapped longitudes along xi, tilt, truth): crossing Greenwich (narrow, wide, tilted), crossing
# the dateline, regional, near-global, and wider than any single wrap
GRIDS = {
    "greenwich": (np.linspace(-2.0, 2.0, 11), 0.0, True),
    "greenwich_tilted": (np.linspace(-2.0, 2.0, 11), 3.0, True),
    "greenwich_wide": (np.linspace(-60.0, 120.0, 25), 5.0, True),
    "dateline": (np.linspace(178.0, 182.0, 11), 0.0, False),
    "pacific": (np.linspace(77.0, 316.0, 40), 4.0, False),
    "regional": (np.linspace(10.0, 20.0, 11), 2.0, False),
    "regional_west": (np.linspace(-100.0, -80.0, 11), 0.0, False),
    "near_global": (np.linspace(0.0, 350.0, 36), 0.0, False),
}


# ----------------------------------------------------------------- wrapping and straddling
LONGITUDES = np.concatenate(
    [
        np.arange(-1080.0, 1081.0, 7.5),
        np.random.default_rng(0).uniform(-2000.0, 2000.0, 500),
        [0.0, 180.0, -180.0, 360.0, 540.0, -540.0, 179.99999, 180.00001, -179.99999, 359.99999],
    ]
)


@pytest.mark.parametrize("straddle", [True, False])
def test_normalize_longitude(straddle):
    theirs = np.array([rt_utils.normalize_longitude(float(lon), straddle) for lon in LONGITUDES])
    ours = xroms.wrap_longitude(LONGITUDES, "-180-180" if straddle else "0-360")
    np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("straddle", [True, False])
def test_normalize_longitude_agrees_on_scalars(straddle):
    for lon in (-190.0, -180.0, 0.0, 10.5, 180.0, 190.0, 360.0, 725.25):
        theirs = rt_utils.normalize_longitude(lon, straddle)
        assert xroms.wrap_longitude(lon, "-180-180" if straddle else "0-360") == pytest.approx(theirs, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("straddle", [True, False])
def test_wrap_longitudes(straddle):
    rng = np.random.default_rng(1)
    # roms-tools only shifts once, so give it values within one wrap of range, and none on the seams
    # it leaves alone (-180 with straddle, 360 without) where xroms wraps to the end of the interval
    low, high = (-179.9, 540.0) if straddle else (-359.9, 359.9)
    shapes = {"rho": (7, 9), "u": (7, 8), "v": (6, 9)}
    coords = {}
    for pos, shape in shapes.items():
        coords[f"lon_{pos}"] = (C.CANONICAL[pos], rng.uniform(low, high, shape), {"units": "degrees_east", "long_name": pos})
        coords[f"lat_{pos}"] = (C.CANONICAL[pos], rng.uniform(-80, 80, shape))
    ds = xr.Dataset(coords=coords)
    theirs = rt_utils.wrap_longitudes(ds, straddle)
    ours = xroms.wrap_longitude(ds, "-180-180" if straddle else "0-360")
    for pos in shapes:
        name = f"lon_{pos}"
        np.testing.assert_allclose(ours[name].values, theirs[name].values, rtol=RTOL, atol=ATOL)
        assert ours[name].dims == theirs[name].dims and ours[name].attrs == theirs[name].attrs
        np.testing.assert_array_equal(ours[f"lat_{pos}"].values, theirs[f"lat_{pos}"].values)
    assert ours.attrs == theirs.attrs


@pytest.mark.parametrize("name", GRIDS)
def test_straddle_criterion(name):
    lon1d, tilt, truth = GRIDS[name]
    for transpose in (False, True):
        ds = curvilinear_rho(lon1d, tilt=tilt, transpose=transpose)
        # roms-tools' own method, run on a stand-in for its Grid (it reads self.ds and sets self.straddle)
        stand_in = types.SimpleNamespace(ds=ds.copy())
        rt_grid.Grid._straddle(stand_in)
        assert stand_in.straddle == truth, "the synthetic grid is not the case it claims to be"
        assert xroms.straddles(ds) == stand_in.straddle
        assert xroms.straddles(ds["lon_rho"]) == stand_in.straddle


@pytest.mark.parametrize("name", GRIDS)
def test_the_convention_xroms_picks_is_the_one_roms_tools_uses(name):
    # roms-tools wraps into -180..180 for a straddling grid and keeps 0..360 otherwise; for a grid
    # stored in 0..360 xroms leaves the contiguous ones alone and moves a straddler, the same way
    lon1d, tilt, truth = GRIDS[name]
    ds = curvilinear_rho(lon1d, tilt=tilt)
    theirs = rt_utils.wrap_longitudes(ds, truth)
    ours = xroms.wrap_longitude(ds)
    np.testing.assert_allclose(ours["lon_rho"].values, theirs["lon_rho"].values, rtol=RTOL, atol=ATOL)
    if truth:
        assert float(ours["lon_rho"].min()) < 0


# ----------------------------------------------------------------- positions at velocity points
@pytest.mark.parametrize("name", GRIDS)
def test_lat_lon_at_velocity_points(name):
    lon1d, tilt, straddle = GRIDS[name]
    ds = curvilinear_rho(lon1d, tilt=tilt)
    theirs = rt_grid._add_lat_lon_at_velocity_points(ds, straddle)
    for pos in ("u", "v"):
        lon, lat = xroms.lonlat_at(ds, pos)
        assert lon.dims == theirs[f"lon_{pos}"].dims == C.CANONICAL[pos]
        np.testing.assert_allclose(lon.values, theirs[f"lon_{pos}"].values, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lat.values, theirs[f"lat_{pos}"].values, rtol=RTOL, atol=ATOL)
        # same words; xroms spells the units the CF way and adds a standard_name, roms-tools writes "degrees East"
        assert lon.attrs["long_name"] == theirs[f"lon_{pos}"].attrs["long_name"]
        assert lat.attrs["long_name"] == theirs[f"lat_{pos}"].attrs["long_name"]
        assert lon.attrs["units"] == "degrees_east" and theirs[f"lon_{pos}"].attrs["units"] == "degrees East"
        assert float(lon.min()) >= 0.0 and float(lon.max()) < 360.0  # both leave a 0-360 grid in 0-360


def test_a_grid_crossing_greenwich_averages_through_the_seam_in_both():
    ds = xr.Dataset(
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), np.array([[359.5, 0.5], [359.5, 0.5]])),
            "lat_rho": (("eta_rho", "xi_rho"), np.array([[1.0, 1.0], [2.0, 2.0]])),
        }
    )
    theirs = rt_grid._add_lat_lon_at_velocity_points(ds, True)
    assert theirs["lon_u"].values.tolist() == [[0.0], [0.0]]
    assert xroms.lonlat_at(ds, "u")[0].values.tolist() == [[0.0], [0.0]]
    assert xroms.lonlat_at(ds, "v")[0].values.tolist() == theirs["lon_v"].values.tolist() == [[359.5, 0.5]]


@pytest.mark.parametrize("name", ["greenwich", "greenwich_wide", "regional", "dateline"])
def test_roms_tools_grid_files_keep_the_positions_they_hold(name):
    # a grid that carries lon_u/lat_u: xroms returns them, and they agree with a fresh average
    lon1d, tilt, straddle = GRIDS[name]
    full = rt_grid._add_lat_lon_at_velocity_points(curvilinear_rho(lon1d, tilt=tilt), straddle)
    for pos in ("u", "v"):
        stored = xroms.lonlat_at(full, pos)
        np.testing.assert_array_equal(stored[0].values, full[f"lon_{pos}"].values)
        again = xroms.lonlat_at(full.drop_vars([f"lon_{pos}", f"lat_{pos}"]), pos)
        np.testing.assert_allclose(again[0].values, stored[0].values, rtol=RTOL, atol=ATOL)


# ----------------------------------------------------------------- staggering and rotation
def velocities(seed=0, nt=3, neta=9, nxi=12):
    """Random u (on u points), v (on v points) and a grid angle, on canonical dims."""
    rng = np.random.default_rng(seed)
    u = xr.DataArray(rng.normal(size=(nt, neta, nxi - 1)), dims=("time", "eta_rho", "xi_u"), name="u")
    v = xr.DataArray(rng.normal(size=(nt, neta - 1, nxi)), dims=("time", "eta_v", "xi_rho"), name="v")
    angle = xr.DataArray(rng.uniform(-np.pi, np.pi, size=(neta, nxi)), dims=("eta_rho", "xi_rho"), name="angle")
    return u, v, angle


def test_staggering_matches():
    u, v, _ = velocities()
    rho = xr.DataArray(np.random.default_rng(3).normal(size=(3, 9, 12)), dims=("time", "eta_rho", "xi_rho"))
    for ours, theirs in (
        (xroms.to_u(rho), rt_utils.interpolate_from_rho_to_u(rho)),
        (xroms.to_v(rho), rt_utils.interpolate_from_rho_to_v(rho)),
        # u and v onto rho points: roms-tools leaves the two edge points NaN, which "fill" does too
        (xroms.to_rho(u, hboundary="fill"), rt_utils.interpolate_from_u_to_rho(u)),
        (xroms.to_rho(v, hboundary="fill"), rt_utils.interpolate_from_v_to_rho(v)),
    ):
        assert ours.dims == theirs.dims
        np.testing.assert_allclose(ours.values, theirs.values, rtol=RTOL)
    # xroms' default, "extend", repeats the nearest value at the edges instead of leaving NaN
    interior = dict(xi_rho=slice(1, -1))
    np.testing.assert_allclose(xroms.to_rho(u).isel(interior).values, rt_utils.interpolate_from_u_to_rho(u).isel(interior).values, rtol=RTOL)
    assert not np.isnan(xroms.to_rho(u).values).any() and np.isnan(rt_utils.interpolate_from_u_to_rho(u).values).any()


def test_rotation_at_rho_points():
    rng = np.random.default_rng(4)
    u, v = (xr.DataArray(rng.normal(size=(3, 9, 12)), dims=("time", "eta_rho", "xi_rho")) for _ in range(2))
    _, _, angle = velocities()
    # roms-tools rotates by +angle (earth -> grid); -angle goes from the grid to the earth, as xroms does
    theirs = rt_utils.rotate_velocities(u, v, -angle)
    ours = xroms.rotate_vectors(u, v, angle, hcoord=None)
    for got, want in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(got.values, want.values, rtol=RTOL, atol=1e-15)
        assert got.dims == want.dims


def test_grid_to_earth_matches_rotate_velocities_with_interpolate_before():
    u, v, angle = velocities()
    theirs = rt_utils.rotate_velocities(u, v, -angle, interpolate_before=True)
    # "fill" puts NaN where roms-tools has it: the edges that u and v do not reach
    east, north = xroms.grid_to_earth(u, v, angle, hboundary="fill")
    for got, want in zip((east, north), theirs, strict=True):
        assert got.dims == want.dims == ("time", "eta_rho", "xi_rho")
        np.testing.assert_allclose(got.values, want.values, rtol=RTOL, atol=1e-15)
        assert np.isnan(got.values).any()
    # by default xroms extends the edges; the interior is the same
    east, north = xroms.grid_to_earth(u, v, angle)
    interior = dict(eta_rho=slice(1, -1), xi_rho=slice(1, -1))
    for got, want in zip((east, north), theirs, strict=True):
        assert not np.isnan(got.values).any()
        np.testing.assert_allclose(got.isel(interior).values, want.isel(interior).values, rtol=RTOL, atol=1e-15)


def test_earth_to_grid_matches_rotate_velocities_with_interpolate_after():
    _, _, angle = velocities()
    rng = np.random.default_rng(5)
    east = xr.DataArray(rng.normal(size=(3, 9, 12)), dims=("time", "eta_rho", "xi_rho"))
    north = xr.DataArray(rng.normal(size=(3, 9, 12)), dims=("time", "eta_rho", "xi_rho"))
    theirs_u, theirs_v = rt_utils.rotate_velocities(east, north, angle, interpolate_after=True)
    ours_u, ours_v = xroms.earth_to_grid(east, north, angle)
    for got, want, dims in ((ours_u, theirs_u, ("time", "eta_rho", "xi_u")), (ours_v, theirs_v, ("time", "eta_v", "xi_rho"))):
        assert got.dims == want.dims == dims
        np.testing.assert_allclose(got.values, want.values, rtol=RTOL, atol=1e-15)


def test_a_round_trip_agrees_where_roms_tools_has_values():
    u, v, angle = velocities()
    east, north = xroms.grid_to_earth(u, v, angle)
    ours = xroms.earth_to_grid(east, north, angle)
    theirs = rt_utils.rotate_velocities(*rt_utils.rotate_velocities(u, v, -angle, interpolate_before=True), angle, interpolate_after=True)
    for got, want in zip(ours, theirs, strict=True):
        # roms-tools leaves NaN along the edges (and so does anything that averages them): xroms
        # extends them, and the two agree wherever roms-tools has a value, the whole interior
        finite = np.isfinite(want.values)
        assert finite.sum() > 0.5 * finite.size
        np.testing.assert_allclose(got.values[finite], want.values[finite], rtol=RTOL, atol=1e-15)


def test_land_masks_at_velocity_points():
    mask = xr.DataArray((np.random.default_rng(2).uniform(size=(9, 12)) > 0.3).astype(float), dims=("eta_rho", "xi_rho"))
    for pos, theirs in (
        ("u", rt_utils.interpolate_from_rho_to_u(mask, method="multiplicative")),
        ("v", rt_utils.interpolate_from_rho_to_v(mask, method="multiplicative")),
    ):
        ours = xroms.mask_at(mask, pos)
        assert ours.dims == theirs.dims
        np.testing.assert_array_equal(ours.values, theirs.values)
    m = mask.values  # roms-tools has no psi mask: ROMS' own definition is the product of the four rho neighbours
    psi = m[:-1, :-1] * m[:-1, 1:] * m[1:, :-1] * m[1:, 1:]
    np.testing.assert_array_equal(xroms.mask_at(mask, "psi").values, psi)


# ----------------------------------------------------------------- the vertical coordinate
VERTICAL = [(5.0, 2.0), (7.0, 0.5), (1.5, 8.0), (10.0, 10.0), (0.1, 0.1)]


@pytest.mark.parametrize("theta_s, theta_b", VERTICAL)
@pytest.mark.parametrize("N", [1, 3, 10, 50])
@pytest.mark.parametrize("loc, kind", [("rho", "r"), ("w", "w")])
def test_sigma_and_stretching(theta_s, theta_b, N, loc, kind):
    theirs_cs, theirs_sigma = rt_vertical.sigma_stretch(theta_s, theta_b, N, kind)
    sigma = xroms.sigma_levels(N, loc)
    cs = xroms.stretching(sigma, theta_s, theta_b)
    assert sigma.dims == theirs_sigma.dims == (f"s_{loc}",) and cs.dims == theirs_cs.dims
    np.testing.assert_allclose(sigma.values, theirs_sigma.values, rtol=RTOL)
    np.testing.assert_allclose(cs.values, theirs_cs.values, rtol=RTOL)


@pytest.mark.parametrize("theta_s, theta_b", VERTICAL)
def test_compute_cs_on_plain_numbers(theta_s, theta_b):
    sigma = np.linspace(-1.0, 0.0, 41)
    np.testing.assert_allclose(xroms.stretching(sigma, theta_s, theta_b).values, rt_vertical.compute_cs(sigma, theta_s, theta_b), rtol=RTOL)
    assert float(xroms.stretching(xr.DataArray(-0.37), theta_s, theta_b)) == pytest.approx(float(rt_vertical.compute_cs(-0.37, theta_s, theta_b)), rel=RTOL)


@pytest.mark.parametrize("kind, scoord", [("r", "rho"), ("w", "w")])
@pytest.mark.parametrize("with_zeta", [True, False])
def test_compute_depth(kind, scoord, with_zeta):
    ds = syn.make_dataset("rutgers", N=8, theta_s=7.0, theta_b=3.0, hc=25.0)
    h = ds["h"].reset_coords(drop=True)
    zeta = ds["zeta"].reset_coords(drop=True).rename(ocean_time="time") if with_zeta else 0
    cs, sigma = rt_vertical.sigma_stretch(7.0, 3.0, 8, kind)
    theirs = rt_vertical.compute_depth(zeta, h, 25.0, cs, sigma)
    xsigma = xroms.sigma_levels(8, scoord)
    ours = xroms.compute_depth(h, zeta, hc=25.0, Cs=xroms.stretching(xsigma, 7.0, 3.0), sigma=xsigma, Vtransform=2, positive="down")
    assert ours.dims == theirs.dims
    np.testing.assert_allclose(ours.values, theirs.values, rtol=RTOL)
    assert float(ours.min()) > -1.0  # depths below the surface are positive


@pytest.mark.parametrize("location", ["rho", "u", "v"])
@pytest.mark.parametrize("depth_type, scoord", [("layer", "s_rho"), ("interface", "s_w")])
@pytest.mark.parametrize("with_zeta", [True, False])
def test_depth_coordinates_at_velocity_points(ucla_romstools, location, depth_type, scoord, with_zeta):
    # roms-tools averages h and zeta onto the u or v points and computes depths there (method="interp_inputs");
    # xroms' default averages the depths instead, as ROMS does, which differs by the curvature of the profile
    output, grid = ucla_romstools
    zeta = output["zeta"] if with_zeta else 0
    theirs = rt_vertical.compute_depth_coordinates(grid, zeta, depth_type, location)
    ours = xroms.z(grid, hcoord=location, scoord=scoord, zeta=zeta, method="interp_inputs", positive="down")
    assert ours.dims == theirs.dims
    np.testing.assert_allclose(ours.values, theirs.values, rtol=RTOL)
    if location != "rho" and with_zeta:
        averaged_depths = xroms.z(grid, hcoord=location, scoord=scoord, zeta=zeta, positive="down")
        assert not np.allclose(averaged_depths.values, theirs.values, rtol=1e-9, atol=0)


# ----------------------------------------------------------------- metrics
@pytest.mark.parametrize("lat", [None, 12.5, -60.0, 0.0])
def test_nominal_resolution_in_degrees(lat):
    ds = syn.make_dataset("rutgers")
    theirs = rt_utils.infer_nominal_horizontal_resolution(ds, lat)
    assert xroms.nominal_resolution(ds, units="degrees", lat=lat) == pytest.approx(theirs, rel=RTOL)


def test_nominal_resolution_uses_the_same_mean_spacing():
    ds = merged("rutgers", neta=14, nxi=20)
    meters = ((1 / ds.pm).mean() + (1 / ds.pn).mean()) / 2
    assert xroms.nominal_resolution(ds) == pytest.approx(float(meters), rel=RTOL)
    ratio = xroms.nominal_resolution(ds, units="degrees", lat=0.0) / xroms.nominal_resolution(ds)
    assert ratio == pytest.approx(rt_utils.infer_nominal_horizontal_resolution(ds, 0.0) / float(meters), rel=RTOL)


def test_cell_area_is_one_over_pm_pn():
    ds = syn.make_dataset("rutgers")
    area = 1 / (ds["pm"] * ds["pn"])  # roms-tools: cdr_analysis (area) and topography (smoothing weights)
    ours = xroms.dA(ds, "rho")
    assert ours.dims == area.dims
    np.testing.assert_allclose(ours.values, area.values, rtol=RTOL)
