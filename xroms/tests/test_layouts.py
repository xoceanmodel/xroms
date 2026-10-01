"""The ``ds.xroms`` accessor on every layout, and what each layout lacks or spells differently.

* every method and property of ``ds.xroms`` (``_sweep.ACCESSOR``) on Rutgers, UCLA, CROCO and REMORA
  Datasets, numpy and dask alike, against the pure function it wraps: the same numbers, at the
  position it documents, in the Dataset's own dim naming, with the Dataset's coordinates for that
  position, and no dims gained when combined with the Dataset's own variable there;
* analytic checks of the same members on a uniform grid (convergence, vorticity, gradients, rotations);
* the errors for what a layout (or a Dataset) does not have;
* CROCO files that give their vertical transform only as ``VertCoordType``;
* land (NaN in the state, 0 in the speeds) and NaNs in a depth average.
"""

import importlib.util

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms._align import GridMismatchError
from xroms.conventions import CANONICAL, RUTGERS, canonicalize, convention, horizontal_coords, hposition, vposition
from xroms.tests import _sweep as S
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked


def _close(got, want, what="", rtol=1e-9):
    """Equal values, to ``rtol`` and 1e-12 of the field's peak (the same code runs on both sides)."""
    assert set(got.dims) == set(want.dims), f"{what}: dims {got.dims} vs {want.dims}"
    want_values = want.values
    finite = np.abs(want_values[np.isfinite(want_values)])
    atol = 1e-12 * (finite.max() if finite.size else 0.0)
    np.testing.assert_allclose(got.transpose(*want.dims).values, want_values, rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


# --- every member, against the pure function ----------------------------------------------------------


@pytest.mark.parametrize("chunks", [False, True], ids=["numpy", "dask"])
@pytest.mark.parametrize("acc", S.ACCESSOR, ids=S.ACC_IDS)
def test_accessor_member_matches_its_pure_function(layout, acc, chunks):
    ds = S.dataset(layout)
    got, want = S.results(acc.accessor(chunked(ds) if chunks else ds)), S.results(acc.pure(ds))
    assert len(got) == len(want)
    names = RUTGERS if convention(ds) == "rutgers" else CANONICAL
    for i, (g, w) in enumerate(zip(got, want)):
        pos = acc.pos[i] if isinstance(acc.pos, tuple) else acc.pos
        # the same numbers as the pure function
        _close(canonicalize(g), canonicalize(w), acc.name)
        # at the documented position
        assert hposition(canonicalize(g)) == pos and vposition(g) == acc.vert, (acc.name, g.dims)
        # in the Dataset's own dim naming for that position
        if pos is not None:
            horizontal = tuple(d for d in g.dims if d.startswith(("eta_", "xi_")))
            assert horizontal == names[pos], f"{acc.name}: {horizontal} is not the Dataset's naming {names[pos]}"
        # combined with the Dataset's own variable there, no dims appear
        same = acc.same[i] if isinstance(acc.same, tuple) else acc.same
        if same is not None:
            combined = ds[same] + g
            assert set(combined.dims) == set(ds[same].dims), f"{acc.name}: combining with {same} gained dims {combined.dims}"
        # the Dataset's own coordinates for the position come along; lon/lat it keeps as data variables (UCLA) do not
        if pos is not None:
            for name in horizontal_coords(ds, pos):
                if name is None or not set(ds[name].dims) <= set(g.dims):
                    continue
                if name not in ds.coords:
                    assert name not in g.coords, f"{acc.name}: the data variable {name} became a coordinate"
                    continue
                assert name in g.coords, f"{acc.name}: lacks the coordinate {name}"
                np.testing.assert_array_equal(g[name].transpose(*ds[name].dims).values, ds[name].values)


def test_accessor_properties_that_are_not_computed(layout):
    ds = S.dataset(layout)
    assert ds.xroms.find_horizontal_velocities() == ("u", "v")
    got, want = ds.xroms.vertical_params, xroms.vertical_params(ds)
    assert (got.Vtransform, got.hc) == (want.Vtransform, want.hc)
    for name in ("Cs_r", "Cs_w", "sigma_r", "sigma_w"):
        np.testing.assert_array_equal(getattr(got, name).values, getattr(want, name).values)
    # the accessor results are views of the Dataset: they never write into it
    before = ds.copy(deep=True)
    for acc in S.ACCESSOR:
        acc.accessor(ds)
    xr.testing.assert_identical(ds, before)


def test_results_can_be_stored_back_in_the_dataset(layout):
    # UCLA files keep lon_rho/lat_rho as data variables: results do not carry them as coords, which would not merge
    ds = S.dataset(layout)
    results = {
        # accessor results
        "speed": ds.xroms.speed, "depth": ds.xroms.z_rho, "dtdx": ds.xroms.ddxi("temp"),
        # pure functions that read the grid hand its coordinates on
        "z": xroms.z(ds), "dz": xroms.dz(ds), "rho": xroms.density(ds.temp, ds.salt, grid=ds), "dx": xroms.dx(ds),
    }
    for name, value in results.items():
        stored = ds.assign({name: value})
        np.testing.assert_array_equal(stored[name].values, value.values)
        assert stored[name].dims == value.dims, name


def test_xgcm_grid_averages_like_to_grid(layout):
    ds = S.dataset(layout)
    grid = ds.xroms.xgcm_grid()
    assert set(grid.axes) == {"X", "Y", "Z"}
    can = canonicalize(ds)
    np.testing.assert_allclose(grid.interp(can.temp, "X").values, xroms.to_u(can.temp).values, rtol=1e-12)
    np.testing.assert_allclose(grid.interp(can.temp, "Y").values, xroms.to_v(can.temp).values, rtol=1e-12)
    np.testing.assert_allclose(grid.interp(can.temp, "Z").values, xroms.to_s_w(can.temp).values, rtol=1e-12)


# --- analytic checks on a uniform grid -------------------------------------------------------------------

ANGLE = 0.3


@pytest.fixture
def uniform_layout(layout):
    return S.dataset(layout, uniform=True, angle=ANGLE)


def test_gradients_and_vorticity_of_the_analytic_fields(uniform_layout):
    ds = uniform_layout
    np.testing.assert_allclose(ds.xroms.ddxi("temp").values, syn.TEMP_A, rtol=1e-9)
    np.testing.assert_allclose(ds.xroms.ddeta("temp").values, 0.0, atol=1e-12)
    np.testing.assert_allclose(ds.xroms.ddz("temp").values, syn.TEMP_B, rtol=1e-9)
    np.testing.assert_allclose(ds.xroms.vort.values, 0.0, atol=1e-12)
    np.testing.assert_allclose(ds.xroms.convergence.values, -(syn.U_A + syn.V_A), rtol=1e-9)
    np.testing.assert_allclose(ds.xroms.divergence.values, syn.U_A + syn.V_A, rtol=1e-9)
    # a field of depth alone has none of these at constant depth; salt is quadratic in z
    np.testing.assert_allclose(ds.xroms.ddxi("salt").values, 0.0, atol=1e-12)


def test_rotations_of_the_analytic_velocities(uniform_layout):
    ds = uniform_layout
    east, north = ds.xroms.eastnorth
    u_rho, v_rho = xroms.to_rho(ds.u.fillna(0)), xroms.to_rho(ds.v.fillna(0))
    # rotating by the grid angle and back
    _close(canonicalize(ds.xroms.east_rotated(-ANGLE)), canonicalize(u_rho), "east_rotated(-angle) is u at rho points", rtol=1e-12)
    _close(canonicalize(ds.xroms.north_rotated(-ANGLE)), canonicalize(v_rho), "north_rotated(-angle) is v at rho points", rtol=1e-12)
    # a rotation keeps the length of a vector
    np.testing.assert_allclose((east**2 + north**2).values, ds.xroms.speed.values**2, rtol=1e-12)
    np.testing.assert_allclose(ds.xroms.KE.values, 0.5 * xroms.rho0(ds) * ds.xroms.speed.values**2, rtol=1e-12)
    # and the east and north components follow the angle
    np.testing.assert_allclose(east.values, (u_rho * np.cos(ANGLE) - v_rho * np.sin(ANGLE)).values, rtol=1e-12)
    np.testing.assert_allclose(north.values, (u_rho * np.sin(ANGLE) + v_rho * np.cos(ANGLE)).values, rtol=1e-12)


def test_grid_aligned_velocities_from_the_earth_components(uniform_layout):
    ds = uniform_layout
    east, north = ds.xroms.eastnorth
    # (the coordinates are dropped: see test_results_can_be_stored_back_in_the_dataset for UCLA's)
    only_earth = ds.drop_vars(["u", "v"]).assign(u_eastward=east.reset_coords(drop=True), v_northward=north.reset_coords(drop=True))
    assert only_earth.xroms.find_horizontal_velocities() == ("u_eastward", "v_northward")
    u, v = only_earth.xroms.u, only_earth.xroms.v
    assert u.dims == ds.u.dims and v.dims == ds.v.dims
    # averaging to and from rho points loses the end points; away from them u and v come back
    np.testing.assert_allclose(u.isel(xi_u=slice(1, -1)).values, ds.u.isel(xi_u=slice(1, -1)).values, atol=1e-12)
    np.testing.assert_allclose(v.isel(eta_v=slice(1, -1)).values, ds.v.isel(eta_v=slice(1, -1)).values, atol=1e-12)
    # the file's own eastward and northward velocities are what speed and vorticity use
    np.testing.assert_allclose(only_earth.xroms.speed.values, np.sqrt(east**2 + north**2).values, rtol=1e-12)


# --- what a Dataset or layout lacks ----------------------------------------------------------------------------

MISSING = [
    ("east", lambda ds: ds.xroms.east, "angle", KeyError, r"'angle'"),
    ("ug", lambda ds: ds.xroms.ug, "f", KeyError, r"'f'"),
    ("ug", lambda ds: ds.xroms.ug, "zeta", KeyError, r"'zeta'"),
    ("rho", lambda ds: ds.xroms.rho, "temp", KeyError, r"'temp'"),
    ("sig0", lambda ds: ds.xroms.sig0, "salt", KeyError, r"'salt'"),
    ("convergence_norm", lambda ds: ds.xroms.convergence_norm, "f", KeyError, r"'f'"),
    ("ddxi", lambda ds: ds.xroms.ddxi("temp"), "pm", GridMismatchError, r"\['pm'\]"),
    ("ddeta", lambda ds: ds.xroms.ddeta("temp"), "pn", GridMismatchError, r"\['pn'\]"),
    ("gridmean", lambda ds: ds.xroms.gridmean("temp", ("X", "Y")), "pm", GridMismatchError, r"\['pm'\]"),
    ("dx", lambda ds: ds.xroms.dx(), "pm", GridMismatchError, r"\['pm'\]"),
    ("dA", lambda ds: ds.xroms.dA(), "pn", GridMismatchError, r"\['pn'\]"),
    ("ddz", lambda ds: ds.xroms.ddz("temp"), "h", GridMismatchError, r"\['h'\]"),
    ("z_rho", lambda ds: ds.xroms.z_rho, "h", GridMismatchError, r"\['h'\]"),
    ("N2", lambda ds: ds.xroms.N2, "h", GridMismatchError, r"\['h'\]"),
    ("mld", lambda ds: ds.xroms.mld(), "h", GridMismatchError, r"\['h'\]"),
    # time-varying data and no free surface: depths would silently be flat, so the call asks
    ("ddz", lambda ds: ds.xroms.ddz("temp"), "zeta", GridMismatchError, r"has no 'zeta'"),
    ("N2", lambda ds: ds.xroms.N2, "zeta", GridMismatchError, r"has no 'zeta'"),
    ("zslice", lambda ds: ds.xroms.zslice("temp", [-5.0]), "temp", KeyError, r"'temp' is not a variable"),
]


@pytest.mark.parametrize("label,call,dropped,error,pattern", MISSING, ids=[f"{m[0]}-without-{m[2]}" for m in MISSING])
def test_missing_variable_is_named(layout, label, call, dropped, error, pattern):
    ds = S.dataset(layout).drop_vars(dropped)
    with pytest.raises(error, match=pattern):
        call(ds)


def test_z_without_a_free_surface_is_at_rest(layout):
    # no zeta and no data in time to match it with: the depths are the static ones, as documented
    ds = S.dataset(layout)
    static = xroms.z(ds, zeta=0)
    np.testing.assert_array_equal(ds.drop_vars("zeta").xroms.z_rho.values, static.transpose(*ds.drop_vars("zeta").xroms.z_rho.dims).values)


@pytest.mark.parametrize("dropped,member", [("v", "dudz"), ("u", "dvdz")])
def test_shear_needs_only_its_own_component(layout, dropped, member):
    ds = S.dataset(layout)
    got = getattr(ds.drop_vars(dropped).xroms, member)
    _close(canonicalize(got), canonicalize(getattr(ds.xroms, member)), member)


@pytest.mark.parametrize("dropped", ["u", "v"])
@pytest.mark.parametrize("member", ["speed", "KE", "vort", "convergence", "vertical_shear", "ertel"])
def test_missing_velocity_names_the_velocity(layout, dropped, member):
    ds = S.dataset(layout).drop_vars(dropped)
    with pytest.raises(KeyError) as err:
        getattr(ds.xroms, member)
    assert "'east'" not in str(err.value) and (f"'{dropped}'" in str(err.value) or "velocit" in str(err.value))


def test_remora_has_no_lonlat_and_says_what_needs_it():
    ds = S.dataset("remora")
    assert not [v for v in ds.variables if str(v).startswith(("lon_", "lat_"))]
    assert xroms.nominal_resolution(ds) == pytest.approx(1400.0)
    with pytest.raises(GridMismatchError, match=r"\['lat_rho'\]"):
        xroms.nominal_resolution(ds, units="degrees")
    # TEOS-10 needs the location to turn practical into absolute salinity
    with pytest.raises(ValueError, match="longitude and latitude.*lon= and lat="):
        xroms.density(ds.temp, ds.salt, grid=ds, eos="teos10")
    with pytest.raises(ValueError, match="longitude and latitude"):
        xroms.potential_density(ds.temp, ds.salt, eos="teos10", grid=ds)
    with pytest.raises(ValueError, match="both lon= and lat="):
        xroms.density(ds.temp, ds.salt, grid=ds, eos="teos10", lon=-90.0)


@pytest.mark.skipif(importlib.util.find_spec("gsw") is None, reason="needs gsw")
def test_teos10_runs_on_every_layout_given_a_location(layout):
    ds = S.dataset(layout)
    kwargs = {"lon": -90.0, "lat": 28.0} if layout == "remora" else {}
    rho = xroms.density(ds.temp, ds.salt, grid=ds, eos="teos10", **kwargs)
    assert rho.dims == canonicalize(ds.temp).dims and bool(np.isfinite(rho).all())
    roms = xroms.density(ds.temp, ds.salt, grid=ds)
    assert 0.0 < float(abs(rho - roms).max()) < 3.0  # two equations of state: the same water to a few g/m3


def test_interpll_needs_xesmf_and_lonlat(layout):
    ds = S.dataset(layout)
    xname = horizontal_coords(ds.temp, "rho")[0]
    lonlat = xname is not None and xname.startswith("lon_")  # UCLA keeps lon/lat as data variables, REMORA has only x/y
    if importlib.util.find_spec("xesmf") is None:
        with pytest.raises(ModuleNotFoundError, match="xESMF"):
            xroms.interpll(ds.temp, [-90.0], [28.0])
    elif not lonlat:
        with pytest.raises(ValueError, match="lon_rho/lat_rho"):
            xroms.interpll(ds.temp, [-90.0], [28.0])
    else:
        out = xroms.interpll(ds.temp, [float(ds.lon_rho[4, 5])], [float(ds.lat_rho[4, 5])])
        assert out.sizes["locations"] == 1


def test_nearest_point_uses_lonlat_or_xy(layout):
    ds = S.dataset(layout)
    xname, yname = horizontal_coords(ds, "rho")
    assert xname is not None and xname.startswith("x_" if layout == "remora" else "lon_")
    x0, y0 = float(ds[xname][3, 5]), float(ds[yname][3, 5])
    assert ds.xroms.argsel2d(x0, y0) == (3, 5)
    point = ds.xroms.sel2d("temp", x0, y0)
    xr.testing.assert_equal(point.reset_coords(drop=True), canonicalize(ds.temp).isel(eta_rho=3, xi_rho=5).reset_coords(drop=True))
    if xname in ds.temp.coords:  # on the DataArray, the coordinates it carries are used
        assert ds.temp.xroms.argsel2d(x0, y0) == (3, 5)
    with pytest.raises(KeyError, match="no lon/lat or x/y coordinates at rho points"):
        ds.drop_vars([xname, yname]).xroms.argsel2d(x0, y0)


# --- CROCO: the vertical transform given as VertCoordType only ---------------------------------------------------------


def _croco_vertcoordtype_only(vtransform, value):
    ds = syn.make_dataset("croco", vtransform=vtransform)
    assert "Vtransform" not in ds.variables
    attrs = {k: v for k, v in ds.attrs.items() if k != "Vtransform"}
    if value is None:
        attrs.pop("VertCoordType")
    else:
        attrs["VertCoordType"] = value
    ds.attrs = attrs
    return ds


@pytest.mark.parametrize("value,expected", [("NEW", 2), ("OLD", 1), ("new", 2), (" Old ", 1), ("nEw", 2)])
def test_croco_vertcoordtype_gives_the_vertical_transform(value, expected):
    ds = _croco_vertcoordtype_only(2, value)
    assert "Vtransform" not in ds.attrs and "Vtransform" not in ds.variables
    assert xroms.vertical_params(ds).Vtransform == expected
    explicit = syn.make_dataset("croco", vtransform=expected)
    assert explicit.attrs["Vtransform"] == float(expected)
    for kwargs in ({}, {"hcoord": "u"}, {"scoord": "s_w"}, {"hcoord": "psi", "scoord": "s_w"}, {"zeta": 0}):
        xr.testing.assert_identical(xroms.z(ds, **kwargs), xroms.z(explicit, **kwargs))
    xr.testing.assert_identical(xroms.dz(ds), xroms.dz(explicit))
    xr.testing.assert_identical(ds.xroms.z_rho, explicit.xroms.z_rho)
    lazy = xroms.z(chunked(ds))
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, xroms.z(explicit).values)


def test_croco_new_and_old_are_different_transforms():
    new, old = _croco_vertcoordtype_only(2, "NEW"), _croco_vertcoordtype_only(2, "OLD")
    assert (xroms.vertical_params(new).Vtransform, xroms.vertical_params(old).Vtransform) == (2, 1)
    # the same grid and parameters, so only the transform tells them apart
    assert float(abs(xroms.z(new) - xroms.z(old)).max()) > 1e-3


def test_croco_vertcoordtype_is_the_last_resort():
    ds = _croco_vertcoordtype_only(2, "NEW")
    # the argument beats everything; a Vtransform variable or attribute beats VertCoordType
    assert xroms.vertical_params(ds, Vtransform=1).Vtransform == 1
    assert xroms.vertical_params(ds.assign(Vtransform=1)).Vtransform == 1
    assert xroms.vertical_params(ds.assign_attrs(Vtransform=1.0)).Vtransform == 1
    assert xroms.vertical_params(ds.assign_attrs(Vtransform=2.0, VertCoordType="OLD")).Vtransform == 2


@pytest.mark.parametrize("value", ["WEIRD", "", "3", 1])
def test_croco_unrecognized_vertcoordtype_raises(value):
    ds = _croco_vertcoordtype_only(2, value)
    with pytest.raises(ValueError, match="cannot determine Vtransform"):
        xroms.vertical_params(ds)
    with pytest.raises(ValueError, match="cannot determine Vtransform"):
        xroms.z(ds)
    # an explicit Vtransform still works
    assert xroms.vertical_params(ds, Vtransform=2).Vtransform == 2


def test_croco_without_any_vertical_transform_raises():
    with pytest.raises(ValueError, match="Vtransform"):
        xroms.vertical_params(_croco_vertcoordtype_only(2, None))


def test_croco_unrecognized_vertcoordtype_is_named_in_the_error():
    with pytest.raises(ValueError) as err:
        xroms.vertical_params(_croco_vertcoordtype_only(2, "WEIRD"))
    assert "WEIRD" in str(err.value)


# --- land, and NaNs in a depth average ------------------------------------------------------------------------------------


def test_land_is_nan_in_the_state_and_in_the_speeds(layout):
    ds = S.dataset(layout, land=True)
    can = canonicalize(ds)
    land = can["mask_rho"] == 0
    assert bool(land.any())

    def over_land(da):
        return canonicalize(da).where(land)

    # temperature, salinity and what is made of them have no value on land ...
    for name, value in (
        ("rho", ds.xroms.rho), ("sig0", ds.xroms.sig0), ("N2", ds.xroms.N2), ("ertel", ds.xroms.ertel),
        ("zslice", ds.xroms.zslice("temp", [-5.0])), ("mld", ds.xroms.mld()),
    ):
        assert bool(over_land(value).isnull().where(land, True).all()), f"{name} has values over land"
    # ... and are finite over water, the mixed layer depth within the water column
    mld = canonicalize(ds.xroms.mld())
    assert bool((np.isfinite(mld) | land).all()) and bool(((mld > 0) & (mld <= can["h"] + 1e-9)).where(~land, True).all())
    # masked velocities count as 0 when averaged (documented), so that the water next to land keeps its values,
    # but land itself, with no velocity around it, is NaN in the speeds and the earth components too
    water = ~land
    for name, value in (("speed", ds.xroms.speed), ("KE", ds.xroms.KE), ("east", ds.xroms.east), ("north", ds.xroms.north)):
        assert bool(over_land(value).isnull().where(land, True).all()), f"{name} has values over land"
        assert bool(canonicalize(value).notnull().where(water, True).all()), f"{name} is missing over water"
    # and a column sum over land has nothing to sum
    assert bool(over_land(ds.xroms.gridsum("temp", "Z")).isnull().where(land, True).all())


def test_depth_average_is_the_gridmean_over_z_where_nothing_is_missing(layout):
    ds = S.dataset(layout)
    for kwargs in ({}, {"zeta": 0}):
        _close(canonicalize(xroms.depth_average(ds.temp, ds, **kwargs)), canonicalize(xroms.gridmean(ds.temp, ds, "Z", **kwargs)), "depth_average vs gridmean", rtol=1e-12)


def test_depth_average_over_land_is_nan_when_depths_are_finite(layout):
    ds = S.dataset(layout, land=True)
    land = canonicalize(ds)["mask_rho"] == 0
    assert bool(land.any())
    avg = xroms.depth_average(ds.temp, ds, zeta=0)
    assert bool(canonicalize(avg).where(land).isnull().where(land, True).all())
    # the grid-weighted mean says the same thing
    assert bool(canonicalize(xroms.gridmean(ds.temp, ds, "Z", zeta=0)).where(land).isnull().where(land, True).all())


def test_depth_average_of_a_column_with_a_missing_level_uses_the_valid_levels(layout):
    ds = S.dataset(layout)
    temp = ds.temp.copy(deep=True)
    temp[{"s_rho": 0, "eta_rho": 4, "xi_rho": 4}] = np.nan
    got = canonicalize(xroms.depth_average(temp, ds, zeta=0))
    want = canonicalize(xroms.gridmean(temp, ds, "Z", zeta=0))
    _close(got, want, "depth average with a NaN level", rtol=1e-12)
