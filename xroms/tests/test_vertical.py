"""Vertical coordinates and grid metrics: what reviewing the rewrite found.

Sections: depth warnings for Vtransform 1 (``compute_depth``), the sign convention
of depth bands, the attrs of depth outputs, the level marked by ``surface`` and
``bottom``, xgcm grids passed where a Dataset belongs, and the ``_align`` helpers
the vertical and metric functions rely on, and the vertical parameters each call needs
(``levels=``, ``hc=``, ``default_Vtransform=``). The behaviours that concern data cut or
edited before computing (explicit zeta/z, vertical subsets, strided subsets, grids
without zeta) are tested in ``test_modify_then_compute.py``.
"""

import warnings

import dask
import numpy as np
import pytest
import xarray as xr

import xroms

from xroms import _align
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


# ----------------------------------------------------- Vtransform 1 and hc
def vtransform1(**kwargs):
    """Vtransform 1 with hc = 60 m over water as shallow as 20 m (h is 20 to 100 m)."""
    return merged("rutgers", vtransform=1, hc=60, theta_s=7, N=10, **kwargs)


def hc_warnings(caught):
    return [w for w in caught if "Vtransform 1 needs hc" in str(w.message)]


def test_vtransform_1_warns_only_when_hc_exceeds_the_shallowest_depth():
    """ROMS stops on this configuration; xroms gave non-monotonic z without a word
    (a layer thickness of -5 m)."""
    ds = vtransform1()
    with pytest.warns(UserWarning, match=r"hc=60 m exceeds the shallowest h=20 m"):
        xroms.z(ds)
    with pytest.warns(UserWarning, match="non-monotonic"):
        dz = xroms.dz(ds)
    assert float(dz.min()) < -1.0  # the warning is deserved

    # the pure function warns too, for in-memory arrays of any kind
    with pytest.warns(UserWarning, match="Vtransform 1"):
        xroms.compute_depth(ds.h, 0, hc=60.0, Cs=ds.Cs_r, sigma=ds.s_rho, Vtransform=1)
    column = (slice(None), None, None)
    with pytest.warns(UserWarning, match="Vtransform 1"):
        xroms.compute_depth(
            ds.h.values, 0.0, hc=60.0, Cs=ds.Cs_r.values[column], sigma=ds.s_rho.values[column], Vtransform=1
        )

    # at or below the shallowest depth, and with Vtransform 2 whatever hc is, it is fine
    for kwargs in (dict(vtransform=1, hc=20), dict(vtransform=1, hc=10), dict(vtransform=2, hc=60)):
        ok = merged("rutgers", theta_s=7, N=10, **kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xroms.z(ok)
            assert float(xroms.dz(ok).min()) > 0
        assert not hc_warnings(caught), kwargs


def test_the_hc_check_skips_dask_backed_h_and_computes_nothing():
    """Checking dask-backed h would compute it: the check is skipped, and lazy input
    stays lazy (no warning here, by design)."""
    ds = vtransform1()
    with pytest.warns(UserWarning, match="Vtransform 1"):  # in memory: checked
        xroms.z(ds)

    c = chunked(ds)

    def no_compute(*args, **kwargs):
        raise AssertionError("the hc check computed dask data")

    with warnings.catch_warnings(record=True) as caught, dask.config.set(scheduler=no_compute):
        warnings.simplefilter("always")
        z = xroms.z(c)
        dz = xroms.dz(c)
        assert z.chunks is not None and dz.chunks is not None
    assert not hc_warnings(caught)


# ------------------------------------------------------------ depth bands
def test_depth_band_weights_respects_the_positive_attr(rutgers):
    """z with positive='down' gave all-zero weights (and depth_average of them NaN);
    the same interfaces counted either way must give the same weights."""
    ds = rutgers
    for reference, band in (("surface", (0, 10)), ("mean_sea_level", (5, 30))):
        up = xroms.z(ds, scoord="w", reference=reference, positive="up")
        down = xroms.z(ds, scoord="w", reference=reference, positive="down")
        w_up = xroms.depth_band_weights(up, *band)
        w_down = xroms.depth_band_weights(down, *band)
        assert float(w_down.sum("s_rho").max()) > 0, reference
        xr.testing.assert_allclose(w_down, w_up)
    top10 = xroms.depth_band_weights(xroms.z(ds, scoord="w", reference="surface", positive="down"), 0, 10)
    np.testing.assert_allclose(top10.sum("s_rho").values, 10.0)

    # z_w without attrs is taken as positive up, as before
    bare = up.copy()
    bare.attrs = {}
    xr.testing.assert_allclose(xroms.depth_band_weights(bare, 5, 30), xroms.depth_band_weights(up, 5, 30))


def test_depth_band_weights_keywords_say_what_z_w_is(rutgers):
    """Attrs xroms did not write (an h's standard_name, a stray positive) decided the sign silently
    (ocean-skill friction 8): positive=/reference= now win over them, and reading them warns."""
    up = xroms.z(rutgers, scoord="w", reference="surface")
    want = xroms.depth_band_weights(up, 0, 10)
    for stray in ({"standard_name": "sea_floor_depth_below_geoid"}, {"positive": "down"}):
        z_w = up.copy()
        z_w.attrs = stray
        xr.testing.assert_allclose(xroms.depth_band_weights(z_w, 0, 10, positive="up", reference="surface"), want)
        with pytest.warns(UserWarning, match="xroms did not write"):
            assert float(xroms.depth_band_weights(z_w, 0, 10).sum()) == 0.0  # still read as depths
    # the keywords win over xroms' own labels too
    down = xroms.z(rutgers, scoord="w", reference="surface", positive="down")
    xr.testing.assert_allclose(xroms.depth_band_weights(-down, 0, 10, positive="up", reference="surface"), want)
    # xroms' labels, and no attrs at all, are read without a warning
    bare = up.copy()
    bare.attrs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xroms.depth_band_weights(up, 0, 10)
        xroms.depth_band_weights(down, 0, 10)
        xroms.depth_band_weights(bare, 0, 10)


def test_depth_band_weights_rejects_what_it_cannot_interpret(rutgers):
    ds = rutgers
    seabed = xroms.z(ds, scoord="w", reference="bottom")
    with pytest.raises(ValueError, match="seabed"):
        xroms.depth_band_weights(seabed, 0, 10)
    # depth_average used to return NaN for a band below the seabed
    with pytest.raises(ValueError, match="seabed"):
        xroms.depth_average(ds.temp, ds, shallow=0, deep=10, reference="bottom")
    # without a band the thickness is fine against any reference
    assert np.isfinite(xroms.depth_average(ds.temp, ds, reference="bottom").values).all()

    up = xroms.z(ds, scoord="w", reference="surface")
    for attrs, match in (
        ({"vertical_reference": "geoid"}, "geoid"),
        ({"positive": "sideways"}, "sideways"),
        ({"standard_name": "depth", "positive": "up"}, "contradicts"),
    ):
        odd = up.copy()
        odd.attrs = {**up.attrs, **attrs}
        with pytest.raises(ValueError, match=match):
            xroms.depth_band_weights(odd, 0, 10)


# ----------------------------------------------------------- output attrs
Z_ATTRS = {"units", "positive", "vertical_reference", "standard_name", "long_name"}


def test_compute_depth_carries_its_own_attrs_and_not_hs(rutgers):
    """On xarray 2025.8 and later arithmetic keeps attrs, so z inherited everything
    h carried: its long_name ("bathymetry at RHO-points"), field, location, ..."""
    ds = rutgers
    h = ds.h.assign_attrs(field="bath, scalar", location="face")
    for positive in ("up", "down"):
        z = xroms.compute_depth(h, 0, hc=20.0, Cs=ds.Cs_r, sigma=ds.s_rho, positive=positive)
        assert set(z.attrs) == Z_ATTRS
        assert "bathymetry" not in z.attrs["long_name"]
        assert z.attrs["units"] == "m" and z.attrs["positive"] == positive
    assert h.attrs["long_name"] == "bathymetry at RHO-points"  # the input is untouched


@pytest.mark.parametrize("scoord", ["s_rho", "s_w"])
@pytest.mark.parametrize("hcoord", ["rho", "u"])
def test_z_and_dz_carry_only_their_own_attrs(rutgers, scoord, hcoord):
    ds = rutgers.assign(h=rutgers.h.assign_attrs(field="bath, scalar", location="face"))
    z = xroms.z(ds, hcoord=hcoord, scoord=scoord)
    assert set(z.attrs) == Z_ATTRS
    assert z.attrs["long_name"] == f"vertical position at {hcoord}/{scoord} points"
    dz = xroms.dz(ds, hcoord=hcoord, scoord=scoord)
    assert dz.attrs == {"units": "m", "long_name": f"layer thickness at {hcoord}/{scoord} points"}


# ---------------------------------------------------- surface and bottom
def level_of(layout, which, n=6):
    """Index of the selected level and whether the layout labels its vertical dims."""
    return (n - 1 if which == "surface" else 0), layout in ("rutgers", "remora")


@pytest.mark.parametrize("which", ["surface", "bottom"])
def test_surface_and_bottom_leave_the_selected_level_as_a_scalar_coord(layout, which):
    """With s_rho labels the label stays behind (as isel leaves it); UCLA and CROCO
    have none, and nothing showed which level had been selected: now its index does."""
    ds = merged(layout)
    index, labelled = level_of(layout, which)
    one = getattr(xroms, which)(ds.temp)
    assert "s_rho" not in one.dims
    assert one["s_rho"].ndim == 0
    assert one["s_rho"].item() == (ds.s_rho.values[index] if labelled else index)

    # w levels too: the variable on s_w has no labels after moving there
    temp_w = xroms.to_s_w(ds.temp)
    w = getattr(xroms, which)(temp_w)
    assert w["s_w"].ndim == 0 and w["s_w"].item() == (temp_w.sizes["s_w"] - 1 if which == "surface" else 0)

    # the accessor and the input: nothing is written into either
    before = ds.temp.copy(deep=True)
    acc = getattr(ds.xroms, which)("temp")
    assert acc["s_rho"].item() == one["s_rho"].item()
    xr.testing.assert_identical(ds.temp, before)


@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
@pytest.mark.parametrize("which", ["surface", "bottom"])
def test_horizontal_derivative_of_a_selected_level_needs_along_s_in_every_layout(layout, name, which):
    """ddxi(bottom(temp), ds) silently returned the along-s derivative on UCLA and
    CROCO output (with the wrong sign, on the synthetic UCLA grid). It now says what it
    is, like Rutgers and REMORA output always did; along_s=True accepts it."""
    ds = merged(layout)
    one = getattr(xroms, which)(ds.temp)
    dd = getattr(xroms, name)
    with pytest.raises(ValueError, match="along_s"):
        dd(one, ds)

    got = dd(one, ds, along_s=True)
    tdim = xroms.conventions.time_dim(one)
    if name == "ddxi":
        assert got.dims == (tdim, "eta_rho", "xi_u")
        metric = 0.5 * (ds.pm.values[:, :-1] + ds.pm.values[:, 1:])
        want = np.diff(one.values, axis=-1) * metric
    else:
        assert got.dims == (tdim, "eta_v", "xi_rho")
        metric = 0.5 * (ds.pn.values[:-1, :] + ds.pn.values[1:, :])
        want = np.diff(one.values, axis=-2) * metric
    np.testing.assert_allclose(got.values, want, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("which", ["surface", "bottom"])
def test_a_marked_level_still_works_with_everything_else_in_every_layout(layout, which):
    """The scalar coord rides along through grid moves, grid-weighted sums and means,
    and the matching of grid fields; and it selects the level of a grid field that
    has labels for it (Rutgers, REMORA), but is never guessed at when the field has
    none (an index says nothing about which level of a cut field it was)."""
    ds = merged(layout)
    pick = getattr(xroms, which)
    one, u1 = pick(ds.temp), pick(ds.u)
    index, labelled = level_of(layout, which)

    assert xroms.to_u(one).dims[-2:] == ("eta_rho", "xi_u")
    assert xroms.to_rho(u1).dims[-2:] == ("eta_rho", "xi_rho")
    assert xroms.to_rho(u1)["s_rho"].item() == u1["s_rho"].item()
    assert np.isfinite(xroms.gridmean(one, ds, ["X", "Y"]).values).all()
    assert np.isfinite(xroms.gridsum(one, ds, "X").values).all()
    np.testing.assert_allclose(
        xroms.gridsum(one, ds, ["X", "Y"]).values,
        (one.values * xroms.dA(ds).values).sum(axis=(-2, -1)),
        rtol=1e-10,
    )
    # 2-D grid fields ignore the level
    xr.testing.assert_identical(_align.select_like(ds.h, one), xroms.canonicalize(ds.h))

    z = xroms.z(ds)
    if labelled:
        got = _align.select_like(z, one)
        assert "s_rho" not in got.dims
        np.testing.assert_allclose(got.values, z.isel(s_rho=index).values)
    else:
        with pytest.raises(_align.GridMismatchError, match="s_rho"):
            _align.select_like(z, one)


# ------------------------------------------------ legacy xgcm grids
@pytest.fixture(params=["tiny", "from_dataset"])
def xgrid(request, rutgers):
    """An xgcm.Grid, as pre-1.0 code passed: a one-axis grid, and xroms' own."""
    xgcm = pytest.importorskip("xgcm")
    if request.param == "from_dataset":
        return rutgers.xroms.xgcm_grid()
    return xgcm.Grid(
        xr.Dataset(coords={"x": [0, 1]}),
        coords={"X": {"center": "x"}},
        padding="fill",
        autoparse_metadata=False,
    )


GRID_CALLS = {
    "z": lambda g, ds: xroms.z(g),
    "dz": lambda g, ds: xroms.dz(g),
    "dx": lambda g, ds: xroms.dx(g),
    "dy": lambda g, ds: xroms.dy(g),
    "dA": lambda g, ds: xroms.dA(g),
    "dV": lambda g, ds: xroms.dV(g),
    "nominal_resolution": lambda g, ds: xroms.nominal_resolution(g),
    "depth_average": lambda g, ds: xroms.depth_average(ds.temp, g),
}


@pytest.mark.parametrize("name", list(GRID_CALLS))
def test_a_positional_xgcm_grid_is_rejected_with_guidance(rutgers, xgrid, name):
    """These raised "'Grid' object has no attribute 'sizes'"."""
    with pytest.raises(TypeError, match=r"pass the Dataset .* instead of an xgcm Grid") as err:
        GRID_CALLS[name](xgrid, rutgers)
    assert f"xroms.{name}" in str(err.value)


def test_a_positional_xgcm_grid_is_rejected_through_the_accessor(rutgers, xgrid):
    with pytest.raises(TypeError, match="xgcm Grid"):
        rutgers.xroms.z(grid=xgrid)
    with pytest.raises(TypeError, match="xgcm Grid"):
        rutgers.xroms.depth_average(rutgers.temp, grid=xgrid)


def test_a_dataarray_is_not_a_grid_either(rutgers):
    with pytest.raises(TypeError, match="must be an xarray Dataset"):
        xroms.dz(rutgers.h)
    with pytest.raises(TypeError, match="must be an xarray Dataset"):
        xroms.dA(rutgers.h)


# ------------------------------------------------------- _align helpers
def test_level_positions(rutgers):
    can = xroms.canonicalize(rutgers)
    labels = can.indexes["s_rho"]
    t = can.temp
    assert _align.level_positions(t, "s_rho", 6, labels) is None
    for levels, want in (([3, 1], [3, 1]), (slice(2, 5), [2, 3, 4])):
        got = _align.level_positions(t.isel(s_rho=levels), "s_rho", 6, labels)
        np.testing.assert_array_equal(got, want)
    assert _align.level_positions(can.h, "s_rho", 6, labels) is None  # no such dim: nothing to select

    # without labels on both sides: the same count is taken as the same levels, any other raises
    bare = xr.DataArray(np.zeros(3), dims="s_rho", name="temp")
    assert _align.level_positions(bare, "s_rho", 3) is None
    assert _align.level_positions(bare, "s_rho", 3, labels[:3]) is None
    with pytest.raises(_align.GridMismatchError, match="select levels on the Dataset"):
        _align.level_positions(bare, "s_rho", 6)
    labelled = bare.assign_coords(s_rho=[0.1, 0.2, 0.3])
    with pytest.raises(_align.GridMismatchError, match="select levels on the Dataset"):
        _align.level_positions(labelled, "s_rho", 6)  # the grid side has no labels
    with pytest.raises(_align.GridMismatchError, match="lacks"):
        _align.level_positions(labelled, "s_rho", 6, labels)  # labels the grid does not have
    with pytest.raises(_align.GridMismatchError, match="lacks"):
        _align.level_positions(labelled, "s_rho", 3, labels[:3])


def test_check_unstrided():
    _align.check_unstrided(np.arange(5) + 100, "xi_rho")
    _align.check_unstrided(np.array([7]), "xi_rho")
    _align.check_unstrided(np.array([], dtype=int), "xi_rho")
    # labels that are not integers may be longitudes, say: not checked
    _align.check_unstrided(np.array([0.0, 2.0, 4.0]), "xi_rho")
    _align.check_unstrided(np.array([0.5, 1.5, 2.5]), "xi_rho")
    for bad in ([0, 2, 4], [3, 2, 1], [0, 1, 3], [1, 1, 2]):
        with pytest.raises(_align.GridMismatchError, match=r"'temp' has 'xi_rho' index labels .* step by 1"):
            _align.check_unstrided(np.array(bad), "xi_rho", "temp")
    with pytest.raises(_align.GridMismatchError, match="step by 1"):
        _align.check_unstrided(np.array([3, 2, 1], dtype="uint8"), "xi_rho")  # no wrap-around


def test_is_time_varying(rutgers, ucla):
    ds = rutgers
    assert _align.is_time_varying(ds.temp)
    assert _align.is_time_varying(ds.temp.isel(ocean_time=0))  # the scalar time coord
    assert _align.is_time_varying(ds.temp.isel(ocean_time=[1]))
    assert not _align.is_time_varying(ds.h)
    assert not _align.is_time_varying(ds.temp.mean("ocean_time"))
    assert not _align.is_time_varying(ds.temp.isel(ocean_time=0, drop=True))

    out, grid = ucla
    assert _align.is_time_varying(out.temp)  # a time dim without a coordinate
    assert not _align.is_time_varying(grid.h)
    decoded = xroms.decode_time(xr.merge([out, grid.drop_vars("spherical")]))
    assert _align.is_time_varying(decoded.temp.isel(time=0))


@pytest.mark.parametrize("select", ["surface", "bottom"])
def test_depth_of_a_selected_level_is_found_by_its_label(layout, select):
    ds = merged(layout)
    one = getattr(xroms, select)
    if C.convention(ds) == "rutgers":  # Rutgers and REMORA label their s levels
        full = xroms.density(ds.temp, ds.salt, grid=ds).isel(s_rho=-1 if select == "surface" else 0)
        xr.testing.assert_allclose(xroms.density(one(ds.temp), one(ds.salt), grid=ds), full)
    else:
        with pytest.raises(ValueError, match="compute on the 3-D fields and select afterwards"):
            xroms.density(one(ds.temp), one(ds.salt), grid=ds)


def test_explicit_zeta_off_rho_points_is_refused():
    ds = merged("rutgers")
    zeta_u = xroms.to_u(C.canonicalize(ds).zeta)
    with pytest.raises(ValueError, match="rho points"):
        xroms.ddxi(ds.temp, ds, zeta=zeta_u)


@pytest.mark.parametrize("layout", ["rutgers", "ucla"])
def test_zeta_renamed_to_its_cf_name_is_still_the_free_surface(layout):
    # ocean-skill renames zeta to its CF standard name; a variable with that standard_name counts too
    ds = merged(layout)
    for renamed in (
        ds.rename(zeta="sea_surface_height_above_geoid"),
        ds.rename(zeta="ssh").assign(ssh=ds.zeta.assign_attrs(standard_name="sea_surface_height_above_geoid")),
    ):
        xr.testing.assert_identical(xroms.z(renamed, scoord="s_w"), xroms.z(ds, scoord="s_w"))
        xr.testing.assert_identical(renamed.xroms.z_rho, ds.xroms.z_rho)
        xr.testing.assert_identical(xroms.ddxi(renamed.temp, renamed), xroms.ddxi(ds.temp, ds))
        xr.testing.assert_identical(xroms.z(renamed, zeta="mean"), xroms.z(ds, zeta="mean"))
        # also when the output, which has it, is kept apart from the grid
        grid = renamed.drop_vars([name for name in renamed.data_vars if renamed[name].ndim > 2 or name in ("ssh", "sea_surface_height_above_geoid")])
        xr.testing.assert_allclose(C.canonicalize(renamed.xroms.ddz("temp", grid=grid)), C.canonicalize(ds.xroms.ddz("temp")))
    # it is not guessed when two variables could be it
    two = ds.rename(zeta="sea_surface_height_above_geoid").assign(ssh=ds.zeta.assign_attrs(standard_name="sea_surface_height"))
    with pytest.raises(ValueError, match="several variables could be the free surface"):
        xroms.z(two)
    xr.testing.assert_identical(xroms.z(two, zeta=two.ssh), xroms.z(ds))


def test_a_renamed_free_surface_is_found_on_a_cut_dataset():
    """ocean-skill renames zeta to its CF name and then cuts the Dataset to sections, rows and columns, where
    positions can't be read from the dims: z used a flat surface there (ocean-skill friction 6)."""
    ds = merged("ucla", romstools_grid=True)
    ds = ds.assign(zeta=ds.zeta + 0.5)
    ssh = ds.rename(zeta="sea_surface_height_above_geoid")
    path = dict(eta_rho=xr.DataArray([1, 2, 3], dims="along"), xi_rho=xr.DataArray([2, 3, 4], dims="along"))
    for cut in (path, dict(eta_rho=2), dict(eta_rho=2, xi_rho=3)):
        want = xroms.z(ds.isel(cut))
        assert "time" in want.dims
        xr.testing.assert_identical(xroms.z(ssh.isel(cut)), want)
    # a sea surface height that is not on h's points, such as a tide gauge's series, is not the free surface
    gauge = ds.drop_vars("zeta").assign(sea_surface_height=("time", np.zeros(ds.sizes["time"])))
    assert C.free_surface_name(gauge) is None


def test_to_grid_rejects_an_xgcm_grid_where_hcoord_goes():
    ds = merged("rutgers")
    with pytest.raises(TypeError, match="xgcm grid"):
        xroms.to_grid(ds.temp, ds.xroms.xgcm_grid(), "u")


# ----------------------------------------------------- vertical parameters: only what a call needs
def _without_pair(ds, names):
    """``ds`` without the parameters ``names`` of one level pair, nor theta_s/theta_b to compute them from."""
    out = ds.drop_vars(names)
    out.attrs = {k: v for k, v in out.attrs.items() if k not in (*names, "theta_s", "theta_b")}
    return out


def test_depths_on_rho_levels_need_only_the_rho_parameters():
    """h, hc, Cs_r and sigma_r are enough for z at rho levels, as in ocean-skill's Datasets with a separate grid
    file; Cs_w used to be required as well (ocean-skill friction 1)."""
    ds = merged("ucla", romstools_grid=True)
    slim = _without_pair(ds, ["Cs_w", "sigma_w"])
    assert "s_w" not in slim.dims
    xr.testing.assert_identical(xroms.z(slim), xroms.z(ds))
    xr.testing.assert_identical(slim.xroms.z_rho, ds.xroms.z_rho)
    xr.testing.assert_identical(xroms.zslice(slim.temp, [-5.0], slim), xroms.zslice(ds.temp, [-5.0], ds))
    xr.testing.assert_identical(xroms.ddxi(slim.temp, slim), xroms.ddxi(ds.temp, ds))
    xr.testing.assert_identical(xroms.density(slim.temp, slim.salt, grid=slim), xroms.density(ds.temp, ds.salt, grid=ds))
    p = C.vertical_params(slim, levels="s_rho")
    assert p.Cs_w is None and p.sigma_w is None
    xr.testing.assert_identical(p.Cs_r, C.vertical_params(ds).Cs_r)
    # whatever needs the interfaces still asks for their parameters
    for call in (
        lambda: xroms.z(slim, scoord="s_w"),
        lambda: xroms.dz(slim),
        lambda: xroms.depth_average(slim.temp, slim),
        lambda: C.vertical_params(slim),
    ):
        with pytest.raises(ValueError, match="cannot find Cs_w"):
            call()


def test_depths_on_w_levels_need_only_the_w_parameters(rutgers):
    w_only = rutgers.drop_vars([name for name in rutgers.variables if "s_rho" in rutgers[name].dims])
    assert "s_rho" not in w_only.dims and "s_w" in w_only.dims
    xr.testing.assert_identical(xroms.z(w_only, scoord="s_w"), xroms.z(rutgers, scoord="s_w"))
    # one rho level selected with an integer keeps every interface (and, like every variable of
    # that Dataset, the scalar s_rho coord)
    one = xroms.z(rutgers.isel(s_rho=3), scoord="s_w")
    xr.testing.assert_identical(one.drop_vars("s_rho"), xroms.z(rutgers, scoord="s_w"))
    for ds in (w_only, rutgers.isel(s_rho=3)):
        with pytest.raises(ValueError, match=r"no 's_rho' one.*xroms.z\(ds\).isel\(s_rho=3\)"):
            xroms.z(ds)


def test_w_parameters_need_one_level_more_than_the_data(ucla):
    """UCLA output has no s_w dim, so a short Cs_w used to give z on too few interfaces, silently."""
    ds = xroms.merge_grid(*ucla)
    assert "s_w" not in ds.dims
    short = ds.assign_attrs(Cs_w=ds.attrs["Cs_w"][:-1])
    with pytest.raises(ValueError, match="Cs_w has 6 levels but the data have 6 along 's_rho', so 7 interfaces"):
        xroms.z(short, scoord="s_w")
    xr.testing.assert_identical(xroms.z(short), xroms.z(ds))


@pytest.mark.parametrize("cut", ["rho levels alone", "one level", "attrs only, then cut", "2-D", "w levels alone"])
def test_hc_and_vtransform_alone_whatever_the_vertical_cut(cut):
    """levels=False looks up only hc and Vtransform, as ocean-skill's own lookup did (ocean-skill friction 4)."""
    ds = merged("ucla", romstools_grid=True)
    attrs_only = ds.drop_vars(["Cs_r", "Cs_w", "sigma_r", "sigma_w"])
    sub = {
        "rho levels alone": ds.isel(s_rho=slice(1, 4)),
        "one level": ds.isel(s_rho=3),
        "attrs only, then cut": attrs_only.isel(s_rho=slice(1, 4)),
        "2-D": ds.isel(s_rho=-1, s_w=-1),
        "w levels alone": ds.isel(s_w=slice(0, 3)),
    }[cut]
    p = C.vertical_params(sub, levels=False)
    assert (p.hc, p.Vtransform) == (20.0, 2)
    assert (p.Cs_r, p.Cs_w, p.sigma_r, p.sigma_w) == (None, None, None, None)
    with pytest.raises(ValueError, match="s_rho"):  # the level arrays still check the levels of the data
        C.vertical_params(sub)


def test_levels_names_a_pair_or_is_true_or_false(rutgers):
    full = C.vertical_params(rutgers)
    for levels, have in (("rho", "Cs_r"), ("s_rho", "Cs_r"), ("w", "Cs_w"), ("s_w", "Cs_w")):
        p = C.vertical_params(rutgers, levels=levels)
        other = "Cs_w" if have == "Cs_r" else "Cs_r"
        xr.testing.assert_identical(getattr(p, have), getattr(full, have))
        assert getattr(p, other) is None
    for bad in (None, "x", 2, ("s_rho", "s_w")):
        with pytest.raises(ValueError, match="levels must be True"):
            C.vertical_params(rutgers, levels=bad)


def _built_from_the_grid(**kw):
    """Every call that builds depths from the grid, taking hc=, Vtransform= and default_Vtransform=."""
    return {
        "vertical_params": lambda ds: (C.vertical_params(ds, **kw).hc, C.vertical_params(ds, **kw).Vtransform),
        "z": lambda ds: xroms.z(ds, **kw),
        "z_w": lambda ds: xroms.z(ds, scoord="s_w", hcoord="u", **kw),
        "dz": lambda ds: xroms.dz(ds, scoord="s_w", **kw),
        "dV": lambda ds: xroms.dV(ds, **kw),
        "zslice": lambda ds: xroms.zslice(ds.temp, [-5.0], ds, **kw),
        "depth_average": lambda ds: xroms.depth_average(ds.temp, ds, shallow=0, deep=10, reference="surface", **kw),
        "gridsum": lambda ds: xroms.gridsum(ds.temp, ds, "Z", **kw),
        "gridmean": lambda ds: xroms.gridmean(ds.temp, ds, ("Z", "X"), **kw),
        "ds.xroms.z": lambda ds: ds.xroms.z(**kw),
        "ds.xroms.dz": lambda ds: ds.xroms.dz(**kw),
        "ds.xroms.dV": lambda ds: ds.xroms.dV(**kw),
        "ds.xroms.assign_z": lambda ds: ds.xroms.assign_z(**kw).z_w,
        "ds.xroms.zslice": lambda ds: ds.xroms.zslice("temp", [-5.0], **kw),
        "ds.xroms.depth_average": lambda ds: ds.xroms.depth_average("temp", **kw),
        "ds.xroms.gridsum": lambda ds: ds.xroms.gridsum("temp", "Z", **kw),
    }


@pytest.mark.parametrize("name", list(_built_from_the_grid()))
def test_hc_keyword_wins_over_the_dataset(rutgers, name):
    """A catalog's hc for a file without one (or with another), as Vtransform= already did (ocean-skill friction 2)."""
    want = _built_from_the_grid()[name](rutgers.assign(hc=5.0))
    for ds in (rutgers.drop_vars("hc"), rutgers):  # no hc, and an hc of 20 m the keyword overrides
        got = _built_from_the_grid(hc=5.0)[name](ds)
        assert got == want if name == "vertical_params" else xr.testing.assert_identical(got, want) is None
    if name == "z":
        with pytest.raises(ValueError, match="pass hc="):
            xroms.z(rutgers.drop_vars("hc"))


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "x", [1.0, 2.0]])
def test_hc_keyword_must_be_a_depth(rutgers, bad):
    with pytest.raises(ValueError, match="hc must be a finite number of metres"):
        xroms.z(rutgers, hc=bad)


@pytest.mark.parametrize("name", list(_built_from_the_grid()))
def test_vtransform_keywords_everywhere_depths_are_built(name):
    raw = syn.make_dataset("rutgers").drop_vars("Vtransform")  # states no Vtransform, nor implies one
    want = _built_from_the_grid()[name](raw.assign(Vtransform=1))
    for kw in (dict(Vtransform=1), dict(default_Vtransform=1)):
        got = _built_from_the_grid(**kw)[name](raw)
        assert got == want if name == "vertical_params" else xr.testing.assert_identical(got, want) is None


def test_default_vtransform_applies_only_when_the_dataset_states_none():
    """ocean-skill's rule "the file's Vtransform, else 2" used to need a pre-check of where xroms looks (friction 5)."""
    raw = syn.make_dataset("rutgers").drop_vars("Vtransform")
    says_1 = syn.make_dataset("rutgers", vtransform=1)
    vt = lambda ds, **kw: C.vertical_params(ds, **kw).Vtransform
    assert (vt(raw, default_Vtransform=2), vt(raw, default_Vtransform=1)) == (2, 1)
    assert vt(says_1, default_Vtransform=2) == 1
    assert vt(says_1, Vtransform=2, default_Vtransform=1) == 2  # the argument still beats everything
    # what the layout says counts as stated: CROCO's VertCoordType, UCLA output and REMORA
    assert vt(raw.assign_attrs(VertCoordType="OLD"), default_Vtransform=2) == 1
    assert vt(merged("ucla"), default_Vtransform=1) == 2
    assert vt(merged("remora"), default_Vtransform=1) == 2
    # a VertCoordType that can't be read is not "nothing stated"
    with pytest.raises(ValueError, match="VertCoordType is 'WEIRD'"):
        vt(raw.assign_attrs(VertCoordType="WEIRD"), default_Vtransform=2)
    # a bad default raises even where the Dataset states one
    with pytest.raises(ValueError, match="default_Vtransform must be 1 or 2"):
        vt(says_1, default_Vtransform=3)
    with pytest.raises(ValueError, match="default_Vtransform= .used only when the Dataset states none"):
        vt(raw)
