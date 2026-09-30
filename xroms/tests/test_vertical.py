"""Vertical coordinates and grid metrics: what reviewing the rewrite found.

Sections: depth warnings for Vtransform 1 (``compute_depth``), the sign convention
of depth bands, the attrs of depth outputs, the level marked by ``surface`` and
``bottom``, xgcm grids passed where a Dataset belongs, and the ``_align`` helpers
the vertical and metric functions rely on. The behaviours that concern data cut or
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
