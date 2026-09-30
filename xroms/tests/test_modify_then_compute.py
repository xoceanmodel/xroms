"""What the new xroms API must do when users modify data before computing.

Each test encodes a failure mode found while planning the overhaul, all of them
situations in which the old code gave a wrong answer or a confusing error after
the user changed the data first: cutting a variable to one time re-broadcast the
result over every time of the grid; a horizontal subset gave "conflicting sizes"
errors or mostly NaN; dropping the attached z coordinates silently removed the
s-coordinate slope term; the accessor held stale state after in-place edits and
wrote a ``speed`` variable into the user's Dataset; output without its grid
variables was not diagnosed; UCLA output has no usable time axis; horizontally
chunked input crashed xgcm (issues #16 and #77); and the Rutgers alias dimensions
(``eta_u``, ``xi_v``, ...) silently turned sums into outer products.

They are written against the NEW API (pure functions taking a ``grid`` Dataset,
plus the stateless ``.xroms`` accessor) on the synthetic datasets of
``_synthetic.py``. Those fields are analytic, so most expected values are exact
rather than regression numbers. Sections, in order: 1 single time, 2 horizontal
subsets and labels, 3 subset validation, 4 halo round trip, 5 single s-level,
6 dropped coordinates and analytic derivatives, 7 in-place edits, 8 output without
grid variables, 9 time-reduced variables, 10 UCLA end to end, 11 chunked input,
12 Rutgers naming, 13 no setup step. Sections 14-17 came from reviewing the
rewrite: 14 explicit zeta and z, 15 grids without a free surface, 16 variables
cut vertically, 17 strided subsets.

The tests were written before the new API existed, each marked
``xfail(strict=True)``, and the markers came off as the features landed; they
now guard against the old failure modes coming back. No new name is used at
module level (names are looked up inside the tests).
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xroms

from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


# Same computation reached by a different route: equal up to rounding only.
TIGHT = {"rtol": 1e-10, "atol": 1e-14}

# Rho-index window used throughout: interior to the 12 (xi) x 9 (eta) rho grid.
# A u (v) point with index i lies between rho points i and i + 1, so the window
# holds xi_u points XU and eta_v points YV.
X, Y = slice(2, 9), slice(1, 7)
XU, YV = slice(X.start, X.stop - 1), slice(Y.start, Y.stop - 1)

# temp = TEMP_A * x + TEMP_B * z + TEMP_0 is linear in x and z, so the horizontal
# derivative at constant depth is TEMP_A in xi and 0 in eta, wherever it is taken.
ANALYTIC = {
    "ddxi": {"desired": syn.TEMP_A, "rtol": 1e-9},
    "ddeta": {"desired": 0.0, "rtol": 0, "atol": 1e-12},
}

# Alias dimensions that only the Rutgers family uses.
RUTGERS_ALIASES = {"eta_u", "xi_v", "eta_psi", "xi_psi"}


# ---------------------------------------------------------------------- helpers
def assert_same(actual, expected, **tol):
    """Same dims in the same order, numerically equal values.

    Coordinates are deliberately not compared here: which labels a result carries
    is tested where it matters, and is not what these comparisons are about.
    """
    assert actual.dims == expected.dims, (
        f"dims differ: {actual.dims} != {expected.dims}"
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), **(tol or TIGHT)
    )


def window(sl, n):
    """(start, stop) of a rho-index slice on an axis with ``n`` rho points."""
    return (0, n) if sl is None else sl.indices(n)[:2]


def ucla_dataset(pair):
    """UCLA output with its separate grid file merged in."""
    out, grid = pair
    return xr.merge([out, grid.drop_vars("spherical", errors="ignore")])


def tapped(ds):
    """``ds`` with its 3-D and 4-D dask fields instrumented; returns ``(ds, loads)``.

    ``loads`` gains an item every time a block of a model field is evaluated, so it
    stays empty for as long as nothing has computed model data. Small things (the
    stretching curves, 2-D grid fields, scalars) are left alone: reading those early
    is harmless, loading the model fields is what makes a result not lazy.
    """
    dask_array = pytest.importorskip("dask.array")
    loads = []

    def tap(block):
        loads.append(1)
        return block

    for name, var in ds.variables.items():
        if var.ndim >= 3 and var.chunks is not None and name not in ds.indexes:
            data = dask_array.map_blocks(
                tap, var.data, dtype=var.dtype, meta=var.data._meta
            )
            new = xr.Variable(var.dims, data, var.attrs)
            assign = ds.assign_coords if name in ds.coords else ds.assign
            ds = assign({name: new})
    return ds, loads


@pytest.fixture
def labelled(rutgers):
    """Rutgers dataset with an index coordinate on every horizontal dimension.

    Labels are offset (xi from 100, eta from 200) so they cannot be mistaken for
    positions: renumbering or dropping them is detectable after any slicing. Alias
    dimensions carry the same labels as the canonical dimension they alias.
    """
    labels = {}
    for dim, n in rutgers.sizes.items():
        if dim.startswith("xi_"):
            labels[dim] = 100 + np.arange(n)
        elif dim.startswith("eta_"):
            labels[dim] = 200 + np.arange(n)
    return rutgers.assign_coords(labels)


@pytest.fixture(params=["ucla", "ucla_romstools"])
def ucla_pair(request):
    """UCLA ``(output, grid)`` with either kind of grid file."""
    return request.getfixturevalue(request.param)


# ------------------------------------------------------------ 1. single time
@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
@pytest.mark.parametrize("t", [0, 1])
def test_single_time_variable_is_not_rebroadcast(rutgers, name, t):
    """A variable cut to one time must not come back broadcast over all times.

    The full grid's zeta is matched to the variable's scalar ``ocean_time``
    coordinate, so the result is the full-domain result at that time.
    """
    ds = rutgers
    dd = getattr(xroms, name)
    got = dd(ds.temp.isel(ocean_time=t), ds)
    assert "ocean_time" not in got.dims
    assert_same(got, dd(ds.temp, ds).isel(ocean_time=t))


@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
@pytest.mark.parametrize("t", [0, 1])
def test_single_time_uses_that_times_free_surface(rutgers, name, t):
    """temp was built from each time's own depths, so the derivative at constant
    depth is exact only if the grid's zeta for *that* time is the one used."""
    ds = rutgers
    got = getattr(xroms, name)(ds.temp.isel(ocean_time=t), ds)
    np.testing.assert_allclose(got.values, **ANALYTIC[name])


@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
def test_dataset_cut_to_one_time_first(rutgers, name):
    """Cutting the Dataset itself to one time, then computing, gives the same answer
    as computing on everything and then cutting, for the functions and the accessor."""
    ds = rutgers
    dd = getattr(xroms, name)
    sub = ds.isel(ocean_time=0)
    want = dd(ds.temp, ds).isel(ocean_time=0)

    got = dd(sub.temp, sub)
    assert "ocean_time" not in got.dims
    assert_same(got, want)

    acc = getattr(sub.xroms, name)("temp")
    assert "ocean_time" not in acc.dims
    np.testing.assert_allclose(acc.values, want.values, **TIGHT)

    # the accessor also takes a DataArray that was cut to one time, with ds as grid
    acc = getattr(ds.xroms, name)(ds.temp.isel(ocean_time=0))
    assert "ocean_time" not in acc.dims
    np.testing.assert_allclose(acc.values, want.values, **TIGHT)


# ------------------------------------------------ 2. horizontal subsets, labels
@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
def test_horizontal_subset_then_derivative_matches_full_domain(rutgers, name):
    """After xroms.subset the derivative used to raise "conflicting sizes" or come
    back mostly NaN. It must equal the full-domain result at the same points."""
    ds = rutgers
    dd = getattr(xroms, name)
    sub = xroms.subset(ds, X=X, Y=Y)
    full = dd(ds.temp, ds)
    if name == "ddxi":
        want = full.isel(eta_rho=Y, xi_u=XU)
    else:
        want = full.isel(eta_v=YV, xi_rho=X)

    got = dd(sub.temp, sub)
    assert not np.isnan(got.values).any()
    assert_same(got, want)

    acc = getattr(sub.xroms, name)("temp")
    assert not np.isnan(acc.values).any()
    np.testing.assert_allclose(acc.values, want.values, **TIGHT)


def test_subset_keeps_original_index_labels(labelled):
    """Labels are never renumbered: subset, pure function and accessor all keep the
    labels the points had in the full dataset."""
    ds = labelled
    sub = xroms.subset(ds, X=X, Y=Y)

    # window of rho indices per dim; u- and v-like dims hold one point fewer
    windows = {
        "xi_rho": (2, 9),
        "xi_v": (2, 9),
        "xi_u": (2, 8),
        "xi_psi": (2, 8),
        "eta_rho": (1, 7),
        "eta_u": (1, 7),
        "eta_v": (1, 6),
        "eta_psi": (1, 6),
    }
    for dim, (a, b) in windows.items():
        np.testing.assert_array_equal(sub[dim].values, ds[dim].values[a:b], dim)

    pure = xroms.ddxi(sub.temp, sub)
    assert pure.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
    np.testing.assert_array_equal(pure["xi_u"].values, ds["xi_u"].values[2:8])
    np.testing.assert_array_equal(pure["eta_rho"].values, ds["eta_rho"].values[1:7])

    acc = sub.xroms.ddxi("temp")
    assert acc.dims == ("ocean_time", "s_rho", "eta_u", "xi_u")
    np.testing.assert_array_equal(acc["xi_u"].values, ds["xi_u"].values[2:8])
    np.testing.assert_array_equal(acc["eta_u"].values, ds["eta_u"].values[1:7])


def test_horizontal_subset_then_speed_matches_full_domain_inside(rutgers):
    """Speed needs u and v at rho points, so the outermost ring of a subset has
    no neighbours to average; everything strictly inside must match the full domain."""
    ds = rutgers
    sub = xroms.subset(ds, X=X, Y=Y)
    full = xroms.speed(ds.u, ds.v)
    # rho (j, i) averages u[i-1], u[i] and v[j-1], v[j]: rho 3..7 in xi, 2..5 in eta
    want = full.isel(
        eta_rho=slice(Y.start + 1, Y.stop - 1), xi_rho=slice(X.start + 1, X.stop - 1)
    )
    inside = {"eta_rho": slice(1, -1), "xi_rho": slice(1, -1)}
    for got in (xroms.speed(sub.u, sub.v), sub.xroms.speed):
        assert got.dims == want.dims
        np.testing.assert_allclose(got.isel(inside).values, want.values, **TIGHT)


# ------------------------------------------------------ 3. subset validation
@pytest.mark.parametrize(
    "xsl, ysl",
    [
        (slice(2, None), None),  # raised TypeError: None - 1
        (slice(None, 9), None),
        (slice(None, None), None),
        (slice(2, 9, 1), None),
        (None, slice(3, None)),
        (None, slice(None, 6)),
        (slice(2, None), slice(None, 6)),
    ],
)
def test_subset_accepts_open_ended_slices(rutgers, xsl, ysl):
    """None bounds mean "to the edge"; every stagger of every dim follows the rho
    window, and the data in the window is untouched."""
    ds = rutgers
    sub = xroms.subset(ds, X=xsl, Y=ysl)
    x0, x1 = window(xsl, ds.sizes["xi_rho"])
    y0, y1 = window(ysl, ds.sizes["eta_rho"])

    nx, ny = x1 - x0, y1 - y0
    sizes = {
        "xi_rho": nx,
        "xi_v": nx,
        "xi_u": nx - 1,
        "xi_psi": nx - 1,
        "eta_rho": ny,
        "eta_u": ny,
        "eta_v": ny - 1,
        "eta_psi": ny - 1,
    }
    assert {d: sub.sizes[d] for d in sizes} == sizes

    np.testing.assert_array_equal(sub.temp.values, ds.temp.values[..., y0:y1, x0:x1])
    np.testing.assert_array_equal(sub.u.values, ds.u.values[..., y0:y1, x0 : x1 - 1])
    np.testing.assert_array_equal(sub.v.values, ds.v.values[..., y0 : y1 - 1, x0:x1])
    np.testing.assert_array_equal(
        sub.mask_psi.values, ds.mask_psi.values[y0 : y1 - 1, x0 : x1 - 1]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"X": slice(2, 9, 2)},
        {"Y": slice(1, 7, 2)},
        {"X": slice(9, 2, -1)},
    ],
)
def test_subset_rejects_steps_other_than_one(rutgers, kwargs):
    """A stride would leave the staggered grids describing different cells."""
    with pytest.raises(ValueError):
        xroms.subset(rutgers, **kwargs)


def test_subset_negative_stop_is_consistent_or_rejected(rutgers):
    """The old code applied ``stop - 1`` blindly to the u grid, so a negative stop
    left rho and u describing different cells. Normalizing it like Python does, or
    rejecting it, are both fine; a silent mismatch is not."""
    ds = rutgers
    try:
        sub = xroms.subset(ds, X=slice(2, -1))
    except ValueError:
        return  # rejecting a negative stop is an acceptable design
    x0, x1 = window(slice(2, -1), ds.sizes["xi_rho"])
    assert sub.sizes["xi_rho"] == x1 - x0
    assert sub.sizes["xi_u"] == x1 - x0 - 1
    np.testing.assert_array_equal(sub.u.values, ds.u.values[..., x0 : x1 - 1])


# ------------------------------------------------------------- 4. halo
@pytest.mark.parametrize("name", ["u", "v"])
def test_halo_round_trip_to_rho(rutgers, name):
    """Averaging u (or v) to rho points needs a neighbour beyond the window, so the
    outermost ring of a subset is wrong. Subset with a halo, compute, then trim the
    halo: the kept region must be identical to the full-domain result."""
    ds = rutgers
    full = xroms.to_rho(ds[name])
    want = full.isel(eta_rho=Y, xi_rho=X)

    padded = xroms.subset(ds, X=X, Y=Y, halo=1)
    got = xroms.trim(xroms.to_rho(padded[name]), 1)
    assert_same(got, want)

    # without the halo the outer ring differs (what the halo is for), the inside not
    bare = xroms.to_rho(xroms.subset(ds, X=X, Y=Y)[name])
    inside = {"eta_rho": slice(1, -1), "xi_rho": slice(1, -1)}
    np.testing.assert_allclose(bare.isel(inside).values, want.isel(inside).values)
    assert not np.allclose(bare.values, want.values)


def test_halo_then_trim_on_a_dataset_is_the_plain_subset(labelled):
    """trim undoes exactly what halo added, on every stagger and on the labels."""
    ds = labelled
    padded = xroms.subset(ds, X=X, Y=Y, halo=1)
    assert padded.sizes["xi_rho"] == X.stop - X.start + 2
    assert padded.sizes["eta_rho"] == Y.stop - Y.start + 2
    xr.testing.assert_equal(xroms.trim(padded, 1), xroms.subset(ds, X=X, Y=Y))


# ---------------------------------------------------------- 5. single s-level
@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
def test_single_s_level_needs_along_s(rutgers, name):
    """A horizontal derivative at constant depth needs the neighbouring levels."""
    surface = rutgers.temp.isel(s_rho=-1)
    with pytest.raises(ValueError, match="along_s"):
        getattr(xroms, name)(surface, rutgers)


@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
def test_single_s_level_along_s_is_two_dimensional_in_space(rutgers, name):
    """With along_s=True the derivative is taken along the s-surface: the plain
    difference over the metric, two-dimensional in space."""
    ds = rutgers
    surface = ds.temp.isel(s_rho=-1)
    got = getattr(xroms, name)(surface, ds, along_s=True)
    if name == "ddxi":
        dims = ("ocean_time", "eta_rho", "xi_u")
        metric = 0.5 * (ds.pm.values[:, :-1] + ds.pm.values[:, 1:])
        want = np.diff(surface.values, axis=-1) * metric
    else:
        dims = ("ocean_time", "eta_v", "xi_rho")
        metric = 0.5 * (ds.pn.values[:-1, :] + ds.pn.values[1:, :])
        want = np.diff(surface.values, axis=-2) * metric
    assert got.dims == dims
    np.testing.assert_allclose(got.values, want, **TIGHT)


# ----------------------------- 6. dropped coordinates, analytic derivatives
@pytest.mark.parametrize("name", ["ddxi", "ddeta", "ddz"])
def test_dropped_coordinates_do_not_change_the_physics(rutgers, name):
    """Dropping coordinates used to drop the attached z, and the s-coordinate slope
    term with it. Depths come from the grid's primitives, so nothing changes."""
    ds = rutgers
    dd = getattr(xroms, name)
    want = dd(ds.temp, ds)
    assert_same(dd(ds.temp.reset_coords(drop=True), ds), want)

    # the old setup step attached z as a coordinate; attaching or dropping it again
    # makes no difference (the synthetic temp carries none of its own)
    with_z = ds.temp.assign_coords(z_rho=xroms.z(ds))
    assert_same(dd(with_z, ds), want)
    assert_same(dd(with_z.reset_coords(drop=True), ds), want)


@pytest.mark.parametrize("name", ["ddxi", "ddeta", "ddz"])
def test_attached_z_coordinate_is_ignored(rutgers, name):
    """Not even a wrong z coordinate attached to the variable is consulted."""
    ds = rutgers
    dd = getattr(xroms, name)
    wrong = (ds.temp.dims, np.full(ds.temp.shape, -5.0))
    assert_same(dd(ds.temp.assign_coords(z_rho=wrong), ds), dd(ds.temp, ds))


def test_a_variable_made_from_numpy_needs_no_attached_z(rutgers):
    """A variable a user builds from bare arrays carries no coordinates at all."""
    ds = rutgers
    bare = xr.DataArray(3.0 * ds.temp.values + 1.0, dims=ds.temp.dims)
    got = xroms.ddxi(bare, ds)
    np.testing.assert_allclose(got.values, 3.0 * syn.TEMP_A, rtol=1e-9)


def test_analytic_xi_derivative_holds_on_every_layer(rutgers):
    """(probe: boundary layers) d/dxi of temp is TEMP_A everywhere, top and bottom
    layers included; the old code zeroed the boundary layers."""
    got = xroms.ddxi(rutgers.temp, rutgers)
    assert got.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
    for k in range(got.sizes["s_rho"]):
        np.testing.assert_allclose(
            got.isel(s_rho=k).values, syn.TEMP_A, rtol=1e-9, err_msg=f"s_rho {k}"
        )


def test_analytic_eta_derivative_vanishes_on_every_layer(rutgers):
    """temp does not depend on y at constant depth, but h slopes in y, so the
    eta derivative only vanishes if the s-coordinate slope term is right."""
    got = xroms.ddeta(rutgers.temp, rutgers)
    assert got.dims == ("ocean_time", "s_rho", "eta_v", "xi_rho")
    for k in range(got.sizes["s_rho"]):
        np.testing.assert_allclose(
            got.isel(s_rho=k).values, err_msg=f"s_rho {k}", **ANALYTIC["ddeta"]
        )


def test_analytic_vertical_derivative_on_every_w_level(rutgers):
    """ddz of an s_rho variable lands on s_w, and is TEMP_B on every level,
    boundaries included (the old code gave 0 at the bottom and top)."""
    ds = rutgers
    got = xroms.ddz(ds.temp, ds)
    assert got.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
    assert got.sizes["s_w"] == ds.sizes["s_w"]
    for k in range(got.sizes["s_w"]):
        np.testing.assert_allclose(
            got.isel(s_w=k).values, syn.TEMP_B, rtol=1e-9, err_msg=f"s_w {k}"
        )


def test_z_is_computed_from_the_grid_primitives(rutgers):
    """z comes from h, zeta and the s-parameters: nothing has to be attached."""
    ds = rutgers
    h, zeta, hc = ds.h.values[None, None], ds.zeta.values[:, None], float(ds.hc)
    for scoord, cs, s in [("s_rho", ds.Cs_r, ds.s_rho), ("s_w", ds.Cs_w, ds.s_w)]:
        want = syn.depths(
            h,
            zeta,
            hc,
            cs.values[None, :, None, None],
            s.values[None, :, None, None],
            int(ds.Vtransform),
        )
        got = xroms.z(ds, scoord=scoord)
        assert got.dims == ("ocean_time", scoord, "eta_rho", "xi_rho")
        np.testing.assert_allclose(got.values, want, rtol=1e-12)


def test_z_honours_the_zeta_argument(rutgers):
    """The top w-level is the free surface and the bottom one is -h, whichever free
    surface is used: None (the grid's), 0, "mean", or an explicit DataArray."""
    ds = rutgers

    def w(**kw):
        return xroms.z(ds, scoord="s_w", **kw)

    assert_same(w().isel(s_w=-1), ds.zeta)
    top_flat = w(zeta=0).isel(s_w=-1)
    assert "ocean_time" not in top_flat.dims
    np.testing.assert_allclose(top_flat.values, 0.0, atol=1e-12)
    assert_same(w(zeta="mean").isel(s_w=-1), ds.zeta.mean("ocean_time"))
    assert_same(w(zeta=ds.zeta + 1.0).isel(s_w=-1), ds.zeta + 1.0)
    for zeta in (None, 0, "mean", ds.zeta + 1.0):
        bottom = w(zeta=zeta).isel(s_w=0)
        np.testing.assert_allclose(
            bottom.values, np.broadcast_to(-ds.h.values, bottom.shape), rtol=1e-12
        )


# ------------------------------------------------------- 7. in-place edits
def test_in_place_edit_of_u_and_v_is_seen_by_speed(rutgers):
    """The accessor kept stale state: speed did not change after u and v were
    edited in place, and a ``speed`` variable appeared in the user's Dataset."""
    ds = rutgers
    before = ds.xroms.speed.copy(deep=True)  # the first use creates the accessor
    ds["u"] = ds.u * 10
    ds["v"] = ds.v * 10
    after = ds.xroms.speed
    np.testing.assert_allclose(after.values, 10 * before.values, rtol=1e-12)
    assert "speed" not in ds


def test_in_place_edit_of_a_data_variable_is_seen_by_ddxi(rutgers):
    ds = rutgers
    before = ds.xroms.ddxi("temp").copy(deep=True)
    ds["temp"] = ds.temp * 2
    after = ds.xroms.ddxi("temp")
    np.testing.assert_allclose(after.values, 2 * before.values, rtol=1e-12)


def test_in_place_edit_of_a_grid_variable_is_seen_by_z(rutgers):
    """Grid variables are not special: editing zeta in place changes z_rho."""
    ds = rutgers
    before = ds.xroms.z_rho.copy(deep=True)
    ds["zeta"] = ds.zeta * 0.0
    after = ds.xroms.z_rho
    assert not np.allclose(after.values, before.values)  # the edit mattered
    flat = xroms.z(ds, zeta=0).broadcast_like(after).transpose(*after.dims)
    np.testing.assert_allclose(after.values, flat.values, rtol=1e-12)


def test_accessor_never_writes_into_the_dataset(rutgers):
    ds = rutgers
    pristine = ds.copy(deep=True)
    ds.xroms.speed
    ds.xroms.z_rho
    ds.xroms.ddxi("temp")
    ds.xroms.to_grid("temp", hcoord="u")
    assert "speed" not in ds
    xr.testing.assert_identical(ds, pristine)


# ---------------------------------------- 8. output without grid variables
@pytest.mark.parametrize("how", ["function", "accessor"])
def test_output_without_grid_variables_raises_naming_them(ucla, how):
    """An output file lacks pm, h, ...: say which ones are missing, right away."""
    out, _ = ucla
    with pytest.raises((KeyError, ValueError)) as err:
        if how == "function":
            xroms.ddxi(out.temp, out)
        else:
            out.xroms.ddxi("temp")
    message = str(err.value).replace("\\", "")  # KeyError repr-escapes quotes
    for missing in ("pm", "h"):
        assert re.search(rf"\b{missing}\b", message), message


def test_output_with_a_separate_grid_works(ucla_pair):
    """The grid does not have to live in the data's Dataset: pass it as ``grid=``."""
    out, grid = ucla_pair
    full = ucla_dataset((out, grid))

    got = xroms.ddxi(out.temp, grid=full)
    assert got.dims == ("time", "s_rho", "eta_rho", "xi_u")
    assert np.isfinite(got.values).all()
    assert_same(got, xroms.ddxi(full.temp, full))

    acc = out.xroms.ddxi("temp", grid=full)
    np.testing.assert_allclose(acc.values, got.values, **TIGHT)


# ------------------------------------------------ 9. time-reduced variables
@pytest.mark.parametrize("name", ["ddxi", "ddeta"])
def test_time_reduced_variable_needs_an_explicit_free_surface(rutgers, name):
    """A time mean has no time to pick zeta at, and guessing would silently give
    wrong depths: the error says how to choose."""
    ds = rutgers
    tm = ds.temp.mean("ocean_time")
    with pytest.raises((ValueError, KeyError)) as err:
        getattr(xroms, name)(tm, ds)
    message = str(err.value).replace("\\", "")
    assert "zeta=0" in message
    assert re.search(r"""zeta=["']mean["']""", message), message
    assert re.search(r"\bz=", message), message


def test_time_reduced_variable_with_mean_free_surface(rutgers):
    """Depth is linear in zeta, so the mean of temp is exactly linear in the depth
    built from the mean zeta: d/dxi at constant depth is TEMP_A again."""
    ds = rutgers
    tm = ds.temp.mean("ocean_time")
    got = xroms.ddxi(tm, ds, zeta="mean")
    assert got.dims == ("s_rho", "eta_rho", "xi_u")
    np.testing.assert_allclose(got.values, syn.TEMP_A, rtol=1e-9)
    # the same choice spelled as a DataArray, and as explicit depths
    assert_same(xroms.ddxi(tm, ds, zeta=ds.zeta.mean("ocean_time")), got)
    assert_same(xroms.ddxi(tm, ds, z=xroms.z(ds, zeta="mean")), got)


def test_time_reduced_variable_with_flat_free_surface(rutgers):
    """zeta=0 means depths at rest: the same as a grid whose zeta is zero."""
    ds = rutgers
    tm = ds.temp.mean("ocean_time")
    got = xroms.ddxi(tm, ds, zeta=0)
    assert got.dims == ("s_rho", "eta_rho", "xi_u")
    assert np.isfinite(got.values).all()

    at_rest = ds.isel(ocean_time=0, drop=True)
    at_rest = at_rest.assign(zeta=xr.zeros_like(at_rest.zeta))
    assert_same(got, xroms.ddxi(tm, at_rest))


# ---------------------------------------------------- 10. UCLA end to end
def test_decode_time_gives_a_datetime_index(ucla_pair):
    """UCLA output has a ``time`` dim with no coordinate and seconds in a data
    variable; decode_time makes ``time`` a real datetime index."""
    raw = ucla_dataset(ucla_pair)
    assert "time" not in raw.indexes  # what the raw output looks like

    ds = xroms.decode_time(raw)
    assert isinstance(ds.indexes["time"], pd.DatetimeIndex)
    epoch = np.datetime64(syn.EPOCH, "D")
    want = epoch + np.arange(ds.sizes["time"]) * np.timedelta64(1, "D")
    np.testing.assert_array_equal(ds["time"].values.astype("datetime64[D]"), want)
    assert "time" not in raw.indexes  # the input is not modified


def test_ucla_single_time_derivative_end_to_end(ucla_pair):
    """Decoded time labels are what lets a single-time variable find its zeta."""
    ds = xroms.decode_time(ucla_dataset(ucla_pair))
    full = xroms.ddxi(ds.temp, ds)
    for t in (0, 1):
        one = ds.temp.isel(time=t)
        assert one.time.ndim == 0
        assert np.issubdtype(one.time.dtype, np.datetime64)
        got = xroms.ddxi(one, ds)
        assert got.dims == ("s_rho", "eta_rho", "xi_u")
        assert_same(got, full.isel(time=t))


def test_ucla_matches_rutgers_for_the_same_model_state(ucla, rutgers):
    """Same ocean, two file conventions, one answer: nothing about the answer may
    depend on where the s-parameters or the time axis were stored."""
    ds = xroms.decode_time(ucla_dataset(ucla))
    got = xroms.ddxi(ds.temp.isel(time=1), ds)
    want = xroms.ddxi(rutgers.temp.isel(ocean_time=1), rutgers)
    assert_same(got, want)


# --------------------------------------------------------- 11. chunked input
CHUNK_CASES = {
    # name: (function of a dataset, variable it reads, result dim -> input dim of
    # the dims the operation does not touch)
    "ddxi": (
        lambda d: xroms.ddxi(d.temp, d),
        "temp",
        {"ocean_time": "ocean_time", "s_rho": "s_rho", "eta_rho": "eta_rho"},
    ),
    "ddeta": (
        lambda d: xroms.ddeta(d.temp, d),
        "temp",
        {"ocean_time": "ocean_time", "s_rho": "s_rho", "xi_rho": "xi_rho"},
    ),
    "to_rho(u)": (
        lambda d: xroms.to_rho(d.u),
        "u",
        {"ocean_time": "ocean_time", "s_rho": "s_rho", "eta_rho": "eta_u"},
    ),
    "to_rho(v)": (
        lambda d: xroms.to_rho(d.v),
        "v",
        {"ocean_time": "ocean_time", "s_rho": "s_rho", "xi_rho": "xi_v"},
    ),
}


@pytest.mark.parametrize("case", list(CHUNK_CASES))
def test_chunked_input_stays_lazy_and_matches_numpy(rutgers, case):
    """(#16/#77) Horizontally chunked input crashed xgcm. Results must stay lazy,
    equal the numpy results, and keep the chunking of the dims not operated on."""
    call, varname, untouched = CHUNK_CASES[case]
    c, loads = tapped(chunked(rutgers))
    want = call(rutgers)

    got = call(c)
    assert got.chunks is not None
    assert not loads, "model fields were evaluated while building the result"
    for out_dim, in_dim in untouched.items():
        n_out = len(got.chunksizes[out_dim])
        n_in = len(c[varname].chunksizes[in_dim])
        assert n_out == n_in, f"{out_dim}: {n_in} chunks became {n_out}"
    xr.testing.assert_allclose(got.compute(), want, **TIGHT)


# ------------------------------------------------------ 12. Rutgers naming
@pytest.mark.parametrize(
    "func, dims",
    [
        ("to_u", ("ocean_time", "s_rho", "eta_rho", "xi_u")),
        ("to_v", ("ocean_time", "s_rho", "eta_v", "xi_rho")),
        ("to_psi", ("ocean_time", "s_rho", "eta_v", "xi_u")),
    ],
)
def test_pure_functions_return_canonical_names(rutgers, func, dims):
    """Whatever the input's naming, pure functions answer in canonical names."""
    assert getattr(xroms, func)(rutgers.temp).dims == dims


@pytest.mark.parametrize(
    "hcoord, partner, dims",
    [
        ("u", "u", ("ocean_time", "s_rho", "eta_u", "xi_u")),
        ("v", "v", ("ocean_time", "s_rho", "eta_v", "xi_v")),
        ("psi", "mask_psi", ("ocean_time", "s_rho", "eta_psi", "xi_psi")),
    ],
)
def test_accessor_returns_the_datasets_own_names(rutgers, hcoord, partner, dims):
    """The accessor answers in the Dataset's own naming, so the result combines
    with the Dataset's own variables without growing extra dimensions."""
    ds = rutgers
    got = ds.xroms.to_grid("temp", hcoord=hcoord)
    assert got.dims == dims
    combined = ds[partner] + got
    assert set(combined.dims) == set(dims)
    assert combined.ndim == len(dims)


@pytest.mark.parametrize("hcoord", ["u", "v"])
def test_accessor_result_combines_with_own_variables_in_every_layout(layout, hcoord):
    """Rutgers and REMORA alias their dims, UCLA and CROCO do not: in all four the
    accessor's result lines up with the Dataset's own u or v."""
    ds = merged(layout)
    own = ds[hcoord]
    assert (own + ds.xroms.to_grid("temp", hcoord=hcoord)).dims == own.dims


def test_canonicalize_makes_pure_results_combinable(rutgers):
    ds = rutgers
    c = xroms.canonicalize(ds)
    assert RUTGERS_ALIASES.isdisjoint(c.dims)
    assert c.u.dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
    assert c.v.dims == ("ocean_time", "s_rho", "eta_v", "xi_rho")
    assert c.mask_psi.dims == ("eta_v", "xi_u")

    combined = c.u + xroms.to_u(c.temp)
    assert combined.dims == c.u.dims
    # both routes add the same numbers
    via_accessor = ds.u + ds.xroms.to_grid("temp", hcoord="u")
    np.testing.assert_allclose(combined.values, via_accessor.values, **TIGHT)

    # without canonicalizing, the same sum silently grows a spurious dimension
    assert (ds.u + xroms.to_u(ds.temp)).ndim > ds.u.ndim
    assert ds.u.dims == ("ocean_time", "s_rho", "eta_u", "xi_u")  # input untouched


# ---------------------------------------------------- 13. no setup step
CALLS = {
    "ddxi": lambda ds: xroms.ddxi(ds.temp, ds),
    "ddeta": lambda ds: xroms.ddeta(ds.temp, ds),
    "ddz": lambda ds: xroms.ddz(ds.temp, ds),
    "z": lambda ds: xroms.z(ds),
    "speed": lambda ds: xroms.speed(ds.u, ds.v),
    "to_rho": lambda ds: xroms.to_rho(ds.u),
    "subset": lambda ds: xroms.subset(ds, X=slice(2, 9), Y=slice(1, 7), halo=1),
    "trim": lambda ds: xroms.trim(ds, 1),
    "canonicalize": lambda ds: xroms.canonicalize(ds),
    "accessor.ddxi": lambda ds: ds.xroms.ddxi("temp"),
    "accessor.speed": lambda ds: ds.xroms.speed,
    "accessor.z_rho": lambda ds: ds.xroms.z_rho,
    "accessor.to_grid": lambda ds: ds.xroms.to_grid("temp", hcoord="u"),
}


@pytest.mark.parametrize("call", list(CALLS))
def test_calling_twice_is_identical_and_leaves_the_dataset_alone(rutgers, call):
    """There is no setup step to run twice, and no hidden state: a call can be
    repeated, and the Dataset is exactly as it was."""
    ds = rutgers
    pristine = ds.copy(deep=True)
    first = CALLS[call](ds)
    second = CALLS[call](ds)
    xr.testing.assert_identical(first, second)
    xr.testing.assert_identical(ds, pristine)


def test_every_layout_works_without_setup_and_is_left_alone(layout):
    """The same holds for every ROMS family, straight from a freshly opened file."""
    ds = merged(layout)
    pristine = ds.copy(deep=True)
    for name, call in CALLS.items():
        first, second = call(ds), call(ds)
        try:
            xr.testing.assert_identical(first, second)
            xr.testing.assert_identical(ds, pristine)
        except AssertionError as err:
            raise AssertionError(f"{name}: {err}") from err


def test_functions_that_prepare_a_dataset_are_idempotent(layout):
    """canonicalize and decode_time can be applied to their own output; with
    nothing to rename or decode they change nothing."""
    ds = merged(layout)
    once = xroms.canonicalize(ds)
    xr.testing.assert_identical(xroms.canonicalize(once), once)
    if not RUTGERS_ALIASES & set(ds.dims):
        xr.testing.assert_equal(once, ds)
    if layout == "ucla":
        decoded = xroms.decode_time(ds)
        xr.testing.assert_identical(xroms.decode_time(decoded), decoded)


# ------------------------------------ 14. explicit zeta and z follow the variable
# name -> (call taking (variable, dataset, **explicit), the dims of its full-domain
# result that hold the points of the X, Y window)
EXPLICIT = {
    "ddxi": (lambda v, d, **kw: xroms.ddxi(v, d, **kw), {"eta_rho": Y, "xi_u": XU}),
    "ddeta": (lambda v, d, **kw: xroms.ddeta(v, d, **kw), {"eta_v": YV, "xi_rho": X}),
    "ddz": (lambda v, d, **kw: xroms.ddz(v, d, **kw), {"eta_rho": Y, "xi_rho": X}),
    "zslice": (
        lambda v, d, **kw: xroms.zslice(v, [-15.0], d, **kw),
        {"eta_rho": Y, "xi_rho": X},
    ),
    "gridsum": (
        lambda v, d, **kw: xroms.gridsum(v, d, "Z", **kw),
        {"eta_rho": Y, "xi_rho": X},
    ),
}
# gridsum takes a free surface but has no argument for explicit depths
EXPLICIT_CASES = [
    (name, how)
    for name in EXPLICIT
    for how in ("zeta", "z")
    if (name, how) != ("gridsum", "z")
]


def given(ds, how):
    """The explicit free surface (``how="zeta"``) or depths (``"z"``) of ``ds``."""
    return {"zeta": ds.zeta} if how == "zeta" else {"z": xroms.z(ds)}


@pytest.mark.parametrize("name, how", EXPLICIT_CASES)
@pytest.mark.parametrize("t", [0, 1])
def test_explicit_zeta_or_z_follows_a_single_time_variable(rutgers, name, how, t):
    """zeta=ds.zeta or z=xroms.z(ds) hold every time of the grid. A variable cut to
    one time must be matched to its own time, like the grid's zeta is, and not come
    back broadcast over all of them (shape (2, 6, 9, 11) instead of (6, 9, 11))."""
    ds = rutgers
    call = EXPLICIT[name][0]
    one = ds.temp.isel(ocean_time=t)
    got = call(one, ds, **given(ds, how))
    assert "ocean_time" not in got.dims
    assert_same(got, call(one, ds))


def test_explicit_zeta_reaches_the_accessor_too(rutgers):
    ds = rutgers
    one = ds.temp.isel(ocean_time=1)
    got = ds.xroms.ddxi(one, zeta=ds.zeta)
    assert "ocean_time" not in got.dims
    np.testing.assert_allclose(got.values, **ANALYTIC["ddxi"])


@pytest.mark.parametrize("name, how", EXPLICIT_CASES)
def test_explicit_zeta_or_z_follows_a_horizontal_subset(labelled, name, how):
    """The explicit field covers the whole domain and the variable only a window of
    it: they are matched by label, as the grid's own h and zeta are (a raw
    AlignmentError before), also when the variable was cut in time as well."""
    ds = labelled
    call, window = EXPLICIT[name]
    sub = ds.temp.isel(xi_rho=X, eta_rho=Y)
    want = call(ds.temp, ds).isel(window)
    assert_same(call(sub, ds, **given(ds, how)), want)

    one = sub.isel(ocean_time=1)
    got = call(one, ds, **given(ds, how))
    assert "ocean_time" not in got.dims
    assert_same(got, want.isel(ocean_time=1))


@pytest.mark.parametrize("how", ["zeta", "z"])
def test_explicit_zeta_or_z_varying_in_time_needs_a_time_reduced_variable(rutgers, how):
    """A time mean has no time to pick from an explicit field either. The error says
    so and what to pass, where the result used to grow a spurious time dimension."""
    ds = rutgers
    tm = ds.temp.mean("ocean_time")
    with pytest.raises(xroms._align.GridMismatchError, match="ocean_time") as err:
        xroms.ddxi(tm, ds, **given(ds, how))
    assert "zeta=0" in str(err.value)

    reduced = {
        "zeta": {"zeta": ds.zeta.mean("ocean_time")},
        "z": {"z": xroms.z(ds, zeta="mean")},
    }[how]
    got = xroms.ddxi(tm, ds, **reduced)
    assert got.dims == ("s_rho", "eta_rho", "xi_u")
    np.testing.assert_allclose(got.values, **ANALYTIC["ddxi"])


def test_matching_explicit_zeta_z_and_levels_stays_lazy(rutgers):
    """Matching uses labels and metadata only: no model field is evaluated."""
    c, loads = tapped(chunked(rutgers))
    cut = c.temp.isel(ocean_time=0)
    results = {
        "zeta": xroms.ddxi(cut, c, zeta=c.zeta),
        "z": xroms.ddxi(cut, c, z=xroms.z(c)),
        "levels": xroms.ddxi(c.temp.isel(s_rho=slice(1, 4)), c),
        "sum": xroms.gridsum(c.temp.isel(s_rho=slice(1, 4)), c, "Z"),
        "mean": xroms.depth_average(c.temp.isel(s_rho=slice(1, 4)), c),
    }
    assert not loads, "model fields were evaluated while building the results"
    for key, got in results.items():
        assert got.chunks is not None, key
    want = xroms.ddxi(rutgers.temp.isel(ocean_time=0), rutgers)
    np.testing.assert_allclose(results["zeta"].values, want.values, **TIGHT)
    np.testing.assert_allclose(results["z"].values, want.values, **TIGHT)
    np.testing.assert_allclose(results["levels"].values, **ANALYTIC["ddxi"])


# ---------------------------------------- 15. grid without a free surface
@pytest.mark.parametrize("name", list(EXPLICIT))
def test_time_varying_variable_needs_a_free_surface_when_the_grid_has_none(rutgers, name):
    """Output without its zeta used to get flat (zeta = 0) depths without a word. A
    variable with a time dim, or the scalar time of a selection, now raises and says
    how to choose; one without any time keeps the resting default."""
    ds = rutgers
    call = EXPLICIT[name][0]
    bare = ds.drop_vars("zeta")
    for var in (ds.temp, ds.temp.isel(ocean_time=1)):
        with pytest.raises(xroms._align.GridMismatchError, match="zeta") as err:
            call(var, bare)
        for hint in ("zeta=0", "zeta=<DataArray>", "merge"):
            assert hint in str(err.value), hint

    assert_same(call(ds.temp, bare, zeta=0), call(ds.temp, ds, zeta=0))
    assert_same(call(ds.temp, bare, zeta=ds.zeta), call(ds.temp, ds))

    tm = ds.temp.mean("ocean_time")
    assert_same(call(tm, bare), call(tm, bare, zeta=0))


def test_output_with_a_grid_file_that_lacks_zeta(ucla_romstools):
    """A roms-tools grid file holds the vertical parameters but no zeta, which the
    output has: passing the grid file alone used to give flat depths. Merging the
    two, or passing the output's zeta, gives the real ones."""
    out, grid = ucla_romstools
    with pytest.raises(xroms._align.GridMismatchError, match="zeta"):
        xroms.ddxi(out.temp, grid=grid)
    want = xroms.ddxi(out.temp, grid=ucla_dataset((out, grid)))
    assert_same(xroms.ddxi(out.temp, grid=grid, zeta=out.zeta), want)


# ------------------------------------------- 16. variables cut vertically
CUTS = {"slice": slice(1, 4), "list": [1, 2, 3]}


@pytest.mark.parametrize("levels", list(CUTS.values()), ids=list(CUTS))
@pytest.mark.parametrize("lay", ["rutgers", "remora"])
def test_variable_cut_vertically_is_matched_to_the_vertical_parameters_by_label(lay, levels):
    """These files label s_rho, yet ``temp.isel(s_rho=...)`` on its own raised a raw
    AlignmentError (join='exact'). The depths are matched to its levels by label, so
    every result is that of its own layers."""
    ds = merged(lay)
    t = ds.temp.isel(s_rho=levels)
    ax = t.dims.index("s_rho")

    got = xroms.ddxi(t, ds)
    assert got.sizes["s_rho"] == 3
    np.testing.assert_allclose(got.values, **ANALYTIC["ddxi"])
    np.testing.assert_allclose(xroms.ddeta(t, ds).values, **ANALYTIC["ddeta"])
    w = xroms.ddz(t, ds)
    assert w.sizes["s_w"] == 4
    np.testing.assert_allclose(w.values, syn.TEMP_B, rtol=1e-9)

    # thicknesses, sums and means over its own layers
    layers = xroms.dz(ds).isel(s_rho=levels)
    assert_same(xroms.dz(ds, like=t), layers)
    np.testing.assert_allclose(
        xroms.gridsum(t, ds, "Z").values, (t.values * layers.values).sum(ax), **TIGHT
    )
    np.testing.assert_allclose(
        xroms.depth_average(t, ds).values,
        (t.values * layers.values).sum(ax) / layers.values.sum(ax),
        **TIGHT,
    )

    # zslice sees the same water as on the full column wherever the cut reaches
    depth = [-15.0]
    part = xroms.zslice(t, depth, ds)
    full = xroms.zslice(ds.temp, depth, ds)
    inside = np.isfinite(part.values)
    assert inside.mean() > 0.5
    np.testing.assert_allclose(part.values[inside], full.values[inside], **TIGHT)


def test_explicit_depths_on_all_levels_follow_a_variable_cut_vertically(rutgers):
    """A z with every level is matched to the variable's levels by label too."""
    ds = rutgers
    t = ds.temp.isel(s_rho=slice(1, 4))
    got = xroms.ddxi(t, ds, z=xroms.z(ds))
    assert got.sizes["s_rho"] == 3
    np.testing.assert_allclose(got.values, **ANALYTIC["ddxi"])


def test_w_level_variable_cut_vertically_is_matched_by_label_too(rutgers):
    """Variables on w levels (like the w velocity, which these files label along s_w)
    are matched to Cs_w and the w-level depths the same way."""
    ds = rutgers
    cut = slice(2, 6)
    on_w = xroms.to_s_w(ds.temp).assign_coords(s_w=ds.s_w)
    sub = on_w.isel(s_w=cut)
    assert_same(xroms.vertical.z_like(sub, ds), xroms.z(ds, scoord="w").isel(s_w=cut))
    assert_same(xroms.dz(ds, scoord="w", like=sub), xroms.dz(ds, scoord="w").isel(s_w=cut))
    assert xroms.ddxi(sub, ds).sizes["s_w"] == 4


@pytest.mark.parametrize("lay", ["ucla", "croco"])
@pytest.mark.parametrize("name", ["ddxi", "ddeta", "ddz", "zslice", "gridsum", "depth_average"])
def test_variable_cut_vertically_without_labels_says_to_cut_the_dataset(lay, name):
    """Without s_rho labels on both sides the levels cannot be told apart: ddxi gave
    "conflicting sizes {3, 6}" and zslice a numba core-dimension error. Now the error
    says to select levels on the Dataset rather than on the variable alone."""
    ds = merged(lay)
    t = ds.temp.isel(s_rho=slice(1, 4))
    calls = {
        **{n: EXPLICIT[n][0] for n in EXPLICIT},
        "depth_average": lambda v, d: xroms.depth_average(v, d),
    }
    with pytest.raises(xroms._align.GridMismatchError) as err:
        calls[name](t, ds)
    assert "select levels on the Dataset rather than on the variable alone" in str(err.value)

    with pytest.raises(xroms._align.GridMismatchError, match="Dataset"):  # the same for a given z
        xroms.ddxi(t, ds, z=xroms.z(ds))
    # cutting z the same way says which levels, so that works
    got = xroms.ddxi(t, ds, z=xroms.z(ds).isel(s_rho=slice(1, 4)))
    np.testing.assert_allclose(got.values, **ANALYTIC["ddxi"])


# ------------------------------------------------ 17. strided subsets
@pytest.fixture
def indexed(rutgers):
    """Rutgers dataset with integer index coordinates on every horizontal dimension."""
    return xroms.add_cf_attrs(rutgers, index_coords=True)


@pytest.mark.parametrize("name, dim", [("ddxi", "xi_rho"), ("ddeta", "eta_rho")])
@pytest.mark.parametrize("step", [2, -1], ids=["every-other", "reversed"])
def test_strided_subset_with_index_coords_raises_instead_of_a_wrong_derivative(indexed, name, dim, step):
    """pm and pn describe neighbouring cells. Every other point gave a derivative
    twice too large, silently, once the dataset had index coordinates to match by."""
    d = indexed
    strided = d.temp.isel({dim: slice(None, None, step)})
    with pytest.raises(xroms._align.GridMismatchError, match=rf"{dim}.*do not step by 1"):
        getattr(xroms, name)(strided, d)
    # the same subset with step 1 is what to do instead, and is exact
    contiguous = d.temp.isel({dim: slice(2, 9)})
    np.testing.assert_allclose(getattr(xroms, name)(contiguous, d).values, **ANALYTIC[name])


def test_strided_index_labels_are_caught_wherever_grid_fields_are_matched(indexed):
    d = indexed
    strided = d.temp.isel(xi_rho=slice(None, None, 2))
    with pytest.raises(xroms._align.GridMismatchError, match="step by 1"):
        xroms._align.select_like(d.h, strided)
    with pytest.raises(xroms._align.GridMismatchError, match="step by 1"):
        xroms.dx(d, like=strided)
    with pytest.raises(xroms._align.GridMismatchError, match="step by 1"):
        xroms.z(d, like=strided)
    # or in the grid field, whatever the variable has
    plain = d.temp.isel(xi_rho=slice(0, 6)).drop_vars("xi_rho")
    thinned = d.h.isel(xi_rho=slice(None, None, 2))
    with pytest.raises(xroms._align.GridMismatchError, match=r"'h' has 'xi_rho' index labels .* step by 1"):
        xroms._align.select_like(thinned, plain, name="h")


@pytest.mark.parametrize(
    "call, dim",
    [
        (lambda d: xroms.to_rho(d.u.isel(xi_u=slice(None, None, 2))), "xi_u"),
        (lambda d: xroms.to_u(d.temp.isel(xi_rho=slice(None, None, 3))), "xi_rho"),
        (lambda d: xroms.to_v(d.temp.isel(eta_rho=slice(None, None, 2))), "eta_rho"),
        (lambda d: xroms.to_rho(d.v.isel(eta_v=slice(None, None, 2))), "eta_v"),
    ],
    ids=["u-to-rho", "rho-to-u", "rho-to-v", "v-to-rho"],
)
def test_moving_strided_index_labels_to_another_stagger_raises(indexed, call, dim):
    """``to_rho(u[..., ::2]).xi_rho`` came out as [0 2 4 6 8 10 11]: labels that
    describe no cell, made up across points that are not neighbours."""
    with pytest.raises(xroms._align.GridMismatchError, match=rf"{dim}.*do not step by 1"):
        call(indexed)


def test_only_the_axis_that_is_operated_on_has_to_be_contiguous(indexed):
    """Moving along xi does not look at eta: every other eta row is what it is. Moving
    along eta does."""
    d = xroms.canonicalize(indexed)
    every_other_row = d.u.isel(eta_rho=slice(None, None, 2))
    got = xroms.to_rho(every_other_row)
    assert got.dims[-2:] == ("eta_rho", "xi_rho")
    np.testing.assert_array_equal(got["eta_rho"].values, d.eta_rho.values[::2])
    with pytest.raises(xroms._align.GridMismatchError, match=r"eta_rho.*do not step by 1"):
        xroms.to_v(every_other_row)


def test_labels_that_are_not_integer_positions_are_not_invented(indexed):
    """Half-integer labels on u cannot be shifted onto rho points without inventing
    values: the new dimension gets none, so adding rho-point data still lines up (it
    came out with an xi_rho of size 0)."""
    d = xroms.canonicalize(indexed)
    half = d.assign_coords(xi_u=np.arange(d.sizes["xi_u"]) + 0.5)
    rho = xroms.to_rho(half.u)
    assert "xi_rho" not in rho.indexes
    combined = rho + half.temp
    assert combined.sizes["xi_rho"] == d.sizes["xi_rho"]
    assert xroms.to_u(half.temp).sizes["xi_u"] == d.sizes["xi_u"]
    # integer positions do carry across, as before
    np.testing.assert_array_equal(xroms.to_rho(d.u)["xi_rho"].values, np.arange(d.sizes["xi_rho"]))
