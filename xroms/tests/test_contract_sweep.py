"""The dimension contract, swept over the public pure functions and unusual inputs.

``test_dimension_contract.py`` sweeps the accessor over four variants of the input.
This sweeps the pure functions (``xroms.f(var, ds, ...)``, see ``_sweep.OPS``) over
more variants, in every ROMS-family layout:

* ``full`` 4-D input, one time (``isel(time=0)``, no time dim left), a time dim of
  length 1 (``isel(time=[0])``, which must stay a time dim), a horizontal subset,
  dask-chunked data, float32 fields, land (NaN), every variable with its dims in reverse order and an
  ensemble dim in front of the others (the results must still come out in
  (time, vertical, eta, xi, extras) order, except for selections and metrics, which keep the order
  of their input);
* a single water column (``isel`` across every staggered dim): the vertical
  operations work and those that need horizontal neighbours say so;
* the surface level of each variable (``xroms.surface``): a single selected s-level
  is not on an s-surface any more, so horizontal derivatives of it raise unless
  ``along_s=True``;
* 2-D fields (``zeta``, ``h``).

For every result that is returned the contract of ``test_dimension_contract.py``
holds: the time dim is present if and only if the input had it, dims are ordered
(time, vertical, eta, xi, extras), and combining the result with the Dataset's own
variable at the same grid position gains no dims. Beyond dims, the variants must also
agree on values with the full-domain computation (one time, a length-1 time, dask,
a subset with a halo that is trimmed afterwards, a column).
"""

import functools
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pytest

import xroms
from xroms.conventions import TIME_NAMES, canonicalize, time_dim
from xroms.tests import _sweep as S
from xroms.tests.conftest import chunked


LEAD = ["s_rho", "s_w", "eta_rho", "eta_u", "eta_v", "eta_psi", "xi_rho", "xi_u", "xi_v", "xi_psi"]
NEW_VERTICAL = ("z", "zz")  # the dims zslice and isoslice put where the level dim was
#: dims a result may have that the Dataset does not: s_w even for UCLA output, which has no w-level variables
NEW_DIMS = {"z", "zz", "s_w"}


def _ordered(da):
    dims = list(da.dims)
    t = [d for d in dims if d in TIME_NAMES]
    if t:
        assert dims[0] == t[0], f"time dim not first: {dims}"
        dims = dims[1:]
    for new in NEW_VERTICAL:
        if new in dims:
            assert dims.index(new) == 0, f"the new level dim {new!r} is not where the levels were: {da.dims}"
            dims.remove(new)
    known = [d for d in dims if d in LEAD]
    assert known == sorted(known, key=LEAD.index), f"not in (vertical, eta, xi) order: {da.dims}"


def check_contract(name, driver, same, ds, p, out, *, ordered=True, finite=True):
    """The three clauses of the dimension contract for every result in ``out``.

    ``p(driver)`` is the input that decides whether there is a time dim; ``same`` names the
    Dataset's own variable(s) at the result's position, which a result may not grow dims against.
    """
    want_time = driver is not None and S.has_time(p(driver))
    # an ensemble dim comes from the inputs, never from the grid alone
    want_member = name not in S.GRID_ONLY and driver is not None and "member" in p(driver).dims
    for i, o in enumerate(S.results(out)):
        assert (time_dim(o) is not None) == want_time, f"{name}: time dim {'lost' if want_time else 'added'}: {o.dims}"
        assert ("member" in o.dims) == want_member, f"{name}: ensemble dim {'lost' if want_member else 'added'}: {o.dims}"
        assert o.dtype.kind == "f", f"{name}: {o.dtype}"
        if ordered:
            _ordered(o)
            assert not want_member or o.dims[-1] == "member", f"{name}: extra dims come last: {o.dims}"
        extra = set(o.dims) - set(canonicalize(ds).dims) - NEW_DIMS - ({"member"} if want_member else set())
        assert not extra, f"{name}: dims {extra} are not in the Dataset"
        var = same[i] if isinstance(same, tuple) else same
        if var is not None:
            ref = canonicalize(ds[var])
            combined = ref + o
            allowed = set(ref.dims) | ({"member"} if want_member else set())  # the ensemble dim is the inputs'
            assert set(combined.dims) == allowed, f"{name}: combining with {var} gained dims {combined.dims}"
        if finite:  # (needs the values: dask results are compared with numpy's in test_variants_agree_...)
            assert np.isfinite(o.values).any(), f"{name}: nothing finite in the result"


def check_op(op, ds, p, *, ordered=True, finite=True):
    out = op.call(ds, p)
    check_contract(op.name, op.driver, op.same, ds, p, out, ordered=ordered, finite=finite)
    return out


# --- whole-Dataset variants ---------------------------------------------------------------


@pytest.mark.parametrize("variant", ["full", "one_time", "time1", "subset", "chunked", "float32", "transposed", "member"])
@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_pure_functions_keep_the_contract(layout, variant, op):
    ds, p = S.make_variant(S.dataset(layout), variant)
    # input dims in reverse order (xi, eta, level, time), or an ensemble dim in front, still come out as
    # (time, vertical, eta, xi, extras): the density family and the grid moves say so in their docstrings;
    # the others are selections and metrics
    scrambled = variant in ("transposed", "member")
    out = check_op(op, ds, p, ordered=not scrambled or op.name not in S.KEEPS_INPUT_ORDER, finite=variant != "chunked")
    if variant == "chunked":
        assert all(o.chunks is not None for o in S.results(out)), f"{op.name}: chunked input computed eagerly"


@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_pure_functions_keep_the_contract_over_land(layout, op):
    # land is NaN in the fields (and in zeta, so in the depths): nothing may be lost or added by it
    ds, p = S.make_variant(S.dataset(layout, land=True), "full")
    check_op(op, ds, p)


@pytest.mark.parametrize("op_name", ["depth_average", "gridmean_over_z"])
def test_means_over_land_do_not_warn_when_computed(layout, op_name):
    ds = chunked(S.dataset(layout, land=True))
    mean = xroms.depth_average(ds.temp, ds) if op_name == "depth_average" else xroms.gridmean(ds.temp, ds, "Z")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mean.compute(scheduler="synchronous")  # in this thread, where the filter is certain to apply


@functools.lru_cache(maxsize=None)
def _full(layout, op_name):
    op = next(o for o in S.OPS if o.name == op_name)
    return tuple(o.load() for o in S.results(op.call(*S.make_variant(S.dataset(layout), "full"))))


def _agree(got, want, what):
    assert set(got.dims) == set(want.dims), f"{what}: dims {got.dims} vs {want.dims}"
    np.testing.assert_allclose(got.transpose(*want.dims).values, want.values, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=what)


@pytest.mark.parametrize("variant", ["one_time", "time1", "chunked", "halo", "transposed"])
@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_variants_agree_with_the_full_computation(layout, variant, op):
    if variant == "halo" and not op.local:
        pytest.skip("reduces over the horizontal: a subset has another answer by construction")
    ds, p = S.make_variant(S.dataset(layout), variant)
    for got, want in zip(S.results(op.call(ds, p)), _full(layout, op.name)):
        tdim = time_dim(want)
        if variant == "one_time" and tdim:
            want = want.isel({tdim: 0})
        elif variant == "time1" and tdim:
            want = want.isel({tdim: [0]})
        elif variant == "halo":
            # the halo makes the staggered operations exact; trim it, and what is left is the window
            got = xroms.trim(got, n=S.HALO)
            want = want.isel({d: S.WINDOW[d] for d in want.dims if d in S.WINDOW})
        _agree(got, want, f"{op.name} on {variant}")


# --- a single water column ----------------------------------------------------------------

#: needs horizontal neighbours: the message says the position cannot be told, or the axis is missing
NEEDS_NEIGHBOURS = r"horizontal grid position|has no '[XY]' dimension|lacks a dimension for axes"
COLUMN_RAISES = {
    "ddxi", "ddeta", "ddxi_along_s", "ddeta_along_s", "hgrad", "relative_vorticity", "convergence",
    "convergence_along_s", "uv_geostrophic", "EKE", "ertel", "M2", "z_w_u", "dz_w_v", "dy_u", "dA_psi", "gridmean_XY",
}
#: at the column's rho point these equal the full-domain result there (the others average over
#: neighbours that a column does not have, or are no-ops)
COLUMN_MATCHES = {
    "ddz", "density", "potential_density", "buoyancy", "N2", "mld", "z_rho", "dz", "dx", "dV", "depth_average",
    "surface", "zslice", "isoslice", "gridsum_Z", "to_s_w", "to_s_rho",
}


@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_water_column(layout, op):
    ds, p = S.make_variant(S.dataset(layout), "column")
    if op.name in COLUMN_RAISES:
        with pytest.raises(ValueError, match=NEEDS_NEIGHBOURS):
            op.call(ds, p)
        return
    out = check_op(op, ds, p)
    if op.name in COLUMN_MATCHES:
        at = {d: S.K if d.startswith("eta_") else S.J for d in canonicalize(_full(layout, op.name)[0]).dims if d in ("eta_rho", "xi_rho")}
        for got, want in zip(S.results(out), _full(layout, op.name)):
            _agree(got, canonicalize(want).isel(at), f"{op.name} on a column")


def test_column_errors_name_the_array(layout):
    # z at u points averages neighbouring rho points; a column has none, which is an error
    # worth naming the array for
    ds, _ = S.make_variant(S.dataset(layout), "column")
    with pytest.raises(ValueError, match="'X' dimension") as err:
        xroms.z(ds, hcoord="u")
    assert not str(err.value).startswith("None ")


# --- the surface level -----------------------------------------------------------------------

SINGLE_LEVEL = r"single selected s-level"
NO_VERTICAL = r"no vertical dimension"
SURFACE_RAISES = {
    "ddxi": SINGLE_LEVEL, "ddeta": SINGLE_LEVEL, "hgrad": SINGLE_LEVEL, "relative_vorticity": SINGLE_LEVEL,
    "convergence": SINGLE_LEVEL, "ertel": SINGLE_LEVEL, "M2": f"{SINGLE_LEVEL}|{NO_VERTICAL}",
    "ddz": NO_VERTICAL, "dudz": NO_VERTICAL, "dvdz": NO_VERTICAL, "vertical_shear": NO_VERTICAL,
    "surface": NO_VERTICAL, "bottom": NO_VERTICAL, "gridsum_Z": NO_VERTICAL, "N2": NO_VERTICAL,
    "mld": r"has no vertical dimension; mld needs profiles", "depth_average": r"needs a variable on s_rho levels",
    "zslice": f"{NO_VERTICAL}|no 'Z' dimension", "isoslice": r"no 'Z' dimension",
}


@pytest.mark.parametrize("op", S.OPS, ids=S.OP_IDS)
def test_surface_level(layout, op):
    base = S.dataset(layout)
    ds, p = S.make_variant(base, "surface")
    pattern = SURFACE_RAISES.get(op.name)
    if op.name == "density":
        # z of a single level is found by the level's label: Rutgers and REMORA have s_rho labels, UCLA and CROCO do not
        pattern = None if "s_rho" in base.indexes else r"without an s-coordinate label"
    if pattern is not None:
        with pytest.raises(ValueError, match=pattern):
            op.call(ds, p)
        return
    check_op(op, ds, p)


def test_along_s_is_the_plain_difference_along_the_surface_level(layout):
    # along_s=True takes the derivative along the s-surface: the plain difference over the u-point
    # spacing, without the chain-rule term of the derivative at constant depth
    base = canonicalize(S.dataset(layout))
    ds, p = S.make_variant(S.dataset(layout), "surface")
    along = xroms.ddxi(p("temp"), ds, along_s=True)
    pm_u = 0.5 * (base.pm.values[:, :-1] + base.pm.values[:, 1:])
    expected = np.diff(base.temp.isel(s_rho=-1).values, axis=-1) * pm_u
    assert along.dims == base.temp.isel(s_rho=-1).dims[:-1] + ("xi_u",)
    np.testing.assert_allclose(along.values, expected, rtol=1e-12, atol=1e-15)


# --- 2-D fields ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoD:
    name: str
    call: Callable  # (ds, q) -> result
    same: Union[None, str, Tuple[Optional[str], ...]] = None
    raises: Optional[str] = None  # message of the error a field without a vertical dim gets


TWO_D = [
    TwoD("ddxi", lambda ds, q: xroms.ddxi(q, ds), "u"),
    TwoD("ddeta", lambda ds, q: xroms.ddeta(q, ds), "v"),
    TwoD("hgrad", lambda ds, q: xroms.hgrad(q, ds), ("u", "v")),
    TwoD("to_rho", lambda ds, q: xroms.to_rho(q), "temp"),
    TwoD("to_u", lambda ds, q: xroms.to_u(q), "u"),
    TwoD("to_v", lambda ds, q: xroms.to_v(q), "v"),
    TwoD("to_psi", lambda ds, q: xroms.to_psi(q)),
    TwoD("to_s_w", lambda ds, q: xroms.to_s_w(q), "temp"),
    TwoD("gridmean_XY", lambda ds, q: xroms.gridmean(q, ds, ("X", "Y"))),
    TwoD("gridsum_X", lambda ds, q: xroms.gridsum(q, ds, "X"), "zeta"),
    TwoD("uv_geostrophic", lambda ds, q: xroms.uv_geostrophic(q, ds["f"], ds), ("u", "v")),
    TwoD("EKE", lambda ds, q: xroms.EKE(*xroms.uv_geostrophic(q, ds["f"], ds)), "temp"),
    TwoD("ddz", lambda ds, q: xroms.ddz(q, ds), raises=NO_VERTICAL),
    TwoD("dudz", lambda ds, q: xroms.dudz(q, ds), raises=NO_VERTICAL),
    TwoD("N2", lambda ds, q: xroms.N2(q, ds), raises=NO_VERTICAL),
    TwoD("mld", lambda ds, q: xroms.mld(q, ds), raises=NO_VERTICAL),
    TwoD("zslice", lambda ds, q: xroms.zslice(q, [-5.0], ds), raises=NO_VERTICAL),
    TwoD("isoslice", lambda ds, q: xroms.isoslice(q, [1.0], q), raises=r"no 'Z' dimension"),
    TwoD("depth_average", lambda ds, q: xroms.depth_average(q, ds), raises=r"needs a variable on s_rho levels"),
    TwoD("surface", lambda ds, q: xroms.surface(q), raises=NO_VERTICAL),
    TwoD("bottom", lambda ds, q: xroms.bottom(q), raises=NO_VERTICAL),
    TwoD("gridsum_Z", lambda ds, q: xroms.gridsum(q, ds, "Z"), raises=NO_VERTICAL),
    TwoD("density_from_grid", lambda ds, q: xroms.density(q, q, grid=ds), raises=NO_VERTICAL),
]


@pytest.mark.parametrize("field", ["zeta", "h", "zeta_one_time"])
@pytest.mark.parametrize("op", TWO_D, ids=[o.name for o in TWO_D])
def test_two_dimensional_fields(layout, field, op):
    ds = S.dataset(layout)
    tdim = time_dim(ds)
    q = ds["h"] if field == "h" else ds["zeta"] if field == "zeta" else ds["zeta"].isel({tdim: 0})
    if op.raises is not None:
        with pytest.raises(ValueError, match=op.raises):
            op.call(ds, q)
        return
    out = op.call(ds, q)
    # a field without a time dim must not grow one from the grid's time-dependent variables
    check_contract(op.name, "q", op.same, ds, lambda name: q, out)
    for o in S.results(out):
        assert "s_rho" not in o.dims and "s_w" not in o.dims


# --- one vertical level ---------------------------------------------------------------------------------------


def test_a_vertical_dim_of_length_one_needs_along_s(layout):
    base = S.dataset(layout)
    if "s_w" not in base.dims:
        pytest.skip("UCLA's s-coordinate parameters are attributes: they cannot be cut with the data")
    one = base.isel(s_rho=slice(-1, None), s_w=slice(-2, None))
    with pytest.raises(ValueError, match="at least 2 vertical levels"):
        xroms.ddxi(one.temp, one)
    along = xroms.ddxi(one.temp, one, along_s=True)
    assert along.sizes["s_rho"] == 1 and time_dim(along) is not None
    _ordered(along)
    with pytest.raises(ValueError, match="at least 2 levels"):
        xroms.ddz(one.temp, one)
    assert xroms.z(one).sizes["s_rho"] == 1 and xroms.mld(xroms.potential_density(one.temp, one.salt), one).ndim == 3
