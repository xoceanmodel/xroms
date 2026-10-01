"""The dimension contract: every result has exactly the dims it should.

For each operation and each input variant (full 4-D, one time, horizontal
subset, dask-chunked) in every ROMS-family layout, the result:

1. carries the time dim if and only if the input did (no re-broadcasting);
2. is ordered (time, vertical, eta, xi, ...);
3. when combined arithmetically with the Dataset's own variable at the same
   grid position, gains no dims (no silent broadcasting to 5-D);
4. carries the Dataset's coordinates for its position (accessor results).
"""

import numpy as np
import pytest

import xroms
from xroms.conventions import TIME_NAMES, canonicalize, time_dim
from xroms.tests.conftest import chunked, merged


LEAD = ["s_rho", "s_w", "eta_rho", "eta_u", "eta_v", "eta_psi", "xi_rho", "xi_u", "xi_v", "xi_psi"]


def _ordered(da):
    dims = list(da.dims)
    t = [d for d in dims if d in TIME_NAMES]
    if t:
        assert dims[0] == t[0], f"time dim not first: {dims}"
        dims = dims[1:]
    known = [d for d in dims if d in LEAD]
    assert known == sorted(known, key=LEAD.index), f"not in (vertical, eta, xi) order: {da.dims}"


def _variant(ds, name):
    tdim = time_dim(ds)
    if name == "full":
        return ds
    if name == "one_time":
        return ds.isel({tdim: 0})
    if name == "subset":
        return xroms.subset(ds, X=slice(2, 9), Y=slice(1, 7))
    if name == "chunked":
        return chunked(ds)
    raise ValueError(name)


# (label, accessor call, name of the Dataset variable at the same position or None)
OPERATIONS = [
    ("ddxi_temp", lambda ds: ds.xroms.ddxi("temp"), "u"),
    ("ddeta_temp", lambda ds: ds.xroms.ddeta("temp"), "v"),
    ("ddz_temp", lambda ds: ds.xroms.ddz("temp"), None),
    ("ddxi_u", lambda ds: ds.xroms.ddxi("u"), "temp"),
    ("to_u", lambda ds: ds.xroms.to_grid("temp", hcoord="u"), "u"),
    ("to_rho_v", lambda ds: ds.xroms.to_grid("v", hcoord="rho"), "temp"),
    ("z_rho", lambda ds: ds.xroms.z_rho, "temp"),
    ("z_u", lambda ds: ds.xroms.z(hcoord="u"), "u"),
    ("dz_w", lambda ds: ds.xroms.dz(scoord="w"), None),
    ("zslice", lambda ds: ds.xroms.zslice("temp", [-5.0]), None),
    ("gridsum_Z", lambda ds: ds.xroms.gridsum("temp", "Z"), "zeta"),
    ("gridmean_XY", lambda ds: ds.xroms.gridmean("temp", ("X", "Y")), None),
    ("speed", lambda ds: ds.xroms.speed, "temp"),
    ("vort", lambda ds: ds.xroms.vort, None),
    ("convergence", lambda ds: ds.xroms.convergence, "temp"),
    ("N2", lambda ds: ds.xroms.N2, None),
]


@pytest.mark.parametrize("variant", ["full", "one_time", "subset", "chunked"])
@pytest.mark.parametrize("label,op,same_pos", OPERATIONS, ids=[o[0] for o in OPERATIONS])
def test_contract(layout, variant, label, op, same_pos):
    ds = _variant(merged(layout), variant)
    out = op(ds)
    has_time = time_dim(ds) is not None
    assert (time_dim(out) is not None) == has_time, f"{label}: time dim {'lost' if has_time else 'added'}: {out.dims}"
    _ordered(out)
    # s_w is legitimate even when the Dataset has no w-level variables (UCLA output)
    extra = set(out.dims) - set(ds.dims) - {"z", "s_w"}
    assert not extra, f"{label}: dims {extra} not in the Dataset"
    if same_pos is not None:
        combined = ds[same_pos] + out if same_pos != "zeta" else ds[same_pos] * 0 + out
        assert set(combined.dims) == set(ds[same_pos].dims), f"{label}: combining with {same_pos} gained dims {combined.dims}"
    if variant == "chunked":
        assert out.chunks is not None, f"{label}: chunked input computed eagerly"
    assert np.isfinite(out.values).any()


@pytest.mark.parametrize("label,op,same_pos", OPERATIONS[:3], ids=[o[0] for o in OPERATIONS[:3]])
def test_single_time_variable_with_full_grid(label, op, same_pos):
    # a variable selected on its own (not the Dataset) must not re-grow a time dim
    ds = merged("rutgers")
    t0 = canonicalize(ds).temp.isel(ocean_time=0)
    for func in (xroms.ddxi, xroms.ddeta, xroms.ddz):
        out = func(t0, ds)
        assert "ocean_time" not in out.dims
        np.testing.assert_allclose(out.values, func(canonicalize(ds).temp, ds).isel(ocean_time=0).values)


# --- coordinates of results that land on another position ---------------------------


class TestGridCoordsAtTheResultsPosition:
    """Pure functions that read a grid give their result the grid's lon/lat (or x/y) where it lands."""

    @pytest.mark.parametrize(
        "compute, names",
        [
            (lambda ds: xroms.relative_vorticity(ds.u, ds.v, ds), ("lon_psi", "lat_psi")),
            (lambda ds: xroms.ddxi(ds.salt, ds), ("lon_u", "lat_u")),
            (lambda ds: xroms.ddeta(ds.salt, ds), ("lon_v", "lat_v")),
            (lambda ds: xroms.z(ds, hcoord="u"), ("lon_u", "lat_u")),
            (lambda ds: xroms.dx(ds, "v"), ("lon_v", "lat_v")),
            (lambda ds: xroms.convergence(ds.u, ds.v, ds), ("lon_rho", "lat_rho")),
        ],
    )
    def test_moved_results_carry_the_grids_coords(self, rutgers, compute, names):
        out = compute(rutgers)
        for name in names:
            expected = canonicalize(rutgers[name].reset_coords(drop=True))
            np.testing.assert_array_equal(out[name].transpose(*expected.dims).values, expected.values)
        # bare variables: no index coordinates are added
        assert set(out.indexes) <= set(rutgers.indexes)

    def test_subsets_and_lazy_results(self, rutgers):
        sub = xroms.subset(rutgers, X=slice(2, 8), Y=slice(1, 6))
        vort = xroms.relative_vorticity(sub.u, sub.v, sub)
        np.testing.assert_array_equal(vort.lon_psi.values, sub.lon_psi.values)
        lazy = xroms.relative_vorticity(chunked(rutgers).u, chunked(rutgers).v, chunked(rutgers))
        assert lazy.chunks is not None and "lon_psi" in lazy.coords

    def test_cartesian_grids_get_x_and_y(self, remora):
        out = xroms.ddxi(remora.temp, remora)
        assert {"x_u", "y_u"} <= set(out.coords)

    def test_positions_the_grid_has_no_coords_for_get_none(self):
        ds = merged("ucla")
        out = xroms.ddxi(ds.temp, ds)
        assert not {"lon_u", "lat_u"} & set(out.coords)
