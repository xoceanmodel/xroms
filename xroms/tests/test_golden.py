"""Regression against outputs recorded from xroms v0.6.2 (``tests/golden``).

* **Exact** where the discretization is unchanged: depths at every stagger,
  metrics, grid moves, speed/KE/EKE, east/north, slices, weighted sums/means,
  density and N2.
* **Interior-exact** where only the boundary treatment changed on purpose
  (v0.6.2 forced zero vertical differences at the top/bottom w levels and used
  a fill value of 0 for the w-level thickness): ``ddz``, ``dudz``, ``dz_w``.
* **Close** (correlation and median relative difference) where the
  discretization changed on purpose: horizontal derivatives are now evaluated
  on the input's own vertical levels with a second-order vertical stencil.

Golden inputs: ``tiny`` (flat 100 m, 3 levels) and ``window`` (a 12x16 piece of
the bundled Rutgers example: sloping, Vtransform 1, 30 levels, float32 fields).
"""

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests.conftest import GOLDEN, INPUT


pytestmark = pytest.mark.skipif(not (GOLDEN / "golden_window.nc").exists(), reason="golden files missing")


def _inputs(name):
    if name == "tiny":
        ds = xr.open_dataset(INPUT / "ocean_his_0001.nc").merge(xr.open_dataset(INPUT / "grid.nc"), overwrite_vars=True, compat="override")
    else:
        ds = xr.open_dataset(GOLDEN / "window_input.nc")
    return ds.load(), xr.open_dataset(GOLDEN / f"golden_{name}.nc").load()


@pytest.fixture(params=["tiny", "window"], scope="module")
def pair(request):
    ds, gold = _inputs(request.param)
    return request.param, ds, C.canonicalize(ds), gold


def _cmp(new, gold, *, rtol=1e-12, atol=1e-12, interior_dim=None):
    new = C.canonicalize(new).reset_coords(drop=True)
    new = new.transpose(*gold.dims)
    a, b = new.values, gold.values
    if interior_dim is not None:
        ax = gold.dims.index(interior_dim)
        a = np.take(a, range(1, a.shape[ax] - 1), axis=ax)
        b = np.take(b, range(1, b.shape[ax] - 1), axis=ax)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def _tol(name, key):
    # window fields are float32; outputs derived from them only agree to ~1e-6
    return 1e-5 if name == "window" else 1e-10


EXACT_Z = {
    "z_rho": dict(), "z_w": dict(scoord="w"),
    "z_rho_u": dict(hcoord="u"), "z_rho_v": dict(hcoord="v"), "z_rho_psi": dict(hcoord="psi"),
    "z_w_u": dict(hcoord="u", scoord="w"), "z_w_v": dict(hcoord="v", scoord="w"), "z_w_psi": dict(hcoord="psi", scoord="w"),
    "z_rho0": dict(zeta=0), "z_w0": dict(scoord="w", zeta=0),
}


@pytest.mark.parametrize("key", list(EXACT_Z))
def test_depths(pair, key):
    name, ds, can, gold = pair
    _cmp(xroms.z(ds, **EXACT_Z[key]), gold[key], rtol=_tol(name, key))


@pytest.mark.parametrize("key,pos", [("dx", "rho"), ("dx_u", "u"), ("dx_v", "v"), ("dx_psi", "psi"), ("dy", "rho"), ("dy_u", "u"), ("dy_v", "v"), ("dy_psi", "psi"), ("dA", "rho"), ("dA_u", "u"), ("dA_v", "v"), ("dA_psi", "psi")])
def test_horizontal_metrics(pair, key, pos):
    name, ds, can, gold = pair
    func = {"dx": xroms.dx, "dy": xroms.dy, "dA": xroms.dA}[key.split("_")[0]]
    _cmp(func(ds, pos), gold[key], rtol=1e-12)


def test_layer_thickness(pair):
    name, ds, can, gold = pair
    _cmp(xroms.dz(ds), gold["dz"], rtol=_tol(name, "dz"))
    _cmp(xroms.dz(ds, hcoord="u"), gold["dz_u"], rtol=_tol(name, "dz"))
    # w-level thickness: interior unchanged; v0.6.2's boundary values were wrong
    _cmp(xroms.dz(ds, scoord="w"), gold["dz_w"], rtol=_tol(name, "dz"), interior_dim="s_w")
    assert (xroms.dz(ds, scoord="w") > 0).all() and (gold["dz_w"].isel(s_w=0) < 0).any()


def test_grid_moves(pair):
    name, ds, can, gold = pair
    rtol = _tol(name, "moves")
    _cmp(xroms.to_rho(can.u), gold["to_rho_u"], rtol=rtol)
    _cmp(xroms.to_rho(can.v), gold["to_rho_v"], rtol=rtol)
    _cmp(xroms.to_u(can.temp), gold["to_u_temp"], rtol=rtol)
    _cmp(xroms.to_v(can.temp), gold["to_v_temp"], rtol=rtol)
    _cmp(xroms.to_psi(can.temp), gold["to_psi_temp"], rtol=rtol)
    _cmp(xroms.to_s_w(can.temp), gold["to_s_w_temp"], rtol=rtol)
    _cmp(xroms.to_s_rho(xroms.z(ds, scoord="w")), gold["to_s_rho_zw"], rtol=rtol)


def test_speed_ke_eke_rotation(pair):
    name, ds, can, gold = pair
    rtol = _tol(name, "speed")
    _cmp(xroms.speed(can.u, can.v), gold["speed"], rtol=rtol)
    _cmp(ds.xroms.KE, gold["KE"], rtol=rtol)
    ug, vg = xroms.uv_geostrophic(can.zeta, can.f, ds)
    _cmp(ug, gold["ug"], rtol=rtol, atol=1e-10)
    _cmp(vg, gold["vg"], rtol=rtol, atol=1e-10)
    _cmp(xroms.EKE(ug, vg), gold["EKE"], rtol=rtol, atol=1e-12)
    east, north = ds.xroms.eastnorth
    _cmp(east, gold["east"], rtol=rtol, atol=1e-12)
    _cmp(north, gold["north"], rtol=rtol, atol=1e-12)


def test_plain_horizontal_derivatives_of_2d_fields(pair):
    name, ds, can, gold = pair
    _cmp(xroms.ddxi(can.zeta, ds), gold["ddxi_zeta"], rtol=_tol(name, "d"), atol=1e-14)
    _cmp(xroms.ddeta(can.zeta, ds), gold["ddeta_zeta"], rtol=_tol(name, "d"), atol=1e-14)


def test_vertical_derivatives_interior(pair):
    name, ds, can, gold = pair
    rtol = _tol(name, "ddz")
    _cmp(xroms.ddz(can.temp, ds), gold["ddz_temp"], rtol=rtol, atol=1e-12, interior_dim="s_w")
    _cmp(xroms.dudz(can.u, ds), gold["dudz"], rtol=rtol, atol=1e-12, interior_dim="s_w")
    _cmp(xroms.dvdz(can.v, ds), gold["dvdz"], rtol=rtol, atol=1e-12, interior_dim="s_w")
    _cmp(xroms.vertical_shear(xroms.dudz(can.u, ds), xroms.dvdz(can.v, ds)), gold["vertical_shear"], rtol=rtol, atol=1e-12, interior_dim="s_w")
    # v0.6.2 forced zeros on the boundary w levels; they are one-sided now
    assert (gold["ddz_temp"].isel(s_w=0) == 0).all()


def test_density_and_stratification(pair):
    name, ds, can, gold = pair
    rtol = _tol(name, "rho")
    rho = xroms.density(can.temp, can.salt, grid=ds)
    _cmp(rho, gold["rho"], rtol=rtol)
    sig0 = xroms.potential_density(can.temp, can.salt, 0)
    _cmp(sig0, gold["sig0"], rtol=rtol)
    _cmp(xroms.buoyancy(sig0), gold["buoy"], rtol=rtol)
    _cmp(xroms.N2(rho, ds), gold["N2"], rtol=max(rtol, 1e-9), atol=1e-12)


def test_slices_and_aggregates(pair):
    name, ds, can, gold = pair
    rtol = _tol(name, "slice")
    depths = gold["zslice_temp"]["z_rho_dim"].values
    out = xroms.zslice(can.temp, depths, ds).rename({"z": "z_rho_dim"})
    _cmp(out.assign_coords(z_rho_dim=depths), gold["zslice_temp"], rtol=rtol, atol=1e-10)
    sal = gold["isoslice_temp_on_salt"]["salt"].values
    iso = xroms.isoslice(can.temp, sal, can.salt, new_dim="salt")
    _cmp(iso, gold["isoslice_temp_on_salt"], rtol=rtol, atol=1e-10)
    _cmp(xroms.gridmean(can.temp, ds, ("Y", "X")), gold["gridmean_temp_YX"], rtol=rtol)
    _cmp(xroms.gridsum(can.temp, ds, "Z"), gold["gridsum_temp_Z"], rtol=rtol)


def _close(new, gold, dims_interior, *, corr=0.98, median_rel=0.1):
    new = C.canonicalize(new).reset_coords(drop=True).transpose(*gold.dims)
    sl = {d: slice(1, -1) for d in dims_interior if d in gold.dims}
    a, b = new.isel(sl).values.ravel(), gold.isel(sl).values.ravel()
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12 * np.nanmax(np.abs(b)))
    if ok.sum() < 10:
        pytest.skip("golden values identically ~0 for this input")
    assert np.corrcoef(a[ok], b[ok])[0, 1] > corr
    assert np.median(np.abs(a[ok] - b[ok]) / np.abs(b[ok])) < median_rel


def test_horizontal_derivatives_close_to_v062(pair):
    # Compared away from the domain edges and the top/bottom layers, where v0.6.2
    # was wrong by construction (zero padded differences; halved boundary layers).
    # The remaining differences (median ~6-9 % on the steep, stratified window) come
    # from the new second-order vertical stencil, which is exact for fields quadratic
    # in depth where v0.6.2's scheme is not (see test_calculus).
    name, ds, can, gold = pair
    interior = ("s_rho", "s_w", "xi_rho", "xi_u", "eta_rho", "eta_v")
    _close(xroms.ddxi(can.temp, ds, hcoord="rho", scoord="s_rho"), gold["ddxi_temp_rho"], interior)
    _close(xroms.ddeta(can.temp, ds, hcoord="rho", scoord="s_rho"), gold["ddeta_temp_rho"], interior)
    _close(xroms.convergence(can.u, can.v, ds), gold["convergence"], interior)
