"""The real fixtures of ``tests/input`` (see its ``README_real_fixtures.md``), opened with plain xarray.

* ``ucla`` -- a real UCLA-ROMS restart and its grid file: no coordinate variables at all,
  vertical parameters only in global attributes, ``ocean_time`` a float of seconds, 100 levels,
  4500 m deep.
* ``romstools`` -- a grid written by roms-tools (float32 ``sigma``/``Cs`` variables, int32 masks, land,
  a rotated 30 km grid, 34-370 m deep). It has no model output, so smooth analytic fields are put
  on it: temperature linear in depth (NaN over land), constant salinity, and velocities with shear.
* ``rutgers`` -- the Rutgers-style history files opened as one with ``xr.open_mfdataset``
  (flat 100 m, 3 levels, ``hc`` 0, parameters read from the first record).

Each goes through the functions the library is for, checking what must hold of any sensible
ocean: finite values over water, depths between ``-h`` and the free surface, positive layer
thicknesses that add up to the water column, stable stratification, magnitudes of the right
order, and the dims and naming that come out.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _sweep as S
from xroms.tests.conftest import INPUT


pytestmark = pytest.mark.skipif(not (INPUT / "ucla_rst.nc").exists(), reason="real fixtures missing")

G = 9.81


@dataclass
class Case:
    name: str
    ds: xr.Dataset
    N: int  # rho levels
    hc: float
    dx: Tuple[float, float]  # bounds of the grid spacing along xi and eta [m]
    dy: Tuple[float, float]
    grad_max: float  # bound on |d temp / dx| at constant depth [K/m]
    vort_max: float  # bound on |relative vorticity| and |convergence| [1/s]

    @property
    def can(self):
        return C.canonicalize(self.ds)

    @property
    def zeta(self):
        return self.can["zeta"] if "zeta" in self.can else 0.0

    def water(self, pos="rho"):
        return xroms.mask_at(self.can["mask_rho"], pos) == 1

    def supported(self):
        """Rho points away from the domain edge whose u and v neighbours are all water: what a divergence there needs.

        (A point on the edge copies its neighbour's derivative, which needs that neighbour's neighbours.)
        """
        mu, mv = (xroms.to_rho(xroms.mask_at(self.can["mask_rho"], p).astype(float)) for p in ("u", "v"))
        ok = ((mu == 1.0) & (mv == 1.0)).copy()
        ok[{"eta_rho": [0, -1]}] = False
        ok[{"xi_rho": [0, -1]}] = False
        return ok


def _ucla():
    rst = xr.open_dataset(INPUT / "ucla_rst.nc")
    grd = xr.open_dataset(INPUT / "ucla_grd.nc")
    return xr.merge([rst, grd.drop_vars("spherical")], compat="override")


def _romstools():
    grid = xr.open_dataset(INPUT / "romstools_grid.nc")
    z = xroms.z(grid)  # at rest: the file has no free surface
    water, water_u, water_v = (xroms.mask_at(grid.mask_rho, p) == 1 for p in ("rho", "u", "v"))
    index = lambda dim: xr.DataArray(np.arange(grid.sizes[dim]), dims=dim)  # noqa: E731
    shear = lambda zz: 1.0 + zz / 400.0  # noqa: E731 - weaker with depth
    fields = {
        "temp": (15.0 + 0.04 * z).where(water),
        "salt": (35.0 + 0.0 * z).where(water),
        "u": (0.1 * np.sin(index("xi_u") / 3.0) * np.cos(index("eta_rho") / 4.0) * shear(xroms.z(grid, hcoord="u"))).where(water_u),
        "v": (0.1 * np.cos(index("xi_rho") / 3.0) * np.sin(index("eta_v") / 2.0) * shear(xroms.z(grid, hcoord="v"))).where(water_v),
    }
    units = {"temp": "Celsius", "salt": "PSU", "u": "m/s", "v": "m/s"}
    return grid.assign({name: xroms.order(field).assign_attrs(units=units[name]) for name, field in fields.items()})


def _rutgers():
    files = [INPUT / "ocean_his_0001.nc", INPUT / "ocean_his_0002.nc"]
    return xr.open_mfdataset(files, data_vars="minimal", coords="minimal", compat="override")


CASES = {
    "ucla": lambda: Case("ucla", _ucla(), 100, 300.0, (24e3, 25e3), (24e3, 25e3), 1e-3, 1e-4),
    "romstools": lambda: Case("romstools", _romstools(), 10, 50.0, (29.9e3, 30.1e3), (29.9e3, 30.1e3), 1e-9, 1e-3),
    "rutgers": lambda: Case("rutgers", _rutgers(), 3, 0.0, (13e3, 15e3), (17e3, 20e3), 1e-5, 1e-5),
}


@pytest.fixture(params=list(CASES), scope="module")
def case(request):
    c = CASES[request.param]()
    yield c
    c.ds.close()


@pytest.fixture(scope="module")
def romstools():
    ds = _romstools()
    yield ds
    ds.close()


def _like(a, b):
    """The values of ``a`` broadcast against, and ordered like, ``b``."""
    a = a if isinstance(a, xr.DataArray) else xr.DataArray(a)
    return a.broadcast_like(b).transpose(*b.dims).values


def _finite_over_water(out, case, *, mask=None, interior_w=False):
    """Every point of ``out`` that is water at its position (or in ``mask``) is finite."""
    out = C.canonicalize(out)
    if interior_w and "s_w" in out.dims:
        out = out.isel(s_w=slice(1, -1))  # the end interfaces have a neighbour on one side only
    water = case.water(C.hposition(out) or "rho") if mask is None else mask
    bad = water & ~np.isfinite(out)
    assert not bool(bad.any()), f"{int(bad.sum())} water points are not finite"


# --- the vertical coordinate -------------------------------------------------------------------------


def test_vertical_params(case):
    p = xroms.vertical_params(case.ds)
    assert (p.Vtransform, p.hc) == (2, case.hc)
    assert p.sigma_r.sizes["s_rho"] == case.N and p.Cs_r.sizes["s_rho"] == case.N
    assert p.sigma_w.sizes["s_w"] == case.N + 1 and p.Cs_w.sizes["s_w"] == case.N + 1
    for name in ("Cs_r", "Cs_w", "sigma_r", "sigma_w"):
        values = getattr(p, name).values
        assert values.dtype.kind == "f" and np.all(np.diff(values) > 0) and values.min() >= -1.0 and values.max() <= 0.0, name
    # every file here uses evenly spaced sigma, and a stretching that reaches the bottom and the surface
    np.testing.assert_allclose(p.sigma_w.values, np.linspace(-1, 0, case.N + 1), atol=1e-6)
    assert p.Cs_w.values[0] == pytest.approx(-1.0) and p.Cs_w.values[-1] == pytest.approx(0.0)


def test_ucla_parameters_are_the_attributes():
    ds = _ucla()
    p = xroms.vertical_params(ds)
    np.testing.assert_array_equal(p.Cs_r.values, ds.attrs["Cs_r"])
    np.testing.assert_array_equal(p.Cs_w.values, ds.attrs["Cs_w"])
    assert (ds.attrs["theta_s"], ds.attrs["theta_b"], ds.attrs["hc"]) == (5.0, 2.0, 300.0)
    assert xroms.rho0(ds) == pytest.approx(1027.4)
    assert "Vtransform" not in ds.variables and "Vtransform" not in ds.attrs  # UCLA only has Vtransform 2


def test_romstools_parameters_are_the_files_variables(romstools):
    p = xroms.vertical_params(romstools)
    np.testing.assert_allclose(p.Cs_r.values, romstools.Cs_r.values)
    np.testing.assert_allclose(p.sigma_w.values, romstools.sigma_w.values)
    assert p.hc == 50.0 and romstools.attrs["theta_s"] == 6.0 and romstools.attrs["theta_b"] == 1.5
    # roms-tools' depth formula at rest (z = zeta + (zeta + h) * (hc * sigma + h * Cs) / (hc + h), with zeta = 0)
    h, sigma, cs = romstools.h.values, romstools.sigma_r.values[:, None, None], romstools.Cs_r.values[:, None, None]
    np.testing.assert_allclose(xroms.z(romstools).values, h * (50.0 * sigma + h * cs) / (50.0 + h), rtol=1e-12)


def test_rutgers_parameters_come_from_the_first_record():
    ds = _rutgers()
    p = xroms.vertical_params(ds)
    assert p.hc == 0.0 and p.Vtransform == 2
    np.testing.assert_allclose(p.Cs_r.values, np.linspace(-0.975, -0.025, 3))
    np.testing.assert_allclose(p.Cs_w.values, np.linspace(-1, 0, 4))


@pytest.mark.parametrize("hcoord", ["rho", "u", "v", "psi"])
def test_depths_lie_between_the_bottom_and_the_free_surface(case, hcoord):
    z_rho, z_w = xroms.z(case.ds, hcoord=hcoord), xroms.z(case.ds, hcoord=hcoord, scoord="s_w")
    h, zeta = case.can["h"], case.zeta
    if hcoord != "rho":
        move = getattr(xroms, f"to_{hcoord}")
        h, zeta = move(h), (move(zeta) if isinstance(zeta, xr.DataArray) else zeta)
    eta, xi = C.CANONICAL[hcoord]
    assert z_rho.dims[-3:] == ("s_rho", eta, xi) and z_w.dims[-3:] == ("s_w", eta, xi)
    assert z_rho.sizes["s_rho"] == case.N and z_w.sizes["s_w"] == case.N + 1
    assert bool(np.isfinite(z_rho).all()) and bool(np.isfinite(z_w).all())
    # layers are ordered bottom to top and sit in the water column
    assert bool((z_rho.diff("s_rho") > 0).all()) and bool((z_w.diff("s_w") > 0).all())
    assert bool((z_rho > -h).all()) and bool((z_rho < zeta).all())
    np.testing.assert_allclose(z_w.isel(s_w=0).values, _like(-h, z_w.isel(s_w=0)), rtol=1e-9)
    np.testing.assert_allclose(z_w.isel(s_w=-1).values, _like(zeta, z_w.isel(s_w=-1)), atol=1e-9)
    # every layer centre sits between its interfaces
    below, above = z_w.isel(s_w=slice(None, -1)).values, z_w.isel(s_w=slice(1, None)).values
    assert bool((below < z_rho.values).all()) and bool((z_rho.values < above).all())


def test_layer_thicknesses_add_up_to_the_water_column(case):
    column = case.can["h"] + case.zeta
    for scoord, n in (("s_rho", case.N), ("s_w", case.N + 1)):
        dz = xroms.dz(case.ds, scoord=scoord)
        assert dz.sizes[scoord] == n and bool((dz > 0).all()), scoord
        total = dz.sum(scoord)
        np.testing.assert_allclose(total.values, _like(column, total), rtol=1e-9)
    dz = xroms.dz(case.ds)
    assert float(dz.min()) > 1e-3 and bool((dz < case.can["h"] + 1.0).all())


# --- horizontal metrics -----------------------------------------------------------------------------


def test_metrics_have_the_grids_spacing(case):
    dx, dy, dA = xroms.dx(case.ds), xroms.dy(case.ds), xroms.dA(case.ds)
    assert dx.dims == dy.dims == dA.dims == ("eta_rho", "xi_rho")
    assert case.dx[0] < float(dx.min()) and float(dx.max()) < case.dx[1]
    assert case.dy[0] < float(dy.min()) and float(dy.max()) < case.dy[1]
    np.testing.assert_allclose(dA.values, (dx * dy).values, rtol=1e-12)
    assert 0.5 * (case.dx[0] + case.dy[0]) < xroms.nominal_resolution(case.ds) < 0.5 * (case.dx[1] + case.dy[1])
    for pos in ("u", "v", "psi"):
        for metric in (xroms.dx, xroms.dy, xroms.dA):
            out = metric(case.ds, pos)
            assert out.dims == C.CANONICAL[pos] and bool((out > 0).all()), (metric.__name__, pos)


# --- derivatives ---------------------------------------------------------------------------------------


def test_derivatives_of_temperature(case):
    temp = case.can["temp"]
    ddxi, ddeta, ddz = xroms.ddxi(temp, case.ds), xroms.ddeta(temp, case.ds), xroms.ddz(temp, case.ds)
    lead = temp.dims[:-3]
    assert ddxi.dims == lead + ("s_rho", "eta_rho", "xi_u") and ddeta.dims == lead + ("s_rho", "eta_v", "xi_rho")
    assert ddz.dims == lead + ("s_w", "eta_rho", "xi_rho") and ddz.sizes["s_w"] == case.N + 1
    for out in (ddxi, ddeta, ddz):
        _finite_over_water(out, case)
    # a sensible gradient in K per metre, and temperature rises upward in these ocean columns
    assert float(abs(ddxi).max()) < case.grad_max and float(abs(ddeta).max()) < case.grad_max
    wet = ddz.where(case.water())
    assert float(wet.max()) < 1.0 and float(wet.mean()) > 0.0


def test_vorticity_convergence_and_shear(case):
    u, v = case.can["u"], case.can["v"]
    vort, conv = xroms.relative_vorticity(u, v, case.ds), xroms.convergence(u, v, case.ds)
    assert vort.dims == u.dims[:-2] + ("eta_v", "xi_u") and conv.dims == u.dims[:-2] + ("eta_rho", "xi_rho")
    _finite_over_water(vort, case)  # a psi point is water if its four rho points are
    _finite_over_water(conv, case, mask=case.supported())  # a rho point needs the u and v points around it
    assert float(abs(vort).max()) < case.vort_max and float(abs(conv).max()) < case.vort_max
    dudz, dvdz = xroms.dudz(u, case.ds), xroms.dvdz(v, case.ds)
    assert dudz.dims[-3:] == ("s_w", "eta_rho", "xi_u") and dvdz.dims[-3:] == ("s_w", "eta_v", "xi_rho")
    shear = xroms.vertical_shear(dudz, dvdz)
    assert shear.dims[-3:] == ("s_w", "eta_rho", "xi_rho")
    wet = shear.where(case.water())
    assert float(wet.min()) >= 0.0 and float(wet.max()) < 1.0  # a magnitude, in 1/s


# --- density, stratification, mixed layer ------------------------------------------------------------------------


def test_density_stratification_and_mixed_layer(case):
    temp, salt = case.can["temp"], case.can["salt"]
    water = case.water()
    rho = xroms.density(temp, salt, grid=case.ds)
    assert rho.dims == temp.dims
    _finite_over_water(rho, case)
    assert 1000.0 < float(rho.min()) and float(rho.max()) < 1060.0
    n2 = xroms.N2(rho, case.ds, xroms.rho0(case.ds))
    assert n2.dims[-3:] == ("s_w", "eta_rho", "xi_rho") and n2.sizes["s_w"] == case.N + 1
    _finite_over_water(n2, case, interior_w=True)
    # statically stable everywhere in the interior; the end interfaces have no value (nothing beyond them)
    interior = n2.isel(s_w=slice(1, -1))
    assert float(interior.min()) > 0.0 and float(interior.max()) < 0.1
    assert bool(n2.isel(s_w=0).isnull().all()) and bool(n2.isel(s_w=-1).isnull().all())
    mld = xroms.mld(xroms.potential_density(temp, salt), case.ds)
    assert mld.dims == temp.dims[:-3] + ("eta_rho", "xi_rho")
    _finite_over_water(mld, case)
    wet = mld.where(water)
    assert float(wet.min()) > 0.0 and bool((wet <= case.can["h"] + 1e-9).where(water, True).all())
    assert bool(mld.where(~water).isnull().all()), "columns without data (land) have no mixed layer depth"


def test_slices_and_depth_averages(case):
    temp, water = case.can["temp"], case.water()
    lo, hi = float(temp.min()), float(temp.max())
    sl = xroms.zslice(temp, [-10.0, -100.0], case.ds)
    assert sl.dims == temp.dims[:-3] + ("z", "eta_rho", "xi_rho") and list(sl.z.values) == [-10.0, -100.0]
    assert lo - 1e-9 <= float(sl.min()) and float(sl.max()) <= hi + 1e-9
    # a slice has a value wherever its depth lies between the column's lowest and highest rho levels
    z = xroms.z(case.ds)
    for k, depth in enumerate((10.0, 100.0)):
        inside = (z.min("s_rho") <= -depth) & (z.max("s_rho") >= -depth)
        _finite_over_water(sl.isel(z=k), case, mask=water & inside)
        assert bool(sl.isel(z=k).where(~inside).isnull().all()), f"{depth} m: a value outside the column's levels"
    avg = xroms.depth_average(temp, case.ds)
    assert avg.dims == temp.dims[:-3] + ("eta_rho", "xi_rho")
    _finite_over_water(avg, case)
    assert lo <= float(avg.where(water).min()) and float(avg.where(water).max()) <= hi
    # the upper layers are warmer than the whole column (temperature rises upward)
    upper = xroms.depth_average(temp, case.ds, deep=50.0, reference="surface")
    assert float(upper.where(water).mean()) >= float(avg.where(water).mean())
    # and the average is the thickness-weighted mean, worked out by hand from z
    dz = xroms.dz(case.ds)
    by_hand = (temp * dz).sum("s_rho") / dz.where(temp.notnull()).sum("s_rho")
    np.testing.assert_allclose(avg.where(water).values, by_hand.transpose(*avg.dims).where(water).values, rtol=1e-12, equal_nan=True)


# --- naming ---------------------------------------------------------------------------------------------------


def test_accessor_uses_the_files_own_names(case):
    ds = case.ds
    alias = C.convention(ds) == "rutgers"
    u_dims, v_dims, psi_dims = (("eta_u", "xi_u"), ("eta_v", "xi_v"), ("eta_psi", "xi_psi")) if alias else (("eta_rho", "xi_u"), ("eta_v", "xi_rho"), ("eta_v", "xi_u"))
    assert ds.xroms.ddxi("temp").dims[-2:] == u_dims and ds.xroms.ddeta("temp").dims[-2:] == v_dims
    assert ds.xroms.vort.dims[-2:] == psi_dims and ds.xroms.z(hcoord="u").dims[-2:] == u_dims
    assert ds.xroms.dA("psi").dims == psi_dims and ds.xroms.dx().dims == ("eta_rho", "xi_rho")
    # the same numbers as the pure functions
    np.testing.assert_allclose(ds.xroms.speed.values, xroms.speed(ds.u, ds.v).values, equal_nan=True)
    np.testing.assert_allclose(ds.xroms.z_w.values, xroms.z(ds, scoord="s_w").values)
    rho = xroms.density(ds.temp, ds.salt, grid=ds)
    np.testing.assert_allclose(ds.xroms.N2.values, xroms.N2(rho, ds, xroms.rho0(ds)).values, equal_nan=True)
    # the grid's longitudes and latitudes come along; the UCLA grid file keeps them as data variables, and they
    # come along once they are coordinates (as data variables they would not merge back into the Dataset)
    if case.name == "ucla":
        assert {"lon_rho", "lat_rho"} <= set(ds.data_vars) and not {"lon_rho", "lat_rho"} & set(ds.xroms.speed.coords)
        ds = ds.set_coords(["lon_rho", "lat_rho"])
    assert {"lon_rho", "lat_rho"} <= set(ds.xroms.speed.coords)
    assert {"lon_u", "lat_u"} <= set(ds.xroms.ddxi("temp").coords) or case.name == "ucla"  # the UCLA grid file has no u-point lon/lat


@pytest.mark.parametrize("acc", S.ACCESSOR, ids=S.ACC_IDS)
def test_every_accessor_member_runs_on_the_real_data(case, acc):
    if case.name == "romstools" and acc.name in ("ug", "vg", "EKE"):
        # the grid file has no free surface, and the geostrophic velocities are made of it
        with pytest.raises(KeyError, match="'zeta'"):
            acc.accessor(case.ds)
        return
    for got, want in zip(S.results(acc.accessor(case.ds)), S.results(acc.pure(case.ds))):
        got, want = C.canonicalize(got), C.canonicalize(want)
        assert set(got.dims) == set(want.dims)
        np.testing.assert_allclose(got.transpose(*want.dims).values, want.values, rtol=1e-12, equal_nan=True)


# --- a grid kept apart, and UCLA's time ---------------------------------------------------------------------------------


def test_a_grid_kept_apart_gives_the_same_results():
    rst, grd = xr.open_dataset(INPUT / "ucla_rst.nc"), xr.open_dataset(INPUT / "ucla_grd.nc")
    merged = _ucla()
    for name, call in {
        "z": lambda d, **kw: d.xroms.z(**kw),
        "z_w_psi": lambda d, **kw: d.xroms.z(hcoord="psi", scoord="s_w", **kw),
        "dz": lambda d, **kw: d.xroms.dz(**kw),
        "ddxi": lambda d, **kw: d.xroms.ddxi("temp", **kw),
        "ddz": lambda d, **kw: d.xroms.ddz("temp", **kw),
        "dA": lambda d, **kw: d.xroms.dA(**kw),
    }.items():
        np.testing.assert_allclose(call(rst, grid=grd).values, call(merged).values, rtol=1e-12, err_msg=name)
    # the same for the Rutgers history files and their grid file
    his, grid = _rutgers(), xr.open_dataset(INPUT / "grid.nc")
    np.testing.assert_allclose(his.xroms.ddxi("temp", grid=grid).values, his.xroms.ddxi("temp").values, rtol=1e-12)
    np.testing.assert_allclose(his.xroms.z(grid=grid).values, his.xroms.z().values, rtol=1e-12)


def test_ucla_time_has_to_be_decoded_to_select_a_time():
    ds = _ucla()
    assert "time" not in ds.coords and ds.ocean_time.attrs["units"] == "second"
    decoded = xroms.decode_time(ds)
    # the epoch is in the long_name ("Time since 1995/01/01"); the two records are one time step (dt) apart
    expected = np.array(["1998-01-05T23:50:00", "1998-01-06T00:00:00"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(decoded.time.values, expected)
    assert float(decoded.time.diff("time").dt.total_seconds()[0]) == ds.attrs["dt"]
    assert "time" not in ds.coords and ds.ocean_time.dtype == np.float64  # the input is not changed
    np.testing.assert_array_equal(decoded.ocean_time.values, ds.ocean_time.values)
    # a variable selected on its own cannot be matched with the grid's time-dependent zeta ...
    with pytest.raises(ValueError, match="decode_time"):
        xroms.ddz(ds.temp.isel(time=1), ds)
    # ... once decoded, it is matched through the scalar time label
    last = xroms.ddz(decoded.temp.isel(time=1), decoded)
    np.testing.assert_allclose(last.values, xroms.ddz(decoded.temp, decoded).isel(time=1).values, rtol=1e-12)
    assert last.dims == ("s_w", "eta_rho", "xi_rho") and last["time"].values == expected[1]
    np.testing.assert_allclose(xroms.z(decoded).values, xroms.z(ds).values)


# --- analytic fields on the roms-tools grid ----------------------------------------------------------------------------------


def test_romstools_linear_temperature_has_the_slope_and_no_horizontal_gradient(romstools):
    ds = romstools
    water = xroms.mask_at(ds.mask_rho, "rho") == 1
    ddz = xroms.ddz(ds.temp, ds).where(water)
    np.testing.assert_allclose(ddz.values[np.isfinite(ddz.values)], 0.04, rtol=1e-9)
    assert int(np.isfinite(ddz.values).sum()) == int(water.sum()) * (ds.sizes["s_rho"] + 1)
    # at constant depth a field of depth alone has no horizontal gradient, whatever the slope of the grid
    for out, pos in ((xroms.ddxi(ds.temp, ds), "u"), (xroms.ddeta(ds.temp, ds), "v")):
        wet = xroms.mask_at(ds.mask_rho, pos) == 1
        vals = out.where(wet).values
        assert np.isfinite(vals[:, wet.values]).all() and np.abs(vals[np.isfinite(vals)]).max() < 1e-9


def test_romstools_depth_average_and_slice_of_a_linear_profile(romstools):
    ds = romstools
    water = xroms.mask_at(ds.mask_rho, "rho") == 1
    h = ds.h
    # the mean over the column of a profile linear in z is its value at mid-depth, up to how far the
    # rho levels sit from the middle of their layers (the stretching is not linear)
    avg = xroms.depth_average(ds.temp, ds).where(water)
    np.testing.assert_allclose(avg.values[water.values], (15.0 - 0.04 * h / 2.0).where(water).values[water.values], rtol=2e-2)
    sl = xroms.zslice(ds.temp, [-10.0], ds).isel(z=0).where(water & (h > 30.0))
    np.testing.assert_allclose(sl.values[np.isfinite(sl.values)], 15.0 - 0.4, rtol=1e-12)
    assert int(np.isfinite(sl.values).sum()) == int((water & (h > 30.0)).sum())


def test_romstools_nothing_is_wrong_over_land(romstools):
    ds = romstools
    assert int((ds.mask_rho == 1).sum()) == 42 and ds.mask_rho.dtype == np.int32
    land = ds.mask_rho == 0
    assert bool(ds.temp.where(land).isnull().where(land, True).all())
    # masked velocities count as 0 in speed (documented), so the water next to land keeps a speed;
    # land itself, with no velocity around it, is NaN
    speed = xroms.speed(ds.u, ds.v)
    assert bool(speed.notnull().where(~land, True).all()) and bool(speed.isnull().where(land, True).all())
    # N2 follows its definition: -g / rho0 times the vertical density gradient
    rho = xroms.density(ds.temp, ds.salt, grid=ds)
    n2, drho = xroms.N2(rho, ds, 1025.0), xroms.ddz(rho, ds)
    inner = np.isfinite(n2.values)
    np.testing.assert_allclose(n2.values[inner], (-G / 1025.0 * drho).values[inner], rtol=1e-12)


def test_romstools_masks_at_u_and_v_points_are_the_files(romstools):
    # roms-tools writes int32 masks at u and v points; mask_at derives the same ones from mask_rho
    for pos in ("u", "v"):
        derived = xroms.mask_at(romstools.mask_rho, pos)
        assert derived.dtype == romstools.mask_rho.dtype == np.int32
        np.testing.assert_array_equal(derived.values, romstools[f"mask_{pos}"].values)


def test_romstools_depth_average_is_nan_over_land(romstools):
    land = romstools.mask_rho == 0
    avg = xroms.depth_average(romstools.temp, romstools)
    assert bool(avg.where(land).isnull().where(land, True).all())
