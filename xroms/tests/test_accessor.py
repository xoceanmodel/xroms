"""The stateless ``ds.xroms`` / ``da.xroms`` accessors."""

import inspect

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms import conventions as C
from xroms.tests import _synthetic as syn
from xroms.tests.conftest import chunked, merged


class TestStateless:
    def test_holds_only_a_reference(self, rutgers):
        assert rutgers.xroms._obj is rutgers
        assert set(vars(rutgers.xroms)) == {"_obj"}

    def test_never_writes_into_the_dataset(self, rutgers):
        before = rutgers.copy(deep=True)
        rutgers.xroms.speed, rutgers.xroms.ddxi("temp"), rutgers.xroms.z_rho, rutgers.xroms.vort
        xr.testing.assert_identical(rutgers, before)

    def test_sees_in_place_edits(self, rutgers):
        s1 = rutgers.xroms.speed.values.copy()
        rutgers["u"] = rutgers.u * 10
        rutgers["v"] = rutgers.v * 10
        np.testing.assert_allclose(rutgers.xroms.speed.values, 10 * s1)


class TestNamingAndCoords:
    def test_results_use_dataset_naming(self, rutgers):
        u_like = rutgers.xroms.to_grid("temp", hcoord="u")
        assert u_like.dims == rutgers.u.dims
        assert (rutgers.u + u_like).dims == rutgers.u.dims
        assert rutgers.xroms.ddxi("temp").dims == rutgers.u.dims
        assert rutgers.xroms.vort.dims == ("ocean_time", "s_rho", "eta_psi", "xi_psi")

    def test_pure_functions_are_canonical(self, rutgers):
        assert xroms.to_u(rutgers.temp).dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
        can = xroms.canonicalize(rutgers)
        assert (can.u + xroms.to_u(can.temp)).dims == can.u.dims

    def test_position_coords_attached(self, rutgers):
        out = rutgers.xroms.ddxi("temp")
        assert {"lon_u", "lat_u", "ocean_time", "s_rho"} <= set(out.coords)
        assert "lon_rho" not in out.coords

    def test_ucla_lonlat_come_along_once_they_are_coords(self, ucla):
        # UCLA's grid keeps lon/lat as data variables; as coords on the results they would not merge back into ds
        out, grid = ucla
        ds = xr.merge([out, grid.drop_vars("spherical")])
        assert not {"lon_rho", "lat_rho"} & set(ds.xroms.speed.coords)
        ds = ds.set_coords(["lon_rho", "lat_rho"])
        assert {"lon_rho", "lat_rho"} <= set(ds.xroms.speed.coords)
        ds["speed"] = ds.xroms.speed


class TestGridFacts:
    def test_z_and_metrics(self, rutgers):
        np.testing.assert_allclose(rutgers.xroms.z_rho.values, xroms.z(rutgers).values)
        assert rutgers.xroms.z_w.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        assert rutgers.xroms.z(hcoord="u").dims == rutgers.u.dims
        assert rutgers.xroms.dz(scoord="w").dims[1] == "s_w"
        assert rutgers.xroms.dx("u").dims == ("eta_u", "xi_u")
        assert rutgers.xroms.dA("psi").dims == ("eta_psi", "xi_psi")
        assert rutgers.xroms.dV().dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        assert rutgers.xroms.vertical_params.Vtransform == 2

    def test_assign_z_is_a_new_dataset(self, rutgers):
        withz = rutgers.xroms.assign_z()
        assert {"z_rho", "z_w"} <= set(withz.coords) and "z_rho" not in rutgers.coords
        assert "z_rho" in withz.temp.coords

    def test_xgcm_grid(self, rutgers):
        import xgcm

        assert isinstance(rutgers.xroms.xgcm_grid(), xgcm.Grid)


class TestCalculations:
    def test_methods_accept_names_or_arrays(self, rutgers):
        xr.testing.assert_allclose(rutgers.xroms.ddxi("temp"), rutgers.xroms.ddxi(rutgers.temp))

    def test_zslice_gridmean_depth_average(self, rutgers):
        assert rutgers.xroms.zslice("temp", [-5.0]).dims == ("ocean_time", "z", "eta_rho", "xi_rho")
        assert rutgers.xroms.gridmean("temp", ("X", "Y")).dims == ("ocean_time", "s_rho")
        assert rutgers.xroms.depth_average("temp").dims == ("ocean_time", "eta_rho", "xi_rho")
        assert rutgers.xroms.surface("temp").dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_separate_grid(self, ucla):
        out, grid = ucla
        merged_grid = xr.merge([out, grid.drop_vars("spherical")])
        a = out.xroms.ddxi("temp", grid=merged_grid)
        np.testing.assert_allclose(a.values, syn.TEMP_A, rtol=1e-9)

    def test_subset_and_selection(self, rutgers, remora):
        sub = rutgers.xroms.subset(X=slice(2, 8), halo=1)
        assert sub.attrs["xroms_halo"] == [1, 1, 0, 0]
        lon, lat = float(rutgers.lon_rho[3, 4]), float(rutgers.lat_rho[3, 4])
        assert rutgers.xroms.argsel2d(lon, lat) == (3, 4)
        assert rutgers.xroms.sel2d("temp", lon, lat).dims == ("ocean_time", "s_rho")
        x0, y0 = float(remora.x_rho[2, 5]), float(remora.y_rho[2, 5])
        assert remora.xroms.argsel2d(x0, y0) == (2, 5)

    def test_every_layout(self, layout):
        ds = merged(layout)
        np.testing.assert_allclose(ds.xroms.ddxi("temp").values, syn.TEMP_A, rtol=1e-9)
        assert ds.xroms.speed.dims[-2:] == ("eta_rho", "xi_rho")

    def test_chunked(self, rutgers):
        c = chunked(rutgers)
        out = c.xroms.convergence
        assert out.chunks is not None
        np.testing.assert_allclose(out.values, rutgers.xroms.convergence.values)


class TestPhysics:
    def test_speed_ke_eke(self, rutgers):
        s = rutgers.xroms.speed
        np.testing.assert_allclose(s.values, xroms.speed(rutgers.u, rutgers.v).values)
        np.testing.assert_allclose(rutgers.xroms.KE.values, 0.5 * 1025.0 * s.values**2)
        assert rutgers.xroms.ug.dims == ("ocean_time", "eta_u", "xi_u")
        assert rutgers.xroms.vg.dims == ("ocean_time", "eta_v", "xi_v")
        assert rutgers.xroms.EKE.dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_rotation(self):
        ds = syn.make_dataset("rutgers", angle=np.pi / 2)
        east, north = ds.xroms.eastnorth
        np.testing.assert_allclose(north.values, xroms.to_rho(ds.u.fillna(0)).values, atol=1e-12)
        np.testing.assert_allclose(east.values, -xroms.to_rho(ds.v.fillna(0)).values, atol=1e-12)
        rot = ds.xroms.east_rotated(90, reference="compass", isradians=False, name="along")
        assert rot.name == "along"

    def test_u_v_from_earth_components(self, rutgers):
        east, north = rutgers.xroms.eastnorth
        only_earth = rutgers.drop_vars(["u", "v"]).assign(u_eastward=east, v_northward=north)
        assert only_earth.xroms.u.dims == rutgers.u.dims
        assert only_earth.xroms.find_horizontal_velocities() == ("u_eastward", "v_northward")

    def test_shear_vorticity_convergence(self, uniform):
        assert uniform.xroms.dudz.dims == ("ocean_time", "s_w", "eta_u", "xi_u")
        assert uniform.xroms.vertical_shear.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        np.testing.assert_allclose(uniform.xroms.vort.values, 0.0, atol=1e-12)
        np.testing.assert_allclose(uniform.xroms.convergence.values, syn.U_A + syn.V_A, rtol=1e-9)
        assert uniform.xroms.convergence_norm.dims == ("ocean_time", "eta_rho", "xi_rho")

    def test_density_family(self, rutgers):
        assert rutgers.xroms.rho.dims == rutgers.temp.dims
        assert rutgers.xroms.sig0.dims == rutgers.temp.dims
        assert rutgers.xroms.N2.dims == ("ocean_time", "s_w", "eta_rho", "xi_rho")
        assert rutgers.xroms.M2.dims == rutgers.temp.dims
        assert rutgers.xroms.ertel.dims == rutgers.temp.dims
        mld = rutgers.xroms.mld()
        assert mld.dims == ("ocean_time", "eta_rho", "xi_rho") and (mld >= 0).all()


class TestDataArrayAccessor:
    def test_grid_free_operations(self, rutgers):
        assert rutgers.u.xroms.to_rho().dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
        assert rutgers.temp.xroms.to_grid("psi", "w").dims == ("ocean_time", "s_w", "eta_v", "xi_u")
        assert rutgers.temp.transpose("xi_rho", ...).xroms.order().dims[0] == "ocean_time"

    def test_selection_uses_own_coords(self, rutgers):
        lon, lat = float(rutgers.lon_u[2, 3]), float(rutgers.lat_u[2, 3])
        assert rutgers.u.xroms.argsel2d(lon, lat) == (2, 3)
        assert rutgers.u.xroms.sel2d(lon, lat).dims == ("ocean_time", "s_rho")

    def test_isoslice(self, rutgers):
        z = xroms.z(rutgers)
        out = C.canonicalize(rutgers).temp.xroms.isoslice([-5.0], z)
        assert "z_rho" in out.dims


# --- a grid kept apart from the output -----------------------------------------------------


def _no_compute(dsk, keys, **kwargs):
    raise AssertionError("dask computed while a lazy result was being built")


def _split(ds):
    """``(data, grid)``: the grid variables (with their lon/lat or x/y) apart from the output."""
    names = [n for n in ds.variables if n in ("h", "pm", "pn", "angle", "f") or n.startswith(("mask_", "lon_", "lat_", "x_", "y_"))]
    return ds.drop_vars(names), ds[names]


@pytest.fixture(params=["ucla", "ucla_romstools", "rutgers", "croco", "remora"])
def split(request):
    """``(data, grid, merged)`` for each way of keeping the grid apart from the output."""
    if request.param.startswith("ucla"):
        romstools = request.param == "ucla_romstools"
        out, grid = syn.make_dataset("ucla", romstools_grid=romstools)
        return out, grid, merged("ucla", romstools_grid=romstools)
    full = syn.make_dataset(request.param)
    return (*_split(full), full)


def assert_same(got, want):
    """Same canonical dims in the same order and equal values (tuples elementwise; coords not compared).

    Alias names are compared canonically: which names a result uses is tested on its own,
    and a Dataset without psi variables has no psi names to give.
    """
    if isinstance(got, tuple):
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            assert_same(g, w)
        return
    got, want = C.canonicalize(got), C.canonicalize(want)
    assert got.dims == want.dims, (got.dims, want.dims)
    np.testing.assert_allclose(got.values, want.values, rtol=1e-10, atol=1e-14, equal_nan=True)


def _assigned(acc, **kw):
    out = acc.assign_z(**kw)
    return out["z_rho"], out["z_w"]


def _xgcm_products(acc, **kw):
    """What an xgcm grid with horizontal and vertical metrics gives on temp."""
    g = acc.xgcm_grid(vertical_metrics=True, **kw)
    temp = C.canonicalize(acc._obj)["temp"]
    return g.derivative(temp, "X"), g.average(temp, ["X", "Y"]), g.integrate(temp, "Z")


#: one call of every accessor method that takes ``grid=``
GRID_CALLS = {
    "z": lambda acc, **kw: acc.z(hcoord="u", scoord="s_w", **kw),
    "dz": lambda acc, **kw: acc.dz(scoord="s_w", **kw),
    "dx": lambda acc, **kw: acc.dx("u", **kw),
    "dy": lambda acc, **kw: acc.dy("v", **kw),
    "dA": lambda acc, **kw: acc.dA("psi", **kw),
    "dV": lambda acc, **kw: acc.dV(**kw),
    "ddxi": lambda acc, **kw: acc.ddxi("temp", **kw),
    "ddeta": lambda acc, **kw: acc.ddeta("temp", **kw),
    "ddz": lambda acc, **kw: acc.ddz("temp", **kw),
    "hgrad": lambda acc, **kw: acc.hgrad("temp", **kw),
    "gridsum": lambda acc, **kw: acc.gridsum("temp", "Z", **kw),
    "gridmean": lambda acc, **kw: acc.gridmean("temp", ("X", "Y"), **kw),
    "depth_average": lambda acc, **kw: acc.depth_average("temp", shallow=0, deep=10, reference="surface", **kw),
    "mld": lambda acc, **kw: acc.mld(threshold=0.03, **kw),
    "zslice": lambda acc, **kw: acc.zslice("temp", [-5.0], **kw),
    "assign_z": _assigned,
    "xgcm_grid": _xgcm_products,
}


def test_reductions_keep_the_datasets_own_names(rutgers):
    """With an eta or xi dim summed out, the one left is still named as the variable's (xi_v, eta_u)."""
    assert rutgers.xroms.gridmean("v", ("Z", "Y")).dims == ("ocean_time", "xi_v")
    assert rutgers.xroms.gridsum("u", ("Z", "X")).dims == ("ocean_time", "eta_u")
    assert rutgers.xroms.gridmean("temp", "Y").dims == ("ocean_time", "s_rho", "xi_rho")
    xr.testing.assert_allclose(
        rutgers.xroms.gridmean("v", ("Z", "Y")).reset_coords(drop=True).rename(xi_v="xi_rho"),
        xroms.gridmean(rutgers.v, rutgers, ("Z", "Y")).reset_coords(drop=True),
    )


def test_grid_calls_cover_every_method_with_grid():
    methods = inspect.getmembers(xroms.accessor.xromsDatasetAccessor, inspect.isfunction)
    with_grid = {name for name, f in methods if not name.startswith("_") and "grid" in inspect.signature(f).parameters}
    assert with_grid == set(GRID_CALLS)


@pytest.mark.parametrize("name", sorted(GRID_CALLS))
def test_separate_grid_matches_the_merged_dataset(split, name):
    data, grid, full = split
    assert_same(GRID_CALLS[name](data.xroms, grid=grid), GRID_CALLS[name](full.xroms))


class TestSeparateGrid:
    def test_zeta_is_the_datasets_not_zero(self, split):
        data, grid, full = split
        moving = data.xroms.z(grid=grid)
        flat = data.xroms.z(zeta=0, grid=grid)
        assert float(abs(moving - flat).max()) > 0.05
        np.testing.assert_allclose(moving.values, full.xroms.z().values, rtol=1e-12)

    def test_the_grids_own_zeta_wins(self, split):
        data, grid, _ = split
        still = grid.assign(zeta=xr.zeros_like(data.zeta.reset_coords(drop=True)))
        got, at_rest = data.xroms.z(grid=still), data.xroms.z(zeta=0, grid=grid)
        np.testing.assert_allclose(got.values, np.broadcast_to(at_rest.values, got.shape))

    def test_zeta_that_does_not_fit_the_grid_raises(self, split):
        data, grid, _ = split
        with pytest.raises(ValueError, match="zeta") as err:
            data.xroms.z(grid=xroms.subset(grid, X=slice(2, 9)))
        assert "merge" in str(err.value) and "same way" in str(err.value)

    def test_index_labels_must_agree(self, rutgers):
        labels = {d: 100 + np.arange(n) for d, n in rutgers.sizes.items() if d.startswith(("xi_", "eta_"))}
        full = rutgers.assign_coords(labels)
        data, grid = _split(full)
        assert_same(data.xroms.ddxi("temp", grid=grid), full.xroms.ddxi("temp"))
        shifted = grid.assign_coords(xi_rho=200 + np.arange(grid.sizes["xi_rho"]))
        with pytest.raises(ValueError, match="different xi_rho points"):
            data.xroms.z(grid=shifted)

    def test_subsetting_both_the_same_way_works(self, split):
        data, grid, full = split

        def sub(ds):
            return xroms.subset(ds, X=slice(2, 9), Y=slice(1, 7))

        assert_same(sub(data).xroms.ddxi("temp", grid=sub(grid)), sub(full).xroms.ddxi("temp"))

    def test_results_use_the_datasets_naming(self, split):
        data, grid, _ = split
        assert data.xroms.ddxi("temp", grid=grid).dims == data.u.dims
        assert data.xroms.ddeta("temp", grid=grid).dims == data.v.dims

    def test_single_time(self, split):
        data, grid, full = split
        tdim = "ocean_time" if "ocean_time" in data.dims else "time"
        assert_same(data.isel({tdim: 1}).xroms.ddxi("temp", grid=grid), full.isel({tdim: 1}).xroms.ddxi("temp"))

    def test_inputs_are_left_alone(self, split):
        data, grid, _ = split
        data0, grid0 = data.copy(deep=True), grid.copy(deep=True)
        for call in GRID_CALLS.values():
            call(data.xroms, grid=grid)
        xr.testing.assert_identical(data, data0)
        xr.testing.assert_identical(grid, grid0)

    def test_lazy_under_dask(self, split):
        dask = pytest.importorskip("dask")
        data, grid, full = split
        names = sorted(set(GRID_CALLS) - {"xgcm_grid"})  # xgcm itself refuses chunked core dims
        with dask.config.set(scheduler=_no_compute):
            lazy = {n: GRID_CALLS[n](chunked(data).xroms, grid=chunked(grid)) for n in names}
        for name, got in lazy.items():
            assert_same(got, GRID_CALLS[name](full.xroms))
            arrays = got if isinstance(got, tuple) else (got,)
            assert all(a.chunks is not None for a in arrays), name

    def test_not_a_dataset_is_left_to_the_function(self, rutgers):
        grid = rutgers.xroms.xgcm_grid()
        with pytest.raises(TypeError, match="instead of an xgcm Grid"):
            rutgers.xroms.ddxi("temp", grid=grid)
        with pytest.raises(TypeError, match="grid must be an xarray Dataset"):
            rutgers.xroms.xgcm_grid(grid=grid)


# --- the xgcm Grid ---------------------------------------------------------------------------


class TestXgcmGrid:
    def test_axes_and_horizontal_metrics_on_every_layout(self, layout):
        g = merged(layout).xroms.xgcm_grid()
        assert set(g.axes) == {"X", "Y", "Z"}
        assert set(g._metrics) == {frozenset("X"), frozenset("Y"), frozenset("XY")}

    def test_derivative_and_average(self, layout):
        ds = merged(layout)
        c = C.canonicalize(ds)
        g = ds.xroms.xgcm_grid()
        np.testing.assert_allclose(g.derivative(c.temp, "X").values, np.diff(c.temp.values, axis=-1) / xroms.dx(ds, "u").values)
        np.testing.assert_allclose(g.derivative(c.temp, "Y").values, np.diff(c.temp.values, axis=-2) / xroms.dy(ds, "v").values)
        for name in ("temp", "u", "v"):
            assert_same(g.average(c[name], ["X", "Y"]), xroms.gridmean(c[name], ds, ("X", "Y")))

    def test_vertical_metrics_are_opt_in(self, rutgers):
        assert frozenset("Z") not in rutgers.xroms.xgcm_grid()._metrics
        assert frozenset("Z") in rutgers.xroms.xgcm_grid(vertical_metrics=True)._metrics

    def test_integrating_over_z_matches_gridsum(self, layout):
        ds = merged(layout)
        c = C.canonicalize(ds)
        g = ds.xroms.xgcm_grid(vertical_metrics=True)
        for name in ("temp", "u", "v"):
            assert_same(g.integrate(c[name], "Z"), xroms.gridsum(c[name], ds, "Z"))
        on_w = xroms.to_s_w(c.temp)
        assert_same(g.integrate(on_w, "Z"), xroms.gridsum(on_w, ds, "Z"))

    def test_vertical_metrics_follow_zeta(self, rutgers):
        temp = C.canonicalize(rutgers).temp
        at_rest = rutgers.xroms.xgcm_grid(vertical_metrics=True, zeta=0).integrate(temp, "Z")
        assert_same(at_rest, xroms.gridsum(temp, rutgers, "Z", zeta=0))
        assert not np.allclose(at_rest.values, rutgers.xroms.xgcm_grid(vertical_metrics=True).integrate(temp, "Z").values)

    def test_ucla_output_gets_a_z_axis_and_is_not_changed(self):
        ds = merged("ucla")
        assert "s_w" not in ds.dims
        before = ds.copy(deep=True)
        g = ds.xroms.xgcm_grid()
        assert set(g.axes) == {"X", "Y", "Z"}
        xr.testing.assert_identical(ds, before)
        assert g.interp(C.canonicalize(ds).temp, "Z").sizes["s_w"] == ds.sizes["s_rho"] + 1

    def test_without_pm_and_pn_there_are_no_metrics(self, ucla):
        out, _ = ucla
        g = out.xroms.xgcm_grid()
        assert set(g.axes) == {"X", "Y", "Z"} and g._metrics == {}

    def test_a_grid_of_another_size_raises(self, split):
        data, grid, _ = split
        with pytest.raises(ValueError, match="different xi_rho points"):
            data.xroms.xgcm_grid(grid=xroms.subset(grid, X=slice(2, 9)))

    def test_the_datasets_index_labels_are_the_grids(self, rutgers):
        labels = {d: 100 + np.arange(n) for d, n in rutgers.sizes.items() if d.startswith(("xi_", "eta_"))}
        sub = xroms.subset(rutgers.assign_coords(labels), X=slice(2, 9), Y=slice(1, 7))
        c = C.canonicalize(sub)
        g = sub.xroms.xgcm_grid(vertical_metrics=True)
        assert_same(g.average(c.temp, ["X", "Y"]), xroms.gridmean(c.temp, sub, ("X", "Y")))
        assert_same(g.integrate(c.temp, "Z"), xroms.gridsum(c.temp, sub, "Z"))
        assert (g.derivative(c.temp, "X").xi_u.values == c.xi_u.values).all()

    def test_lazy_under_dask(self, split):
        dask = pytest.importorskip("dask")
        data, grid, full = split
        lazy = chunked(data)
        with dask.config.set(scheduler=_no_compute):
            g = lazy.xroms.xgcm_grid(vertical_metrics=True, grid=chunked(grid))
            total = g.integrate(C.canonicalize(lazy).temp, "Z")
        assert total.chunks is not None
        assert_same(total, xroms.gridsum(C.canonicalize(full).temp, full, "Z"))


# --- da.xroms names ----------------------------------------------------------------------------


class TestDataArrayNaming:
    def test_alias_inputs_give_alias_results(self, rutgers):
        u, v = rutgers.u, rutgers.v
        assert u.xroms.to_rho().dims == rutgers.temp.dims
        assert u.xroms.to_v().dims == v.dims
        assert v.xroms.to_u().dims == u.dims
        assert u.xroms.to_psi().dims == ("ocean_time", "s_rho", "eta_psi", "xi_psi")
        assert u.xroms.to_s_w().dims == ("ocean_time", "s_w", "eta_u", "xi_u")
        assert v.xroms.to_grid("psi", "w").dims == ("ocean_time", "s_w", "eta_psi", "xi_psi")

    def test_results_combine_with_the_datasets_own_variables(self, rutgers):
        assert (rutgers.u + rutgers.v.xroms.to_u()).dims == rutgers.u.dims
        assert (rutgers.v + rutgers.u.xroms.to_v()).dims == rutgers.v.dims

    def test_isoslice_keeps_the_position_names(self, rutgers):
        for var, dims in (("u", ("eta_u", "xi_u")), ("v", ("eta_v", "xi_v"))):
            out = rutgers[var].xroms.isoslice([-5.0], rutgers.xroms.z(hcoord=var))
            assert out.dims[-2:] == dims

    def test_canonical_inputs_give_canonical_results(self, rutgers):
        can = C.canonicalize(rutgers)
        assert can.u.xroms.to_v().dims == ("ocean_time", "s_rho", "eta_v", "xi_rho")
        assert can.u.xroms.to_psi().dims == ("ocean_time", "s_rho", "eta_v", "xi_u")
        assert can.u.xroms.isoslice([-5.0], xroms.z(can, hcoord="u")).dims[-2:] == ("eta_rho", "xi_u")

    def test_a_rho_point_array_cannot_tell_and_the_dataset_route_can(self, rutgers):
        assert rutgers.temp.xroms.to_u().dims == ("ocean_time", "s_rho", "eta_rho", "xi_u")
        assert rutgers.xroms.to_grid("temp", "u").dims == rutgers.u.dims
