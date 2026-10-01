"""Parity with roms-tools' own calculations (dev-only: runs where roms_tools is importable).

roms-tools is adopting these xroms functions in place of its copies; these tests
pin down where the two agree exactly and where xroms differs on purpose.
"""

import types

import numpy as np
import pytest
import xarray as xr

import xroms
from xroms.tests.conftest import merged
from xroms.tests import _synthetic as syn

pytest.importorskip("roms_tools")
from roms_tools.datasets.roms_dataset import ROMSDataset  # noqa: E402
from roms_tools.setup.mask import add_velocity_masks  # noqa: E402
from roms_tools.setup.utils import compute_barotropic_velocity, compute_mld, compute_potential_density  # noqa: E402
from roms_tools.utils import (  # noqa: E402
    interpolate_from_rho_to_u,
    interpolate_from_rho_to_v,
    interpolate_from_u_to_rho,
    interpolate_from_v_to_rho,
)


@pytest.fixture
def with_land():
    return syn.make_dataset("rutgers", land=True)


class TestMixedLayerDepth:
    @pytest.mark.parametrize("reference_depth", [0.0, 5.0, 10.0])
    @pytest.mark.parametrize("threshold", [0.01, 0.03])
    def test_transform_method_is_compute_mld(self, with_land, reference_depth, threshold):
        # compute_mld fills a column without a crossing with its deepest level: xroms without a grid
        z = xroms.z(with_land)
        sig0 = xroms.potential_density(with_land.temp, with_land.salt)
        expected = compute_mld(sig0, -z, "s_rho", reference_depth=reference_depth, threshold=threshold)
        for method, rtol in (("transform", 0), ("interp", 1e-13)):
            md = xroms.mld(sig0, z=z, reference_depth=reference_depth, threshold=threshold, method=method)
            # columns shallower than the reference depth differ on purpose: roms-tools falls back to
            # the shallowest level as the reference, xroms finds no crossing below the reference depth
            deep = (with_land.h > reference_depth).broadcast_like(md)
            np.testing.assert_allclose(md.where(deep).values, expected.transpose(*md.dims).where(deep).values, rtol=rtol, atol=0)

    def test_teos10_potential_density_differs_from_the_practical_salinity_shortcut(self, with_land):
        # compute_potential_density treats practical salinity as absolute salinity and potential
        # temperature as conservative temperature; xroms converts both, so sigma0 is larger by
        # about the absolute-salinity anomaly (SA - SP ~ 0.165 g/kg, ~0.13 kg/m^3)
        shortcut = compute_potential_density(with_land.temp, with_land.salt)
        proper = xroms.potential_density(with_land.temp, with_land.salt, eos="teos10", grid=with_land) - 1000
        offset = (proper - shortcut.transpose(*proper.dims)).values
        offset = offset[np.isfinite(offset)]
        assert 0.1 < offset.mean() < 0.16 and offset.std() < 0.02


class TestStaggering:
    def test_rho_to_u_and_v(self):
        ds = merged("ucla")
        h = ds.h.reset_coords(drop=True)
        np.testing.assert_array_equal(xroms.to_u(h).values, interpolate_from_rho_to_u(h).values)
        np.testing.assert_array_equal(xroms.to_v(h).values, interpolate_from_rho_to_v(h).values)

    def test_u_and_v_to_rho_fill_the_edges(self):
        # roms-tools pads the rho points beyond the staggered ones with NaN: hboundary="fill"
        ds = merged("ucla")
        u, v = ds.u.isel(time=0, s_rho=0).reset_coords(drop=True), ds.v.isel(time=0, s_rho=0).reset_coords(drop=True)
        rt_u = interpolate_from_u_to_rho(u).transpose("eta_rho", "xi_rho")
        rt_v = interpolate_from_v_to_rho(v).transpose("eta_rho", "xi_rho")
        np.testing.assert_array_equal(xroms.to_rho(u, hboundary="fill").values, rt_u.values)
        np.testing.assert_array_equal(xroms.to_rho(v, hboundary="fill").values, rt_v.values)

    def test_velocity_masks(self, with_land):
        masks = add_velocity_masks(xr.Dataset({"mask_rho": xroms.canonicalize(with_land.mask_rho).reset_coords(drop=True)}))
        np.testing.assert_array_equal(xroms.mask_at(with_land.mask_rho, "u").values, masks.mask_u.values)
        np.testing.assert_array_equal(xroms.mask_at(with_land.mask_rho, "v").values, masks.mask_v.values)


class TestBarotropicVelocity:
    def test_depth_average_is_compute_barotropic_velocity(self, rutgers):
        ds = xroms.canonicalize(rutgers)
        interface_depth = -xroms.z(ds, hcoord="u", scoord="s_w")
        # roms-tools' data has no s-coordinate labels (with them, its renamed diff would not align)
        bare = [da.drop_vars(list(da.coords)) for da in (ds.u, interface_depth)]
        expected = compute_barotropic_velocity(*bare)
        ubar = xroms.depth_average(ds.u, ds)
        np.testing.assert_allclose(ubar.values, expected.transpose(*ubar.dims).values, rtol=1e-12)


class TestTimeDecoding:
    def test_decode_time_is_roms_tools_absolute_time(self):
        out, _ = syn.make_dataset("ucla")
        dummy = types.SimpleNamespace(model_reference_date=None)
        ROMSDataset._infer_model_reference_date_from_metadata(dummy, out)
        expected = ROMSDataset._add_absolute_time(dummy, out)
        decoded = xroms.decode_time(out)
        np.testing.assert_array_equal(decoded.indexes["time"].values, expected.indexes["time"].values.astype("datetime64[ns]"))


class TestVerticalRegrid:
    @pytest.mark.parametrize("mask_edges", [False, True])
    def test_isoslice_onto_model_depths_is_vertical_regrid(self, rutgers, mask_edges):
        # a z-level source (a climatology, say) regridded onto the model's s-levels, as roms-tools'
        # initial conditions do: target depths vary in space, the source depths do not
        from roms_tools.regrid import VerticalRegrid

        depth = xr.DataArray([0.0, 5.0, 15.0, 30.0, 60.0, 120.0], dims="depth")
        h = xroms.canonicalize(rutgers.h).reset_coords(drop=True)
        source = (20.0 - 0.1 * depth + 0.01 * h).rename("temp").transpose("depth", "eta_rho", "xi_rho")
        target = -xroms.z(rutgers, zeta=0).reset_coords(drop=True)
        expected = VerticalRegrid(source.to_dataset(), "depth").apply(source, depth, target, mask_edges=mask_edges)
        out = xroms.isoslice(source, target, depth.broadcast_like(source), dim="depth", new_dim="s_rho", mask_edges=mask_edges)
        np.testing.assert_allclose(out.values, expected.transpose(*out.dims).values, rtol=1e-13, atol=0)
