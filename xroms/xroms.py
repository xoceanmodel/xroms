"""Pre-1.0 setup and opening functions, kept only to explain their replacements.

xroms 1.0 has no setup step: open model output with xarray and call xroms
directly; everything grid-related is computed on demand from the data you pass.
"""

_MIGRATION = (
    "xroms 1.0 has no setup step and no stored xgcm grid. Open output with xarray "
    "(xr.open_dataset / xr.open_mfdataset / xr.open_zarr) and call xroms directly, "
    "e.g. ds.xroms.ddxi('temp') or xroms.ddxi(ds.temp, ds). If the grid is in a separate "
    "file, merge it (xr.merge([ds, grid], compat='override')) or pass grid=. Depths and "
    "metrics that roms_dataset used to attach are now computed on demand: ds.xroms.z(), "
    "ds.xroms.dz(), ds.xroms.dA(), or ds.xroms.assign_z() to attach depth coordinates. "
    "See the migration guide in the documentation."
)


_FLAGS = (
    "Its flags map to: include_Z0=True -> zeta=0 on the depth functions, e.g. ds.xroms.z(zeta=0) or "
    "ds.xroms.assign_z(zeta=0) (depths at rest); include_cell_area -> ds.xroms.dA(); include_cell_volume -> "
    "ds.xroms.dV(); include_3D_metrics -> nothing to do (dz, dV and the 3-D derivatives are always computed "
    "on demand); add_verts and proj -> gone (plot cell-centre data with shading='nearest' or 'auto'). "
)


def roms_dataset(*args, **kwargs):
    """Removed in xroms 1.0; raises an error explaining the replacement."""
    raise RuntimeError("xroms.roms_dataset was removed in 1.0. " + _FLAGS + _MIGRATION)


def open_netcdf(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_dataset``."""
    raise RuntimeError("xroms.open_netcdf was removed in 1.0; use xr.open_dataset. " + _MIGRATION)


def open_mfnetcdf(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_mfdataset``."""
    raise RuntimeError(
        "xroms.open_mfnetcdf was removed in 1.0; use xr.open_mfdataset. Its v0.6.2 defaults were "
        'xr.open_mfdataset(files, data_vars="minimal", coords="minimal", compat="override"). ' + _MIGRATION
    )


def open_zarr(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_zarr``."""
    raise RuntimeError("xroms.open_zarr was removed in 1.0; use xr.open_zarr. " + _MIGRATION)


def grid_interp(*args, **kwargs):
    """Removed in xroms 1.0; use ``xroms.to_grid``."""
    raise RuntimeError("xroms.grid_interp was removed in 1.0; use xroms.to_grid(var, hcoord=..., scoord=...).")
