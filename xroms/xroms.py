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


def roms_dataset(*args, **kwargs):
    """Removed in xroms 1.0; raises an error explaining the replacement."""
    raise RuntimeError("xroms.roms_dataset was removed in 1.0. " + _MIGRATION)


def open_netcdf(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_dataset``."""
    raise RuntimeError("xroms.open_netcdf was removed in 1.0; use xr.open_dataset. " + _MIGRATION)


def open_mfnetcdf(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_mfdataset``."""
    raise RuntimeError("xroms.open_mfnetcdf was removed in 1.0; use xr.open_mfdataset. " + _MIGRATION)


def open_zarr(*args, **kwargs):
    """Removed in xroms 1.0; use ``xarray.open_zarr``."""
    raise RuntimeError("xroms.open_zarr was removed in 1.0; use xr.open_zarr. " + _MIGRATION)


def grid_interp(*args, **kwargs):
    """Removed in xroms 1.0; use ``xroms.to_grid``."""
    raise RuntimeError("xroms.grid_interp was removed in 1.0; use xroms.to_grid(var, hcoord=..., scoord=...).")
