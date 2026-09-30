"""xroms: stateless analysis of ROMS-family ocean model output with xarray.

Open output with xarray, then call xroms functions or the ``ds.xroms`` accessor.
Nothing is precomputed or stored: depths, metrics and derivatives are derived at
call time from the data you pass, so subsetting or editing never leaves stale
results behind. Rutgers and UCLA ROMS, CROCO and REMORA layouts are detected
automatically.
"""

from importlib.metadata import PackageNotFoundError, version

from . import conventions, datasets, metrics, vertical
from .conventions import (
    VerticalParams,
    add_cf_attrs,
    canonicalize,
    decode_time,
    rename_like,
    rho0,
    sigma_levels,
    stretching,
    vertical_params,
)
from .derived import (
    EKE,
    KE,
    convergence,
    dudz,
    dvdz,
    ertel,
    relative_vorticity,
    speed,
    uv_geostrophic,
    vertical_shear,
)
from .interp import interpll, isoslice, make_regridder, zslice
from .metrics import dA, dV, dx, dy, mask_at, nominal_resolution
from .roms_seawater import M2, N2, buoyancy, density, mld, potential_density
from .utilities import (
    argsel2d,
    ddeta,
    ddxi,
    ddz,
    gridmean,
    gridsum,
    hgrad,
    order,
    sel2d,
    subset,
    to_grid,
    to_psi,
    to_rho,
    to_s_rho,
    to_s_w,
    to_u,
    to_v,
    trim,
    xisoslice,
)
from .vector import earth_to_grid, grid_to_earth, rotate_vectors
from .vertical import bottom, compute_depth, depth_average, depth_band_weights, dz, surface, z
from .xroms import grid_interp, open_mfnetcdf, open_netcdf, open_zarr, roms_dataset

from . import accessor  # noqa: E402  (registers ds.xroms / da.xroms)


try:
    __version__ = version("xroms")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "unknown"
