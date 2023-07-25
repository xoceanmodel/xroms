"""Initialize xroms."""

from importlib.metadata import PackageNotFoundError, version

import xroms.accessor
import xroms.datasets

from .derived import (
    EKE,
    KE,
    divergence,
    dudz,
    dvdz,
    ertel,
    relative_vorticity,
    speed,
    uv_geostrophic,
    vertical_shear,
)
from .interp import interpll, isoslice
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
    xisoslice,
)
from .vector import rotate_vectors
from .xroms import grid_interp, open_mfnetcdf, open_netcdf, open_zarr, roms_dataset


try:
    __version__ = version("xroms")
except PackageNotFoundError:
    # package is not installed
    pass

# to manage whether xesmf is installed or not
try:
    import xesmf as xe

    XESMF_AVAILABLE = True
except ImportError:  # pragma: no cover
    XESMF_AVAILABLE = False  # pragma: no cover
