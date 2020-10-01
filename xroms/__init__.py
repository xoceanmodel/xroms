try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

import xroms.accessor

from .derived import (
    EKE,
    KE,
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
    sel2d,
    to_grid,
    to_psi,
    to_rho,
    to_s_rho,
    to_s_w,
    to_u,
    to_v,
    xisoslice,
)
from .xroms import open_mfnetcdf, open_netcdf, open_zarr, roms_dataset
