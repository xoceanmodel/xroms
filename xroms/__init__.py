"""Initialize xroms."""

from pkg_resources import DistributionNotFound, get_distribution

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
from .xroms import open_mfnetcdf, open_netcdf, open_zarr, roms_dataset


try:
    __version__ = get_distribution("xroms").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

# try:
#     from ._version import __version__
# except ImportError:
#     __version__ = "unknown"

# to manage whether xesmf is installed or not
try:
    import xesmf as xe

    XESMF_AVAILABLE = True
except ImportError:  # pragma: no cover
    XESMF_AVAILABLE = False  # pragma: no cover
