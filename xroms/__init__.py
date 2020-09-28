try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

import xroms.interp  # order of reading in `from scipy.spatial import Delaunay` matters
from .xroms import (roms_dataset, open_netcdf, open_mfnetcdf, open_zarr)
from .roms_seawater import (density, buoyancy, potential_density, N2,
                            M2, mld)
from .derived import (relative_vorticity, KE, speed,
                     ertel, uv_geostrophic, EKE,
                     dudz, dvdz,
                     vertical_shear)
from .utilities import (hgrad, to_rho, to_psi, to_u, to_v,
                        to_s_w, to_s_rho,
                        to_grid, ddz, ddxi, ddeta,
                        xisoslice, sel2d, argsel2d,
                        gridmean, gridsum)
# from .basic_calcs import (gridmean, gridsum, groupbytime, downsampletime)
# # from .interp import setup, ll2xe, calc_zslices, interp
import xroms.accessor

