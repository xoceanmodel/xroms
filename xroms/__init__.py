try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .xroms import (roms_dataset,
                    open_netcdf,
                    open_zarr,
                    hgrad,
                    relative_vorticity, KE, speed,
                    ertel, uv_geostrophic, EKE)
from .roms_seawater import (density, buoyancy, sig0, N2,
                            M2, mld)
from .utilities import (to_rho, to_psi, to_u, to_v,
                        to_s_w, to_s_rho,
                        to_grid, ddz, ddxi, ddeta,
                        dudz, dvdz,
                        xisoslice, sel2d, argsel2d,
                       build_indexer, id_grid)
from .basic_calcs import (gridmean, gridsum,
                          mean, sum, max, min,
                          std, var, median,
                          groupbytime, downsampletime)
# from .interp import setup, ll2xe, calc_zslices, interp
import xroms.interp
import xroms.accessor

