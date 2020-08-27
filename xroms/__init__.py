try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .xroms import (roms_dataset,
                    open_netcdf,
                    open_zarr,
                    hgrad,
                    relative_vorticity)#,
#                     ertel)
from .roms_seawater import density, buoyancy
from .utilities import (to_rho, to_psi, to_s_w, to_s_rho,
                        to_grid, ddz, ddxi, ddeta,
                        xisoslice, sel2d, argsel2d,
                       build_indexer, id_grid)
# from .interp import setup, ll2xe, calc_zslices, interp
import xroms.interp
import xroms.accessor
