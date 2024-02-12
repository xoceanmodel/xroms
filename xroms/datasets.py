"""
Load sample data.
"""

import pooch
import xarray as xr


CLOVER = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("xroms"),
    # The remote data is on Github
    # base_url="https://github.com/xoceanmodel/xroms/raw/{version_dev}/data/",
    base_url="https://github.com/xoceanmodel/xroms/raw/main/xroms/data/",
    # version=version,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry={
        "ROMS_example_full_grid.nc": None,
    },
)


def fetch_ROMS_example_full_grid():
    """
    Load the ROMS_example_full_grid sample data as an xarray Dataset.
    """
    # The file will be downloaded automatically the first time this is run
    # returns the file path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = CLOVER.fetch("ROMS_example_full_grid.nc")
    # The "fetch" method returns the full path to the downloaded data file.
    # All we need to do now is load it with our standard Python tools.
    data = xr.open_dataset(fname, engine="netcdf4", chunks={})
    return data
