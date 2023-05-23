"""
Load sample data.
"""

import xarray as xr
import pooch

# from . import version  # The version string of your project


CLOVER = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("xroms"),
    # The remote data is on Github
    base_url=f"https://github.com/xoceanmodel/xroms/raw/{version_dev}/data/",
    # version=version,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry={
        "ROMS_example_full_grid.nc": "sha256:3376da101d0fddec10a9aa2aecd8cd8e57b2e0df2235c279292cfcf3cf75fbdc",
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
    data = xr.open_dataset(fname)
    return data
