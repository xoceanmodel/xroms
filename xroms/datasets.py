"""Example data (downloaded on first use; needs the optional ``pooch`` package)."""

_REGISTRY = {"ROMS_example_full_grid.nc": None}
_BASE_URL = "https://github.com/xoceanmodel/xroms/raw/main/xroms/data/"


def _fetcher():
    try:
        import pooch
    except ImportError:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("example data needs pooch: pip install 'xroms[examples]'") from None
    return pooch.create(path=pooch.os_cache("xroms"), base_url=_BASE_URL, registry=_REGISTRY)


def fetch_ROMS_example_full_grid():
    """Load the ``ROMS_example_full_grid`` sample data (Rutgers ROMS) as a Dataset."""
    import xarray as xr

    fname = _fetcher().fetch("ROMS_example_full_grid.nc")
    return xr.open_dataset(fname, chunks={})
