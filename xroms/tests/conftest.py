"""Shared fixtures: synthetic ROMS-family datasets with analytic fields."""

import contextlib
from pathlib import Path

import dask
import pytest

from xroms.tests import _synthetic as syn


INPUT = Path(__file__).parent / "input"
GOLDEN = Path(__file__).parent / "golden"

LAYOUTS = ["rutgers", "ucla", "croco", "remora"]

# chunk every dim that can be chunked, including horizontal ones (issue #16/#77)
CHUNKS = {
    "ocean_time": 1,
    "time": 1,
    "s_rho": 3,
    "s_w": 4,
    "eta_rho": 4,
    "eta_u": 4,
    "eta_v": 4,
    "eta_psi": 4,
    "xi_rho": 5,
    "xi_u": 5,
    "xi_v": 5,
    "xi_psi": 5,
}


def chunked(ds):
    """Chunk ``ds`` along every dim it has, splitting horizontal dims too."""
    return ds.chunk({d: c for d, c in CHUNKS.items() if d in ds.dims})


class Computes:
    """A dask scheduler that refuses to compute, and counts the attempts."""

    def __init__(self):
        self.count = 0

    def __call__(self, dsk, keys, **kwargs):
        self.count += 1
        raise RuntimeError("dask data was computed while the result was being built")


@contextlib.contextmanager
def no_computes():
    counter = Computes()
    with dask.config.set(scheduler=counter):
        yield counter
    assert counter.count == 0, f"{counter.count} dask computes"


def merged(layout, **kwargs):
    """Synthetic dataset with the grid merged in, whatever the layout."""
    out = syn.make_dataset(layout, **kwargs)
    if isinstance(out, tuple):
        data, grid = out
        return data.merge(grid.drop_vars("spherical", errors="ignore"), compat="override")
    return out


@pytest.fixture(params=LAYOUTS)
def layout(request):
    """Every ROMS-family layout."""
    return request.param


@pytest.fixture(params=[1, 2])
def vtransform(request):
    """Both ROMS vertical transforms."""
    return request.param


@pytest.fixture
def rutgers():
    """Rutgers-style dataset (aliased stagger dims, Vtransform 2, sloping h)."""
    return syn.make_dataset("rutgers")


@pytest.fixture
def ucla():
    """UCLA-style ``(output, grid)`` pair: attrs-only s-params, no coords."""
    return syn.make_dataset("ucla")


@pytest.fixture
def ucla_romstools():
    """UCLA-style output with a roms-tools-style grid (sigma/Cs variables)."""
    return syn.make_dataset("ucla", romstools_grid=True)


@pytest.fixture
def croco():
    """CROCO-style dataset."""
    return syn.make_dataset("croco")


@pytest.fixture
def remora():
    """REMORA-style dataset (Cartesian, SGRID, time-varying masks)."""
    return syn.make_dataset("remora")


@pytest.fixture
def uniform():
    """Rutgers-style dataset on a uniform grid (for u/v analytic derivatives)."""
    return syn.make_dataset("rutgers", uniform=True)


@pytest.fixture
def with_land():
    """Rutgers-style dataset with a land patch and NaN over land."""
    return syn.make_dataset("rutgers", land=True)
