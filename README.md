# `xroms`

[![Build Status](https://img.shields.io/github/actions/workflow/status/xoceanmodel/xroms/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/xoceanmodel/xroms/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoceanmodel/xroms.svg?style=for-the-badge)](https://codecov.io/gh/xoceanmodel/xroms)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/xroms/latest.svg?style=for-the-badge)](https://xroms.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/actions/workflow/status/xoceanmodel/xroms/pre-commit.yml?branch=main&label=Code%20Style&style=for-the-badge)](https://github.com/xoceanmodel/xroms/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xroms.svg?style=for-the-badge)](https://anaconda.org/conda-forge/xroms)
[![Python Package Index](https://img.shields.io/pypi/v/xroms.svg?style=for-the-badge)](https://pypi.org/project/xroms)

[![DOI](https://zenodo.org/badge/265067025.svg?style=for-the-badge)](https://zenodo.org/badge/latestdoi/265067025)

`xroms` analyses output of the ROMS family of ocean models (Rutgers ROMS, UCLA ROMS, CROCO and REMORA) with
[xarray](https://docs.xarray.dev). It knows the staggered C-grid and the terrain-following vertical coordinate.

There is no setup step and nothing is stored. Open the output with xarray and call xroms; each call computes what it
needs from the data in hand, lazily under dask, so results always match the data you pass, however you subset, select
or edit it.

```python
import xarray as xr
import xroms

ds = xr.open_dataset("ocean_his.nc", chunks={})
ds.xroms.speed                                  # on rho points, in the Dataset's own naming
ds.xroms.ddz("salt")                            # vertical derivative, on w levels
xroms.zslice(ds.temp, [-10, -50], ds)           # temperature at 10 m and 50 m below mean sea level
```

xroms can:
* compute depths at any grid position, relative to mean sea level, the moving free surface or the seabed, plus layer
  thicknesses and grid lengths, areas and volumes;
* take derivatives at constant depth along xi, eta and z, accounting for the curvilinear grid and the s coordinate;
* calculate derived quantities:
  * horizontal speed, kinetic energy and eddy kinetic energy;
  * geostrophic velocities and vertical shear;
  * vertical vorticity, horizontal divergence and convergence, and Ertel potential vorticity;
  * density (ROMS' equation of state or TEOS-10), potential density and buoyancy;
  * $N^2$, $M^2$ and the mixed layer depth;
* move variables between grid positions, and compute grid-weighted sums and means, depth averages, and surface and
  bottom layers;
* interpolate to fixed depths, onto any other field (density surfaces, latitudes), and to lon/lat points (with xESMF);
* subset with the staggered grids kept consistent, select the nearest grid point, rotate velocities to east/north, and
  convert longitude conventions;
* decode UCLA ROMS time and merge output with a separate grid file.

Everything stays lazy and keeps your dask chunks. Coming from xroms 0.6? The
[migration guide](https://xroms.readthedocs.io/en/latest/migration.html) shows what changed.

## Installation

From PyPI:

```
pip install xroms
```

From conda-forge:

```
conda install -c conda-forge xroms
```

Optional features come with extras: `teos10` (gsw, for the TEOS-10 equation of state), `geodesic` (pyproj, for the
ellipsoidal nearest-point search) and `examples` (pooch and netCDF4, for the example data), or `all` for these three,
e.g. `pip install "xroms[all]"`. `xroms.interpll` needs xESMF, which is best installed from conda-forge
(`conda install -c conda-forge xesmf`).

### Development

```
git clone https://github.com/xoceanmodel/xroms.git
cd xroms
mamba env create -f environment.yml
mamba activate xroms
pip install -e . --no-deps
pytest
```

Without conda, `pip install -e ".[dev]"` installs xroms with the test and optional dependencies (except xESMF).
