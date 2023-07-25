# `xroms`

[![Build Status](https://img.shields.io/github/actions/workflow/status/xoceanmodel/xroms/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/xoceanmodel/xroms/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoceanmodel/xroms.svg?style=for-the-badge)](https://codecov.io/gh/xoceanmodel/xroms)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/xroms/latest.svg?style=for-the-badge)](https://xroms.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/actions/workflow/status/xoceanmodel/xroms/pre-commit.yml?branch-main&label=Code%20Style&style=for-the-badge)](https://github.com/xoceanmodel/xroms/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xroms.svg?style=for-the-badge)](https://anaconda.org/conda-forge/xroms)
[![Python Package Index](https://img.shields.io/pypi/v/xroms.svg?style=for-the-badge)](https://pypi.org/project/xroms)

[![DOI](https://zenodo.org/badge/265067025.svg?style=for-the-badge)](https://zenodo.org/badge/latestdoi/265067025)

`xroms` contains functions for commonly used scripts for working with ROMS output in xarray.

There are functions to...
* help read in model output with automatically-calculated z coordinates
* calculate many derived variables with correct grid metrics in one line including:
  * horizontal speed
  * kinetic energy
  * eddy kinetic energy
  * vertical shear
  * vertical vorticity
  * horizontal divergence
  * normalized surface divergence
  * Ertel potential vorticity
  * density as calculated in ROMS
  * potential density
  * buoyancy
  * $N^2$ (buoyancy frequency/vertical buoyancy gradient)
  * $M^2$ (horizontal buoyancy gradient)
* useful functions including:
  * derivatives in all dimensions, accounting for curvilinear grids and sigma layers
  * grid metrics (i.e., grid lengths, areas, and volumes)
  * subset horizontal grid such that the staggered grids are consistent
  * easily change horizontal and vertical grids using `xgcm` grid objects
  * easily reorder to dimensional convention
  * slice along a fixed value
  * wrapper for interpolation in longitude/latitude and for fixed depths
  * mixed-layer depth
* Demonstrations:
  * selecting data in many different ways
  * interpolation
  * changing time sampling
  * calculating climatologies
  * various calculations
* provide/track attributes and coordinates through functions
  * wraps [`cf-xarray`](https://cf-xarray.readthedocs.io/en/latest/) to generalize coordinate and dimension calling.
* ability to automatically choose colormaps for plotting with `xarray`
  * wraps `xcmocean` for this


## Installation

You need to have `conda` installed for these installation instructions. You'll have best results if you use the channel `conda-forge`, which you can prioritize with `conda config --add channels conda-forge --force`.

### Install, the easy way

PyPI:

  ```
  pip install xroms
  ```

conda-forge:

  ```
  mamba install -c conda-forge xroms
  ```

### Create environment if needed

As a first step, you can create an environment for this package with conda if you want. If you do this, you'll need to git clone the package first as below. Note that `mamba` and `conda` can be used interchangeably, but `mamba` is faster for installation.

    mamba env create -f environment.yml

You can choose to install with conda the optional dependencies for full functionality:

    conda install --file requirements-opt.txt

and to install optional dependency `xcmocean`:

    pip install git+git://github.com/pangeo-data/xcmocean

Then choose one of the following to install `xroms` from GitHub:

1. Clone `xroms` into a particular directory then install so that it is editable (`-e`)

    ```
    git clone git@github.com:xoceanmodel/xroms.git
    cd xroms
    pip install -e .
    ```

1. Directly install `xroms` from github

    ```
    pip install git+git://github.com/xoceanmodel/xroms
    ```
