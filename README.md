# `xroms`

`xroms` contains functions for commonly used scripts for working with ROMS output in xarray.

There are functions to...
* help read in model output with automatically-calculated z coordinates
* calculate many derived variables with correct grid metrics in one line including:
  * horizontal speed
  * kinetic energy
  * eddy kinetic energy
  * vertical shear
  * vertical vorticity
  * Ertel potential vorticity
  * density as calculated in ROMS
  * potential density
  * buoyancy
  * N^2 (buoyancy frequency/vertical buoyancy gradient)
  * M^2 (horizontal buoyancy gradient)
* useful functions including:
  * derivatives in all dimensions, accounting for curvilinear grids and sigma layers
  * grid metrics (i.e., grid lengths, areas, and volumes)
  * easily change horizontal and vertical grids using `xgcm` grid objects
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
  * wraps `cf-xarray` to generalize coordinate and dimension calling.


## Installation

You can create an environment for this package with conda:

    conda create --name XROMS python=3.8 --file requirements-conda.txt

Then you can install additional required packages with xroms so that it is editable (`-e`):

    git clone git@github.com:hetland/xroms.git
    cd xroms
    pip install -r requirements-pip.txt -e .

Or:

    pip install git+git://github.com/hetland/xroms

If you already have an environment you'd like to use or just want to install in your regular Python, you can install with conda, then with pip:

    conda install --file requirements-conda.txt
    git clone git@github.com:hetland/xroms.git
    cd xroms
    pip install -r requirements-pip.txt -e .

Additionally installing [bottleneck](https://github.com/pydata/bottleneck/) is supposed to improve the speed of `numpy`-based calculations.


## Quick Start

After installation, read in model output with one of three load methods:
 * `xroms.open_netcdf(filename)`: if model output is in a single netcdf file or at a single thredds address;
 * `xroms.open_mfnetcdf(filenames)`: if model output is available in multiple accessible local netcdf files;
 * `xroms.open_zarr(locations)`: if model output is avilable in multiple accessible zarr directories.
More information about reading in model output is available in Jupyter notebook `examples/io.pynb`.

Other common tasks to do with model output using `xroms` as well as other packages are demonstrated in additional Jupyter notebooks:
 * select_data
 * calc
 * interpolation
 * plotting
