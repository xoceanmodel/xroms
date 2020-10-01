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

You can install this locally and so that it is editable (`-e`), and with the required packages:

    git clone git@github.com:hetland/xroms.git
    cd xroms
    pip install -r requirements.txt -e .

Or:

    pip install git+git://github.com/hetland/xroms
    
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
