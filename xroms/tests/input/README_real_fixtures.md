# Fixtures from real UCLA-ROMS output

`grid.nc` and `ocean_his_000?.nc` in this directory (and the zarr copies of the latter)
are synthetic, Rutgers-style files made by `make_files.py`. The three files described
here are different: they come from real UCLA-ROMS files and from roms-tools, so that xroms
can be tested against the layout those really produce. They are made by
`make_real_fixtures.py` (see "Regenerating").

| File | What it is | Size |
| --- | --- | --- |
| `ucla_rst.nc` | UCLA-ROMS restart file, cut to `ocean_time, zeta, ubar, vbar, u, v, temp, salt` and to 10 x 8 rho points | 433,623 bytes (424 KB) |
| `ucla_grd.nc` | The matching UCLA grid file (Easy Grid), all variables, cut to the same 10 x 8 rho points | 25,169 bytes (25 KB) |
| `romstools_grid.nc` | A tiny grid made with `roms_tools.Grid`, with land and ocean | 37,032 bytes (36 KB) |

## Provenance

### UCLA restart and grid

Both come from the roms-tools test data,
<https://github.com/CWorthy-ocean/roms-tools-test-data>, which roms-tools' own tests also
use. The local clone they were read from was at commit `89ea11de`.

- `eastpac25km_rst.19980106000000.nc`: 8,128,612 bytes, last changed in that repository
  by commit `1a468b5` (2025-01-17), SHA-256
  `8f56d72bd8daf72eb736cc6705f93f478f4ad0ae4a95e98c4c9393a38e032f4c`.
- `epac25km_grd.nc`: 34,481 bytes, last changed by commit `1ec12b5` (2025-01-17), SHA-256
  `ec26c69cda4c4e96abde5b7756c955a7e1074931ab5a0641f598b099778fb617`.

They are from the UCLA-ROMS `eastpac25km` configuration (25 km, East Pacific, with MARBL
biogeochemistry), restart of 1998-01-06. The original files hold only 15 x 10 horizontal
points, although the `Title` attribute of the grid still says `nx: 120 ny: 160`. The
fixtures are cut further, to 10 x 8 (see "What was kept").

### roms-tools grid

Made with `roms_tools.Grid` from a roms-tools checkout at commit
`4f7be0eb7821f40134daf4253f211769d0f366a1` (5.1.0 plus one commit), run in a Python 3.13
environment with xarray 2025.7.1, numpy 2.2.6 and netCDF4 1.7.4. The inputs are local
files from the same test-data repository, so no network access is needed: topography
`EMODnet_C2_coarse100.nc` (EMODnet bathymetry, SHA-256 `4202a6a5877de726...`) and
coastlines `GSHHS_l_L1.shp` with its `.shx`, `.dbf` and `.prj` (GSHHG).

Settings: `nx=8, ny=6, size_x=240, size_y=180` (km), `center_lon=-22.0, center_lat=64.2`,
`rot=20`, `N=10`, `theta_s=6.0, theta_b=1.5, hc=50.0`, `topography_source` EMOD and
`mask_shapefile` GSHHS. That is a 240 x 180 km domain off the southwest coast of
Iceland, rotated by 20 degrees, with the coast along its east side. `nx` and `ny` count
interior cells, so the grid has `eta_rho=8, xi_rho=10` (30 km spacing). `theta_s`,
`theta_b` and `hc` differ from the roms-tools defaults (5, 2, 300), which the UCLA
restart also uses, so a test cannot pass by falling back on those values.

Two attributes of the file need a comment. `roms_tools_version` reads
`4.0.0a3.dev6+g18150aeee.d20260702`, which is the stale version in the editable
install's metadata, not the commit above. `mask_shapefile` and `topography_source_path`
hold bare file names, because the script ran roms-tools in a scratch directory holding
only links to the inputs (so no home directory ends up in the file).

## What was kept

### The horizontal subset (UCLA files only)

The full 15 x 10 restart is 758 KB. That is over the 500 KB limit of the
`check-added-large-files` hook in this repository's pre-commit configuration (the hook
has no arguments there, so its default applies), and the real float64 fields compress by
only about 20%. Dropping variables, time records or vertical levels was ruled out (the
`Cs_r` and `Cs_w` attributes have 100 and 101 values, so the vertical cannot be cut), so
both UCLA files are cut horizontally instead.

Both keep the first 10 rho points in `eta_rho` (`0:10`) and the first 8 in `xi_rho`
(`0:8`) of the original 15 x 10. The u and v points follow: `xi_u` is `0:7` and `eta_v`
is `0:9` (u[i] lies between rho[i] and rho[i+1], and likewise v), so `xi_u = xi_rho - 1`
and `eta_v = eta_rho - 1` still hold and the restart and the grid still agree point for
point. Nothing else was cut.

### `ucla_rst.nc`

- Variables: `ocean_time, zeta, ubar, vbar, u, v, temp, salt` (8 of the 55 in the
  original). Dropped: `time_step` (and with it the `auxil` dimension), 5 `MARBL_*`
  diagnostics and 32 biogeochemical tracers (`PO4` ... `diazFe`), the time-stepping
  internals `DU_avg2`, `DV_avg2`, `DU_avg_bak` and `DV_avg_bak`, the boundary-layer
  depths `hbls` and `hbbl`, and `u_slow`, `v_slow` and `p_slow`.
- Both time records and all 100 `s_rho` levels. Horizontally `zeta`, `temp` and `salt` are
  10 x 8 (`eta_rho`, `xi_rho`), `u` and `ubar` are 10 x 7 (`eta_rho`, `xi_u`), and `v` and
  `vbar` are 9 x 8 (`eta_v`, `xi_rho`).
- All 35 global attributes, unchanged in name, order, dtype and value. Nothing was added.
- Variable attributes, dtypes (everything is float64) and `_FillValue` (NaN), chunking per
  time record, and the unlimited `time` dimension.

### `ucla_grd.nc`

- All 14 variables, including the char array `spherical` (on `one` and `string1`) and
  the length-1 `tra_lon, tra_lat, rotate, xy_flip`, and all 3 global attributes
  (`Title`, `Date`, `Type`), with dtypes and variable attributes unchanged. Only the
  `eta_rho` and `xi_rho` dimensions are cut, to 10 x 8 as in the restart.
- It is no longer a byte-for-byte copy of `epac25km_grd.nc`, so the SHA-256 above does
  not match. The data of every variable equal the corresponding slice of the original.

The netCDF-level check in the script compares dimensions, variables, all attributes (with
their dtypes) and the data of the subset (bytewise) with the original, for both files.

### `romstools_grid.nc`

The file exactly as `Grid.save()` writes it. It is not cut.

### Compression and size

Only the four large restart variables (`u, v, temp, salt`) are compressed, with zlib
level 4 and the shuffle filter, which does not change any dtype. Compressing smaller
variables makes files larger (the HDF5 chunk index costs more than zlib saves: 87 KB
instead of 36 KB for the roms-tools grid), so the grid files and the small restart
variables are not compressed.

`ucla_rst.nc` is 433,623 bytes. The limit of the pre-commit hook is 500 KB, that is
512,000 bytes (it rounds sizes up to whole KB), so the file is 78,377 bytes under it, and
it is also under 500,000 bytes. The script stops with an error if a fixture goes over the
hook's limit.

## Regenerating

From the repository root:

```
python xroms/tests/input/make_real_fixtures.py
```

This makes the two UCLA files, which need only numpy, xarray and netCDF4, and then the
roms-tools grid if `roms_tools` can be imported. Otherwise it says so and skips that
file, so run it with the Python of an environment in which roms_tools is installed to get
all three. `--only ucla` and `--only romstools` make just one group.

The sources are read from a local clone of the test-data repository: by default
`~/packages/roms-tools-test-data`, or set `--src-dir` or the environment variable
`ROMS_TOOLS_TEST_DATA`. `--out-dir` changes where the files go (default: this directory).
Nothing is downloaded.

The output is deterministic: two runs gave byte-identical files, and the UCLA files were
also byte-identical when made in two different Python environments. The script checks
what it wrote: both UCLA files against their sources at the netCDF level and through
xarray (with the `netcdf4` and `h5netcdf` engines), that the restart and grid sizes match,
that every file is under the pre-commit size limit, and that the roms-tools grid has the
variables, coordinates and attributes listed below.

## What these files look like

UCLA restart (`ucla_rst.nc`):

- There are no coordinate variables at all (`ds.coords` is empty): not `s_rho`,
  `lon_rho` or `time`. `ocean_time` is a plain data variable on `time`, float64, with
  `units = "second"` and `long_name = "Time since 1995/01/01"`. That is not
  a CF time unit, so xarray leaves it as a float (checked with xarray 2025.7.1 and
  2026.7.0, including with `decode_timedelta=True`). The values are 95125800 and
  95126400 s, that is 1998-01-05 23:50 and 1998-01-06 00:00: two time steps 600 s (`dt`)
  apart.
- The vertical coordinate exists only as global attributes: `theta_s=5.0`,
  `theta_b=2.0`, `hc=300.0`, `rho0=1027.4` (float64 scalars), and `Cs_r` (100 values) and
  `Cs_w` (101 values) as float64 arrays. There is no `Vtransform`, `Vstretching`, `s_rho`
  or `s_w`.
- `u` and `ubar` are on `(eta_rho, xi_u)` and `v` and `vbar` on `(eta_v, xi_rho)`. There
  is no `eta_u` or `xi_v` dimension. The sizes are `time=2`, `eta_rho=10`, `xi_rho=8`,
  `xi_u=7`, `eta_v=9` and `s_rho=100`.
- The region is all ocean: the fields have no NaNs and the grid's `mask_rho` is 1
  everywhere.

UCLA grid (`ucla_grd.nc`):

- Dimensions `one=1, string1=1, eta_rho=10, xi_rho=8`, again without coordinate
  variables. `eta_rho` and `xi_rho` match the restart.
- Variable attributes use `Long_name` with a capital L. `spherical` is a char array on
  `(one, string1)` holding `T` (xarray shows it as `(one,)` of `|S1`). The scalars
  `tra_lon, tra_lat, rotate, xy_flip` are arrays of length 1 on `one`, and `xy_flip`
  holds netCDF's default fill value, 9.97e36.
- It has `h` and `hraw`, `f`, `pm`, `pn`, `angle` and `lon_rho`/`lat_rho` (0 to 360
  convention, 230.8 to 233.2 E and 7.7 to 10.1 N), but no u/v-point longitudes,
  latitudes or masks. `h` is 4497 to 4674 m.

roms-tools grid (`romstools_grid.nc`):

- Dimensions `eta_rho=8, xi_rho=10, xi_u=9, eta_v=7, s_rho=10, s_w=11`, plus
  `eta_coarse=5, xi_coarse=6` for the coarsened variables and a length-1 `string1` for
  `spherical`. Note that this grid is 8 (eta) x 10 (xi), while the UCLA fixtures are
  10 x 8: they are separate grids, so do not mix them.
- `lon_*` and `lat_*` at rho, u and v points (and the coarse grid) are coordinates
  through the `coordinates` attributes. Longitudes are in 0 to 360 (334.6 to 341.3),
  and `straddle` is the string `"False"`.
- `mask_rho`, `mask_u` and `mask_v` are int32 with both values (42 of 80 rho points are
  water, the water is on the west side). `angle` is 0.31 to 0.39 rad and `h` is 34 to
  370 m.
- `sigma_r`, `Cs_r` (on `s_rho`) and `sigma_w`, `Cs_w` (on `s_w`) are float32 variables,
  not attributes, and `theta_s`, `theta_b` and `hc` are float32 global attributes. The
  other global attributes are `title`, `roms_tools_version`, `size_x`, `size_y` and `rot`
  (int64), `center_lon`, `center_lat`, `hmin`, `straddle`, `mask_shapefile`,
  `close_narrow_channels`, `topography_source_name` and `topography_source_path`.
- There is no `Vtransform`. roms-tools computes depth (positive down) as
  `-(zeta + (zeta + h) * (hc * sigma + h * Cs) / (hc + h))`
  (`roms_tools.vertical_coordinate.compute_depth`).
