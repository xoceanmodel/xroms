import warnings

import numpy as np
import xarray as xr
import xgcm


try:
    import xesmf as xe
except ImportError:
    warnings.warn("xESMF is not installed, so `interpll` will not run.", ImportWarning)


def interpll(var, lons, lats, which="pairs"):
    """Interpolate var to lons/lats positions.

    Wraps xESMF to perform proper horizontal interpolation on non-flat Earth.

    Inputs
    ------
    var: DataArray
        Variable to operate on.
    lons: list, ndarray
        Longitudes to interpolate to. Will be flattened upon input.
    lats: list, ndarray
        Latitudes to interpolate to. Will be flattened upon input.
    which: str, optional
        Which type of interpolation to do:
        * "pairs": lons/lats as unstructured coordinate pairs
          (in xESMF language, LocStream).
        * "grid": 2D array of points with 1 dimension the lons and
          the other dimension the lats.

    Returns
    -------
    DataArray of var interpolated to lons/lats. Dimensionality will be the
    same as var except the Y and X dimensions will be 1 dimension called
    "locations" that lons.size if which=='pairs', or 2 dimensions called
    "lat" and "lon" if which=='grid' that are of lats.size and lons.size,
    respectively.

    Notes
    -----
    var cannot have chunks in the Y or X dimensions.

    cf-xarray should still be usable after calling this function.

    Example usage
    -------------
    To return 1D pairs of points, in this case 3 points:
    >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='pairs')
    To return 2D pairs of points, in this case a 3x3 array of points:
    >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='grid')
    """

    # rename coords for use with xESMF
    lonkey = [coord for coord in var.coords if "lon_" in coord][0]
    latkey = [coord for coord in var.coords if "lat_" in coord][0]
    var = var.rename({lonkey: "lon", latkey: "lat"})

    # make sure dimensions are in typical cf ordering (T, Z, Y, X)
    var = var.cf.transpose(
        *[dim for dim in ["T", "Z", "Y", "X"] if dim in var.cf.get_valid_keys()]
    )

    # force lons/lats to be 1D arrays
    lats = np.asarray(lats).flatten()
    lons = np.asarray(lons).flatten()

    # whether inputs are
    if which == "pairs":
        locstream_out = True
    elif which == "grid":
        locstream_out = False

    # set up for output
    varint = xr.Dataset({"lat": (["lat"], lats), "lon": (["lon"], lons)})

    # Calculate weights.
    regridder = xe.Regridder(var, varint, "bilinear", locstream_out=locstream_out)

    # Perform interpolation
    varint = regridder(var, keep_attrs=True)

    # check for presence of z coord with interpolated output
    zkey_varint = [
        coord for coord in varint.coords if "z_" in coord and "0" not in coord
    ]

    # get z coordinates to go with interpolated output if not available
    if zkey_varint == []:
        zkey = [coord for coord in var.coords if "z_" in coord and "0" not in coord][
            0
        ]  # str
        zint = regridder(var[zkey], keep_attrs=True)

        # add coords
        varint = varint.assign_coords({zkey: zint})

    # add attributes for cf-xarray
    if which == "pairs":
        varint["locations"] = (
            "locations",
            np.arange(varint.sizes["locations"]),
            {"axis": "X"},
        )
    elif which == "grid":
        varint["lon"].attrs["axis"] = "X"
        varint["lat"].attrs["axis"] = "Y"
    varint["lon"].attrs["standard_name"] = "longitude"
    varint["lat"].attrs["standard_name"] = "latitude"

    return varint


def isoslice(var, iso_values, grid=None, iso_array=None, axis="Z"):
    """Interpolate var to iso_values.

    This wraps `xgcm` `transform` function for slice interpolation,
    though `transform` has additional functionality.

    Inputs
    ------
    var: DataArray
        Variable to operate on.
    iso_values: list, ndarray
        Values to interpolate to. If calculating var at fixed depths,
        iso_values are the fixed depths, which should be negative if
        below mean sea level. If input as array, should be 1D.
    grid: xgcm.grid, optional
        Grid object associated with var. Optional because checks var
        attributes for grid.
    iso_array: DataArray, optional
        Array that var is interpolated onto (e.g., z coordinates or
        density). If calculating var on fixed depth slices, iso_array
        contains the depths [m] associated with var. In that case and
        if None, will use z coordinate attached to var. Also use this
        option if you want to interpolate with z depths constant in
        time and input the appropriate z coordinate.
    dim: str, optional
        Dimension over which to calculate isoslice. If calculating var
        onto fixed depths, `dim='Z'`. Options are 'Z', 'Y', and 'X'.

    Returns
    -------
    DataArray of var interpolated to iso_values. Dimensionality will be the
    same as var except with dim dimension of size of iso_values.

    Notes
    -----
    var cannot have chunks in the dimension dim.

    cf-xarray should still be usable after calling this function.

    Example usage
    -------------
    To calculate temperature onto fixed depths:
    >>> xroms.isoslice(ds.temp, np.linspace(0, -30, 50))

    To calculate temperature onto salinity:
    >>> xroms.isoslice(ds.temp, np.arange(0, 36), iso_array=ds.salt, axis='Z')

    Calculate lat-z slice of salinity along a constant longitude value (-91.5):
    >>> xroms.isoslice(ds.salt, -91.5, iso_array=ds.lon_rho, axis='X')

    Calculate slice of salt at 28 deg latitude
    >>> xroms.isoslice(ds.salt, 28, iso_array=ds.lat_rho, axis='Y')

    Interpolate temp to salinity values between 0 and 36 in the X direction
    >>> xroms.isoslice(ds.temp, np.linspace(0, 36, 50), iso_array=ds.salt, axis='X')

    Interpolate temp to salinity values between 0 and 36 in the Z direction
    >>> xroms.isoslice(ds.temp, np.linspace(0, 36, 50), iso_array=ds.salt, axis='Z')

    Calculate the depth of a specific isohaline (33):
    >>> xroms.isoslice(ds.salt, 33, iso_array=ds.z_rho, axis='Z')

    Calculate dye 10 meters above seabed. Either do this on the vertical
    rho grid, or first change to the w grid and then use `isoslice`. You may prefer
    to do the latter if there is a possibility that the distance above the seabed you are
    interpolating to (10 m) could be below the deepest rho grid depth.
    * on rho grid directly:
    >>> height_from_seabed = ds.z_rho + ds.h
    >>> height_from_seabed.name = 'z_rho'
    >>> xroms.isoslice(ds.dye_01, 10, iso_array=height_from_seabed, axis='Z')
    * on w grid:
    >>> var_w = ds.dye_01.xroms.to_grid(scoord='w').chunk({'s_w': -1})
    >>> ds['dye_01_w'] = var_w  # currently this is the easiest way to reattached coords xgcm variables
    >>> height_from_seabed = ds.z_w + ds.h
    >>> height_from_seabed.name = 'z_w'
    >>> xroms.isoslice(ds['dye_01_w'], 10, iso_array=height_from_seabed, axis='Z')
    """

    words = "Either grid should be input or var should be DataArray with grid in attributes."
    assert (grid is not None) or (
        isinstance(var, xr.DataArray) and "grid" in var.attrs
    ), words

    if grid is None:
        grid = var.attrs["grid"]

    assert isinstance(grid, xgcm.Grid), "grid must be `xgcm` grid object."

    attrs = var.attrs  # save to reinstitute at end

    # make sure iso_values are array-like
    if isinstance(iso_values, (int, float)):
        iso_values = [iso_values]

    # interpolate to the z coordinates associated with var
    if iso_array is None:
        key = [coord for coord in var.coords if "z_" in coord and "0" not in coord][
            0
        ]  # str
        assert (
            len(key) > 0
        ), "z coordinates associated with var could not be identified."
        iso_array = var[key]
    else:
        if isinstance(iso_array, xr.DataArray) and iso_array.name is not None:
            key = iso_array.name
        else:
            key = "z"

    # save key names for later
    lonkey = var.cf["longitude"].name
    latkey = var.cf["latitude"].name
    zkey = var.cf["vertical"].name

    # perform interpolation
    transformed = grid.transform(var, axis, iso_values, target_data=iso_array)

    if key not in transformed.coords:
        transformed = transformed.assign_coords({key: iso_array})

    # perform interpolation for other coordinates if needed
    if zkey not in transformed.coords:
        transformedZ = grid.transform(
            var[zkey], axis, iso_values, target_data=iso_array
        )
        transformed = transformed.assign_coords({zkey: transformedZ})

    if latkey not in transformed.coords:
        # this interpolation won't work for certain combinations of var[latkey] and iso_array
        # without the following step
        if "T" in iso_array.cf.get_valid_keys():
            iso_array = iso_array.cf.isel(T=0).drop_vars(
                iso_array.cf["T"].name, errors="ignore"
            )
        if "Z" in iso_array.cf.get_valid_keys():
            iso_array = iso_array.cf.isel(Z=0).drop_vars(
                iso_array.cf["Z"].name, errors="ignore"
            )
        transformedlat = grid.transform(
            var[latkey], axis, iso_values, target_data=iso_array
        )
        transformed = transformed.assign_coords({latkey: transformedlat})

    if lonkey not in transformed.coords:
        # this interpolation won't work for certain combinations of var[latkey] and iso_array
        # without the following step
        if "T" in iso_array.cf.get_valid_keys():
            iso_array = iso_array.cf.isel(T=0).drop_vars(
                iso_array.cf["T"].name, errors="ignore"
            )
        if "Z" in iso_array.cf.get_valid_keys():
            iso_array = iso_array.cf.isel(Z=0).drop_vars(
                iso_array.cf["Z"].name, errors="ignore"
            )
        transformedlon = grid.transform(
            var[lonkey], axis, iso_values, target_data=iso_array
        )
        transformed = transformed.assign_coords({lonkey: transformedlon})

    transformed[zkey].attrs["positive"] = "up"
    transformed[lonkey].attrs["standard_name"] = "longitude"
    transformed[latkey].attrs["standard_name"] = "latitude"

    # bring along attributes for cf-xarray
    transformed[key].attrs["axis"] = axis
    transformed.attrs["grid"] = grid
    # add original attributes back in
    transformed.attrs = {**attrs, **transformed.attrs}

    # reorder back to normal ordering in case changed
    transformed = transformed.cf.transpose(
        *[dim for dim in ["T", "Z", "Y", "X"] if dim in transformed.cf.get_valid_keys()]
    )

    return transformed
