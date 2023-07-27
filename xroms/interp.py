"""
Interpolation functions.
"""

import sys
import warnings

import numpy as np
import xarray as xr
import xgcm

import xroms


# try:
#     import xesmf as xe
# except ModuleNotFoundError:
#     warnings.warn("xESMF is not installed, so `interpll` will not run.")


def interpll(var, lons, lats, which="pairs", **kwargs):
    """Interpolate var to lons/lats positions.

    Wraps xESMF to perform proper horizontal interpolation on non-flat Earth.

    Parameters
    ----------
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
    **kwargs:
        passed on to xESMF Regridder class

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

    Examples
    --------
    To return 1D pairs of points, in this case 3 points:
    >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='pairs')
    To return 2D pairs of points, in this case a 3x3 array of points:
    >>> xroms.interpll(var, [-96, -97, -96.5], [26.5, 27, 26.5], which='grid')
    """

    # make sure that xesmf was read in for this function to run
    if not xroms.XESMF_AVAILABLE:
        raise ModuleNotFoundError("xESMF is not installed, so `interpll` will not run.")
    else:
        import xesmf as xe
    # try:
    #     xe
    # except NameError:
    #     print("xESMF is not installed, so `interpll` will not run.")
    #     return None

    # rename coords for use with xESMF
    lonkey = [coord for coord in var.coords if "lon_" in coord][0]
    latkey = [coord for coord in var.coords if "lat_" in coord][0]
    var = var.rename({lonkey: "lon", latkey: "lat"})

    # make sure dimensions are in typical cf ordering (T, Z, Y, X)
    var = xroms.order(var)

    # force lons/lats to be 1D arrays
    lats = np.asarray(lats).flatten()
    lons = np.asarray(lons).flatten()

    # whether inputs are
    if which == "pairs":
        locstream_out = True
        # set up for output
        varint = xr.Dataset(
            {"lat": (["locations"], lats), "lon": (["locations"], lons)}
        )

    elif which == "grid":
        locstream_out = False
        # set up for output
        varint = xr.Dataset({"lat": (["lat"], lats), "lon": (["lon"], lons)})

    # Calculate weights.
    regridder = xe.Regridder(
        var, varint, "bilinear", locstream_out=locstream_out, **kwargs
    )

    # Perform interpolation
    varint = regridder(var, keep_attrs=True)

    # the following commented code was supposed to add the z coordinate if it was missing
    # but was incorrect. A variation of it may need to be added back in. Oct 2022 KMT
    # # check for presence of z coord with interpolated output
    # zkey_varint = [
    #     coord for coord in varint.coords if "z_" in coord and "0" not in coord
    # ]
    # get z coordinates to go with interpolated output if not available
    zkeys = [coord for coord in var.coords if "z_" in coord and "0" not in coord]
    if len(zkeys) > 0:
        zkey = zkeys[0]  # str

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


def isoslice(var, iso_values, xgrid, iso_array=None, axis="Z"):
    """Interpolate var to iso_values.

    This wraps `xgcm` `transform` function for slice interpolation,
    though `transform` has additional functionality.

    Parameters
    ----------
    var: DataArray
        Variable to operate on.
    iso_values: list, ndarray
        Values to interpolate to. If calculating var at fixed depths,
        iso_values are the fixed depths, which should be negative if
        below mean sea level. If input as array, should be 1D.
    xgrid: xgcm.grid, optional
        Grid object associated with var.
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

    Examples
    --------
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

    words = "Grid should be input."
    assert xgrid is not None, words

    assert isinstance(xgrid, xgcm.Grid), "xgrid must be `xgcm` grid object."

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

    # perform interpolation
    transformed = xgrid.transform(var, axis, iso_values, target_data=iso_array)

    if key not in transformed.coords:
        transformed = transformed.assign_coords({key: iso_array})

    # bring along attributes for cf-xarray
    transformed[key].attrs["axis"] = axis
    # add original attributes back in
    transformed.attrs = {**attrs, **transformed.attrs}

    # save key names for later
    # perform interpolation for other coordinates if needed
    if "longitude" in var.cf.coordinates:
        lonkey = var.cf["longitude"].name

        if lonkey not in transformed.coords:
            # this interpolation won't work for certain combinations of var[latkey] and iso_array
            # without the following step
            if "T" in iso_array.reset_coords(drop=True).cf.axes:
                iso_array = iso_array.cf.isel(T=0).drop_vars(
                    iso_array.cf["T"].name, errors="ignore"
                )
            if "Z" in iso_array.reset_coords(drop=True).cf.axes:
                iso_array = iso_array.cf.isel(Z=0).drop_vars(
                    iso_array.cf["Z"].name, errors="ignore"
                )
            transformedlon = xgrid.transform(
                var[lonkey], axis, iso_values, target_data=iso_array
            )
            transformed = transformed.assign_coords({lonkey: transformedlon})

        transformed[lonkey].attrs["standard_name"] = "longitude"

    if "latitude" in var.cf.coordinates:
        latkey = var.cf["latitude"].name

        if latkey not in transformed.coords:
            # this interpolation won't work for certain combinations of var[latkey] and iso_array
            # without the following step
            if "T" in iso_array.reset_coords(drop=True).cf.axes:
                iso_array = iso_array.cf.isel(T=0).drop_vars(
                    iso_array.cf["T"].name, errors="ignore"
                )
            if "Z" in iso_array.reset_coords(drop=True).cf.axes:
                iso_array = iso_array.cf.isel(Z=0).drop_vars(
                    iso_array.cf["Z"].name, errors="ignore"
                )
            transformedlat = xgrid.transform(
                var[latkey], axis, iso_values, target_data=iso_array
            )
            transformed = transformed.assign_coords({latkey: transformedlat})

        transformed[latkey].attrs["standard_name"] = "latitude"

    if "vertical" in var.cf.coordinates:
        zkey = var.cf["vertical"].name

        if zkey not in transformed.coords:
            transformedZ = xgrid.transform(
                var[zkey], axis, iso_values, target_data=iso_array
            )
            transformed = transformed.assign_coords({zkey: transformedZ})

        transformed[zkey].attrs["positive"] = "up"

    transformed = transformed.squeeze().cf.guess_coord_axis()

    # reorder back to normal ordering in case changed
    transformed = xroms.order(transformed)

    return transformed
