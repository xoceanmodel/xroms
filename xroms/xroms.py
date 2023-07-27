"""
Functions to help read in ROMS output.
"""


import pathlib
import pickle
import warnings

import cf_xarray
import numpy as np
import xarray as xr
import xgcm

# import xroms
from .utilities import grid_interp, order


try:
    import cartopy
except ImportError:
    warnings.warn(
        "cartopy is not installed, so the `add_verts` options in `roms_dataset` will not run.",
        ImportWarning,
    )


xr.set_options(keep_attrs=True)

# from .utilities import xisoslice, to_grid
# from .roms_seawater import buoyancy

g = 9.81  # m/s^2


def roms_dataset(
    ds,
    Vtransform=None,
    add_verts=False,
    proj=None,
    include_Z0=False,
    include_3D_metrics=True,
    include_cell_volume=False,
    include_cell_area=False,
):
    """Modify Dataset to be aware of ROMS coordinates, with matching xgcm grid object.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset with model output
    Vtransform: int, optional
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean, optional
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection
        to be input too.
    proj: cartopy crs projection, optional
        Should match geographic area of model domain. Required if `add_verts=True`,
        otherwise not used. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)
    include_Z0 : bool
        If True, calculate depths for quiescient state, which can be used for faster
        approximations for depth calculations since they are 3D instead of 4D.
    include_3D_metrics : bool
        If True, calculate necessary grid metrics for 3D calculations with xgcm. Note that you need the 3D metrics for horizontal derivatives for ROMS.
    include_cell_volume : bool
        If True, calculate necessary grid metrics for cell volumes. I think this is for cf-xarray.
    include_cell_area : bool
        If True, calculate necessary grid metrics for cell areas (besides dA). I think this is for cf-xarray.

    Returns
    -------
    ds: Dataset
        Same dataset as input, but with dimensions renamed to be consistent with `xgcm` and
        with vertical coordinates and metrics added.
    xgrid: xgcm grid object
        Includes ROMS metrics so can be used for xgcm grid operations, which mostly have
        been wrapped into xroms.

    Notes
    -----
    Note that this could be very slow if dask is not on.

    This does not need to be run by the user if `xroms` functions `open_netcdf` or
    `open_zarr` are used for reading in model output, since run in those functions.

    This also uses `cf-xarray` to manage dimensions of variables.

    Examples
    --------
    >>> ds, xgrid = xroms.roms_dataset(ds)
    """

    if add_verts:
        assert proj is not None, 'To add "verts" grid, input projection "proj".'

    rename = {}
    if "eta_u" in ds.dims:
        rename["eta_u"] = "eta_rho"
    if "xi_v" in ds.dims:
        rename["xi_v"] = "xi_rho"
    if "xi_psi" in ds.dims:
        rename["xi_psi"] = "xi_u"
    if "eta_psi" in ds.dims:
        rename["eta_psi"] = "eta_v"
    ds = ds.rename(rename)

    # Use spherical flag to determine if has lat/lon or not
    # If not present, try to guess its value
    if (not "spherical" in ds) or (not ds["spherical"] in [0, 1]):
        if "lon_rho" in ds:
            ds["spherical"] = 1
        else:
            ds["spherical"] = 0

    # make sure psi grid in coords
    if ds.spherical:
        if "lon_psi" in ds.keys():
            ds = ds.assign_coords({"lon_psi": ds.lon_psi})
        if "lat_psi" in ds.keys():
            ds = ds.assign_coords({"lat_psi": ds.lat_psi})

    # check if dataset has depths or is just 1 layer
    if ("s_rho" in ds) and (ds.s_rho.size > 1):
        ds["3d"] = True
    else:
        ds["3d"] = False

    if not ds["3d"] and include_3D_metrics:
        warnings.warn(
            "need 3D Dataset in order to calculate 3D metrics.", RuntimeWarning
        )

    # modify attributes for using cf-xarray
    tdims = [dim for dim in ds.dims if dim[:3] == "xi_"]
    for dim in tdims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {"axis": "X"})
    tdims = [dim for dim in ds.dims if dim[:4] == "eta_"]
    for dim in tdims:
        ds[dim] = (dim, np.arange(ds.sizes[dim]), {"axis": "Y"})
    if "ocean_time" in ds.keys():
        ds.ocean_time.attrs["axis"] = "T"
        ds.ocean_time.attrs["standard_name"] = "time"

    if ds["3d"]:
        tcoords = [coord for coord in ds.coords if coord[:2] == "s_"]
        for coord in tcoords:
            ds[coord].attrs["axis"] = "Z"

    # make sure lon/lat have standard names
    if ds.spherical:
        tcoords = [coord for coord in ds.coords if coord[:4] == "lon_"]
        for coord in tcoords:
            ds[coord].attrs["standard_name"] = "longitude"
        tcoords = [coord for coord in ds.coords if coord[:4] == "lat_"]
        for coord in tcoords:
            ds[coord].attrs["standard_name"] = "latitude"

    # Xdict = {"center": "xi_rho"}
    # # subsetted ROMS output might be missing some info so don't assume it is present
    # if "xi_u" in ds.coords:
    #     Xdict.update({"inner": "xi_u"})
    # Ydict = {"center": "eta_rho"}
    # if "eta_v" in ds.coords:
    #     Ydict.update({"inner": "eta_v"})
    # Zdict = {"center": "s_rho"}
    # if "s_w" in ds.coords:
    #     Zdict.update({"outer": "s_w"})

    # coords = {"X": Xdict, "Y": Ydict, "Z": Zdict}

    coords = {
        "X": {"center": "xi_rho", "inner": "xi_u"},
        "Y": {"center": "eta_rho", "inner": "eta_v"},
    }

    if ds["3d"]:
        coords.update(
            {
                "Z": {"center": "s_rho", "outer": "s_w"},
            }
        )

    xgrid = xgcm.Grid(ds, coords=coords, periodic=[])

    if "Vtransform" in ds.variables.keys():
        Vtransform = ds.Vtransform

    if ds["3d"]:
        assert Vtransform in [
            1,
            2,
        ], "Need a Vtransform of 1 or 2, either in the Dataset or input to the function."

    if ds["3d"]:
        if Vtransform == 1:
            Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
            z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / ds.h)
            Zo_w = ds.hc * (ds.s_w - ds.Cs_w) + ds.Cs_w * ds.h
            z_w = Zo_w + ds.zeta * (1 + Zo_w / ds.h)
            # also include z coordinates with mean sea level (constant over time)
            if include_Z0:
                z_rho0 = Zo_rho
                z_w0 = Zo_w
        elif Vtransform == 2:
            Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
            z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho
            Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
            z_w = ds.zeta + (ds.zeta + ds.h) * Zo_w
            # also include z coordinates with mean sea level (constant over time)
            if include_Z0:
                z_rho0 = ds.h * Zo_rho
                z_w0 = ds.h * Zo_w
        z_rho.attrs = {
            "long_name": "depth of RHO-points",
            "time": "ocean_time",
            "field": "z_rho, scalar, series",
            "units": "m",
        }
        z_w.attrs = {
            "long_name": "depth of W-points",
            "time": "ocean_time",
            "field": "z_w, scalar, series",
            "units": "m",
        }
        if include_Z0:
            z_rho0.attrs = {
                "long_name": "depth of RHO-points",
                "field": "z_rho0, scalar",
                "units": "m",
            }
            z_w0.attrs = {
                "long_name": "depth of W-points",
                "field": "z_w0, scalar",
                "units": "m",
            }

        ds.coords["z_w"] = order(z_w)
        ds.coords["z_w_u"] = grid_interp(xgrid, ds["z_w"], "X")
        # ds.coords["z_w_u"] = xgrid.interp(ds.z_w, "X")
        ds.coords["z_w_u"].attrs = {
            "long_name": "depth of U-points on vertical W grid",
            "time": "ocean_time",
            "field": "z_w_u, scalar, series",
            "units": "m",
        }
        ds.coords["z_w_v"] = grid_interp(xgrid, ds["z_w"], "Y")
        # ds.coords["z_w_v"] = xgrid.interp(ds.z_w, "Y")
        ds.coords["z_w_v"].attrs = {
            "long_name": "depth of V-points on vertical W grid",
            "time": "ocean_time",
            "field": "z_w_v, scalar, series",
            "units": "m",
        }
        ds.coords["z_w_psi"] = grid_interp(xgrid, ds["z_w_u"], "Y")
        # ds.coords["z_w_psi"] = xgrid.interp(ds.z_w_u, "Y")
        ds.coords["z_w_psi"].attrs = {
            "long_name": "depth of PSI-points on vertical W grid",
            "time": "ocean_time",
            "field": "z_w_psi, scalar, series",
            "units": "m",
        }

        ds.coords["z_rho"] = order(z_rho)
        ds.coords["z_rho_u"] = grid_interp(xgrid, ds["z_rho"], "X")
        # ds.coords["z_rho_u"] = xgrid.interp(ds.z_rho, "X")
        ds.coords["z_rho_u"].attrs = {
            "long_name": "depth of U-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_u, scalar, series",
            "units": "m",
        }

        ds.coords["z_rho_v"] = grid_interp(xgrid, ds["z_rho"], "Y")
        # ds.coords["z_rho_v"] = xgrid.interp(ds.z_rho, "Y")
        ds.coords["z_rho_v"].attrs = {
            "long_name": "depth of V-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_v, scalar, series",
            "units": "m",
        }

        ds.coords["z_rho_psi"] = grid_interp(xgrid, ds["z_rho_u"], "Y")
        # ds.coords["z_rho_psi"] = xgrid.interp(ds.z_rho_u, "Y")
        # also include z coordinates with mean sea level (constant over time)
        ds.coords["z_rho_psi"].attrs = {
            "long_name": "depth of PSI-points on vertical RHO grid",
            "time": "ocean_time",
            "field": "z_rho_psi, scalar, series",
            "units": "m",
        }

        # replace s_rho with z_rho, etc, to make z_rho the vertical coord
        name_dict = {"s_rho": "z_rho", "s_w": "z_w"}
        name_dict.update(
            {
                "filler1": "z_rho_u",
                "filler2": "z_rho_v",
                "filler3": "z_rho_psi",
                "filler4": "z_w_u",
                "filler5": "z_w_v",
                "filler6": "z_w_psi",
            }
        )
        for sname, zname in name_dict.items():
            for var in ds.data_vars:
                if ds[var].ndim == 4:
                    if "coordinates" in ds[var].encoding:
                        coords_here = ds[var].encoding["coordinates"]
                        if sname in coords_here:  # replace if present
                            coords_here = coords_here.replace(sname, zname)
                        else:  # still add z_rho or z_w
                            if (
                                zname in ds[var].coords
                                and ds[zname].shape == ds[var].shape
                            ):
                                coords_here += f" {zname}"
                        ds[var].encoding["coordinates"] = coords_here
                    # same but coordinates not inside encoding. Do same processing
                    # but also move coordinates from attrs to encoding.
                    elif "coordinates" in ds[var].attrs:
                        coords_here = ds[var].attrs["coordinates"]
                        if sname in coords_here:  # replace if present
                            coords_here = coords_here.replace(sname, zname)
                        else:  # still add z_rho or z_w
                            if (
                                zname in ds[var].coords
                                and ds[zname].shape == ds[var].shape
                            ):
                                coords_here += f" {zname}"
                        # move coords to encoding and delete from attrs
                        ds[var].encoding["coordinates"] = coords_here
                        del ds[var].attrs["coordinates"]

        if include_Z0:
            ds.coords["z_rho0"] = order(z_rho0)
            ds.coords["z_rho_u0"] = xgrid.interp(ds.z_rho0, "X")
            ds.coords["z_rho_u0"].attrs = {
                "long_name": "depth of U-points on vertical RHO grid",
                "field": "z_rho_u0, scalar",
                "units": "m",
            }

            ds.coords["z_rho_v0"] = xgrid.interp(ds.z_rho0, "Y")
            ds.coords["z_rho_v0"].attrs = {
                "long_name": "depth of V-points on vertical RHO grid",
                "field": "z_rho_v0, scalar",
                "units": "m",
            }

            ds.coords["z_rho_psi0"] = xgrid.interp(ds.z_rho_u0, "Y")
            ds.coords["z_rho_psi0"].attrs = {
                "long_name": "depth of PSI-points on vertical RHO grid",
                "field": "z_rho_psi0, scalar",
                "units": "m",
            }

            ds.coords["z_w0"] = order(z_w0)
            ds.coords["z_w_u0"] = xgrid.interp(ds.z_w0, "X")
            ds.coords["z_w_u0"].attrs = {
                "long_name": "depth of U-points on vertical W grid",
                "field": "z_w_u0, scalar",
                "units": "m",
            }

            ds.coords["z_w_v0"] = xgrid.interp(ds.z_w0, "Y")
            ds.coords["z_w_v0"].attrs = {
                "long_name": "depth of V-points on vertical W grid",
                "field": "z_w_v0, scalar",
                "units": "m",
            }

            ds.coords["z_w_psi0"] = xgrid.interp(ds.z_w_u0, "Y")
            ds.coords["z_w_psi0"].attrs = {
                "long_name": "depth of PSI-points on vertical W grid",
                "field": "z_w_psi0, scalar",
                "units": "m",
            }

    # add vert grid, esp for plotting pcolormesh
    if ds.spherical and add_verts:
        import pygridgen

        pc = cartopy.crs.PlateCarree()
        # project points for this calculation
        xr, yr = proj.transform_points(pc, ds.lon_rho.values, ds.lat_rho.values)[
            ..., :2
        ].T
        xr = xr.T
        yr = yr.T
        # calculate vert locations
        xv, yv = pygridgen.grid.rho_to_vert(xr, yr, ds.pm, ds.pn, ds.angle)
        # project back
        lon_vert, lat_vert = pc.transform_points(proj, xv, yv)[..., :2].T
        lon_vert = lon_vert.T
        lat_vert = lat_vert.T
        # add new coords to ds
        ds.coords["lon_vert"] = (("eta_vert", "xi_vert"), lon_vert)
        ds.coords["lat_vert"] = (("eta_vert", "xi_vert"), lat_vert)

    # just keep these local instead of saving to Dataset — doesn't look like they
    # are used in any functions outside of this function.
    pm_v = xgrid.interp(ds.pm, "Y")
    # ds["pm_v"].attrs = {
    #     "long_name": "curvilinear coordinate metric in XI on V grid",
    #     "units": "meter-1",
    #     "field": "pm_v, scalar",
    # }

    pn_u = xgrid.interp(ds.pn, "X")
    # ds["pn_u"].attrs = {
    #     "long_name": "curvilinear coordinate metric in ETA on U grid",
    #     "units": "meter-1",
    #     "field": "pn_u, scalar",
    # }

    pm_u = xgrid.interp(ds.pm, "X")
    # ds["pm_u"].attrs = {
    #     "long_name": "curvilinear coordinate metric in XI on U grid",
    #     "units": "meter-1",
    #     "field": "pm_u, scalar",
    # }

    pn_v = xgrid.interp(ds.pn, "Y")
    # ds["pn_v"].attrs = {
    #     "long_name": "curvilinear coordinate metric in ETA on V grid",
    #     "units": "meter-1",
    #     "field": "pn_v, scalar",
    # }

    pm_psi = xgrid.interp(xgrid.interp(ds.pm, "Y"), "X")  # at psi points (eta_v, xi_u)
    # ds["pm_psi"].attrs = {
    #     "long_name": "curvilinear coordinate metric in XI on PSI grid",
    #     "units": "meter-1",
    #     "field": "pm_psi, scalar",
    # }

    pn_psi = xgrid.interp(xgrid.interp(ds.pn, "X"), "Y")  # at psi points (eta_v, xi_u)
    # ds["pn_psi"].attrs = {
    #     "long_name": "curvilinear coordinate metric in ETA on PSI grid",
    #     "units": "meter-1",
    #     "field": "pn_psi, scalar",
    # }

    ds["dx"] = 1 / ds.pm
    ds["dx"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in XI",
        "units": "meter",
        "field": "dx, scalar",
    }

    ds["dx_u"] = 1 / pm_u
    ds["dx_u"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in XI on U grid",
        "units": "meter",
        "field": "dx_u, scalar",
    }

    ds["dx_v"] = 1 / pm_v
    ds["dx_v"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in XI on V grid",
        "units": "meter",
        "field": "dx_v, scalar",
    }

    ds["dx_psi"] = 1 / pm_psi
    ds["dx_psi"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in XI on PSI grid",
        "units": "meter",
        "field": "dx_psi, scalar",
    }

    ds["dy"] = 1 / ds.pn
    ds["dy"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in ETA",
        "units": "meter",
        "field": "dy, scalar",
    }

    ds["dy_u"] = 1 / pn_u
    ds["dy_u"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in ETA on U grid",
        "units": "meter",
        "field": "dy_u, scalar",
    }

    ds["dy_v"] = 1 / pn_v
    ds["dy_v"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in ETA on V grid",
        "units": "meter",
        "field": "dy_v, scalar",
    }

    ds["dy_psi"] = 1 / pn_psi
    ds["dy_psi"].attrs = {
        "long_name": "inverse curvilinear coordinate metric in ETA on PSI grid",
        "units": "meter",
        "field": "dy_psi, scalar",
    }

    if ds["3d"] and include_3D_metrics:
        ds["dz"] = xgrid.diff(ds.z_w.chunk({ds.z_w.cf["Z"].name: -1}), "Z")
        ds["dz"].attrs = {
            "long_name": "vertical layer thickness on vertical RHO grid",
            "time": "ocean_time",
            "field": "dz, scalar, series",
            "units": "m",
        }

        ds["dz_w"] = xgrid.diff(ds.z_rho, "Z", boundary="fill")
        ds["dz_w"].attrs = {
            "long_name": "vertical layer thickness on vertical W grid",
            "time": "ocean_time",
            "field": "dz_w, scalar, series",
            "units": "m",
        }

        ds["dz_u"] = grid_interp(xgrid, ds["dz"], "X")
        # ds["dz_u"] = xgrid.interp(ds.dz, "X")
        ds["dz_u"].attrs = {
            "long_name": "vertical layer thickness on vertical RHO grid on U grid",
            "time": "ocean_time",
            "field": "dz_u, scalar, series",
            "units": "m",
        }

        ds["dz_w_u"] = grid_interp(xgrid, ds["dz_w"], "X")
        # ds["dz_w_u"] = xgrid.interp(ds.dz_w, "X")
        ds["dz_w_u"].attrs = {
            "long_name": "vertical layer thickness on vertical W grid on U grid",
            "time": "ocean_time",
            "field": "dz_w_u, scalar, series",
            "units": "m",
        }

        ds["dz_v"] = grid_interp(xgrid, ds["dz"], "Y")
        # ds["dz_v"] = xgrid.interp(ds.dz, "Y")
        ds["dz_v"].attrs = {
            "long_name": "vertical layer thickness on vertical RHO grid on V grid",
            "time": "ocean_time",
            "field": "dz_v, scalar, series",
            "units": "m",
        }

        ds["dz_w_v"] = grid_interp(xgrid, ds["dz_w"], "Y")
        # ds["dz_w_v"] = xgrid.interp(ds.dz_w, "Y")
        ds["dz_w_v"].attrs = {
            "long_name": "vertical layer thickness on vertical W grid on V grid",
            "time": "ocean_time",
            "field": "dz_w_v, scalar, series",
            "units": "m",
        }

        ds["dz_psi"] = grid_interp(xgrid, ds["dz_v"], "X")
        # ds["dz_psi"] = xgrid.interp(ds.dz_v, "X")
        ds["dz_psi"].attrs = {
            "long_name": "vertical layer thickness on vertical RHO grid on PSI grid",
            "time": "ocean_time",
            "field": "dz_psi, scalar, series",
            "units": "m",
        }

        ds["dz_w_psi"] = grid_interp(xgrid, ds["dz_w_v"], "X")
        # ds["dz_w_psi"] = xgrid.interp(ds.dz_w_v, "X")
        ds["dz_w_psi"].attrs = {
            "long_name": "vertical layer thickness on vertical W grid on PSI grid",
            "time": "ocean_time",
            "field": "dz_w_psi, scalar, series",
            "units": "m",
        }

        if include_Z0:

            # also include z coordinates with mean sea level (constant over time)
            ds["dz0"] = xgrid.diff(ds.z_w0, "Z")
            ds["dz0"].attrs = {
                "long_name": "vertical layer thickness on vertical RHO grid",
                "field": "dz0, scalar",
                "units": "m",
            }

            ds["dz_w0"] = xgrid.diff(ds.z_rho0, "Z", boundary="fill")
            ds["dz_w0"].attrs = {
                "long_name": "vertical layer thickness on vertical W grid",
                "field": "dz_w0, scalar",
                "units": "m",
            }

            ds["dz_u0"] = xgrid.interp(ds.dz0, "X")
            ds["dz_u0"].attrs = {
                "long_name": "vertical layer thickness on vertical RHO grid on U grid",
                "field": "dz_u0, scalar",
                "units": "m",
            }

            ds["dz_w_u0"] = xgrid.interp(ds.dz_w0, "X")
            ds["dz_w_u0"].attrs = {
                "long_name": "vertical layer thickness on vertical W grid on U grid",
                "field": "dz_w_u0, scalar",
                "units": "m",
            }

            ds["dz_v0"] = xgrid.interp(ds.dz0, "Y")
            ds["dz_v0"].attrs = {
                "long_name": "vertical layer thickness on vertical RHO grid on V grid",
                "field": "dz_v0, scalar",
                "units": "m",
            }

            ds["dz_w_v0"] = xgrid.interp(ds.dz_w0, "Y")
            ds["dz_w_v0"].attrs = {
                "long_name": "vertical layer thickness on vertical W grid on V grid",
                "field": "dz_w_v0, scalar",
                "units": "m",
            }

            ds["dz_psi0"] = xgrid.interp(ds.dz_v0, "X")
            ds["dz_psi0"].attrs = {
                "long_name": "vertical layer thickness on vertical RHO grid on PSI grid",
                "field": "dz_psi0, scalar",
                "units": "m",
            }

            ds["dz_w_psi0"] = xgrid.interp(ds.dz_w_v0, "X")
            ds["dz_w_psi0"].attrs = {
                "long_name": "vertical layer thickness on vertical W grid on PSI grid",
                "field": "dz_w_psi0, scalar",
                "units": "m",
            }

    # grid areas
    ds["dA"] = ds.dx * ds.dy
    ds["dA"].attrs = {
        "long_name": "area metric in XI and ETA on RHO grid",
        "units": "meter2",
        "field": "dA, scalar",
    }

    if include_cell_area:
        ds["dA_u"] = ds.dx_u * ds.dy_u
        ds["dA_u"].attrs = {
            "long_name": "area metric in XI and ETA on U grid",
            "units": "meter2",
            "field": "dA_u, scalar",
        }

        ds["dA_v"] = ds.dx_v * ds.dy_v
        ds["dA_v"].attrs = {
            "long_name": "area metric in XI and ETA on V grid",
            "units": "meter2",
            "field": "dA_v, scalar",
        }

        ds["dA_psi"] = ds.dx_psi * ds.dy_psi
        ds["dA_psi"].attrs = {
            "long_name": "area metric in XI and ETA on PSI grid",
            "units": "meter2",
            "field": "dA_psi, scalar",
        }

    # volume
    if ds["3d"] and include_cell_volume:
        ds["dV"] = ds.dz * ds.dx * ds.dy  # rho vertical, rho horizontal
        ds["dV"].attrs = {
            "long_name": "volume metric in XI and ETA and S on RHO/RHO grids",
            "units": "meter3",
            "field": "dV, scalar",
        }

        ds["dV_w"] = ds.dz_w * ds.dx * ds.dy  # w vertical, rho horizontal
        ds["dV_w"].attrs = {
            "long_name": "volume metric in XI and ETA and S on RHO/W grids",
            "units": "meter3",
            "field": "dV_w, scalar",
        }

        ds["dV_u"] = ds.dz_u * ds.dx_u * ds.dy_u  # rho vertical, u horizontal
        ds["dV_u"].attrs = {
            "long_name": "volume metric in XI and ETA and S on U/RHO grids",
            "units": "meter3",
            "field": "dV_u, scalar",
        }

        ds["dV_w_u"] = ds.dz_w_u * ds.dx_u * ds.dy_u  # w vertical, u horizontal
        ds["dV_w_u"].attrs = {
            "long_name": "volume metric in XI and ETA and S on U/W grids",
            "units": "meter3",
            "field": "dV_w_u, scalar",
        }

        ds["dV_v"] = ds.dz_v * ds.dx_v * ds.dy_v  # rho vertical, v horizontal
        ds["dV_v"].attrs = {
            "long_name": "volume metric in XI and ETA and S on V/RHO grids",
            "units": "meter3",
            "field": "dV_v, scalar",
        }

        ds["dV_w_v"] = ds.dz_w_v * ds.dx_v * ds.dy_v  # w vertical, v horizontal
        ds["dV_w_v"].attrs = {
            "long_name": "volume metric in XI and ETA and S on V/W grids",
            "units": "meter3",
            "field": "dV_w_v, scalar",
        }

        ds["dV_psi"] = ds.dz_psi * ds.dx_psi * ds.dy_psi  # rho vertical, psi horizontal
        ds["dV_psi"].attrs = {
            "long_name": "volume metric in XI and ETA and S on PSI/RHO grids",
            "units": "meter3",
            "field": "dV_psi, scalar",
        }

        ds["dV_w_psi"] = (
            ds.dz_w_psi * ds.dx_psi * ds.dy_psi
        )  # w vertical, psi horizontal
        ds["dV_w_psi"].attrs = {
            "long_name": "volume metric in XI and ETA and S on PSI/W grids",
            "units": "meter3",
            "field": "dV_w_psi, scalar",
        }

    if "rho0" not in ds:
        ds["rho0"] = 1025  # kg/m^3

    # cf-xarray
    # areas
    #     ds.coords["cell_area"] = ds['dA']
    #     ds.coords["cell_area_u"] = ds['dA_u']
    #     ds.coords["cell_area_v"] = ds['dA_v']
    #     ds.coords["cell_area_psi"] = ds['dA_psi']
    #     # and set proper attributes
    #     ds.temp.attrs["cell_measures"] = "area: cell_area, volume: cell_volume"
    #     ds.salt.attrs["cell_measures"] = "area: cell_area"
    #     ds.u.attrs["cell_measures"] = "area: cell_area_u"
    #     ds.v.attrs["cell_measures"] = "area: cell_area_v"
    #     # volumes
    #     ds.coords["cell_volume"] = ds['dV']
    # #     ds.temp.attrs["cell_measures"] = "volume: cell_volume"

    #     ds['temp'].attrs['cell_measures'] = 'area: cell_area'
    #     tcoords = [coord for coord in ds.variables if coord[:2] == 'dA']
    #     for coord in tcoords:
    #         ds[coord].attrs['cell_measures'] = 'area: cell_area'
    #     # add coordinates attributes for variables
    if ds["3d"]:  # and include_3D_metrics:
        if "positive" in ds.s_rho.attrs:
            ds.s_rho.attrs.pop("positive")
        if "positive" in ds.s_w.attrs:
            ds.s_w.attrs.pop("positive")
        #     ds['z_rho'].attrs['positive'] = 'up'
        tcoords = [
            coord for coord in ds.coords if coord[:2] == "z_" and "0" not in coord
        ]
        for coord in tcoords:
            ds[coord].attrs["positive"] = "up"
            ds[coord].attrs["standard_name"] = "depth"
        #         ds[dim] = (dim, np.arange(ds.sizes[dim]), {'axis': 'Y'})
        #     ds['z_rho'].attrs['vertical'] = 'depth'
        #     ds['temp'].attrs['coordinates'] = 'lon_rho lat_rho z_rho ocean_time'
        #     [del ds[var].encoding['coordinates'] for var in ds.variables if 'coordinates' in ds[var].encoding]

    # for var in ds.variables:
    #     if "coordinates" in ds[var].encoding:
    #         del ds[var].encoding["coordinates"]
    #     if "coordinates" in ds[var].attrs:
    #         del ds[var].attrs["coordinates"]

    if ds["3d"] and include_3D_metrics:
        metrics = {
            ("X",): ["dx", "dx_u", "dx_v", "dx_psi"],  # X distances
            ("Y",): ["dy", "dy_u", "dy_v", "dy_psi"],  # Y distances
            ("Z",): [
                "dz",
                "dz_u",
                "dz_v",
                "dz_w",
                "dz_w_u",
                "dz_w_v",
                "dz_psi",
                "dz_w_psi",
            ],  # Z distances
            ("X", "Y"): ["dA"],  # Areas
        }
    else:  # 2d
        metrics = {
            ("X",): ["dx", "dx_u", "dx_v", "dx_psi"],  # X distances
            ("Y",): ["dy", "dy_u", "dy_v", "dy_psi"],  # Y distances
            ("X", "Y"): ["dA"],  # Areas
        }

    xgrid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

    # #     ds.attrs['grid'] = grid  # causes recursion error
    # # also put grid into every variable with at least 2D
    # for var in ds.data_vars:
    #     if ds[var].ndim > 1:
    #         ds[var].attrs["grid"] = grid

    return ds, xgrid


def open_netcdf(
    file,
    chunks={"ocean_time": 1},
    xrargs={},
    Vtransform=None,
    add_verts=False,
    proj=None,
):
    """Return Dataset based on a single thredds or physical location.

    Parameters
    ----------
    file: str
        Where to find the model output. `file` could be:
        * a string of a single netCDF file name, or
        * a string of a thredds server address containing model output.
    chunks: dict, optional
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict, optional
        Keyword arguments to be passed to `xarray.open_dataset`. See `xarray` docs
        for options.
    Vtransform: int, optional
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean, optional
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, optional
        Should match geographic area of model domain. Required if `add_verts=True`,
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and
        output is read in lazily instead of eagerly.

    Examples
    --------
    >>> ds = xroms.open_netcdf(file)
    """

    msg = """
Recommended usage going forward is to read in your model output with xarray directly, then subsequently run
`ds, xgrid = xroms.roms_dataset(ds)` to preprocess your Dataset for use with `cf-xarray` and `xgcm`, and get
the necessary grid object for use with `xgcm`. This function will be removed at some future
"""
    warnings.warn(msg, PendingDeprecationWarning)

    words = (
        "Model location should be given as string or `pathlib.Path`."
        "If you have list of multiple locations, use `open_mfdataset`."
    )
    assert isinstance(file, (str, pathlib.Path)), words

    ds = xr.open_dataset(file, chunks=chunks, **xrargs)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, xgrid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)

    return ds


def open_mfnetcdf(
    files,
    chunks={"ocean_time": 1},
    xrargs={},
    Vtransform=None,
    add_verts=False,
    proj=None,
):
    """Return Dataset based on a list of netCDF files.

    Parameters
    ----------
    files: list of strings
        Where to find the model output. `files` can be a list of netCDF file names.
    chunks: dict, optional
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict, optional
        Keyword arguments to be passed to `xarray.open_mfdataset`.
        Anything input by the user overwrites the default selections saved in this
        function. Defaults are:
            {'compat': 'override', 'combine': 'by_coords',
             'data_vars': 'minimal', 'coords': 'minimal', 'parallel': True}
        Many other options are available; see xarray docs.
    Vtransform: int, optional
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean, optional
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, optional
        Should match geographic area of model domain. Required if `add_verts=True`,
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and
        output is read in lazily instead of eagerly.

    Examples
    --------
    >>> ds = xroms.open_mfnetcdf(files)
    """

    msg = """
Recommended usage going forward is to read in your model output with xarray directly, then subsequently run
`ds, xgrid = xroms.roms_dataset(ds)` to preprocess your Dataset for use with `cf-xarray` and `xgcm`, and get
the necessary grid object for use with `xgcm`. This function will be removed at some future
"""
    warnings.warn(msg, PendingDeprecationWarning)

    words = "Model location should be given as list of strings. If have single location, use `open_dataset`."
    assert isinstance(files, list), words

    xrargsin = {
        "compat": "override",
        "combine": "by_coords",
        "data_vars": "minimal",
        "coords": "minimal",
        "parallel": True,
    }

    # use input xarray arguments and prioritize them over xroms defaults.
    xrargsin = {**xrargsin, **xrargs}

    ds = xr.open_mfdataset(files, chunks=chunks, **xrargsin)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, xgrid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)

    return ds


def open_zarr(
    files,
    chunks={"ocean_time": 1},
    xrargs={},
    xrconcatargs={},
    Vtransform=None,
    add_verts=False,
    proj=None,
):
    """Return a Dataset based on a list of zarr files

    Parameters
    ----------
    files: list of strings
        A list of zarr file directories.
    chunks: dict, optional
        The specified chunks for the Dataset. Use chunks to read in output using dask.
    xrargs: dict, optional
        Keyword arguments to be passed to `xarray.open_zarr`.
        Anything input by the user overwrites the default selections saved in this
        function. Defaults are:
            {'consolidated': True, 'drop_variables': 'dstart'}
        Many other options are available; see xarray docs.
    xrconcatargs: dict, optional
        Keyword arguments to be passed to `xarray.concat` for combining zarr files
        together. Anything input by the user overwrites the default selections saved in this
        function. Defaults are:
            {'dim': 'ocean_time', 'data_vars': 'minimal', 'coords': 'minimal'}
        Many other options are available; see xarray docs.
    Vtransform: int, optional
        Vertical transform for ROMS model. Should be either 1 or 2 and only needs
        to be input if not available in ds.
    add_verts: boolean, optional
        Add 'verts' horizontal grid to ds if True. This requires a cartopy projection
        to be input too. This is passed to `roms_dataset`.
    proj: cartopy crs projection, optional
        Should match geographic area of model domain. Required if `add_verts=True`,
        otherwise not used. This is passed to `roms_dataset`. Example:
        >>> proj = cartopy.crs.LambertConformal(central_longitude=-98, central_latitude=30)

    Returns
    -------
    ds: Dataset
        Model output, read into an `xarray` Dataset. If 'chunks' keyword
        argument is input, dask is used when reading in model output and
        output is read in lazily instead of eagerly.

    Examples
    --------
    >>> ds = xroms.open_zarr(files)
    """

    msg = """
Recommended usage going forward is to read in your model output with xarray directly, then subsequently run
`ds, xgrid = xroms.roms_dataset(ds)` to preprocess your Dataset for use with `cf-xarray` and `xgcm`, and get
the necessary grid object for use with `xgcm`. This function will be removed at some future
"""
    warnings.warn(msg, PendingDeprecationWarning)

    # keyword arguments to go to `open_zarr`:
    xrargsin = {"consolidated": True, "drop_variables": "dstart"}
    # use input xarray arguments and prioritize them over xroms defaults.
    xrargsin = {**xrargsin, **xrargs}

    # keyword arguments to go to `concat`:
    xrconcatargsin = {"dim": "ocean_time", "data_vars": "minimal", "coords": "minimal"}
    # use input xarray arguments and prioritize them over xroms defaults.
    xrconcatargsin = {**xrconcatargsin, **xrconcatargs}

    # open files
    ds = xr.concat([xr.open_zarr(file, **xrargsin) for file in files], **xrconcatargsin)

    # modify Dataset with useful ROMS z coords and make xgcm grid operations usable.
    ds, xgrid = roms_dataset(ds, Vtransform=Vtransform, add_verts=add_verts, proj=proj)

    return ds


# def save(ds, filename="output.nc"):
#     """Save to file."""

#     # have to remove the grid objects because they can't be saved
#     for var in ds.data_vars:
#         if "grid" in ds[var].attrs:
#             del ds[var].attrs["grid"]

#     ds.to_netcdf(filename)
