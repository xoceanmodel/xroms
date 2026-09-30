"""Synthetic ROMS-family datasets with analytic fields, for tests.

Every layout is built from one internally consistent grid:

* ``pm`` varies only along xi and ``pn`` only along eta, and the rho-point
  positions satisfy ``x_rho[i+1] - x_rho[i] == 1 / (0.5 * (pm[i] + pm[i+1]))``,
  the u-point metric xroms uses. Differencing any field that is linear in ``x``
  therefore reproduces its slope to machine precision.
* ``h`` slopes in both directions and ``hc > 0``, so the s-coordinate
  (sigma-slope) terms of horizontal derivatives are exercised.
* ``temp = TEMP_A * x + TEMP_B * z_rho + TEMP_0`` (linear in x and z) and
  ``salt = SALT_0 + SALT_C * z_rho**2`` (quadratic in z).

``make_dataset(layout)`` returns the dataset in the on-disk conventions of that
model family; for ``"ucla"`` it returns ``(output, grid)`` because UCLA ROMS
keeps the grid in a separate file.
"""

import numpy as np
import xarray as xr


TEMP_A = 1.0e-4  # degC per m of x
TEMP_B = 0.05  # degC per m of z
TEMP_0 = 10.0
SALT_0 = 35.0
SALT_C = 1.0e-4  # psu per m**2
U_A = 2.0e-5  # u = U_0 + U_A * x_u   (uniform grids only)
U_0 = 0.1
V_A = -1.0e-5  # v = V_0 + V_A * y_v
V_0 = -0.05
F0 = 1.0e-4
EPOCH = "2000-01-01"


def stretching(sigma, theta_s, theta_b):
    """Shchepetkin & McWilliams (2009) stretching (ROMS Vstretching=4)."""
    sigma = np.asarray(sigma, dtype=float)
    if theta_s > 0:
        c = (1.0 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1.0)
    else:
        c = -(sigma**2)
    if theta_b > 0:
        c = (np.exp(theta_b * c) - 1.0) / (1.0 - np.exp(-theta_b))
    return c


def depths(h, zeta, hc, cs, sigma, vtransform):
    """ROMS z (positive up) for broadcastable numpy inputs."""
    if vtransform == 1:
        zo = hc * (sigma - cs) + cs * h
        return zo + zeta * (1.0 + zo / h)
    zo = (hc * sigma + cs * h) / (hc + h)
    return zeta + (zeta + h) * zo


def _axis(n, d0, stretch, uniform):
    """Cell widths at rho points and positions consistent with u-point metrics."""
    if uniform:
        d_rho = np.full(n, d0)
    else:
        d_rho = d0 * (1.0 + stretch * np.linspace(0.0, 1.0, n))
    p_rho = 1.0 / d_rho
    d_inner = 1.0 / (0.5 * (p_rho[:-1] + p_rho[1:]))
    pos = np.concatenate([[0.0], np.cumsum(d_inner)])
    return p_rho, pos


def _canonical(nt, N, neta, nxi, vtransform, hc, theta_s, theta_b, uniform, angle, land):
    """Build the canonical (UCLA-named) dataset plus helper arrays."""
    pm1, x1 = _axis(nxi, 1000.0, 0.3, uniform)
    pn1, y1 = _axis(neta, 1500.0, 0.2, uniform)
    xr_ = np.broadcast_to(x1[None, :], (neta, nxi)).copy()
    yr_ = np.broadcast_to(y1[:, None], (neta, nxi)).copy()
    pm = np.broadcast_to(pm1[None, :], (neta, nxi)).copy()
    pn = np.broadcast_to(pn1[:, None], (neta, nxi)).copy()
    xmax, ymax = x1[-1], y1[-1]
    h = 20.0 + 60.0 * xr_ / xmax + 20.0 * yr_ / ymax
    t = np.arange(nt, dtype=float)
    zeta = (
        0.2
        * np.sin(2 * np.pi * xr_ / xmax)[None]
        * np.cos(np.pi * yr_ / ymax)[None]
        * (1.0 + 0.5 * t)[:, None, None]
    )
    k = np.arange(N)
    s_rho = (k - N + 0.5) / N
    s_w = (np.arange(N + 1) - N) / N
    cs_r = stretching(s_rho, theta_s, theta_b)
    cs_w = stretching(s_w, theta_s, theta_b)
    z_rho = depths(h[None, None], zeta[:, None], hc, cs_r[None, :, None, None], s_rho[None, :, None, None], vtransform)

    temp = TEMP_A * xr_[None, None] + TEMP_B * z_rho + TEMP_0
    salt = SALT_0 + SALT_C * z_rho**2

    # velocity-point positions (averages of rho positions, as ROMS does)
    x_u = 0.5 * (xr_[:, :-1] + xr_[:, 1:])
    y_v = 0.5 * (yr_[:-1, :] + yr_[1:, :])
    u = np.broadcast_to(U_0 + U_A * x_u, (nt, N) + x_u.shape).copy()
    v = np.broadcast_to(V_0 + V_A * y_v, (nt, N) + y_v.shape).copy()

    mask_rho = np.ones((neta, nxi))
    if land:
        mask_rho[:2, :3] = 0.0
    mask_u = mask_rho[:, :-1] * mask_rho[:, 1:]
    mask_v = mask_rho[:-1, :] * mask_rho[1:, :]
    mask_psi = mask_u[:-1, :] * mask_u[1:, :]
    if land:
        temp = np.where(mask_rho[None, None] == 1, temp, np.nan)
        salt = np.where(mask_rho[None, None] == 1, salt, np.nan)
        u = np.where(mask_u[None, None] == 1, u, np.nan)
        v = np.where(mask_v[None, None] == 1, v, np.nan)
        zeta = np.where(mask_rho[None] == 1, zeta, np.nan)

    lat0 = 28.0
    lon_rho = -90.0 + xr_ / (111.2e3 * np.cos(np.deg2rad(lat0)))
    lat_rho = lat0 + yr_ / 111.2e3

    def avg_xi(a):
        return 0.5 * (a[..., :-1] + a[..., 1:])

    def avg_eta(a):
        return 0.5 * (a[..., :-1, :] + a[..., 1:, :])

    rho, uu, vv, pp = ("eta_rho", "xi_rho"), ("eta_rho", "xi_u"), ("eta_v", "xi_rho"), ("eta_v", "xi_u")
    four = lambda hd: ("time", "s_rho") + hd  # noqa: E731
    ds = xr.Dataset(
        {
            "zeta": (("time",) + rho, zeta, {"long_name": "free-surface", "units": "meter"}),
            "u": (four(uu), u, {"long_name": "u-momentum component", "units": "meter second-1"}),
            "v": (four(vv), v, {"long_name": "v-momentum component", "units": "meter second-1"}),
            "temp": (four(rho), temp, {"long_name": "potential temperature", "units": "Celsius"}),
            "salt": (four(rho), salt, {"long_name": "salinity", "units": "PSU"}),
            "h": (rho, h, {"long_name": "bathymetry at RHO-points", "units": "meter"}),
            "pm": (rho, pm, {"long_name": "curvilinear coordinate metric in XI", "units": "meter-1"}),
            "pn": (rho, pn, {"long_name": "curvilinear coordinate metric in ETA", "units": "meter-1"}),
            "angle": (rho, np.full((neta, nxi), float(angle)), {"long_name": "angle between XI-axis and EAST", "units": "radians"}),
            "f": (rho, np.full((neta, nxi), F0), {"long_name": "Coriolis parameter at RHO-points", "units": "second-1"}),
            "mask_rho": (rho, mask_rho),
            "mask_u": (uu, mask_u),
            "mask_v": (vv, mask_v),
            "mask_psi": (pp, mask_psi),
            "lon_rho": (rho, lon_rho, {"units": "degree_east"}),
            "lat_rho": (rho, lat_rho, {"units": "degree_north"}),
            "lon_u": (uu, avg_xi(lon_rho), {"units": "degree_east"}),
            "lat_u": (uu, avg_xi(lat_rho), {"units": "degree_north"}),
            "lon_v": (vv, avg_eta(lon_rho), {"units": "degree_east"}),
            "lat_v": (vv, avg_eta(lat_rho), {"units": "degree_north"}),
            "lon_psi": (pp, avg_eta(avg_xi(lon_rho)), {"units": "degree_east"}),
            "lat_psi": (pp, avg_eta(avg_xi(lat_rho)), {"units": "degree_north"}),
        }
    )
    extra = dict(
        s_rho=s_rho, s_w=s_w, cs_r=cs_r, cs_w=cs_w, x_rho=xr_, y_rho=yr_, z_rho=z_rho,
        hc=hc, theta_s=theta_s, theta_b=theta_b, vtransform=vtransform, t=t,
    )
    return ds, extra


def make_dataset(
    layout="rutgers",
    *,
    nt=2,
    N=6,
    neta=9,
    nxi=12,
    vtransform=2,
    hc=20.0,
    theta_s=5.0,
    theta_b=2.0,
    uniform=False,
    angle=0.0,
    land=False,
    romstools_grid=False,
):
    """Return a synthetic dataset in the conventions of ``layout``.

    ``layout`` is one of ``"rutgers"``, ``"ucla"``, ``"croco"``, ``"remora"``.
    For ``"ucla"`` a ``(output, grid)`` tuple is returned; ``romstools_grid``
    adds the variables and attributes roms-tools writes to its grid files.
    """
    ds, ex = _canonical(nt, N, neta, nxi, vtransform, hc, theta_s, theta_b, uniform, angle, land)
    seconds = ex["t"] * 86400.0
    if layout == "rutgers":
        return _to_rutgers(ds, ex, seconds)
    if layout == "ucla":
        return _to_ucla(ds, ex, seconds, romstools_grid)
    if layout == "croco":
        return _to_croco(ds, ex, seconds)
    if layout == "remora":
        return _to_remora(ds, ex, seconds)
    raise ValueError(f"unknown layout {layout!r}")


def _vertical_vars(ex, **names):
    return {
        names.get("cs_r", "Cs_r"): ("s_rho", ex["cs_r"], {"long_name": "S-coordinate stretching curves at RHO-points"}),
        names.get("cs_w", "Cs_w"): ("s_w", ex["cs_w"], {"long_name": "S-coordinate stretching curves at W-points"}),
    }


def _to_rutgers(ds, ex, seconds):
    ds = ds.rename({"time": "ocean_time"})
    # Rutgers names every stagger separately
    for var in [v for v in ds.variables if set(ds[v].dims) & {"xi_u"} and "eta_rho" in ds[v].dims]:
        ds[var] = ds[var].rename({"eta_rho": "eta_u"})
    for var in [v for v in ds.variables if "eta_v" in ds[v].dims and "xi_rho" in ds[v].dims]:
        ds[var] = ds[var].rename({"xi_rho": "xi_v"})
    for var in [v for v in ds.variables if "eta_v" in ds[v].dims and "xi_u" in ds[v].dims]:
        ds[var] = ds[var].rename({"eta_v": "eta_psi", "xi_u": "xi_psi"})
    ds = ds.assign_coords(
        ocean_time=("ocean_time", np.datetime64(EPOCH) + (seconds * 1e9).astype("timedelta64[ns]"), {"long_name": "time since initialization"}),
        s_rho=("s_rho", ex["s_rho"], {"long_name": "S-coordinate at RHO-points"}),
        s_w=("s_w", ex["s_w"], {"long_name": "S-coordinate at W-points"}),
    )
    ds = ds.set_coords([c for c in ds.variables if c.startswith(("lon_", "lat_"))])
    ds = ds.assign(_vertical_vars(ex))
    ds["hc"] = ((), ex["hc"], {"long_name": "S-coordinate parameter, critical depth", "units": "meter"})
    ds["theta_s"] = ((), ex["theta_s"])
    ds["theta_b"] = ((), ex["theta_b"])
    ds["Vtransform"] = ((), ex["vtransform"])
    ds["Vstretching"] = ((), 4)
    ds["spherical"] = ((), 1)
    return ds


def _to_ucla(ds, ex, seconds, romstools_grid):
    out = ds[["zeta", "u", "v", "temp", "salt"]].copy()
    out["ocean_time"] = ("time", seconds, {"long_name": "Time since 2000/01/01", "units": "second"})
    out.attrs.update(
        theta_s=ex["theta_s"], theta_b=ex["theta_b"], hc=ex["hc"],
        Cs_r=ex["cs_r"], Cs_w=ex["cs_w"], rho0=1027.4, title="synthetic UCLA ROMS output",
    )
    grid_vars = ["h", "pm", "pn", "angle", "f", "mask_rho", "lon_rho", "lat_rho"]
    grid = ds[grid_vars].copy()
    grid["spherical"] = ("one", np.array([b"T"], dtype="S1"))
    if romstools_grid:
        grid = grid.assign(
            {
                "lon_u": ds.lon_u, "lat_u": ds.lat_u, "lon_v": ds.lon_v, "lat_v": ds.lat_v,
                "mask_u": ds.mask_u, "mask_v": ds.mask_v,
                "sigma_r": ("s_rho", ex["s_rho"]), "sigma_w": ("s_w", ex["s_w"]),
            }
        )
        grid = grid.assign(_vertical_vars(ex))
        grid = grid.set_coords(["lon_rho", "lat_rho", "lon_u", "lat_u", "lon_v", "lat_v"])
        grid = grid.drop_vars("spherical")
        grid.attrs.update(theta_s=ex["theta_s"], theta_b=ex["theta_b"], hc=ex["hc"], straddle="False")
    return out, grid


def _to_croco(ds, ex, seconds):
    ds = ds.assign_coords(time=("time", seconds, {"long_name": "time since initialization", "units": "second"}))
    ds = ds.set_coords([c for c in ds.variables if c.startswith(("lon_", "lat_"))])
    ds = ds.drop_vars(["lon_psi", "lat_psi", "mask_psi"])
    ds = ds.assign(_vertical_vars(ex))
    ds["sc_r"] = ("s_rho", ex["s_rho"], {"long_name": "S-coordinate at RHO-points"})
    ds["sc_w"] = ("s_w", ex["s_w"], {"long_name": "S-coordinate at W-points"})
    ds["hc"] = ((), ex["hc"])
    ds["theta_s"] = ((), ex["theta_s"])
    ds["theta_b"] = ((), ex["theta_b"])
    ds.attrs.update(Vtransform=float(ex["vtransform"]), VertCoordType="NEW" if ex["vtransform"] == 2 else "OLD", hc=ex["hc"])
    return ds


def _to_remora(ds, ex, seconds):
    ds = _to_rutgers(ds, ex, seconds)
    ds = ds.drop_vars(["Vtransform", "Vstretching", "spherical"])
    # Cartesian positions instead of lon/lat
    xr_, yr_ = ex["x_rho"], ex["y_rho"]
    ds = ds.drop_vars([c for c in ds.variables if c.startswith(("lon_", "lat_"))])
    ds = ds.assign_coords(
        x_rho=(("eta_rho", "xi_rho"), xr_), y_rho=(("eta_rho", "xi_rho"), yr_),
        x_u=(("eta_u", "xi_u"), 0.5 * (xr_[:, :-1] + xr_[:, 1:])), y_u=(("eta_u", "xi_u"), 0.5 * (yr_[:, :-1] + yr_[:, 1:])),
        x_v=(("eta_v", "xi_v"), 0.5 * (xr_[:-1] + xr_[1:])), y_v=(("eta_v", "xi_v"), 0.5 * (yr_[:-1] + yr_[1:])),
    )
    # masks can change in time (wetting and drying)
    for m in ["mask_rho", "mask_u", "mask_v"]:
        ds[m] = ds[m].expand_dims(ocean_time=ds.ocean_time).copy()
    ds["grid"] = (
        (),
        0,
        {
            "cf_role": "grid_topology",
            "topology_dimension": 2,
            "node_dimensions": "xi_psi eta_psi",
            "face_dimensions": "xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: both)",
            "edge1_dimensions": "xi_u: xi_psi eta_u: eta_psi (padding: both)",
            "edge2_dimensions": "xi_v: xi_psi (padding: both) eta_v: eta_psi",
            "vertical_dimensions": "s_rho: s_w (padding: none)",
        },
    )
    return ds


def analytic_z(ds_layout_extra):
    """Convenience: the exact z_rho used to build the analytic fields."""
    return ds_layout_extra["z_rho"]
