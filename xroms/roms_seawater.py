import numpy as np
import xroms

g = 9.81





def density(T, S, Z, grid, hcoord=None, scoord=None, attrs=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan):
    """Return the density based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       array-like, temperature
    S       array-like, salinity
    Z       array-like, depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, density based on ROMS Nonlinear/rho_eos.F EOS
    """
    A00 = +19092.56
    A01 = +209.8925
    A02 = -3.041638
    A03 = -1.852732e-3
    A04 = -1.361629e-5
    B00 = +104.4077
    B01 = -6.500517
    B02 = +0.1553190
    B03 = +2.326469e-4
    D00 = -5.587545
    D01 = +0.7390729
    D02 = -1.909078e-2
    E00 = +4.721788e-1
    E01 = +1.028859e-2
    E02 = -2.512549e-4
    E03 = -5.939910e-7
    F00 = -1.571896e-2
    F01 = -2.598241e-4
    F02 = +7.267926e-6
    G00 = +2.042967e-3
    G01 = +1.045941e-5
    G02 = -5.782165e-10
    G03 = +1.296821e-7
    H00 = -2.595994e-7
    H01 = -1.248266e-9
    H02 = -3.508914e-9
    Q00 = +999.842594
    Q01 = +6.793952e-2
    Q02 = -9.095290e-3
    Q03 = +1.001685e-4
    Q04 = -1.120083e-6
    Q05 = +6.536332e-9
    U00 = +0.824493e0
    U01 = -4.08990e-3
    U02 = +7.64380e-5
    U03 = -8.24670e-7
    U04 = +5.38750e-9
    V00 = -5.72466e-3
    V01 = +1.02270e-4
    V02 = -1.65460e-6
    W00 = +4.8314e-4
    g = 9.81
    sqrtS = np.sqrt(S)
    den1 = (
        Q00
        + Q01 * T
        + Q02 * T ** 2
        + Q03 * T ** 3
        + Q04 * T ** 4
        + Q05 * T ** 5
        + U00 * S
        + U01 * S * T
        + U02 * S * T ** 2
        + U03 * S * T ** 3
        + U04 * S * T ** 4
        + V00 * S * sqrtS
        + V01 * S * sqrtS * T
        + V02 * S * sqrtS * T ** 2
        + W00 * S ** 2
    )
    K0 = (
        A00
        + A01 * T
        + A02 * T ** 2
        + A03 * T ** 3
        + A04 * T ** 4
        + B00 * S
        + B01 * S * T
        + B02 * S * T ** 2
        + B03 * S * T ** 3
        + D00 * S * sqrtS
        + D01 * S * sqrtS * T
        + D02 * S * sqrtS * T ** 2
    )
    K1 = (
        E00
        + E01 * T
        + E02 * T ** 2
        + E03 * T ** 3
        + F00 * S
        + F01 * S * T
        + F02 * S * T ** 2
        + G00 * S * sqrtS
    )
    K2 = G01 + G02 * T + G03 * T ** 2 + H00 * S + H01 * S * T + H02 * S * T ** 2
    bulk = K0 - K1 * Z + K2 * Z ** 2
    var = (den1 * bulk) / (bulk + 0.1 * Z)

    if attrs is None:
        attrs = {'name': 'rho', 'long_name': 'density', 'units': 'kg/m^3', 'grid': grid}
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)

    return var


def buoyancy(T, S, Z, grid, rho0=1025.0, hcoord=None, scoord=None, hboundary='extend', hfill_value=np.nan, sboundary='extend', sfill_value=np.nan):
    """Return the buoyancy based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       (DataArray). temperature
    S       (DataArray). salinity
    Z       (DataArray). depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, buoyancy based on ROMS Nonlinear/rho_eos.F EOS
            rho = -g * rho / rho0

    Options:
    -------

    rho0    Constant. The reference density. Default rho0=1025.0
    """

    attrs = {'name': 'buoyancy', 'long_name': 'buoyancy', 
             'units': 'm/s^2', 'grid': grid}

    var = -g * density(T, S, Z, grid) / rho0
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)

    return var


def sig0(T, S, grid, hcoord=None, scoord=None):
    '''Calculate potential density from salt/temp.

    Inputs:
    hcoord     string (None). Name of horizontal grid to interpolate variable
               to. Options are 'rho' and 'psi'.
    scoord     string (None). Name of vertical grid to interpolate variable
               to. Options are 's_rho' and 's_w'.        
    '''
    
    attrs = {'name': 'sig0', 'long_name': 'potential density', 'units': 'kg/m^3', 'grid': grid}
    return density(T, S, 0, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    
def N2(rho, grid, rho0=1025, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan):
    '''Calculate buoyancy frequency squared, or vertical buoyancy gradient.'''

    drhodz = xroms.ddz(rho, grid, hcoord=hcoord, scoord=scoord, sboundary=sboundary, sfill_value=sfill_value)
    var = -g*drhodz/rho0
    attrs = {'name': 'N2', 'long_name': 'buoyancy frequency squared, or vertical buoyancy gradient', 
             'units': '1/s^2', 'grid': grid}
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    return var
    
    
def M2(rho, grid, rho0=1025, hcoord=None, scoord='s_w', hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan, z=None):
    '''Calculate the horizontal buoyancy gradient.

    z          DataArray. The vertical depths associated with q. Default is to find the
               coordinate of var that starts with 'z_', and use that.

    '''

    drhodxi = xroms.ddxi(rho, grid, hcoord=hcoord, scoord=scoord, 
                                  hboundary=hboundary, hfill_value=hfill_value, 
                                  sboundary=sboundary, sfill_value=sfill_value, z=None)
    drhodeta = xroms.ddeta(rho, grid, hcoord=hcoord, scoord=scoord, 
                                  hboundary=hboundary, hfill_value=hfill_value, 
                                  sboundary=sboundary, sfill_value=sfill_value, z=None)
    var = np.sqrt(drhodxi**2 + drhodeta**2) * g/rho0
    attrs = {'name': 'M2', 'long_name': 'horizontal buoyancy gradient', 
             'units': '1/s^2', 'grid': grid}
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)
    return var


def mld(sig0, h, mask, grid, z=None, thresh=0.03, hcoord=None, scoord=None, hboundary='extend', hfill_value=None, sboundary='fill', sfill_value=np.nan):
    '''Calculate the mixed layer depth.
    
    Mixed layer depth is based on the fixed Potential Density (PD) threshold.
    
    Inputs:
    sig0       DataArray. Potential density.
    h          depths [m].
    mask       mask to match sig0
    z          DataArray (None). The vertical depths associated with sig0. Should be on 'rho'
               grid horizontally and vertically. Use coords associated with DataArray sig0
               if not input.
    thresh     float (0.03). In kg/m^3
    
    Converted to xroms by K. Thyng Aug 2020 from:
    
    Update history:
    v1.0 DL 2020Jun07
        
    References:
    ncl mixed_layer_depth function at https://github.com/NCAR/ncl/blob/ed6016bf579f8c8e8f77341503daef3c532f1069/ni/src/lib/nfpfort/ocean.f
    de Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004). Mixed layer depth over the global ocean: An examination of profile data and a profile‐based climatology. Journal of  Geophysical Research: Oceans, 109(C12).
    '''
    
    if h.mean() > 0:  # if depths are positive, change to negative
        h = -h
    
    # xisoslice will operate over the relevant s dimension
    skey = sig0.dims[[dim[:2] == "s_" for dim in sig0.dims].index(True)]
    
    if z is None:
        z = sig0.z_rho
    
    # the mixed layer depth is the isosurface of depth where the potential density equals the surface + a threshold
    mld = xroms.xisoslice(sig0 - sig0.isel(s_rho=-1) - thresh, 0.0, z, skey)
    
    # Replace nan's that are not masked with the depth of the water column.
    cond = (mld.isnull()) & (mask == 1)
    mld = mld.where(~cond, h)

    attrs = {'name': 'mld', 'long_name': 'mixed layer depth', 
             'units': 'm', 'grid': grid}
    mld = xroms.to_grid(mld, grid, hcoord=hcoord, scoord=scoord, attrs=attrs,
                       hboundary=hboundary, hfill_value=hfill_value, sboundary=sboundary, sfill_value=sfill_value)

    return mld
