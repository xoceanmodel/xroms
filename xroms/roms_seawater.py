import numpy as np

def density(T, S, Z):
    '''Return the density based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       array-like, temperature
    S       array-like, salinity
    Z       array-like, depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, density based on ROMS Nonlinear/rho_eos.F EOS
    '''
    A00=+19092.56;    A01=+209.8925;     A02=-3.041638;     A03=-1.852732E-3
    A04=-1.361629E-5; B00=+104.4077;     B01=-6.500517;     B02=+0.1553190
    B03=+2.326469E-4; D00=-5.587545;     D01=+0.7390729;    D02=-1.909078E-2
    E00=+4.721788E-1; E01=+1.028859E-2;  E02=-2.512549E-4;  E03=-5.939910E-7
    F00=-1.571896E-2; F01=-2.598241E-4;  F02=+7.267926E-6;  G00=+2.042967E-3
    G01=+1.045941E-5; G02=-5.782165E-10; G03=+1.296821E-7;  H00=-2.595994E-7
    H01=-1.248266E-9; H02=-3.508914E-9;  Q00=+999.842594;   Q01=+6.793952E-2
    Q02=-9.095290E-3; Q03=+1.001685E-4;  Q04=-1.120083E-6;  Q05=+6.536332E-9
    U00=+0.824493E0;  U01=-4.08990E-3;   U02=+7.64380E-5;   U03=-8.24670E-7
    U04=+5.38750E-9;  V00=-5.72466E-3;   V01=+1.02270E-4;   V02=-1.65460E-6
    W00=+4.8314E-4
    g=9.81
    sqrtS=np.sqrt(S)
    den1 = (Q00 + Q01*T + Q02*T**2 + Q03*T**3 + Q04*T**4 + Q05*T**5 +
            U00*S + U01*S*T + U02*S*T**2 + U03*S*T**3 + U04*S*T**4 +
            V00*S*sqrtS + V01*S*sqrtS*T + V02*S*sqrtS*T**2 +
            W00*S**2)
    K0 = (A00 + A01*T + A02*T**2 + A03*T**3 + A04*T**4 +
          B00*S + B01*S*T + B02*S*T**2 + B03*S*T**3 +
          D00*S*sqrtS + D01*S*sqrtS*T + D02*S*sqrtS*T**2)
    K1 = (E00 + E01*T + E02*T**2 + E03*T**3 +
          F00*S + F01*S*T + F02*S*T**2 +
          G00*S*sqrtS)
    K2 = G01 + G02*T + G03*T**2 + H00*S + H01*S*T + H02*S*T**2
    bulk = K0 - K1*Z + K2*Z**2

    return (den1*bulk) / (bulk + 0.1*Z)


def buoyancy(T, S, Z, rho0=1000.0):
    '''Return the buoyancy based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       array-like, temperature
    S       array-like, salinity
    Z       array-like, depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, buoyancy based on ROMS Nonlinear/rho_eos.F EOS
            rho = -g * rho / rho0

    Options:
    -------

    rho0    Constant. The reference density. Default rho0=1000.0
    '''

    g = 9.81
    return -g * density(T, S, Z) / rho0


# def stratification_frequency(ds):
#     '''Calculate buoyancy frequency squared, or N^2.'''
    
    
#     T = ds.temp
#     S = ds.salt
#     Zw = ds.z_w
#     Zr = ds.z_rho

#     sqrtS=np.sqrt(S)
#     A00=+19092.56;    A01=+209.8925;     A02=-3.041638;     A03=-1.852732E-3
#     A04=-1.361629E-5; B00=+104.4077;     B01=-6.500517;     B02=+0.1553190
#     B03=+2.326469E-4; D00=-5.587545;     D01=+0.7390729;    D02=-1.909078E-2
#     E00=+4.721788E-1; E01=+1.028859E-2;  E02=-2.512549E-4;  E03=-5.939910E-7
#     F00=-1.571896E-2; F01=-2.598241E-4;  F02=+7.267926E-6;  G00=+2.042967E-3
#     G01=+1.045941E-5; G02=-5.782165E-10; G03=+1.296821E-7;  H00=-2.595994E-7
#     H01=-1.248266E-9; H02=-3.508914E-9;  Q00=+999.842594;   Q01=+6.793952E-2
#     Q02=-9.095290E-3; Q03=+1.001685E-4;  Q04=-1.120083E-6;  Q05=+6.536332E-9
#     U00=+0.824493E0;  U01=-4.08990E-3;   U02=+7.64380E-5;   U03=-8.24670E-7
#     U04=+5.38750E-9;  V00=-5.72466E-3;   V01=+1.02270E-4;   V02=-1.65460E-6
#     W00=+4.8314E-4

#     g=9.81

#     den1 = (Q00 + Q01*T + Q02*T**2 + Q03*T**3 + Q04*T**4 + Q05*T**5 +
#             U00*S + U01*S*T + U02*S*T**2 + U03*S*T**3 + U04*S*T**4 +
#             V00*S*sqrtS + V01*S*sqrtS*T + V02*S*sqrtS*T**2 +
#             W00*S**2)

#     K0 = (A00 + A01*T + A02*T**2 + A03*T**3 + A04*T**4 +
#           B00*S + B01*S*T + B02*S*T**2 + B03*S*T**3 +
#           D00*S*sqrtS + D01*S*sqrtS*T + D02*S*sqrtS*T**2)

#     K1 = (E00 + E01*T + E02*T**2 + E03*T**3 +
#           F00*S + F01*S*T + F02*S*T**2 +
#           G00*S*sqrtS)

#     K2 = G01 + G02*T + G03*T**2 + H00*S + H01*S*T + H02*S*T**2

#     # below is some fugly coordinate wrangling to keep things in
#     # the xarray universe. This could probably be cleaned up.
#     Nm = len(ds.s_w) - 1
#     Nmm = len(ds.s_w) - 2

#     upw = {'s_w': slice(1, None)}
#     dnw = {'s_w': slice(None, -1)}

#     Zw_up = Zw.isel(**upw)
#     Zw_up = Zw_up.rename({'s_w': 's_rho'})
#     Zw_up.coords['s_rho'] = np.arange(Nm)
#     Zw_dn = Zw.isel(**dnw)
#     Zw_dn = Zw_dn.rename({'s_w': 's_rho'})
#     Zw_dn.coords['s_rho'] = np.arange(Nm)

#     K0.coords['s_rho'] = np.arange(Nm)
#     K1.coords['s_rho'] = np.arange(Nm)
#     K2.coords['s_rho'] = np.arange(Nm)
#     den1.coords['s_rho'] = np.arange(Nm)

#     bulk_up = K0 - Zw_up*(K1-Zw_up*K2)
#     bulk_dn = K0 - Zw_dn*(K1-Zw_dn*K2)

#     den_up = (den1*bulk_up) / (bulk_up + 0.1*Zw_up)
#     den_dn = (den1*bulk_dn) / (bulk_dn + 0.1*Zw_dn)

#     upr = {'s_rho': slice(1, None)}
#     dnr = {'s_rho': slice(None, -1)}

#     den_up = den1.isel(**upr)
#     den_up = den_up.rename({'s_rho': 's_w'})
#     den_up.coords['s_w'] = np.arange(Nmm)
#     den_up = den_up.drop('z_rho')

#     den_dn = den1.isel(**dnr)
#     den_dn = den_dn.rename({'s_rho': 's_w'})
#     den_dn.coords['s_w'] = np.arange(Nmm)
#     den_dn = den_dn.drop('z_rho')

#     Zr_up = Zr.isel(**upr)
#     Zr_up = Zr_up.rename({'s_rho': 's_w'})
#     Zr_up.coords['s_w'] = np.arange(Nmm)
#     Zr_up = Zr_up.drop('z_rho')

#     Zr_dn = Zr.isel(**dnr)
#     Zr_dn = Zr_dn.rename({'s_rho': 's_w'})
#     Zr_dn.coords['s_w'] = np.arange(Nmm)
#     Zr_dn = Zr_dn.drop('z_rho')

#     N2 = -g * (den_up-den_dn)/ (0.5*(den_up+den_dn)*(Zr_up-Zr_dn))

#     # Put the rght vertical coordinates back, for plotting etc.
#     N2.coords['z_w'] = ds.z_w.isel(s_w=slice(1, -1))
#     N2.coords['s_w'] = ds.s_w.isel(s_w=slice(1, -1))

#     return N2
