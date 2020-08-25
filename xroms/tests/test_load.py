'''Test package.'''

import xroms
from glob import glob
import os


def test_open_netcdf():
    '''Test xroms.open_netcdf().'''
    
    base = os.path.join(xroms.__path__[0],'tests','input')
    files = glob('%s/ocean_his_000?.nc' % base)
    ds = xroms.open_netcdf(files, Vtransform=2)
    
    assert ds
    
def test_open_zarr():
    '''Test xroms.open_zarr().'''
    
    base = os.path.join(xroms.__path__[0],'tests','input')
    files = glob('%s/ocean_his_000?' % base)
    ds = xroms.open_zarr(files, chunks={'ocean_time':2}, Vtransform=2)
    
    assert ds
    
