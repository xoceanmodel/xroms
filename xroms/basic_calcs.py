import xarray as xr
import numpy as np
import xroms


def mean(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take mean over all or selected dimensions. Provide attributes.

    NOTE: You may want to be using `gridmean` instead of `mean` to account 
    for the variable spatial grid.
    
    var         (DataArray)

    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', mean over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.mean(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def sum(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take sum over all or selected dimensions. Provide attributes.

    NOTE: You may want to be using `gridsum` instead of `sum` to account 
    for the variable spatial grid.
    
    var         (DataArray)

    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', sum over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.sum(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def max(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take max over all or selected dimensions. Provide attributes.
    
    var         (DataArray)
    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', max over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.max(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def min(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take min over all or selected dimensions. Provide attributes.
    
    var         (DataArray)
    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', min over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.min(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def std(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take std over all or selected dimensions. Provide attributes.
    
    var         (DataArray)
    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', std over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.std(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def var(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take var over all or selected dimensions. Provide attributes.
    
    var         (DataArray)
    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', variance over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.var(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var


def median(var, grid, dim=None, attrs=None, hcoord=None, scoord=None):
    '''Take median over all or selected dimensions. Provide attributes.
    
    var         (DataArray)
    dim         (None) Can be None to average over all dimensions, a string of
                one dimension to average over, or a list or tuple of dimension
                names to average over.
    '''

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', median over dim ' + dimstr
        attrs['grid'] = grid
        
    var = var.median(dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var



def gridmean(var, grid, dim, attrs=None, hcoord=None, scoord=None):
    '''Take mean of var accounting for variable spatial grid with xgcm.
    
    dim         dimension names in the `xgcm` convention are 'Z', 'Y', 
                or 'X'. dim can be a string, list, or tuple of combinations  
                of these names for dimensions to average over.
    
    Note that the following two approaches are equivalent:
    > app1 = xroms.gridmean(ds.u, ds.attrs['grid'], ('Y','X'))
    > app2 = (ds.u*ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))/(ds.dy_u*ds.dx_u).sum(('eta_rho','xi_u'))
    > np.allclose(app1, app2)
    '''
    
    assert isinstance(dim, (str,list,tuple)), 'dim must be a string of or a list or tuple containing "X", "Y", and/or "Z"'
    
    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', grid mean over dim ' + dimstr
        attrs['grid'] = grid
        
    var = grid.average(var, dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)
        
    return var
    

def gridsum(var, grid, dim, attrs=None, hcoord=None, scoord=None):
    '''Take sum of var accounting for variable spatial grid with xgcm.
    
    dim         dimension names in the `xgcm` convention are 'Z', 'Y', 
                or 'X'. dim can be a string, list, or tuple of combinations  
                of these names for dimensions to average over.
                                
    Note that the following two approaches are equivalent:
    > app1 = xroms.gridsum(ds.u, ds.attrs['grid'], ('Z','X'))
    > app2 = (ds.u*ds.dz_u * ds.dx_u).sum(('s_rho','xi_u'))
    > np.allclose(app1, app2)    
    '''
    
    assert isinstance(dim, (str,list,tuple)), 'dim must be a string of or a list or tuple containing "X", "Y", and/or "Z"'

    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = attrs.setdefault('name', 'var') 
        attrs['units'] = attrs.setdefault('units', 'units')
        dimstr = dim if isinstance(dim, str) else ', '.join(dim)
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', grid sum over dim ' + dimstr
        attrs['grid'] = grid
    
    var = grid.integrate(var, dim)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)
         
    return var
    

def groupbytime(var, grid, timeperiod, attrs=None, hcoord=None, scoord=None):
    '''DOCS'''

    timeperiods = ['season', 'year', 'month', 'day', 'hour', 'minute', 'second', 'dayofyear', 'week', 'dayofweek', 'weekday', 'quarter']
    
    assert timeperiod in timeperiods, 'timeperiod should be one of ' + ', '.join(timeperiods)
    
    # find time key for dims
    tkey = var.dims[['time' in dim for dim in var.dims].index(True)]
        
    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = var.name
        attrs['units'] = attrs.setdefault('units', 'units')
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', time mean over ' + timeperiod
        attrs['grid'] = grid        
 
    var = var.groupby(tkey + '.' + timeperiod).mean(tkey)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var
    

def downsampletime(var, grid, timefrequency, aggfunction=np.mean, attrs=None, hcoord=None, scoord=None):
    '''DOCS
    
    timefrequency   (string) Accepts pandas/xarray options, such as: "D" (calendar day frequency); 
                    "W" (weekly frequency); "M" (month end frequency); "MS" (month start frequency);
                    "QS" (quarter start frequency); "QS-DEC" (quarter start frequency with quarter
                    starting in DEC instead of JAN); "A", "Y" (year end frequency); "AS", "YS" 
                    (year start frequency); "H" (hourly frequency); "T", "min" (minutely frequency); 
                    "S" (secondly frequency).
                    (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
    aggfunction     (np.mean) Numpy function by which to aggregate the model output for downsampling. 
                    Other options include np.sum, np.max, np.min, np.std, np.var, np.median.
                    
    
    '''
    
    # find time key for dims
    tkey = var.dims[['time' in dim for dim in var.dims].index(True)]
        
    if attrs is None and isinstance(var, xr.DataArray):
        attrs = var.attrs.copy()
        attrs['name'] = var.name
        attrs['units'] = attrs.setdefault('units', 'units')
        funcname = str(aggfunction).split()[1]
        attrs['long_name']  = attrs.setdefault('long_name', 'var') + ', downsampled by time ' + funcname + ' per ' + timefrequency
        attrs['grid'] = grid    
        
    var = var.resample({tkey: timefrequency}).reduce(aggfunction)
    var = xroms.to_grid(var, grid, hcoord=hcoord, scoord=scoord, attrs=attrs)

    return var