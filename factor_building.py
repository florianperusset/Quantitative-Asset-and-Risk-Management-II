import numpy as np

def factor_building(metric, quantile=0.5, long_above_quantile=True, ew_position=True):
    """
    Builds a factor, that is, determines the weights in each security within the factor
    
    Parameters
    ----------
    metric : DataFrame
        the metric used to build the factor.
    quantile : int, optional
        quantile to build the factor. The default is 0.5.
    long_above_quantile : boolean, optional
        determines if we long the securities above or below the quantile in the construction of the factor. The default is True.
    Returns
    -------
    position_factor : DataFrame
        the DataFrame with the weights in each security in the factor.
    """
    quantile_factor = metric.quantile(q=quantile, axis=1)
    position_factor = metric.copy()
    
    if long_above_quantile:
        for i in position_factor.columns:
            position_factor.loc[metric[i] >= quantile_factor, i] = 1
            position_factor.loc[metric[i] < quantile_factor, i] = 0
    else:
        for i in position_factor.columns:
            position_factor.loc[metric[i] >= quantile_factor, i] = 0
            position_factor.loc[metric[i] < quantile_factor, i] = 1
    
    if ew_position:
        position_factor = position_factor.div(position_factor.sum(axis=1), axis=0).replace(np.nan,0)
        
    return position_factor
