"""
-----------------------------------------------------------------------
QUANTITATIVE ASSET & RISK MANAGEMENT II

HEC LAUSANNE - AUTUMN 2021

Title: Style Rotation on Swiss Long-Only Equity Factors

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

File Name: factor_building.py
-----------------------------------------------------------------------

This is an external file for main.py which build our various factors to create our portfolio
of factors

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def factor_building(metric, quantile, long_above_quantile=True, ew_position=True):
    """
    This function builds a factor, 
    that is, determines the weights in each security within the factor
    
    Parameters
    ----------
    metric : DataFrame
        the metric used to build the factor.
    quantile : int
        Quantile to build the factor.
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
        
    return position_factor.iloc[12:]

def run_ff_regression(returns_ptf, returns_ff, interest_rate):
    """
    This function will perform a Fama French analysis by running an 
    OLS regression

    Parameters
    ----------
    returns_ptf : DataFrame
        Returns of the portfolio.
    returns_ff : DataFrame
        Returns of the FF factors.
    interest_rate : FLoat
        The risk free rate.

    Returns
    -------
    DataFrame
        The results of the OLS regression (coefficient, p-val, R2).

    """
    
    excess_returns = returns_ptf - interest_rate
    
    index_low = returns_ff.iloc[0].name
    index_high = returns_ff.iloc[-1].name
    
    ff_reg = sm.OLS(excess_returns[index_low:index_high], returns_ff[index_low:index_high]).fit()
    
    ## Merge Results
    df_ff_results = pd.DataFrame({'Coeff. ': ff_reg.params, 'Pval ': ff_reg.pvalues}).T

    df_ff_results.rename(columns={'const': 'Intercept'}, inplace=True)

    df_ff_r2 = pd.DataFrame({'R2': [ff_reg.rsquared, np.nan]}, index = df_ff_results.index)
    
    df_ff_merged = pd.concat([df_ff_results, df_ff_r2], axis=1)
    
    return df_ff_merged.round(3)

