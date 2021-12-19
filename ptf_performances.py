"""
-----------------------------------------------------------------------
QUANTITATIVE ASSET & RISK MANAGEMENT II

HEC LAUSANNE - AUTUMN 2021

Title: Style Rotation on Swiss Long-Only Equity Factors

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

File Name: ptf_performances.py
-----------------------------------------------------------------------

This is an external file for main.py which compute all necessary performances metrics to analyse
the performances of our portfolios. 

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def cum_prod(returns):
    """
    This function determine the the cumulative returns.

    Parameters
    ----------
    returns : TYPE
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the cumulative returns.

    """    
    return (returns + 1).cumprod()*100

def hit_ratio(return_dataset):
    """
    This function determine the hit ratio of any time series returns

    Parameters
    ----------
    return_dataset : TYPE
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the hit ratio.

    """
    return len(return_dataset[return_dataset >= 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    """
    It determines the maximum drawdown over the cumulative returns
    of a time series.

    Parameters
    ----------
    cum_returns : TYPE
        Cumulative Return.

    Returns
    -------
    max_monthly_drawdown : TYPE
        Evolution of the max drawdown (negative output).

    """
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown
  
def risk_historical(returns, q, n):
    """
    This function compute the VaR and ES using historical method. 

    Parameters
    ----------
    returns : Dataframe
        The returns of a given strategy, asset, etc.
    q : Integer
        The quantile selected to compute the VaR and ES.
    n : Integer
        The number of months to compute the VaR and ES.

    Returns
    -------
    df : Dataframe
        It returns the evolution of VaR and ES.

    """
    VaR_list = []
    ES_list = []
    for i in tqdm(range(returns.shape[0] - n - 1)):
        temp = - returns[i:n+i].dropna()
        temp_sort = temp.sort_values(ascending=False) #Sort in descending order
        #Var#
        VaR_temp = temp_sort.quantile(q)
        #ES#
        ES_temp = temp[temp > VaR_temp].mean()
        VaR_list.append(VaR_temp)
        ES_list.append(ES_temp)
    
    df = pd.DataFrame({'VaR': VaR_list, 'ES': ES_list}, index=returns[n+1:].index)
        
    return df

def TE_exante(weight_ptf, weight_target, returns_ptf):
    """
    This function computes the ex-ante tracking error between
    a portfolio and a benchmark.

    Parameters
    ----------
    weight_ptf : DataFrame
        Weight of our portfolio.
        
    weight_target : DataFrame
        Weight the benchmark portfolio.
    
    returns_ptf : DataFrame
        Returns of portfolio.

    Returns
    -------
    vol_month : DataFrame
        Monthly Tracking Error.

    """
    
    sigma = returns_ptf.cov().values
    diff_alloc = weight_ptf - weight_target
    temp =  np.matmul(diff_alloc.T, sigma)
    var_month = np.matmul(temp, diff_alloc)
    vol_month = np.power(var_month, 0.5)
    return vol_month 

def TE_expost(return_ptf, return_benchmark):
    """
    This function computes the ex-post tracking error between
    the portfolio and the benchmark.    

    Parameters
    ----------
    return_ptf : DataFrame
        DESCRIPTION.
    return_benchmark : DataFrame
        DESCRIPTION.

    Returns
    -------
    expost_TE : Float
        It computes the annualized ex-post TE.

    """
    active_return = return_ptf - return_benchmark
    expost_TE = active_return.dropna().std()*np.sqrt(12)
    return expost_TE

def info_ratio(return_p, return_b):
    """
    This function determine the information ratio of an investment.

    Parameters
    ----------
    return_p : TYPE
        Returns of the actual portfolio.
    return_b : TYPE
        Returns of the benchmark.

    Returns
    -------
    TYPE
        It returns the annualized info. ratio.

    """
    try:
        excess = return_p - return_b
        ir_result = (excess.mean(axis=0)*12)/(excess.std(axis=0)*(12**0.5))
    except:
        ir_result = 0
    return ir_result

def perf(returns_ptf, returns_benchmark, rf, name):
    """
    This function compute all the required performances of a time series.
    It also plot the cumulative return of the portfolio vs. benchmark

    Parameters
    ----------
    returns_ptf : DataFrame
        Returns of a given portfolio.
    returns_benchmark : DataFrame
        Returns of the benchmark.
    name : String
        Name given to the plot.

    Returns
    -------
    df : DataFrame
        Return a dataframe that contains the annualized returns, volatility,
        Sharpe ratio, max drawdown, hit ratio, ex post TE and info. ratio . 

    """
    
    exp = np.mean(returns_ptf,0)*12
    vol = np.std(returns_ptf,0)*np.power(12,0.5)
    sharpe = (exp-rf)/vol
    max_dd = max_drawdown((returns_ptf+1).cumprod())
    hit = hit_ratio(returns_ptf)
    expost_TE = TE_expost(returns_ptf, returns_benchmark)
    ir = info_ratio(returns_ptf, returns_benchmark)    
    risk = risk_historical(returns_ptf, 0.95, 12)
    VaR = risk['VaR'].mean()
    ES = risk['ES'].mean()
    
    df = pd.DataFrame({name: [exp*100, vol*100, sharpe, max_dd.min()*100, hit*100, 
                              expost_TE*100, ir, VaR*100, ES*100]}, 
                      index = ['Ann. Return (%)', 'Ann. STD (%)', 'SR', 
                               'Max DD (%)', 'Hit Ratio (%)', 'TE Ex-Post (%)', 
                               'Info. Ratio', 'VaR (%)', 'ES (%)']).replace(np.nan, 0)
    return df.round(3)

def avg_returns(returns):
    
    avg_1y = returns.iloc[-12:].mean()*12
    avg_3y = returns.iloc[-12*3:].mean()*12
    avg_5y = returns.iloc[-12*5:].mean()*12
    avg_10y = returns.iloc[-12*10:].mean()*12
    avg_all = returns.mean()*12
    
    df = pd.DataFrame({returns.name: [avg_1y*100, avg_3y*100, avg_5y*100, avg_10y*100, avg_all*100]}, 
                                      index = ['1 Year', '3 Years', '5 Years', '10 Years', 'Since Inception']).round(2).T
    
    return df