import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

#Create files in the working directory
#if not os.path.isdir('Plot'):
#    os.makedirs('Plot')

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


def perf(data, name):
    """
    This function compute all the required performances of a time series.
    It also plot the monthly returns, the evolution of the mayx drawdown and 
    the cumulative return of the portfolio vs. benchmark

    Parameters
    ----------
    data : TYPE
        Returns of a given portfolio.
    benchmark : TYPE
        Returns of the benchmark.
    name : TYPE
        Name of the dataframe.
    name_plt : TYPE
        Name given to the plot.

    Returns
    -------
    df : TYPE
        Return a dataframe that contains the annualized returns, volatility,
        Sharpe ratio, max drawdown and hit ratio.

    """
    plt.figure(figsize=(20,7))
    exp = np.mean(data,0)*12
    vol = np.std(data,0)*np.power(12,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((data+1).cumprod())
    plt.subplot(121)
    plt.plot(max_dd, 'g')
    plt.title("Evolution of Max Drawdown", fontsize=15)
    hit = hit_ratio(data)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, index = ['Annualized Return', 'Annualized STD', 'Sharpe Ratio', 'Max Drawdown', 'Hit Ratio'])
    plt.subplot(122)
    plt.plot(cum_prod(data), 'b', label=name)
    plt.legend(loc='upper left', frameon=True)
    plt.title("Cumulative Return", fontsize=15)
    #plt.savefig('Plot/'+name+'.png')
    plt.show()
    plt.close()
    return df
  
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

def TE(weight_ptf, weight_target, returns_ptf):
    """
    This function computes the tracking error between
    a portfolio and a benchmark.

    Parameters
    ----------
    weight_ptf : TYPE
        Weight of our portfolio.
        
    weight_target : TYPE
        Weight the benchmark portfolio.
    
    sigma : TYPE
        Covariance matrix.

    Returns
    -------
    vol_month : TYPE
        Monthly Tracking Error.

    """
    
    sigma = returns_ptf.cov().values
    diff_alloc = weight_ptf - weight_target
    temp =  np.matmul(diff_alloc.T, sigma)
    var_month = np.matmul(temp, diff_alloc)
    vol_month = np.power(var_month, 0.5)
    return vol_month 
