#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:28:28 2021

@author: sebastiengorgoni
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import re
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred

sns.set_theme(style="darkgrid")

os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Quantitative Asset & Risk Management 2/Project")

from import_data import get_spi

# =============================================================================
# Import Data
# =============================================================================

"""Swiss Performance Index"""
#Price Constituents
price_spi_cons = get_spi()[0] 
index =  price_spi_cons.index

returns_spi_cons = (price_spi_cons/price_spi_cons.shift(1) - 1).replace(np.nan, 0)
returns_spi_cons = returns_spi_cons.replace([np.inf, -np.inf], 0)

cov_spi_cons = returns_spi_cons.cov()

m_range = range(0,12)

roll_var_spi_cons = returns_spi_cons.copy()
roll_var_spi_cons = abs(roll_var_spi_cons*0)

for i in m_range:
    roll_var_spi_cons += (returns_spi_cons.shift(i) - returns_spi_cons.mean())**2

roll_vol_spi_cons = np.sqrt(roll_var_spi_cons/(len(m_range)-1)).dropna()

#PE Ratio Constituents
pe_spi_cons = get_spi()[1] 

#Dividend Yield Constituents
dividend_spi_cons = get_spi()[2]

#Market Cap Constituents
mktcap_spi_cons = get_spi()[3]

#Beta Constituents
beta_spi_cons = get_spi()[4]

#Volatility Constituents
vol_spi_cons = get_spi()[5]

#ROE Constituents
roe_spi_cons = get_spi()[6]

#ROA Constituents
roa_spi_cons = get_spi()[7]

#Gross Margin Constituents
gm_spi = get_spi()[8]

#Benchmark SPI
price_spi_cons = pd.read_excel("Data/SPI_DATA_ALL.xlsx", sheet_name='SPI Index')
price_spi_cons.index = price_spi_cons['Date']
price_spi_cons = price_spi_cons[(price_spi_cons.index >= '2000-01-01')]
del price_spi_cons['Date']
price_spi_cons = price_spi_cons.groupby(pd.Grouper(freq="M")).mean() 
price_spi_cons.index = index

plt.plot(price_spi_cons)

# Alpha Vantage Key: O6PSHZOQS29QHD3E

"""Macro Data"""
#FRED Key: 2fd4cf1862f877db032b4a6a3a5f1c77
fred = Fred(api_key='2fd4cf1862f877db032b4a6a3a5f1c77')

#Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland (Monthly)
gov_bond_ch = fred.get_series('IRLTLT01CHM156N')
gov_bond_ch = gov_bond_ch[(gov_bond_ch.index >= '2000-01-01') & (gov_bond_ch.index < '2021-01-01')]

#CBOE Volatility Index: VIX (Daily)
vix =  fred.get_series('VIXCLS')
vix = vix[(vix.index >= '2000-01-01') & (vix.index < '2021-01-01')]
vix = vix.groupby(pd.Grouper(freq="M")).mean() 
vix.index = index

#Consumer Price Index: All Items for Switzerland (Monthly)
cpi_CH =  fred.get_series('CHECPIALLMINMEI')
cpi_CH = cpi_CH[(cpi_CH.index >= '2000-01-01') & (cpi_CH.index < '2021-01-01')]

#TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill (Daily)
spread_US = fred.get_series('TEDRATE')
spread_US = spread_US[(spread_US.index >= '2000-01-01') & (spread_US.index < '2021-01-01')]
spread_US = spread_US.groupby(pd.Grouper(freq="M")).mean() 
spread_US.index = index

macro_data = pd.DataFrame({'LT 10y Gov. Bond Yield': gov_bond_ch, 'VIX': vix, 
                           'CPI CH': cpi_CH, 'Spread US': spread_US}).dropna()

# =============================================================================
# Factor Construction
# =============================================================================

# Create a function to compute the cumulative returns 
def cum_prod(returns):
    return (returns + 1).cumprod()*100

"""MOMENTUM"""
returns_past12_mom = (returns_spi_cons + 1).rolling(12).apply(np.prod) - 1
returns_past12_mom = returns_past12_mom.dropna()

quantile_mom = returns_past12_mom.quantile(q=0.90, axis=1)

position_mom = returns_past12_mom.copy()

for i in position_mom.columns:
    position_mom.loc[returns_past12_mom[i] >= quantile_mom, i] = 1
    position_mom.loc[returns_past12_mom[i] < quantile_mom, i] = 0

#Equal Weight
position_mom = position_mom.div(position_mom.sum(axis=1), axis=0)

returns_mom = (returns_spi_cons*position_mom).replace(-0, 0).dropna()
returns_mom = returns_mom.sum(axis=1)

plt.plot(cum_prod(returns_mom))

"""VALUE"""
quantile_value = pe_spi_cons.quantile(q=0.25, axis=1)
quantile_value.index = index

position_value = pe_spi_cons.copy()

for i in position_value.columns:
    position_value.loc[pe_spi_cons[i] <= quantile_value, i] = 1
    position_value.loc[pe_spi_cons[i] > quantile_value, i] = 0
    
position_value = position_value.replace(np.nan, 0)

#Equal Weight
position_value = position_value.div(position_value.sum(axis=1), axis=0)

returns_value = (returns_spi_cons*position_value).replace(-0, 0).dropna()
returns_value = returns_value.sum(axis=1)

plt.plot(cum_prod(returns_value))

"""SIZE (SMALL VS. BIG)"""
quantile_size = mktcap_spi_cons.quantile(q=0.10, axis=1)

position_size = pe_spi_cons.copy()

for i in position_size.columns:
    position_size.loc[mktcap_spi_cons[i] <= quantile_size, i] = 1
    position_size.loc[mktcap_spi_cons[i] > quantile_size, i] = 0
    
position_size = position_size.replace(np.nan, 0)

#Equal Weight
position_size = position_size.div(position_size.sum(axis=1), axis=0)

returns_size = (returns_spi_cons*position_size).replace(-0, 0).dropna()
returns_size = returns_size.sum(axis=1)

plt.plot(cum_prod(returns_size))

"""PROFITABILITY"""
quantile_profit = roa_spi_cons.quantile(q=0.75, axis=1)

position_profit = roa_spi_cons.copy()

for i in position_profit.columns:
    position_profit.loc[roa_spi_cons[i] >= quantile_profit, i] = 1
    position_profit.loc[roa_spi_cons[i] < quantile_profit, i] = 0
    
position_profit = position_profit.replace(np.nan, 0)

#Equal Weight
position_profit = position_profit.div(position_profit.sum(axis=1), axis=0)

returns_profit = (returns_spi_cons*position_profit).replace(-0, 0).dropna()
returns_profit = returns_profit.sum(axis=1)

plt.plot(cum_prod(returns_profit))

"""BETA"""
quantile_beta = beta_spi_cons.quantile(q=0.50, axis=1)

position_beta = beta_spi_cons.copy()

for i in position_beta.columns:
    position_beta.loc[beta_spi_cons[i] <= quantile_beta, i] = 1
    position_beta.loc[beta_spi_cons[i] > quantile_beta, i] = 0
    
position_beta = position_beta.replace(np.nan, 0)

#Equal Weight
position_profit = position_profit.div(position_profit.sum(axis=1), axis=0)

returns_beta = (returns_spi_cons*position_beta).replace(-0, 0).dropna()
returns_beta = returns_beta.sum(axis=1)

plt.plot(cum_prod(returns_beta))

"""VOLATILITY"""
quantile_vol = roll_vol_spi_cons.quantile(q=0.25, axis=1)

position_vol = roll_vol_spi_cons.copy()

for i in position_vol.columns:
    position_vol.loc[roll_vol_spi_cons[i] >= quantile_vol, i] = 0
    position_vol.loc[roll_vol_spi_cons[i] < quantile_vol, i] = 1
    
position_vol = position_vol.replace(np.nan, 0)

#Equal Weight
position_profit = position_profit.div(position_profit.sum(axis=1), axis=0)

returns_vol = (returns_spi_cons*position_vol).replace(-0, 0).dropna()
returns_vol = returns_vol.sum(axis=1)

plt.plot(cum_prod(returns_vol))


#WORK UNDER PROGRESS


def MCR_calc(alloc, Returns):
    
    global cov_spi_cons
    
    """ 
    This function computes the marginal contribution to risk (MCR), which 
    determine how much the portfolio volatility would change if we increase
    the weight of a particular asset.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    MCR : Object
        Marginal contribution to risk (MCR)
    """
    ptf = np.multiply(Returns,alloc);
    ptfReturns = np.sum(ptf,1); # Summing across columns
    vol_ptf = np.std(ptfReturns);
    Sigma = cov_spi_cons
    MCR = np.matmul(Sigma,np.transpose(alloc))/vol_ptf;
    return MCR

###ERC Allocation###
def ERC(alloc,Returns):
    """ 
    This function computes the Equally-Weighted Risk Contribution Portfolio (ERC),
    which attributes the same risk contribution to all the assets.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    criterions : Object
        Optimal weights of assets in the portfolio.
    """
    global cov_spi_cons
    ptf = np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns = np.sum(ptf,1); # Summing across columns
    vol_ptf = np.std(ptfReturns);
    indiv_ERC = alloc*MCR_calc(alloc,Returns);
    criterion = np.power(indiv_ERC-vol_ptf/len(alloc),2)
    criterion = np.sum(criterion)*1000000000
    return criterion

x0 = np.array(returns_mom.shape[1]*[0])+0.00001 #Set the first weights of the Gradient Descent

cons=({'type':'eq', 'fun': lambda x: sum(x)-1}) #Sum of weights is equal to 1

Bounds= [(0 , 1) for i in range(0,returns_mom.shape[1])] #Long only positions

#Optimisation
res_ERC = minimize(ERC, x0, method='SLSQP', args=(returns_mom),bounds=Bounds,constraints=cons,options={'disp': True})
