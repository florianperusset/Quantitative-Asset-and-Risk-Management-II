#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:31:27 2021

@author: Florian
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

os.chdir("/Users/Florian/UNIL/Master Finance/2ème année/Premier Semestre/QARM II/Projects/Project")

from import_data import get_spi
from erc_methods import mcr, erc

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


### LOAD THE DATA
spi = get_spi()

pe_spi_cons = spi[1] # PE ratios for all constituents
dividend_spi_cons = spi[2] # Dividend Yield for all consistuents
mktcap_spi_cons = spi[3] # Market cap for all consituents
beta_spi_cons = spi[4] # Beta of all constituents
vol_spi_cons = spi[5] # Volatility of all constituents
roe_spi_cons = spi[6] # ROE of all constituents
roa_spi_cons = spi[7] # ROA of all constituents
gm_spi = spi[8] # Gross Margin of all constituents


#Benchmark SPI
price_spi_index = pd.read_excel("Data/SPI_DATA_ALL.xlsx", sheet_name='SPI Index')
price_spi_index.index = price_spi_index['Date']
price_spi_index = price_spi_index[(price_spi_index.index >= '2000-01-01')]
del price_spi_index['Date']
price_spi_index = price_spi_index.groupby(pd.Grouper(freq="M")).mean() 
price_spi_index.index = index

returns_spi = price_spi_index / price_spi_index.shift(1) - 1

plt.plot(price_spi_cons)

# Alpha Vantage Key: O6PSHZOQS29QHD3E

"""Macro Data"""
#FRED Key: 2fd4cf1862f877db032b4a6a3a5f1c77
fred = Fred(api_key='2fd4cf1862f877db032b4a6a3a5f1c77')

#Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland (Monthly)
gov_bond_ch = fred.get_series('IRLTLT01CHM156N')
gov_bond_ch = gov_bond_ch[(gov_bond_ch.index >= '2000-01-01')]

#CBOE Volatility Index: VIX (Daily)
vix =  fred.get_series('VIXCLS')
vix = vix[(vix.index >= '2000-01-01')]
vix = vix.groupby(pd.Grouper(freq="M")).mean() 
vix.index = index

#Consumer Price Index: All Items for Switzerland (Monthly)
cpi_CH =  fred.get_series('CHECPIALLMINMEI')
cpi_CH = cpi_CH[(cpi_CH.index >= '2000-01-01')]

#TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill (Daily)
spread_US = fred.get_series('TEDRATE')
spread_US = spread_US[(spread_US.index >= '2000-01-01')]
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

# quantile_mom = returns_past12_mom.quantile(q=0.90, axis=1)
quantile_mom = returns_past12_mom.quantile(q=0.50, axis=1)


position_mom = returns_past12_mom.copy()

for i in position_mom.columns:
    position_mom.loc[returns_past12_mom[i] >= quantile_mom, i] = 1
    position_mom.loc[returns_past12_mom[i] < quantile_mom, i] = 0

#Equal Weight
position_mom = position_mom.div(position_mom.sum(axis=1), axis=0)

returns_mom = (returns_spi_cons*position_mom).replace(-0, 0).dropna()
returns_mom = returns_mom.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_mom))
plt.title("Momentum")

"""VALUE"""
# quantile_value = pe_spi_cons.quantile(q=0.25, axis=1)
quantile_value = pe_spi_cons.quantile(q=0.5, axis=1)
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

plt.figure()
plt.plot(cum_prod(returns_value))
plt.title("Value")


"""SIZE (SMALL VS. BIG)"""
# quantile_size = mktcap_spi_cons.quantile(q=0.10, axis=1)
quantile_size = mktcap_spi_cons.quantile(q=0.50, axis=1)

position_size = pe_spi_cons.copy()

for i in position_size.columns:
    position_size.loc[mktcap_spi_cons[i] <= quantile_size, i] = 1
    position_size.loc[mktcap_spi_cons[i] > quantile_size, i] = 0
    
position_size = position_size.replace(np.nan, 0)

#Equal Weight
position_size = position_size.div(position_size.sum(axis=1), axis=0)

returns_size = (returns_spi_cons*position_size).replace(-0, 0).dropna()
returns_size = returns_size.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_size))
plt.title("Size")


"""PROFITABILITY"""
# quantile_profit = roa_spi_cons.quantile(q=0.75, axis=1)
quantile_profit = roa_spi_cons.quantile(q=0.5, axis=1)

position_profit = roa_spi_cons.copy()

for i in position_profit.columns:
    position_profit.loc[roa_spi_cons[i] >= quantile_profit, i] = 1
    position_profit.loc[roa_spi_cons[i] < quantile_profit, i] = 0
    
position_profit = position_profit.replace(np.nan, 0)

#Equal Weight
position_profit = position_profit.div(position_profit.sum(axis=1), axis=0)

returns_profit = (returns_spi_cons*position_profit).replace(-0, 0).dropna()
returns_profit = returns_profit.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_profit))
plt.title("Profitability")


"""BETA"""
quantile_beta = beta_spi_cons.quantile(q=0.50, axis=1)

position_beta = beta_spi_cons.copy()

for i in position_beta.columns:
    position_beta.loc[beta_spi_cons[i] <= quantile_beta, i] = 1
    position_beta.loc[beta_spi_cons[i] > quantile_beta, i] = 0
    
position_beta = position_beta.replace(np.nan, 0)

#Equal Weight
position_beta = position_beta.div(position_beta.sum(axis=1), axis=0)

returns_beta = (returns_spi_cons*position_beta).replace(-0, 0).dropna()
returns_beta = returns_beta.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_beta))
plt.title("Beta")

"""VOLATILITY"""
# quantile_vol = roll_vol_spi_cons.quantile(q=0.25, axis=1)
quantile_vol = roll_vol_spi_cons.quantile(q=0.5, axis=1)


position_vol = roll_vol_spi_cons.copy()

for i in position_vol.columns:
    position_vol.loc[roll_vol_spi_cons[i] >= quantile_vol, i] = 0
    position_vol.loc[roll_vol_spi_cons[i] < quantile_vol, i] = 1
    
position_vol = position_vol.replace(np.nan, 0)

#Equal Weight
position_vol = position_vol.div(position_vol.sum(axis=1), axis=0)

returns_vol = (returns_spi_cons*position_vol).replace(-0, 0).dropna()
returns_vol = returns_vol.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_vol))
plt.title("Volatility")

#WORK UNDER PROGRESS



# Create a df of factor returns to then make an ERC of factors
returns_factors = pd.DataFrame({"Momentum":returns_mom.values, "Value":returns_value.values[11:],
                               "Size":returns_size.values[11:], "Profitability":returns_profit.values[8:],
                               "Beta":returns_beta.values[11:], "Volatility":returns_vol.values},
                              index = returns_mom.index)

# start the optimization
x0 = np.zeros(len(returns_factors.columns))+0.01 # initial values

# constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})


Bounds= [(0 , 1) for i in range(len(returns_factors.columns))]

res_erc = minimize(erc,x0,args=(returns_factors), bounds=Bounds, method='SLSQP',constraints=constraint_set)
weights_factors_erc = res_erc.x


## Results
erc_returns = np.multiply(returns_factors, weights_factors_erc).sum(1)
erc_perf = cum_prod(erc_returns)
plt.figure()
plt.title("Factor-ERC portfolio performance")
erc_perf.plot()




####################################### Parametric weights
macro_variable = (macro_data['VIX'].iloc[10:] - macro_data['VIX'].mean()) / macro_data['VIX'].std()
returns_factors_shifted = returns_factors.iloc[:-1]
risk_aversion = 3

numerator = 0
denominator = 0

for t in range(len(returns_factors_shifted)):
    denominator += (macro_variable.iloc[t]**2) * (returns_factors_shifted.iloc[t] @ returns_factors_shifted.iloc[t].transpose())
    numerator += macro_variable.iloc[t] * returns_factors_shifted.iloc[t]
theta = np.array((1/risk_aversion) * (numerator/denominator))
theta = theta / (theta.sum()) # rescale

parametric_returns = (theta * returns_factors).sum(axis=1)
parametric_returns.plot()
np.cumprod(1+parametric_returns).plot()


parametric_weights = (theta[0] * position_mom + theta[1] * position_value + theta[2] * position_size 
                      + theta[3] * position_profit + theta[4] * position_beta + theta[5] * position_vol)


full_returns = erc_returns + parametric_returns
plt.figure()
cum_prod(full_returns).plot()
plt.title("ERC + Parametric weights cumulated performance")
























