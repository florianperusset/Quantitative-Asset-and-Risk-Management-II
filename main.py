import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred

sns.set_theme(style="darkgrid")

os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Quantitative Asset & Risk Management 2/Project")

from import_data import get_spi
from optimization_criteria import criterion_erc, criterion_ridge
from ptf_performances import cum_prod, perf, risk_historical, TE_exante, TE_expost

# Alpha Vantage Key: O6PSHZOQS29QHD3E
# FRED Key: 2fd4cf1862f877db032b4a6a3a5f1c77

## Start the Creation of the portfolio
start_ptf = '2009-01-01'

# =============================================================================
# Import Data
# =============================================================================

spi = get_spi()

"""Swiss Performance Index"""
#Price Constituents
price_spi_cons = spi[0] 
index =  price_spi_cons.index

#Compute the returns
returns_spi_cons = (price_spi_cons/price_spi_cons.shift(1) - 1).replace(np.nan, 0)
returns_spi_cons = returns_spi_cons.replace([np.inf, -np.inf], 0)

#Compute the Covariance matrix
cov_spi_cons = returns_spi_cons.cov()

#Compute the 12-months rolling variance
roll_vol_spi_cons = returns_spi_cons.rolling(12).std()

"""Load the fundamental data"""
pe_spi_cons = spi[1] # PE ratios for all constituents
dividend_spi_cons = spi[2] # Dividend Yield for all consistuents
mktcap_spi_cons = spi[3] # Market cap for all consituents
beta_spi_cons = spi[4] # Beta of all constituents
vol_spi_cons = spi[5] # Volatility of all constituents
roe_spi_cons = spi[6] # ROE of all constituents
roa_spi_cons = spi[7] # ROA of all constituents
gm_spi_cons = spi[8] # Gross Margin of all constituents
eps_spi_cons = spi[9] #EPS of all constituents
trade_spi_cons = spi[10] #Volume traded of all constituents


"""Benchmark SPI"""
price_spi_index = pd.read_excel("Data_SPI/SPI_DATA_ALL.xlsx", sheet_name='SPI Index')
price_spi_index = price_spi_index.set_index('Date')
price_spi_index = price_spi_index[(price_spi_index.index >= '2000-01-01')]
price_spi_index = price_spi_index.groupby(pd.Grouper(freq="M")).mean() 
price_spi_index.index = index

#Compute the returns
returns_spi = price_spi_index / price_spi_index.shift(1) - 1

"""Macro Data"""
fred = Fred(api_key='2fd4cf1862f877db032b4a6a3a5f1c77')

#Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland (Monthly)
gov_bond_CH = fred.get_series('IRLTLT01CHM156N')
gov_bond_CH = gov_bond_CH[(gov_bond_CH.index >= '2000-01-01') & (gov_bond_CH.index < '2021-10-01')]

#Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for the United States
gov_bond_US = fred.get_series('IRLTLT01USM156N')
gov_bond_US = gov_bond_US[(gov_bond_US.index >= '2000-01-01') & (gov_bond_US.index < '2021-10-01')]

#CBOE Volatility Index: VIX (Daily)
vix =  fred.get_series('VIXCLS')
vix = vix[(vix.index >= '1999-12-01') & (vix.index < '2021-10-01')]
vix = vix.groupby(pd.Grouper(freq="M")).mean()
vix.index = index

#Consumer Price Index: All Items for Switzerland (Monthly)
cpi_CH =  fred.get_series('CHECPIALLMINMEI')
cpi_CH = cpi_CH[(cpi_CH.index >= '2000-01-01') & (cpi_CH.index < '2021-10-01')]

#Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (Monthly)
cpi_US = fred.get_series('CPIAUCSL')
cpi_US = cpi_US[(cpi_US.index >= '2000-01-01') & (cpi_US.index < '2021-10-01')]

#TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill (Daily)
TEDspread_US = fred.get_series('TEDRATE')
TEDspread_US = TEDspread_US[(TEDspread_US.index >= '1999-12-01') & (TEDspread_US.index < '2021-10-01')]
TEDspread_US = TEDspread_US.groupby(pd.Grouper(freq="M")).mean() 
TEDspread_US.index = index

#3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor3M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='3M LIBOR')
libor3M_US = libor3M_US.set_index('Date')
libor3M_US = libor3M_US[(libor3M_US.index >= '1999-12-01') & (libor3M_US.index < '2021-10-01')]
libor3M_US = libor3M_US.groupby(pd.Grouper(freq="M")).mean() 
libor3M_US.index = index

#12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor12M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='12M LIBOR')
libor12M_US = libor12M_US.set_index('Date')
libor12M_US = libor12M_US[(libor12M_US.index >= '1999-12-01') & (libor12M_US.index < '2021-10-01')]
libor12M_US = libor12M_US.groupby(pd.Grouper(freq="M")).mean() 
libor12M_US.index = index

#Merge all macro data
macro_data_df = pd.DataFrame({'LT 10y Gov. Bond Yield US': gov_bond_US, 'VIX': vix,
                              'CPI US': cpi_US, 'Spread US': TEDspread_US}).dropna()

macro_data = pd.concat([macro_data_df, libor3M_US, libor12M_US], axis=1).dropna()

#Lag the Macro Data
macro_data = macro_data.shift(3) 

# =============================================================================
# Create a Cap-Weighted Benchmark
# =============================================================================

"""Cap-Weighted Benchmark"""
cw_spi_cons = mktcap_spi_cons.divide(mktcap_spi_cons.sum(axis=1), axis='index')

cw_spi_index = (cw_spi_cons*returns_spi_cons).sum(axis=1)
cw_spi_cons.index = pd.to_datetime(cw_spi_cons.index)

perf_cwbenchmark = perf(cw_spi_index[start_ptf:], cw_spi_index[start_ptf:], 'CW Benchmark')

# =============================================================================
# Trade Constraint: Trade only Liquid Equities
# =============================================================================

"""Trade Only Liquid Equities"""
trade_liq = pd.DataFrame(np.zeros(price_spi_cons.shape), columns = price_spi_cons.columns, index = price_spi_cons.index)

trade_liq_quantile = trade_spi_cons.quantile(0.25, axis=1)

for i in trade_liq.columns:
    trade_liq.loc[trade_spi_cons[i] >= trade_liq_quantile, i] = 1
    
trade_liq = trade_liq.replace(0, np.nan)

#price_spi_cons = price_spi_cons.loc[:, ~(trade_liq  == 0).all()] (https://stackoverflow.com/questions/30351125/python-pandas-drop-a-df-column-if-condition)

price_spi_cons = (price_spi_cons*trade_liq)
pe_spi_cons = (pe_spi_cons*trade_liq)
dividend_spi_cons = (dividend_spi_cons*trade_liq)
mktcap_spi_cons = (mktcap_spi_cons*trade_liq)
beta_spi_cons = (beta_spi_cons*trade_liq)
vol_spi_cons = (vol_spi_cons*trade_liq)
roe_spi_cons = (roe_spi_cons*trade_liq)
roa_spi_cons = (roa_spi_cons*trade_liq)
gm_spi_cons = (gm_spi_cons*trade_liq)
eps_spi_cons = (eps_spi_cons*trade_liq)

# =============================================================================
# Factor Construction
# =============================================================================

"""MOMENTUM (Price)"""
# returns_past12_mom = (returns_spi_cons + 1).rolling(12).apply(np.prod) - 1
# returns_past12_mom = returns_past12_mom.dropna()

returns_past12_mom = returns_spi_cons.rolling(12,closed='left').mean()  #.replace(np.nan, 0)

#quantile_mom = returns_past12_mom.quantile(q=0.90, axis=1)
quantile_mom = returns_past12_mom.quantile(q=0.50, axis=1)

position_mom = returns_past12_mom.copy()

for i in position_mom.columns:
    position_mom.loc[returns_past12_mom[i] >= quantile_mom, i] = 1
    position_mom.loc[returns_past12_mom[i] < quantile_mom, i] = 0

#Equal Weight
position_mom = position_mom.div(position_mom.sum(axis=1), axis=0).replace(np.nan, 0)

#Compute the returns of the factor
returns_mom = position_mom.mul(returns_spi_cons).sum(axis=1)

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

#Compute the returns of the factor
returns_value = position_value.mul(returns_spi_cons).sum(axis=1)

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

#Compute the returns of the factor
returns_size = position_size.mul(returns_spi_cons).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_size))
plt.title("Size")

"""PROFITABILITY"""
# quantile_profit = roa_spi_cons.quantile(q=0.75, axis=1)
quantile_profit = gm_spi_cons.quantile(q=0.5, axis=1)

position_profit = gm_spi_cons.copy()

for i in position_profit.columns:
    position_profit.loc[gm_spi_cons[i] >= quantile_profit, i] = 1
    position_profit.loc[gm_spi_cons[i] < quantile_profit, i] = 0
    
position_profit = position_profit.replace(np.nan, 0)

#Equal Weight
position_profit = position_profit.div(position_profit.sum(axis=1), axis=0).replace(np.nan, 0)

#Compute the returns of the factor
returns_profit = position_profit.mul(returns_spi_cons).sum(axis=1)

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

#Compute the returns of the factor
returns_beta = position_beta.mul(returns_spi_cons).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_beta))
plt.title("Beta")

"""VOLATILITY"""
# quantile_vol = roll_vol_spi_cons.quantile(q=0.25, axis=1)
quantile_vol = roll_vol_spi_cons.quantile(q=0.50, axis=1)

position_vol = roll_vol_spi_cons.copy()

for i in position_vol.columns:
    position_vol.loc[roll_vol_spi_cons[i] >= quantile_vol, i] = 0
    position_vol.loc[roll_vol_spi_cons[i] < quantile_vol, i] = 1
    
position_vol = position_vol.replace(np.nan, 0)

#Equal Weight
position_vol = position_vol.div(position_vol.sum(axis=1), axis=0)

#Compute the returns of the factor
returns_vol = position_vol.mul(returns_spi_cons).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_vol))
plt.title("Volatility")

"""Dividend"""
quantile_div = dividend_spi_cons.quantile(q=0.5, axis=1)

position_div = dividend_spi_cons.copy()

for i in position_div.columns:
    position_div.loc[dividend_spi_cons[i] >= quantile_div, i] = 1
    position_div.loc[dividend_spi_cons[i] < quantile_div, i] = 0
    
position_div = position_div.replace(np.nan, 0)

#Equal Weight
position_div = position_div.div(position_div.sum(axis=1), axis=0)

#Compute the returns of the factor
returns_div = position_div.mul(returns_spi_cons).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_div))
plt.title("Dividend Yield")

"""EPS (Quality Earnings)"""
quantile_eps = eps_spi_cons.quantile(q=0.5, axis=1)

position_eps = eps_spi_cons.copy()

for i in position_eps.columns:
    position_eps.loc[eps_spi_cons[i] >= quantile_eps, i] = 1
    position_eps.loc[eps_spi_cons[i] < quantile_eps, i] = 0
    
position_eps = position_eps.replace(np.nan, 0)

#Equal Weight
position_eps = position_eps.div(position_eps.sum(axis=1), axis=0)

#Compute the returns of the factor
returns_eps = position_eps.mul(returns_spi_cons).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_div))
plt.title("EPS (Quality Earnings")

# Create a df of factor returns
returns_factors = pd.DataFrame({"Momentum":returns_mom, "Value":returns_value,
                               "Size":returns_size, "Profitability":returns_profit,
                               "Beta":returns_beta, "Volatility":returns_vol,
                               "Dividend": returns_div, 'EPS (Earnings Quality)': returns_eps}).dropna()['2001-01-01':]

"""Momentum of Factors"""
returns_factors_past12_mom = returns_factors.rolling(12, closed='left').mean().dropna()

quantile_mom_factor = returns_factors_past12_mom.quantile(q=0.50, axis=1)

position_mom_factor  = returns_factors_past12_mom.copy()

for i in position_mom_factor.columns:
    position_mom_factor.loc[returns_factors_past12_mom[i] >= quantile_mom_factor, i] = 1
    position_mom_factor.loc[returns_factors_past12_mom[i] < quantile_mom_factor, i] = 0

#Equal Weight
position_mom_factor = position_mom_factor.div(position_mom_factor.sum(axis=1), axis=0)

returns_mom_factors = position_mom_factor.mul(returns_factors).sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_mom_factors))
plt.plot(cum_prod(cw_spi_index))
plt.title("Momentum")

# =============================================================================
# ERC of Factors
# =============================================================================

## Start the optimization
x0 = np.zeros(len(returns_factors.columns))+0.01 # initial values

## Constraint set
#constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

Bounds = [(0 , 1) for i in range(len(returns_factors.columns))]

weights_factors_erc = returns_factors.copy()*0

for row in range(returns_factors.loc[:start_ptf].shape[0]-1, len(returns_factors)): #returns_factors.loc[:start_ptf].shape[0]-1
    
    exp_returns_factors = returns_factors.iloc[:row-1]
    
    constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                      {'type':'ineq', 'fun': lambda x: 0.05/np.sqrt(12) - TE_exante((position_mom.iloc[row-1].values * x[0]
                                                                         + position_value.iloc[row-1].values * x[1]
                                                                         + position_size.iloc[row-1].values * x[2]
                                                                         + position_profit.iloc[row-1].values * x[3]
                                                                         + position_beta.iloc[row-1].values * x[4]
                                                                         + position_vol.iloc[row-1].values * x[5]
                                                                         + position_div.iloc[row-1].values * x[6]
                                                                         + position_eps.iloc[row-1].values * x[7]), cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1])})
    
    res_erc = minimize(criterion_erc, x0, args=(exp_returns_factors), bounds=Bounds, method='SLSQP', constraints=constraint_set)
    weights_factors_erc.iloc[row] = res_erc.x

## Compute the returns of the ERC model
erc_returns = np.multiply(returns_factors, weights_factors_erc).sum(axis=1)

## Performances ERC model
perf_erc = perf(erc_returns[start_ptf:], cw_spi_index[start_ptf:], 'ERC Returns')
risk_erc = risk_historical(erc_returns[start_ptf:], 0.95, 12)
risk_erc.plot(figsize=(7,5))

## Evolution of Factor Weigths
plt.figure(figsize=(20, 10))
weights_factors_erc[start_ptf:].plot()
plt.title("Weights Evolution for ERC")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

## Create a DF of the total weights of each SPI constituents
# weights_spi_cons_erc = (position_mom.mul(weights_factors_erc['Momentum'], axis=0) 
#                                 + position_value.mul(weights_factors_erc['Value'], axis=0)
#                                 + position_size.mul(weights_factors_erc['Size'], axis=0)
#                                 + position_profit.mul(weights_factors_erc['Profitability'], axis=0)
#                                 + position_beta.mul(weights_factors_erc['Beta'], axis=0)
#                                 + position_vol.mul(weights_factors_erc['Volatility'], axis=0)
#                                 + position_div.mul(weights_factors_erc['Dividend'], axis=0)
#                                 + position_eps.mul(weights_factors_erc['EPS (Earnings Quality)'], axis=0)).dropna()[start_ptf:]

## TE ex-ante Between Parametric Weights and CW Benchmark
# TE_exante_erc = []
# for row in weights_spi_cons_erc.loc[start_ptf:].index:
#     temp_TE = TE_exante(weights_spi_cons_erc.loc[row].values, cw_spi_cons.loc[row].replace(np.nan,0).values, returns_spi_cons.loc[:row])
#     TE_exante_erc.append(temp_TE)
    
# TE_exante_erc = pd.DataFrame({'TE Parametrics': TE_exante_erc}, index=weights_spi_cons_erc.loc[start_ptf:].index)
# TE_exante_erc.plot(figsize=(7,5))

## TE ex-post Between Parametric Weights and CW Benchmark
TE_expost_erc = TE_expost(erc_returns[start_ptf:], cw_spi_index[start_ptf:])

# =============================================================================
# Ridge Regression of Factors
# =============================================================================
ridge_weights_factors = returns_factors.copy()*0

constraint_set_ridge = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

bounds_ridge = [(0, 1/5) for i in range(len(returns_factors.columns))]

for row in range(returns_factors.loc[:start_ptf].shape[0]-1,len(returns_factors)): #returns_factors.loc[:start_ptf].shape[0]-1
    expected_return = returns_factors.iloc[:row-1].mean()
    varcov_matrix = returns_factors.iloc[:row-1].cov()
    
    #constraint_set_ridge = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

    constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                      {'type':'ineq', 'fun': lambda x: 0.05/np.sqrt(12) - TE_exante((position_mom.iloc[row-1].values * x[0]
                                                                          + position_value.iloc[row-1].values * x[1]
                                                                          + position_size.iloc[row-1].values * x[2]
                                                                          + position_profit.iloc[row-1].values * x[3]
                                                                          + position_beta.iloc[row-1].values * x[4]
                                                                          + position_vol.iloc[row-1].values * x[5]
                                                                          + position_div.iloc[row-1].values * x[6]
                                                                          + position_eps.iloc[row-1].values * x[7]), cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1])})

    
    res_ridge = minimize(criterion_ridge, x0, args=(expected_return,varcov_matrix), bounds=bounds_ridge, method='SLSQP',constraints=constraint_set)
    ridge_weights_factors.iloc[row] = res_ridge.x

## Compute the returns of ridge regression
ridge_returns = np.multiply(returns_factors, ridge_weights_factors).sum(axis=1)    

## Performances Ridge Regression
perf_ridge = perf(ridge_returns[start_ptf:], cw_spi_index[start_ptf:], 'Ridge Returns')
risk_parametric = risk_historical(ridge_returns[start_ptf:], 0.95, 12)
risk_parametric.plot(figsize=(7,5))

## Evolution of Weigths
plt.figure(figsize=(20, 10))
ridge_weights_factors[start_ptf:].plot()
plt.title("Weights Evolution for Ridge Regression")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

## Create a DF of the total weights of each SPI constituents
weights_spi_cons_ridge = (position_mom.mul(ridge_weights_factors['Momentum'], axis=0) 
                                + position_value.mul(ridge_weights_factors['Value'], axis=0)
                                + position_size.mul(ridge_weights_factors['Size'], axis=0)
                                + position_profit.mul(ridge_weights_factors['Profitability'], axis=0)
                                + position_beta.mul(ridge_weights_factors['Beta'], axis=0)
                                + position_vol.mul(ridge_weights_factors['Volatility'], axis=0)
                                + position_div.mul(ridge_weights_factors['Dividend'], axis=0)
                                + position_eps.mul(ridge_weights_factors['EPS (Earnings Quality)'], axis=0)).dropna()[start_ptf:]

## TE ex-ante Between Parametric Weights and CW Benchmark
TE_exante_ridge = []
for row in weights_spi_cons_ridge.loc[start_ptf:].index:
    temp_TE = TE_exante(weights_spi_cons_ridge.loc[row].values, cw_spi_cons.loc[row].replace(np.nan,0).values, returns_spi_cons.loc[:row])
    TE_exante_ridge.append(temp_TE)
    
TE_exante_ridge = pd.DataFrame({'TE Parametrics': TE_exante_ridge}, index=weights_spi_cons_ridge.loc[start_ptf:].index)
TE_exante_ridge.plot(figsize=(7,5))

## TE ex-post Between Parametric Weights and CW Benchmark
TE_expost_ridge = TE_expost(ridge_returns[start_ptf:], cw_spi_index[start_ptf:])

# =============================================================================
# Parametrics using Macro Data (Factor Timing)
# =============================================================================

"""PARAMETRIC WEIGHTS WITH ALL MACRO VARIABLES"""
returns_factors_parametric = returns_factors.iloc[1:-1].copy()
macro_variables_parametric = macro_data.iloc[12:, 1:2].copy() #keep as a dataframe and not series: macro_data.iloc[10:, 1:2].copy()

shape = returns_factors_parametric.shape[1]*macro_variables_parametric.shape[1]

risk_aversion = (perf_cwbenchmark.T['Sharpe Ratio']/perf_cwbenchmark.T['Annualized STD']).iloc[0]

conditional_weights_factors = returns_factors_parametric.copy()
for time in range(0, len(macro_variables_parametric)):
    
    numerator = np.zeros((shape,1))
    denominator = np.zeros((shape,shape))
    
    for row in range(time): 
        z_t = macro_variables_parametric.iloc[row].to_numpy().reshape((len(macro_variables_parametric.iloc[row])),1)
        r_t1 = returns_factors_parametric.iloc[row].to_numpy().reshape((len(returns_factors_parametric.iloc[row]),1))
        numerator += np.kron(z_t,r_t1)
        denominator += np.kron(np.matmul(z_t,np.transpose(z_t)),np.matmul(r_t1,np.transpose(r_t1)))
          
    if ((denominator == 0).mean() != 1) or ((numerator == 0).mean() != 1): #Forcing a non-singular matrix
        
        unconditional_weights = (1/risk_aversion) * np.matmul(np.linalg.inv(denominator),numerator)
        
        theta = np.asarray(np.split(unconditional_weights, macro_variables_parametric.shape[1])).reshape(macro_variables_parametric.shape[1], returns_factors_parametric.shape[1]).T.clip(0)
        
        conditional_weights_factors.iloc[time] = np.matmul(theta,macro_variables_parametric.iloc[time])
        conditional_weights_factors.iloc[time] = conditional_weights_factors.iloc[time] / conditional_weights_factors.iloc[time].sum()

        print('Negative values ? (True: Yes, False: No):', (conditional_weights_factors.iloc[time] < 0).values.any())

parametric_returns = np.multiply(conditional_weights_factors,returns_factors_parametric).sum(axis=1)

## Performances Parametrics
perf_parametric = perf(parametric_returns[start_ptf:], cw_spi_index[start_ptf:], 'Parametric Returns')
risk_parametric = risk_historical(parametric_returns[start_ptf:], 0.95, 12)
risk_parametric.plot(figsize=(7,5))

## Evolution of Weigths
plt.figure(figsize=(20, 10))
conditional_weights_factors[start_ptf:].plot()
plt.title("Weights Evolution for Parametrics Weights")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

## Create a DF of the total weights of each SPI constituents
weights_spi_cons_parametrics = (position_mom.mul(conditional_weights_factors['Momentum'], axis=0) 
                                + position_value.mul(conditional_weights_factors['Value'], axis=0)
                                + position_size.mul(conditional_weights_factors['Size'], axis=0)
                                + position_profit.mul(conditional_weights_factors['Profitability'], axis=0)
                                + position_beta.mul(conditional_weights_factors['Beta'], axis=0)
                                + position_vol.mul(conditional_weights_factors['Volatility'], axis=0)
                                + position_div.mul(conditional_weights_factors['Dividend'], axis=0)
                                + position_eps.mul(conditional_weights_factors['EPS (Earnings Quality)'], axis=0)).dropna()

## TE ex-ante Between Parametric Weights and CW Benchmark
TE_exante_parametrics = []
for row in weights_spi_cons_parametrics.loc[start_ptf:].index:
    temp_TE = TE_exante(weights_spi_cons_parametrics.loc[row].values, cw_spi_cons.loc[row].replace(np.nan,0).values, returns_spi_cons.loc[:row])
    TE_exante_parametrics.append(temp_TE)
    
TE_exante_parametrics = pd.DataFrame({'TE Parametrics': TE_exante_parametrics}, index=weights_spi_cons_parametrics.loc[start_ptf:].index)
TE_exante_parametrics.plot(figsize=(7,5))

## TE ex-post Between Parametric Weights and CW Benchmark
TE_expost_parametrics = TE_expost(parametric_returns[start_ptf:], cw_spi_index[start_ptf:])

# =============================================================================
# Merge Performance Dash
# =============================================================================

perf_merged = pd.concat([perf_erc, perf_ridge, perf_parametric, perf_cwbenchmark ], axis=1)

df_dash = pd.DataFrame({'ERC': cum_prod(erc_returns[start_ptf:]), 'Ridge': cum_prod(ridge_returns[start_ptf:]), 
                        'Parametrics': cum_prod(parametric_returns[start_ptf:]), 'CW Benchmark': cum_prod(cw_spi_index[start_ptf:])}).dropna()
df_dash.index.name = 'Date'
df_dash.to_csv('dash-financial-report/data/perf_ptf.csv')

test = pd.read_csv('dash-financial-report/data/perf_ptf.csv')


