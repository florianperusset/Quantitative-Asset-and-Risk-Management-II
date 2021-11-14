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
from ptf_performances import cum_prod, perf, risk_historical, TE

# Alpha Vantage Key: O6PSHZOQS29QHD3E
# FRED Key: 2fd4cf1862f877db032b4a6a3a5f1c77

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

"""Benchmark SPI"""
price_spi_index = pd.read_excel("Data_SPI/SPI_DATA_ALL.xlsx", sheet_name='SPI Index')
price_spi_index = price_spi_index.set_index('Date')
price_spi_index = price_spi_index[(price_spi_index.index >= '2000-01-01')]
price_spi_index = price_spi_index.groupby(pd.Grouper(freq="M")).mean() 
price_spi_index.index = index

#Compute the returns
returns_spi = price_spi_index / price_spi_index.shift(1) - 1

plt.plot(cum_prod(returns_spi))

"""Cap-Weighted Benchmark"""
cw_spi_cons = mktcap_spi_cons.divide(mktcap_spi_cons.sum(axis=1), axis='index')

cw_spi_index = (cw_spi_cons*returns_spi_cons).replace(-0, 0).sum(axis=1)
cw_spi_cons.index = pd.to_datetime(cw_spi_cons.index)

plt.plot(cum_prod(cw_spi_index))

"""Macro Data"""
fred = Fred(api_key='2fd4cf1862f877db032b4a6a3a5f1c77')

#Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland (Monthly)
gov_bond_ch = fred.get_series('IRLTLT01CHM156N')
gov_bond_ch = gov_bond_ch[(gov_bond_ch.index >= '2000-01-01') & (gov_bond_ch.index < '2021-10-01')]

#CBOE Volatility Index: VIX (Daily)
vix =  fred.get_series('VIXCLS')
vix = vix[(vix.index >= '1999-12-01') & (vix.index < '2021-10-01')]
vix = vix.groupby(pd.Grouper(freq="M")).mean()
vix.index = index

#Consumer Price Index: All Items for Switzerland (Monthly)
cpi_CH =  fred.get_series('CHECPIALLMINMEI')
cpi_CH = cpi_CH[(cpi_CH.index >= '2000-01-01') & (cpi_CH.index < '2021-10-01')]

#TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill (Daily)
spread_US = fred.get_series('TEDRATE')
spread_US = spread_US[(spread_US.index >= '1999-12-01') & (spread_US.index < '2021-10-01')]
spread_US = spread_US.groupby(pd.Grouper(freq="M")).mean() 
spread_US.index = index

#3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor3M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='3M LIBOR')
libor3M_US = libor3M_US.set_index('Date')
libor3M_US = libor3M_US[(libor3M_US.index >= '1999-12-01') & (libor3M_US.index < '2021-10-01')]
libor3M_US = libor3M_US.groupby(pd.Grouper(freq="M")).mean() 
libor3M_US.index = index

#3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor12M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='12M LIBOR')
libor12M_US = libor12M_US.set_index('Date')
libor12M_US = libor12M_US[(libor12M_US.index >= '1999-12-01') & (libor12M_US.index < '2021-10-01')]
libor12M_US = libor12M_US.groupby(pd.Grouper(freq="M")).mean() 
libor12M_US.index = index

#Merge all macro data
macro_data_df = pd.DataFrame({'LT 10y Gov. Bond Yield': gov_bond_ch, 'VIX': vix, 
                           'CPI CH': cpi_CH, 'Spread US': spread_US}).dropna()

macro_data = pd.concat([macro_data_df, libor3M_US, libor12M_US], axis=1).dropna()

# =============================================================================
# Factor Construction
# =============================================================================

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

returns_profit = (returns_spi_cons*position_profit).replace(-0, 0) #.dropna()
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

"""Dividend"""
quantile_div = dividend_spi_cons.quantile(q=0.5, axis=1)

position_div = dividend_spi_cons.copy()

for i in position_div.columns:
    position_div.loc[dividend_spi_cons[i] >= quantile_div, i] = 1
    position_div.loc[dividend_spi_cons[i] < quantile_div, i] = 0
    
position_div = position_div.replace(np.nan, 0)

#Equal Weight
position_div = position_div.div(position_div.sum(axis=1), axis=0)

returns_div = (returns_spi_cons*position_div).replace(-0, 0).dropna()
returns_div = returns_div.sum(axis=1)

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

returns_eps = (returns_spi_cons*position_eps).replace(-0, 0).dropna()
returns_eps = returns_eps.sum(axis=1)

plt.figure()
plt.plot(cum_prod(returns_div))
plt.title("EPS (Quality Earnings")

# Create a df of factor returns
returns_factors = pd.DataFrame({"Momentum":returns_mom, "Value":returns_value,
                               "Size":returns_size, "Profitability":returns_profit,
                               "Beta":returns_beta, "Volatility":returns_vol,
                               "Dividend": returns_div, 'EPS (Earnings Quality)': returns_eps}).dropna()

# =============================================================================
# ERC of Factors
# =============================================================================

## Start the Creation of the portfolio
start_ptf = '2009-01-01'

## Start the optimization
x0 = np.zeros(len(returns_factors.columns))+0.01 # initial values

## Constraint set
constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
Bounds= [(0 , 1) for i in range(len(returns_factors.columns))]

weights_factors_erc = returns_factors.copy()*0

for row in range(returns_factors.loc[:start_ptf].shape[0]-1,len(returns_factors)):
    exp_returns_factors = returns_factors.iloc[:row-1]

    res_erc = minimize(criterion_erc,x0,args=(exp_returns_factors), bounds=Bounds, method='SLSQP',constraints=constraint_set)
    weights_factors_erc.iloc[row] = res_erc.x

## Results
erc_returns = np.multiply(returns_factors, weights_factors_erc).sum(1)
plt.plot(cum_prod(erc_returns[start_ptf:]), 'b', label='ERC Regression')
plt.plot(cum_prod(cw_spi_index[start_ptf:]), 'r', label='CW Benchmark')
plt.legend(loc='upper left', frameon=True)

## Evolution of Weigths
plt.figure()
weights_factors_erc[start_ptf:].plot()
plt.title("Weights Evolution for ERC")
plt.tight_layout()

## Performances ERC
perf_erc = perf(erc_returns[start_ptf:], 'ERC')
risk_erc = risk_historical(erc_returns[start_ptf:], 0.95, 12)
risk_erc.plot(figsize=(7,5))

# =============================================================================
# Ridge Regression of Factors
# =============================================================================
ridge_weights_factors = returns_factors.copy()*0

constraint_set_ridge = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
bounds_ridge = [(0, 1/5) for i in range(len(returns_factors.columns))]

for row in range(returns_factors.loc[:start_ptf].shape[0]-1,len(returns_factors)):
    expected_return = returns_factors.iloc[:row-1].mean()
    varcov_matrix = returns_factors.iloc[:row-1].cov()
    
    res_ridge = minimize(criterion_ridge, x0, args=(expected_return,varcov_matrix), bounds=bounds_ridge, method='SLSQP',constraints=constraint_set_ridge)
    ridge_weights_factors.iloc[row] = res_ridge.x

## Results
ridge_returns = np.multiply(returns_factors, ridge_weights_factors).sum(1)
plt.plot(cum_prod(ridge_returns[start_ptf:]), 'b', label='Ridge Regression')
plt.plot(cum_prod(cw_spi_index[start_ptf:]), 'r', label='CW Benchmark')
plt.legend(loc='upper left', frameon=True)
plt.show()
plt.close()

## Evolution of Weigths
plt.figure()
ridge_weights_factors[start_ptf:].plot()
plt.title("Weights Evolution for Ridge Regression")
plt.tight_layout()

## Performances Ridge Regression
perf_erc = perf(ridge_returns[start_ptf:], 'ERC')
risk_erc = risk_historical(ridge_returns[start_ptf:], 0.95, 12)
risk_erc.plot(figsize=(7,5))

# =============================================================================
# Parametrics using Macro Data (Factor Timing)
# =============================================================================

"""PARAMETRIC WEIGHTS WITH ALL MACRO VARIABLES"""
returns_factors_parametric = returns_factors.iloc[:-1].copy()
macro_variables_parametric = macro_data.iloc[11:, 1:].copy() #keep as a dataframe and not series: macro_data.iloc[10:, 1:2].copy()

shape = returns_factors_parametric.shape[1]*macro_variables_parametric.shape[1]

perf_bench = perf(cw_spi_index[start_ptf:], 'CW Benchmark')
risk_aversion = (perf_bench.T['Sharpe Ratio']/perf_bench.T['Annualized STD']).iloc[0]

conditional_weights_factors = returns_factors_parametric.copy()
for time in range(0, len(macro_variables_parametric)):
    
    numerator = np.zeros((shape,1))
    denominator = np.zeros((shape,shape))
    
    for row in range(time): 
        z_t = macro_variables_parametric.iloc[row].to_numpy().reshape((len(macro_variables_parametric.iloc[row])),1)
        r_t1 = returns_factors_parametric.iloc[row].to_numpy().reshape((len(returns_factors_parametric.iloc[row]),1))
        numerator += np.kron(z_t,r_t1)
        denominator += np.kron(np.matmul(z_t,np.transpose(z_t)),np.matmul(r_t1,np.transpose(r_t1)))
          
    if (denominator == 0).mean() != 1: #Forcing a non-singular matrix
        
        unconditional_weights = (1/risk_aversion) * np.matmul(np.linalg.inv(denominator),numerator)
        
        theta = np.asarray(np.split(unconditional_weights, macro_variables_parametric.shape[1])).reshape(macro_variables_parametric.shape[1], returns_factors_parametric.shape[1]).T.clip(0)
        
        conditional_weights_factors.iloc[time] = np.matmul(theta,macro_variables_parametric.iloc[time])
        conditional_weights_factors.iloc[time] = conditional_weights_factors.iloc[time] / conditional_weights_factors.iloc[time].sum()

        print('Negative values ? (True: Yes, False: No):', (conditional_weights_factors.iloc[time] < 0).values.any())

parametric_returns = np.multiply(conditional_weights_factors,returns_factors_parametric).sum(axis=1)

plt.plot(cum_prod(cw_spi_index[start_ptf:]), 'r', label='CW Benchmark')
plt.plot(cum_prod(parametric_returns[start_ptf:]), 'b', label='Long-Only Parametric Returns')
plt.legend(loc='upper left', frameon=True)

## Performances Parametrics
perf_parametric = perf(parametric_returns[start_ptf:], 'Parametrics')
risk_parametric = risk_historical(parametric_returns[start_ptf:], 0.95, 12)
risk_parametric.plot(figsize=(7,5))

## Create a DF of the total weights of each SPI constituents
weights_spi_cons_parametrics = (position_mom.mul(conditional_weights_factors['Momentum'], axis=0) 
                                + position_value.mul(conditional_weights_factors['Value'], axis=0)
                                + position_size.mul(conditional_weights_factors['Size'], axis=0)
                                + position_profit.mul(conditional_weights_factors['Profitability'], axis=0)
                                + position_beta.mul(conditional_weights_factors['Beta'], axis=0)
                                + position_vol.mul(conditional_weights_factors['Volatility'], axis=0)
                                + position_div.mul(conditional_weights_factors['Dividend'], axis=0)
                                + position_eps.mul(conditional_weights_factors['EPS (Earnings Quality)'], axis=0)).dropna()

## TE Between Parametric Weights and CW Benchmark
TE_parametrics = []
for row in weights_spi_cons_parametrics.loc[start_ptf:].index:
    temp_TE = TE(weights_spi_cons_parametrics.loc[row].values, cw_spi_cons.loc[row].replace(np.nan,0).values, returns_spi_cons.loc[:row])
    TE_parametrics.append(temp_TE)
    
TE_parametrics = pd.DataFrame({'TE Parametrics': TE_parametrics}, index=weights_spi_cons_parametrics.loc[start_ptf:].index)
TE_parametrics.plot(figsize=(7,5))

# =============================================================================
# Merge Performance Dash
# =============================================================================

df_dash = pd.DataFrame({'ERC': cum_prod(erc_returns[start_ptf:]), 'Ridge': cum_prod(ridge_returns[start_ptf:]), 
                        'Parametrics': cum_prod(parametric_returns[start_ptf:]), 'CW Benchmark': cum_prod(cw_spi_index[start_ptf:])}).dropna()
df_dash.index.name = 'Date'
df_dash.to_csv('dash-financial-report/data/perf_ptf.csv')

test = pd.read_csv('dash-financial-report/data/perf_ptf.csv')
