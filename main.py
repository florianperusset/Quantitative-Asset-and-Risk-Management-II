import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred

sns.set_theme(style="darkgrid")

os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Quantitative Asset & Risk Management 2/Project")

from import_data import get_spi
from optimization_criteria import criterion_erc, criterion_ridge
from ptf_performances import cum_prod, perf, risk_historical, TE_exante, TE_expost
from factor_building import factor_building

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
industry_spi_cons = spi[11] #Industry of all constituents
mb_spi_cons = spi[12] #Market-to-book ratio of all constituents
investment_spi_cons = spi[13] #Investments of all constituents
profit_spi_cons = spi[14] #Operating Profit Margin of all constituents


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

#1-Month London Interbank Offered Rate (LIBOR), based on Swiss Franc
libor1M_CHF = pd.read_excel("Data_SPI/FRED.xls", sheet_name='1M LIBOR CHF')
libor1M_CHF = libor1M_CHF.set_index('Date')
libor1M_CHF = libor1M_CHF[(libor1M_CHF.index >= '1999-12-01') & (libor1M_CHF.index < '2021-10-01')]
libor1M_CHF = libor1M_CHF.groupby(pd.Grouper(freq="M")).mean() 
libor1M_CHF.index = index

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

def run_factor_building(quantile):
    
    """MOMENTUM (Price)"""
    returns_past12_mom = returns_spi_cons.rolling(12,closed='left').mean()*trade_liq #Include trade constraint
    
    position_mom = factor_building(returns_past12_mom, quantile)
    returns_mom = position_mom.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_mom))
    plt.title("Momentum")
    
    """VALUE"""
    position_value = factor_building(pe_spi_cons, quantile, long_above_quantile=False)
    returns_value = position_value.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_value))
    plt.title("Value")
    
    """SIZE (SMALL VS. BIG)"""
    position_size = factor_building(mktcap_spi_cons, quantile, long_above_quantile=False)
    returns_size = position_size.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_size))
    plt.title("Size")
    
    """PROFITABILITY"""
    position_profit = factor_building(gm_spi_cons, quantile)
    returns_profit = position_profit.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_profit))
    plt.title("Profitability")
    
    """BETA"""
    position_beta = factor_building(beta_spi_cons, quantile, long_above_quantile=False)
    returns_beta = position_beta.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_beta))
    plt.title("Beta")
    
    """VOLATILITY"""
    position_vol = factor_building(roll_vol_spi_cons, quantile, long_above_quantile=False)
    returns_vol = position_vol.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_vol))
    plt.title("Volatility")
    
    """Dividend"""
    position_div = factor_building(dividend_spi_cons, quantile)
    returns_div = position_div.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_div))
    plt.title("Dividend Yield")
    
    """EPS (Quality Earnings)"""
    position_eps = factor_building(eps_spi_cons, quantile)
    returns_eps = position_eps.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_eps))
    plt.title("EPS (Quality Earnings)")
    
    # Create a df of factor returns
    returns_factors = pd.DataFrame({"Momentum":returns_mom, "Value":returns_value,
                                   "Size":returns_size, "Profitability":returns_profit,
                                   "Beta":returns_beta, "Volatility":returns_vol,
                                   "Dividend": returns_div, 'EPS (Earnings Quality)': returns_eps}).dropna()['2001-01-01':]
    
    # Create a dictionary of factor weight
    position_factors = {"Momentum": position_mom, "Value": position_value,
                                   "Size": position_size, "Profitability": position_profit,
                                   "Beta": position_beta, "Volatility": position_vol,
                                   "Dividend": position_div, 'EPS (Earnings Quality)': position_eps}
    
    return (returns_factors, position_factors)

returns_factors, position_factors = run_factor_building(quantile = 0.5)

"""Momentum of Factors"""

def run_momentum_factors(quantile, combine_CW_weight, combine_CW, name):
    
    returns_factors_past12_mom = returns_factors.rolling(12, closed='left').mean().dropna()
    
    position_mom_factor  = factor_building(returns_factors_past12_mom, quantile)
    returns_mom_factors = position_mom_factor.mul(returns_factors).sum(axis=1)
    
    if combine_CW:
        
        returns_mom_factors = returns_mom_factors*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances ERC model
    perf_mom_factors = perf(returns_mom_factors[start_ptf:], cw_spi_index[start_ptf:], 'Momentum of Factor Returns ' + name)
    risk_mom_factors = risk_historical(returns_mom_factors[start_ptf:], 0.95, 12)
    risk_mom_factors.plot(figsize=(7,5))
    
    return (position_mom_factor, returns_mom_factors, perf_mom_factors, risk_mom_factors)

run_mom_factors_noTE = run_momentum_factors(quantile = 0.5, combine_CW_weight = 0, combine_CW = False, name = '(No TE Monitor)')
return_mom_factors_noTE = run_mom_factors_noTE[1]

run_mom_factors_combineCW_80 = run_momentum_factors(quantile = 0.5, combine_CW_weight = 0.8, combine_CW = True, name = "(80% Portfolio, 20% CW Benchmark)")
return_mom_factors_combineCW_80 = run_mom_factors_combineCW_80[1]

run_mom_factors_combineCW_60 = run_momentum_factors(quantile = 0.5, combine_CW_weight = 0.6, combine_CW = True, name = "(60% Portfolio, 40% CW Benchmark)")
return_mom_factors_combineCW_60 = run_mom_factors_combineCW_60[1]

# returns_factors_past12_mom = returns_factors.rolling(12, closed='left').mean().dropna()

# quantile_mom_factor = returns_factors_past12_mom.quantile(q=0.50, axis=1)

# position_mom_factor  = returns_factors_past12_mom.copy()

# for i in position_mom_factor.columns:
#     position_mom_factor.loc[returns_factors_past12_mom[i] >= quantile_mom_factor, i] = 1
#     position_mom_factor.loc[returns_factors_past12_mom[i] < quantile_mom_factor, i] = 0
    
# returns_factors_mom = (position_mom_factor*returns_factors)['2002-01-01':]

# x0 = np.zeros(len(returns_factors_mom.columns))+0.01 # initial values

# Bounds = [(0 , 1) for i in range(len(returns_factors_mom.columns))]

# weights_factors_erc = returns_factors_mom.copy()*0

# for row in range(returns_factors_mom.loc[:start_ptf].shape[0]-1, len(returns_factors_mom)): #returns_factors.loc[:start_ptf].shape[0]-1
    
#     exp_returns_factors = returns_factors_mom.iloc[:row-1]
    
#     constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

#     res_erc = minimize(criterion_erc, x0, args=(exp_returns_factors), bounds=Bounds, method='SLSQP', constraints=constraint_set)
#     weights_factors_erc.iloc[row] = res_erc.x    

# erc_returns = np.multiply(returns_factors['2002-01-01':], weights_factors_erc).sum(axis=1)

## Performances ERC model
# perf_erc = perf(erc_returns[start_ptf:], cw_spi_index[start_ptf:], 'ERC Returns')
# risk_erc = risk_historical(erc_returns[start_ptf:], 0.95, 12)
# risk_erc.plot(figsize=(7,5))

# =============================================================================
# ERC of Factors
# =============================================================================

def build_erc(TE_target, check_TE=True):
    ## Start the optimization
    x0 = np.zeros(len(returns_factors.columns))+0.01 # initial values
    
    Bounds = [(0 , 1) for i in range(len(returns_factors.columns))]
    
    weights_factors_erc = returns_factors.copy()*0
    
    for row in range(returns_factors.loc[:start_ptf].shape[0]-1, len(returns_factors)): #returns_factors.loc[:start_ptf].shape[0]-1
        
        exp_returns_factors = returns_factors.iloc[:row-1]
        
        if check_TE: 
        
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                              {'type':'ineq', 'fun': lambda x: TE_target/np.sqrt(12) - TE_exante((position_factors['Momentum'].iloc[row-1].values * x[0]
                                                                                  + position_factors['Value'].iloc[row-1].values * x[1]
                                                                                  + position_factors['Size'].iloc[row-1].values * x[2]
                                                                                  + position_factors['Profitability'].iloc[row-1].values * x[3]
                                                                                  + position_factors['Beta'].iloc[row-1].values * x[4]
                                                                                  + position_factors['Volatility'].iloc[row-1].values * x[5]
                                                                                  + position_factors['Dividend'].iloc[row-1].values * x[6]
                                                                                  + position_factors['EPS (Earnings Quality)'].iloc[row-1].values * x[7]), cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1])})
        
        else:
            
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

        res_erc = minimize(criterion_erc, x0, args=(exp_returns_factors), bounds=Bounds, method='SLSQP', constraints=constraint_set)
        
        weights_factors_erc.iloc[row] = res_erc.x    
        
    return weights_factors_erc

def run_erc(TE_target, TE_check, combine_CW_weight, combine_CW, name):
    
    weights_factors_erc = build_erc(TE_target, TE_check)
    
    ## Compute the returns of the ERC model
    erc_returns = np.multiply(returns_factors, weights_factors_erc).sum(axis=1)
    
    if combine_CW:
        
        erc_returns = erc_returns*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances ERC model
    perf_erc = perf(erc_returns[start_ptf:], cw_spi_index[start_ptf:], 'ERC Returns ' + name)
    risk_erc = risk_historical(erc_returns[start_ptf:], 0.95, 12)
    risk_erc.plot(figsize=(7,5))
    
    ## Evolution of Factor Weigths
    plt.figure(figsize=(20, 10))
    weights_factors_erc[start_ptf:].plot()
    plt.title("Weights Evolution for ERC")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Create a DF of the total weights of each SPI constituents
    weights_spi_cons_erc = (position_factors['Momentum'].mul(weights_factors_erc['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(weights_factors_erc['Value'], axis=0)
                                    + position_factors['Size'].mul(weights_factors_erc['Size'], axis=0)
                                    + position_factors['Profitability'].mul(weights_factors_erc['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(weights_factors_erc['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(weights_factors_erc['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(weights_factors_erc['Dividend'], axis=0)
                                    + position_factors['EPS (Earnings Quality)'].mul(weights_factors_erc['EPS (Earnings Quality)'], axis=0)).dropna()[start_ptf:]
    
    return (weights_factors_erc, erc_returns, perf_erc, risk_erc, weights_spi_cons_erc)

run_erc_noTE = run_erc(TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "(No TE Monitor)")
return_erc_noTE = run_erc_noTE[1]

run_erc_checkTE_06 = run_erc(TE_target = 0.06, TE_check = True, combine_CW_weight = 0, combine_CW = False, name = "(6% TE Target)")
run_erc_checkTE_03 = run_erc(TE_target = 0.03, TE_check = True, combine_CW_weight = 0, combine_CW = False, name = "(3% TE Target)")

run_erc_combineCW_80 = run_erc(TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "(80% Portfolio, 20% CW Benchmark)")
return_erc_combineCW_80 = run_erc_combineCW_80[1]

run_erc_combineCW_60 = run_erc(TE_target = 0, TE_check = False, combine_CW_weight = 0.60, combine_CW = True, name = "(60% Portfolio, 40% CW Benchmark)")
return_erc_combineCW_60 = run_erc_combineCW_60[1]

# =============================================================================
# Ridge Regression of Factors
# =============================================================================

def build_ridge(TE_target, check_TE=True):
    
    x0 = np.zeros(len(returns_factors.columns))+0.01
    
    ridge_weights_factors = returns_factors.copy()*0
    
    bounds_ridge = [(0, 1) for i in range(len(returns_factors.columns))]
    
    for row in range(returns_factors.loc[:start_ptf].shape[0]-1,len(returns_factors)): #returns_factors.loc[:start_ptf].shape[0]-1
        expected_return = returns_factors.iloc[:row-1].mean()
        varcov_matrix = returns_factors.iloc[:row-1].cov()
        
        if check_TE:
    
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                              {'type':'ineq', 'fun': lambda x: TE_target/np.sqrt(12) - TE_exante((position_factors['Momentum'].iloc[row-1].values * x[0]
                                                                                  + position_factors['Value'].iloc[row-1].values * x[1]
                                                                                  + position_factors['Size'].iloc[row-1].values * x[2]
                                                                                  + position_factors['Profitability'].iloc[row-1].values * x[3]
                                                                                  + position_factors['Beta'].iloc[row-1].values * x[4]
                                                                                  + position_factors['Volatility'].iloc[row-1].values * x[5]
                                                                                  + position_factors['Dividend'].iloc[row-1].values * x[6]
                                                                                  + position_factors['EPS (Earnings Quality)'].iloc[row-1].values * x[7]), cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1])})
        
        else:
            
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
        
        res_ridge = minimize(criterion_ridge, x0, args=(expected_return,varcov_matrix), bounds=bounds_ridge, method='SLSQP',constraints=constraint_set)
        ridge_weights_factors.iloc[row] = res_ridge.x
        
    return ridge_weights_factors

def run_rigde(TE_target, TE_check, combine_CW_weight, combine_CW, name):

    ridge_weights_factors = build_erc(TE_target, TE_check)
    
    ## Compute the returns of ridge regression
    ridge_returns = np.multiply(returns_factors, ridge_weights_factors).sum(axis=1)  
    
    if combine_CW:
        
        ridge_returns = ridge_returns*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances Ridge Regression
    perf_ridge = perf(ridge_returns[start_ptf:], cw_spi_index[start_ptf:], 'Ridge Returns ' + name)
    risk_ridge = risk_historical(ridge_returns[start_ptf:], 0.95, 12)
    risk_ridge.plot(figsize=(7,5))
    
    ## Evolution of Weigths
    plt.figure(figsize=(20, 10))
    ridge_weights_factors[start_ptf:].plot()
    plt.title("Weights Evolution for Ridge Regression")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    ## Create a DF of the total weights of each SPI constituents
    weights_spi_cons_ridge = (position_factors['Momentum'].mul(ridge_weights_factors['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(ridge_weights_factors['Value'], axis=0)
                                    + position_factors['Size'].mul(ridge_weights_factors['Size'], axis=0)
                                    + position_factors['Profitability'].mul(ridge_weights_factors['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(ridge_weights_factors['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(ridge_weights_factors['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(ridge_weights_factors['Dividend'], axis=0)
                                    + position_factors['EPS (Earnings Quality)'].mul(ridge_weights_factors['EPS (Earnings Quality)'], axis=0)).dropna()[start_ptf:]

    return (ridge_weights_factors, ridge_returns, perf_ridge, risk_ridge, weights_spi_cons_ridge)

run_ridge_noTE = run_rigde(TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "(No TE Monitor)")
return_ridge_noTE = run_ridge_noTE[1]

run_ridge_checkTE_06 = run_rigde(TE_target = 0.06, TE_check = True, combine_CW_weight = 0, combine_CW = False, name = "(6% TE Target)")
run_ridge_checkTE_03 = run_rigde(TE_target = 0.03, TE_check = True, combine_CW_weight = 0, combine_CW = False, name = "3% TE Target")

run_ridge_combineCW_80 = run_rigde(TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "(80% Portfolio, 20% CW Benchmark)")
return_ridge_combineCW_80 = run_ridge_combineCW_80[1]

run_ridge_combineCW_60 = run_rigde(TE_target = 0, TE_check = False, combine_CW_weight = 0.60, combine_CW = True, name = "(60% Portfolio, 40% CW Benchmark)")
return_ridge_combineCW_60 = run_ridge_combineCW_60[1]

# =============================================================================
# Parametrics using Macro Data (Factor Timing)
# =============================================================================

"""PARAMETRIC WEIGHTS WITH ALL MACRO VARIABLES"""

def build_parametrics(select_macro_data):
    
    returns_factors_parametric = returns_factors.iloc[:-1].copy()
    macro_variables_parametric = macro_data.loc['2001-01-01':, select_macro_data].copy() #keep as a dataframe and not series: macro_data.iloc[10:, 1:2].copy()
         
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
    
    return (conditional_weights_factors, returns_factors_parametric)

# macro_returns_parametrics = macro_data.loc['2001-01-01':].copy()
# for i in macro_data.columns: 
#     conditional_weights_factors, returns_factors_parametric = build_parametrics([i])
    
#     parametric_returns = np.multiply(conditional_weights_factors,returns_factors_parametric) #.sum(axis=1)
    
#     macro_returns_parametrics[i] = parametric_returns
    
# macro_returns_parametrics.apply(cum_prod).plot()

# parametric_returns = macro_returns_parametrics.sum(axis=1)/macro_returns_parametrics.shape[1]

def run_parametrics(select_macro_data, combine_CW_weight, combine_CW, name):
    
    conditional_weights_factors, returns_factors_parametric = build_parametrics(['VIX'])
    
    parametric_returns = np.multiply(conditional_weights_factors,returns_factors_parametric).sum(axis=1)
    
    if combine_CW:
        
        parametric_returns = parametric_returns*combine_CW_weight + cw_spi_index*(1 - combine_CW_weight)
    
    ## Performances Parametrics
    perf_parametric = perf(parametric_returns[start_ptf:], cw_spi_index[start_ptf:], 'Parametric Returns ' + name)
    risk_parametric = risk_historical(parametric_returns[start_ptf:], 0.95, 12)
    risk_parametric.plot(figsize=(7,5))
    
    ## Evolution of Weigths
    plt.figure(figsize=(20, 10))
    conditional_weights_factors[start_ptf:].plot()
    plt.title("Weights Evolution for Parametrics Weights")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    ## Create a DF of the total weights of each SPI constituents
    weights_spi_cons_parametrics = (position_factors['Momentum'].mul(conditional_weights_factors['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(conditional_weights_factors['Value'], axis=0)
                                    + position_factors['Size'].mul(conditional_weights_factors['Size'], axis=0)
                                    + position_factors['Profitability'].mul(conditional_weights_factors['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(conditional_weights_factors['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(conditional_weights_factors['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(conditional_weights_factors['Dividend'], axis=0)
                                    + position_factors['EPS (Earnings Quality)'].mul(conditional_weights_factors['EPS (Earnings Quality)'], axis=0)).dropna()
    
    return (conditional_weights_factors, parametric_returns, perf_parametric, risk_parametric, weights_spi_cons_parametrics)

run_parametrics_VIX_noTE = run_parametrics(select_macro_data = 'VIX', combine_CW_weight = 0, combine_CW = False, name = "(VIX, No TE Monitor)")
returns_parametrics_VIX_noTE = run_parametrics_VIX_noTE[1]

run_parametrics_VIX_combineCW_80 = run_parametrics(select_macro_data = 'VIX', combine_CW_weight = 0.8, combine_CW = True, name = "(VIX - 80% Portfolio, 20% CW Benchmark)")
returns_parametrics_VIX_combineCW_80 = run_parametrics_VIX_combineCW_80[1]

run_parametrics_VIX_combineCW_60 = run_parametrics(select_macro_data = 'VIX', combine_CW_weight = 0.6, combine_CW = True, name = "(VIX - 60% Portfolio, 40% CW Benchmark)")
returns_parametrics_VIX_combineCW_60 = run_parametrics_VIX_combineCW_60[1]

# =============================================================================
# Fama-French Factor Analysis
# =============================================================================

price_spi_cons = spi[0] # Price of all constituents
pe_spi_cons = spi[1] # PE ratios for all constituents
mktcap_spi_cons = spi[3] # Market cap for all consituents
mb_spi_cons = spi[12] #Market-to-book ratio of all constituents
bm_spi_cons = 1/spi[12] #Book-to-market of all constituents
profit_spi_cons = spi[14] #Operating Profit Margin of all constituents


#Compute the returns
returns_spi_cons = (price_spi_cons/price_spi_cons.shift(1) - 1).replace(np.nan, 0)
returns_spi_cons = returns_spi_cons.replace([np.inf, -np.inf], 0)

returns_past12 = returns_spi_cons.rolling(12,closed='left').mean()

"""Market Factor"""
excess_return_market = returns_spi['SPI INDEX'] - libor1M_CHF['1M Libor CHF']/100

"""SMB Factor"""
def SMB_bm():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_value = factor_building((bm_spi_cons*position_small), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_small_growth = factor_building((bm_spi_cons*position_small), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_value = factor_building((bm_spi_cons*position_big), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_big_growth = factor_building((bm_spi_cons*position_big), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    returns_small_value = (position_small_value*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_growth = (position_small_growth*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_value = (position_big_value*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_growth = (position_big_growth*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    smb_bm = 0.5*(returns_small_value + returns_small_growth) - 0.5*(returns_big_value + returns_big_growth)
    
    return smb_bm

smb_bm = SMB_bm()

def SMB_op():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_robust = factor_building((profit_spi_cons*position_small), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_small_weak = factor_building((profit_spi_cons*position_small), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_robust = factor_building((profit_spi_cons*position_big), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_big_weak = factor_building((profit_spi_cons*position_big), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    returns_small_robust = (position_small_robust*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_weak = (position_small_weak*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_robust = (position_big_robust*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_weak = (position_big_weak*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    smb_op = 0.5*(returns_small_robust + returns_small_weak) - 0.5*(returns_big_robust + returns_big_weak)
    
    return smb_op

def SMB_inv():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_aggressive = factor_building((investment_spi_cons*position_small), quantile=0.55, long_above_quantile=True, ew_position=False)
    position_small_conservative = factor_building((investment_spi_cons*position_small), quantile=0.45, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_aggressive = factor_building((investment_spi_cons*position_big), quantile=0.55, long_above_quantile=True, ew_position=False)
    position_big_conservative = factor_building((investment_spi_cons*position_big), quantile=0.45, long_above_quantile=False, ew_position=False)
    
    returns_small_aggressive = (position_small_aggressive*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_conservative = (position_small_conservative*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_aggressive = (position_big_aggressive*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_conservative = (position_big_conservative*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    smb_inv = 0.5*(returns_small_aggressive + returns_small_conservative) - 0.5*(returns_big_aggressive + returns_big_conservative)
    
    return smb_inv
    
smb = (SMB_bm() + SMB_op() + SMB_inv())/3

"""HML Factor"""
def HML():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_value = factor_building((bm_spi_cons*position_small), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_small_growth = factor_building((bm_spi_cons*position_small), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_value = factor_building((bm_spi_cons*position_big), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_big_growth = factor_building((bm_spi_cons*position_big), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    returns_small_value = (position_small_value*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_growth = (position_small_growth*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_value = (position_big_value*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_growth = (position_big_growth*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    hml = 0.5*(returns_small_value + returns_big_value) - 0.5*(returns_small_growth + returns_big_growth)
    
    return hml

hml = HML()

"""WML Factor"""
def WML():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_high = factor_building(returns_past12*position_small, quantile=0.75, long_above_quantile=True, ew_position=False)
    position_small_low = factor_building(returns_past12*position_small, quantile=0.25, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_high = factor_building(returns_past12*position_big, quantile=0.75, long_above_quantile=True, ew_position=False)
    position_big_low = factor_building(returns_past12*position_big, quantile=0.25, long_above_quantile=False, ew_position=False)
    
    returns_small_high = (position_small_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_low = (position_small_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_high = (position_big_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_low = (position_big_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    wml = 0.5*(returns_small_high + returns_big_high) - 0.5*(returns_small_low + returns_big_low)
    
    return wml

wml = WML()

"""RMW Factor"""
def RMW():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_robust = factor_building((profit_spi_cons*position_small), quantile=0.75, long_above_quantile=True, ew_position=False)
    position_small_weak = factor_building((profit_spi_cons*position_small), quantile=0.25, long_above_quantile=False, ew_position=False)
    
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_robust = factor_building((profit_spi_cons*position_big), quantile=0.75, long_above_quantile=False, ew_position=False)
    position_big_weak = factor_building((profit_spi_cons*position_big), quantile=0.25, long_above_quantile=True, ew_position=False)
    
    returns_small_robust = (position_small_robust*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_weak = (position_small_weak*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_robust = (position_big_robust*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_weak = (position_big_weak*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    rmw = 0.5*(returns_small_robust + returns_big_robust) - 0.5*(returns_small_weak + returns_big_weak)
    
    return rmw

rmw = RMW()

"""CMA Factor"""
def CMA():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_aggressive = factor_building((investment_spi_cons*position_small), quantile=0.55, long_above_quantile=True, ew_position=False)
    position_small_conservative = factor_building((investment_spi_cons*position_small), quantile=0.45, long_above_quantile=False, ew_position=False)

    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_aggressive = factor_building((investment_spi_cons*position_big), quantile=0.55, long_above_quantile=True, ew_position=False)
    position_big_conservative = factor_building((investment_spi_cons*position_big), quantile=0.45, long_above_quantile=False, ew_position=False)

    returns_small_aggressive = (position_small_aggressive*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_conservative = (position_small_conservative*returns_spi_cons).replace(0, np.nan).mean(axis=1)

    returns_big_aggressive = (position_big_aggressive*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_conservative = (position_big_conservative*returns_spi_cons).replace(0, np.nan).mean(axis=1)

    cma = 0.5*(returns_small_conservative + returns_big_conservative) - 0.5*(returns_small_aggressive + returns_big_aggressive)

    return cma

cma = CMA()

"""VOL Factor"""
def VOL():
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_high = factor_building((roll_vol_spi_cons*position_small), quantile=0.65, long_above_quantile=True, ew_position=False)
    position_small_low = factor_building((roll_vol_spi_cons*position_small), quantile=0.35, long_above_quantile=False, ew_position=False)
        
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_high = factor_building((roll_vol_spi_cons*position_big), quantile=0.65, long_above_quantile=True, ew_position=False)
    position_big_low= factor_building((roll_vol_spi_cons*position_big), quantile=0.35, long_above_quantile=False, ew_position=False)
    
    returns_small_high= (position_small_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_low= (position_small_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_high = (position_big_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_low= (position_big_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    vol = 0.5*(returns_small_low + returns_big_low) - 0.5*(returns_small_high + returns_big_high)
    
    return vol

vol = VOL()

"""Fama-French 3 Factor Model Analysis"""
ff3_merged = pd.DataFrame({'Rm-Rf': excess_return_market, 'SMB': smb, 'HML': hml}).dropna()
ff3_merged_constant = sm.add_constant(ff3_merged[start_ptf:])

ff3_index_low = ff3_merged_constant.iloc[0].name
ff3_index_high = ff3_merged_constant.iloc[-1].name

def run_ff3(mom_factor_returns, erc_returns, ridge_returns, parametric_returns):
    
    ## FF Regression for Momentum of Factor
    mom_factor_excess_returns = mom_factor_returns - libor1M_CHF['1M Libor CHF']/100
    ff3_reg_mom_factor = sm.OLS(mom_factor_excess_returns[ff3_index_low:ff3_index_high], ff3_merged_constant).fit()
    
    ff3_reg_mom_factor.summary()
    
    ## FF Regression for ERC factor
    erc_excess_returns = erc_returns - libor1M_CHF['1M Libor CHF']/100
    ff3_reg_erc = sm.OLS(erc_excess_returns[ff3_index_low:ff3_index_high], ff3_merged_constant).fit()
    
    ff3_reg_erc.summary()
    
    ## FF Regression for Ridge Regression
    ridge_excess_returns = ridge_returns - libor1M_CHF['1M Libor CHF']/100
    ff3_reg_ridge = sm.OLS(ridge_excess_returns[ff3_index_low:ff3_index_high], ff3_merged_constant).fit()
    
    ff3_reg_ridge.summary()
    
    ## FF Regression for Parametrics
    parametric_excess_returns = parametric_returns - libor1M_CHF['1M Libor CHF']/100
    ff3_reg_parametrics = sm.OLS(parametric_excess_returns[ff3_index_low:ff3_index_high], ff3_merged_constant).fit()
    
    ff3_reg_parametrics.summary()
    
    ## Merge Results
    df_ff3_results = pd.DataFrame({'Coefficient Mom. Factor': ff3_reg_mom_factor.params, 'T-Test Mom. Factor': ff3_reg_mom_factor.tvalues,
                                   'Coefficient ERC': ff3_reg_erc.params, 'T-Test ERC': ff3_reg_erc.tvalues,
                                   'Coefficient Ridge': ff3_reg_ridge.params, 'T-Test Ridge': ff3_reg_ridge.tvalues,
                                   'Coefficient Parametrics': ff3_reg_parametrics.params, 'T-Test Parametrics': ff3_reg_parametrics.tvalues}).T
    
    df_ff3_results.rename(columns={'const': 'Intercept'}, inplace=True)
    
    df_ff3_r2 = pd.DataFrame({'R2': [ff3_reg_mom_factor.rsquared, np.nan, ff3_reg_erc.rsquared, np.nan, ff3_reg_ridge.rsquared, np.nan, ff3_reg_parametrics.rsquared, np.nan]}, index=df_ff3_results.index)
    
    df_ff3_merged = pd.concat([df_ff3_results, df_ff3_r2], axis=1)
    
    return df_ff3_merged

ff3_analysis_noTE = run_ff3(return_mom_factors_noTE, return_erc_noTE, return_ridge_noTE, returns_parametrics_VIX_noTE)

ff3_analysis_combineCW_80 = run_ff3(return_mom_factors_combineCW_80, return_erc_combineCW_80, return_ridge_combineCW_80, returns_parametrics_VIX_combineCW_80)

ff3_analysis_combineCW_60 = run_ff3(return_mom_factors_combineCW_60, return_erc_combineCW_60, return_ridge_combineCW_60, returns_parametrics_VIX_combineCW_60)

"""Fama-French Multiple Factor Model Analysis"""
ff_merged = pd.DataFrame({'Rm-Rf': excess_return_market, 'SMB': smb, 'HML': hml, 'WML': wml, 'RMW': rmw, 'CMA': cma, 'VOL': vol}).dropna()
ff_merged_constant = sm.add_constant(ff_merged[start_ptf:])

ff_index_low = ff_merged_constant.iloc[0].name
ff_index_high = ff_merged_constant.iloc[-1].name

def run_ff(mom_factor_returns, erc_returns, ridge_returns, parametric_returns):
    
    ## FF Regression for Momentum of Factor
    mom_factor_excess_returns = mom_factor_returns - libor1M_CHF['1M Libor CHF']/100
    ff_reg_mom_factor = sm.OLS(mom_factor_excess_returns[ff_index_low:ff_index_high], ff_merged_constant).fit()
    
    ff_reg_mom_factor.summary()
    
    ## FF Regression for ERC factor
    erc_excess_returns = erc_returns - libor1M_CHF['1M Libor CHF']/100
    ff_reg_erc = sm.OLS(erc_excess_returns[ff_index_low:ff_index_high], ff_merged_constant).fit()
    
    ff_reg_erc.summary()
    
    ## FF Regression for Ridge Regression
    ridge_excess_returns = ridge_returns - libor1M_CHF['1M Libor CHF']/100
    ff_reg_ridge = sm.OLS(ridge_excess_returns[ff_index_low:ff_index_high], ff_merged_constant).fit()
    
    ff_reg_erc.summary()
    
    ## FF Regression for Parametrics
    parametric_excess_returns = parametric_returns - libor1M_CHF['1M Libor CHF']/100
    ff_reg_parametrics = sm.OLS(parametric_excess_returns[ff_index_low:ff_index_high], ff_merged_constant).fit()
    
    ff_reg_parametrics.summary()
    
    ## Merge Results
    df_ff_results = pd.DataFrame({'Coefficient Mom. Factor': ff_reg_mom_factor.params, 'T-Test Mom. Factor': ff_reg_mom_factor.tvalues, 
                                  'Coefficient ERC': ff_reg_erc.params, 'T-Test ERC': ff_reg_erc.tvalues,
                                  'Coefficient Ridge': ff_reg_ridge.params, 'T-Test Ridge': ff_reg_ridge.tvalues,
                                  'Coefficient Parametrics': ff_reg_parametrics.params, 'T-Test Parametrics': ff_reg_parametrics.tvalues}).T
    
    df_ff_results.rename(columns={'const': 'Intercept'}, inplace=True)
    
    df_ff_r2 = pd.DataFrame({'R2': [ff_reg_mom_factor.rsquared, np.nan, ff_reg_erc.rsquared, np.nan, ff_reg_ridge.rsquared, np.nan, ff_reg_parametrics.rsquared, np.nan]}, index=df_ff_results.index)
    
    df_ff_merged = pd.concat([df_ff_results, df_ff_r2], axis=1)
    
    return df_ff_merged

ff_analysis_noTE = run_ff(return_mom_factors_noTE, return_erc_noTE, return_ridge_noTE, returns_parametrics_VIX_noTE)

ff_analysis_combineCW_80 = run_ff(return_mom_factors_combineCW_80, return_erc_combineCW_80, return_ridge_combineCW_80, returns_parametrics_VIX_combineCW_80)

ff_analysis_combineCW_60 = run_ff(return_mom_factors_combineCW_60, return_erc_combineCW_60, return_ridge_combineCW_60, returns_parametrics_VIX_combineCW_60)

# =============================================================================
# Merge Performance Dash
# =============================================================================

# perf_merged = pd.concat([perf_erc, perf_ridge, perf_parametric, perf_cwbenchmark], axis=1)

# df_dash = pd.DataFrame({'ERC': cum_prod(erc_returns[start_ptf:]), 'Ridge': cum_prod(ridge_returns[start_ptf:]), 
#                         'Parametrics': cum_prod(parametric_returns[start_ptf:]), 'CW Benchmark': cum_prod(cw_spi_index[start_ptf:])}).dropna()
# df_dash.index.name = 'Date'
# df_dash.to_csv('dash-financial-report/data/perf_ptf.csv')

# test = pd.read_csv('dash-financial-report/data/perf_ptf.csv')


