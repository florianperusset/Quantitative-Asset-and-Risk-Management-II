"""
-----------------------------------------------------------------------
QUANTITATIVE ASSET & RISK MANAGEMENT II

HEC LAUSANNE - AUTUMN 2021

Title: Style Rotation on Swiss Long-Only Equity Factors

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

File Name: main.py
-----------------------------------------------------------------------

Objective:
    
    - Our client is a large Swiss pension fund who has a substantial allocation to Swiss equities. He is a firm believer of risk premia and is fully convinced by the long-term benefits of tilting his portfolio to reap the benefits of well-known risk premia.
    - With no particular view on which risk premia is best suited for him, he wants to go for a diversified approach. He is nevertheless concerned by the time-varying nature of factor returns and fears of being unable to cope with a too long period of underperformance of one given factor.
    - He is therefore thinking about the potentials of adjusting his exposure to the various risk premia over time and make his portfolio more dynamic.
    - He is willing to give a mandate for managing a dynamic long-only portfolio of risk premia on the Swiss market. Tracking error is also a concern for him.

Data:
    
    - Monthly data of the SPI constituents
    - Macro data from FRED

This is the main file of the project called "Style Rotation on Swiss Long-Only Equity Factors". 
It is divided into 8 part:
    
    1) Import Data
        1.1) Import the necessary data of the SPI constituents from Datastream 
        1.2) Import the necessary macro data from FRED
    2) Create a Cap-Weighted Benchmark
    3) Add a Trading Constraint: Trade only Liquid Equities based on Volume traded
    4) Factor Construction for the Strategies
    5) Creation of the Strategies
        5.1) Strategy 1: Momentum of Factors
        5.2) Strategy 2: ERC of Factors
        5.3) Strategy 3: Ridge Regression of Factors
        5.4) Strategy 4: Parametrics using Macro Data (Factor Timing)
    6) Fama-French Factor Analysis
        6.1) Creation of the FF Factors in the Swiss Market (SPI Constitutents)
        6.2) Regress a 3 factors Fama-French 
        6.3) Regress a multiple factors Fama-French
    7) Conduct a Sensitivity Analyis
        7.1) Liquidity Constraint Sensitivity
        7.2) Factor Construction Quantile Sensitivity
        7.3) Combination with CW Benchmark Sensitivity (TE Reduction)
    8) Collect all Data for the Dashboard

External files for the main are:

    - import_data.py    
    - ptf_performances.py
    - factor_building.py
    - optimization_criteria.py

"""

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
from ptf_performances import cum_prod, perf, risk_historical, TE_exante, avg_returns
from factor_building import factor_building, run_ff_regression

# Alpha Vantage Key: O6PSHZOQS29QHD3E
# FRED Key: 2fd4cf1862f877db032b4a6a3a5f1c77

## Start the Creation of the portfolio
start_ptf = '2009-01-01'

# Create files in the working directory
if not os.path.isdir('Plot'):
    os.makedirs('Plot')
    
if not os.path.isdir('Plot/Summary'):
    os.makedirs('Plot/Summary')

if not os.path.isdir('Plot/Basis'):
    os.makedirs('Plot/Basis')
    
if not os.path.isdir('Plot/Weights'):
    os.makedirs('Plot/Weights')
    
if not os.path.isdir('Plot/Sensitivity'):
    os.makedirs('Plot/Sensitivity')

if not os.path.isdir('Plot/Sensitivity/Liquidity'):
    os.makedirs('Plot/Sensitivity/Liquidity')
    
if not os.path.isdir('Plot/Sensitivity/FactorQuantile'):
    os.makedirs('Plot/Sensitivity/FactorQuantile')

if not os.path.isdir('Plot/Sensitivity/CW'):
    os.makedirs('Plot/Sensitivity/CW')

if not os.path.isdir('Output'):
    os.makedirs('Output')
    
if not os.path.isdir('Output/Basis'):
    os.makedirs('Output/Basis')
    
if not os.path.isdir('Output/Summary'):
    os.makedirs('Output/Summary')
    
if not os.path.isdir('Output/FF'):
    os.makedirs('Output/FF')

if not os.path.isdir('Output/Sensitivity'):
    os.makedirs('Output/Sensitivity')

if not os.path.isdir('Output/Sensitivity/Liquidity'):
    os.makedirs('Output/Sensitivity/Liquidity')
    
if not os.path.isdir('Output/Sensitivity/FactorQuantile'):
    os.makedirs('Output/Sensitivity/FactorQuantile')

if not os.path.isdir('Output/Sensitivity/CW'):
    os.makedirs('Output/Sensitivity/CW')

# =============================================================================
# =============================================================================
# 1) Import Data
# =============================================================================
# =============================================================================

# =============================================================================
# 1.1) Import the necessary data of the SPI constituents from Datastream 
# =============================================================================

"""
This section aims to collect all data required to build the factors. We collected 
(from Reuters Datastream) and processed the following metrics of the SPI:
    
    - Price
    - Price-to-Earnings
    - Dividend yield
    - Market Cap
    - Beta
    - Volatility (unused)
    - ROE (unused)
    - ROA (unused)
    - Gross Margin 
    - EPS
    - Volume traded
    - Industry Classification
    - Market-to-book 
    - Investments
    - Operating Profitability

"""


# Import and process all data required
spi = get_spi()

"""Swiss Performance Index"""
#Price Constituents
price_spi_cons_fix = spi[0] 
price_spi_cons_fix.index = pd.to_datetime(price_spi_cons_fix.index)
index =  price_spi_cons_fix.index

#Compute the returns
returns_spi_cons = (price_spi_cons_fix/price_spi_cons_fix.shift(1) - 1)
returns_spi_cons.loc['2000-01-01'] = 0
returns_spi_cons = returns_spi_cons.replace([np.inf, -np.inf], 0)

#Compute the Covariance matrix
cov_spi_cons = returns_spi_cons.cov()

#Compute the 12-months rolling variance
roll_vol_spi_cons = returns_spi_cons.rolling(12).std()

"""Load the fundamental data"""
pe_spi_cons_fix = spi[1] # PE ratios for all constituents
dividend_spi_cons_fix = spi[2] # Dividend Yield for all consistuents
mktcap_spi_cons_fix = spi[3] # Market cap for all consituents
beta_spi_cons_fix = spi[4] # Beta of all constituents
vol_spi_cons_fix = spi[5] # Volatility of all constituents
roe_spi_cons_fix = spi[6] # ROE of all constituents
roa_spi_cons_fix = spi[7] # ROA of all constituents
gm_spi_cons_fix = spi[8] # Gross Margin of all constituents
eps_spi_cons_fix = spi[9] #EPS of all constituents
trade_spi_cons_fix = spi[10] #Volume traded of all constituents
industry_spi_cons = spi[11] #Industry of all constituents
mb_spi_cons = spi[12] #Market-to-book ratio of all constituents
investment_spi_cons = spi[13] #Investments of all constituents
profit_spi_cons = spi[14] #Operating Profit Margin of all constituents

# Count the number of each constituents in each industry type
sns.histplot(industry_spi_cons, discrete=True, shrink=0.8, legend=False)
plt.xticks(range(1,7), ['Industrial', 'Utility', 'Transporation', 'Bank', 'Insurance', 'Other Financial'], rotation=90)
plt.savefig('Plot/Summary/industry.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

"""Benchmark SPI"""
# Collect the SPI
price_spi_index = pd.read_excel("Data_SPI/SPI_DATA_ALL.xlsx", sheet_name='SPI Index')
price_spi_index = price_spi_index.set_index('Date')
price_spi_index = price_spi_index[(price_spi_index.index >= '2000-01-01')]
price_spi_index = price_spi_index.groupby(pd.Grouper(freq="M")).mean() 
price_spi_index.index = index

#Compute the returns
returns_spi = price_spi_index / price_spi_index.shift(1) - 1

# =============================================================================
#  1.2) Import the necessary macro data from FRED
# =============================================================================

"""

This section aims to collect all necessary macro data to time the factors and 
perform an analysis of the porfolio. We collected from the Federal Reserve of 
Economic Data the following metrics (monthly data):
    
    - Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for Switzerland
    - Long-Term Government Bond Yields: 10-year: Main (Including Benchmark) for the United States
    - CBOE Volatility Index: VIX
    - Consumer Price Index: All Items for Switzerland
    - Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    - TED rate spread between 3-Month LIBOR based on US dollars and 3-Month Treasury Bill
    - 3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
    - 12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
    - 1-Month London Interbank Offered Rate (LIBOR), based on Swiss Franc
    
"""

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
libor3M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='3M LIBOR US')
libor3M_US = libor3M_US.set_index('Date')
libor3M_US = libor3M_US[(libor3M_US.index >= '1999-12-01') & (libor3M_US.index < '2021-10-01')]
libor3M_US = libor3M_US.groupby(pd.Grouper(freq="M")).mean() 
libor3M_US.index = index

#12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
libor12M_US = pd.read_excel("Data_SPI/FRED.xls", sheet_name='12M LIBOR US')
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

risk_free = (libor1M_CHF['1M Libor CHF'][start_ptf:]/100).mean()

# Merge all macro data
macro_data_df = pd.DataFrame({'10y Bond Yield US': gov_bond_US, 'VIX': vix,
                              'CPI US': cpi_US, 'TED Spread': TEDspread_US}).dropna()

macro_data = pd.concat([macro_data_df, libor3M_US, libor12M_US], axis=1).dropna()

# Lag the Macro Data
macro_data = macro_data.shift(3) 

macro_data.describe().round(2).to_latex('Output/Summary/macro_summary.tex', column_format = 'lcccccc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 2) Create a Cap-Weighted Benchmark
# =============================================================================
# =============================================================================

"""

This section aims to create a cap-weighted benchmark of our available assets,
based on their market cap.

"""

"""Cap-Weighted Benchmark"""
# Build the CW benchmark
cw_spi_cons = mktcap_spi_cons_fix.divide(mktcap_spi_cons_fix.sum(axis=1), axis='index')
cw_spi_cons.index = pd.to_datetime(cw_spi_cons.index)

cw_spi_index = (cw_spi_cons*returns_spi_cons).sum(axis=1)

# Plot the cumulative performances of the CW benchmark
plt.figure(figsize=(10,7))
plt.plot(cum_prod(cw_spi_index[start_ptf:]), 'r', label='CW Benchmark')
plt.legend(loc='upper left', frameon=True)
plt.savefig('Plot/Basis/cw_basis.png', dpi=400)
plt.show()
plt.close()

# Determine the performances of the benchmark
perf_cwbenchmark = perf(cw_spi_index[start_ptf:], cw_spi_index[start_ptf:], risk_free, 'CW')
perf_cwbenchmark.to_latex('Output/Basis/perf_cwbenchmark_basis.tex', column_format = 'lc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 3) Trade Constraint: Trade only Liquid Equities
# =============================================================================
# =============================================================================

"""

In this section, we set a liquidity constraint in our swiss equities universe, as some
small cap companies in the Swiss market tends to be illiquid, so we won't allocate them in 
our portfolios.

"""


"""Trade Only Liquid Equities"""
def liqudity_constraint(quantile):
    """
    
    This function sets a liquidity constraint to avoid trading illiquid
    stocks. It is based on the monthly volume traded in the markets
    
    Parameters
    ----------
    quantile : Float
        The quantile to set the liquidity constraint.

    Returns
    -------
    It results all metrics required to build factors with a liquidity constraint.

    """
    
    trade_liq_l = pd.DataFrame(np.zeros(price_spi_cons_fix.shape), columns = price_spi_cons_fix.columns, index = price_spi_cons_fix.index)
    
    trade_liq_quantile = trade_spi_cons_fix.quantile(quantile, axis=1)
    
    for i in trade_liq_l.columns:
        trade_liq_l.loc[trade_spi_cons_fix[i] >= trade_liq_quantile, i] = 1
        
    trade_liq_l = trade_liq_l.replace(0, np.nan)
    
    price_spi_cons_l = (price_spi_cons_fix*trade_liq_l)
    pe_spi_cons_l = (pe_spi_cons_fix*trade_liq_l)
    dividend_spi_cons_l = (dividend_spi_cons_fix*trade_liq_l)
    mktcap_spi_cons_l = (mktcap_spi_cons_fix*trade_liq_l)
    beta_spi_cons_l = (beta_spi_cons_fix*trade_liq_l)
    vol_spi_cons_l = (vol_spi_cons_fix*trade_liq_l)
    roe_spi_cons_l = (roe_spi_cons_fix*trade_liq_l)
    roa_spi_cons_l = (roa_spi_cons_fix*trade_liq_l)
    gm_spi_cons_l = (gm_spi_cons_fix*trade_liq_l)
    eps_spi_cons_l = (eps_spi_cons_fix*trade_liq_l)
    
    return (price_spi_cons_l, pe_spi_cons_l, dividend_spi_cons_l, mktcap_spi_cons_l, beta_spi_cons_l, vol_spi_cons_l, roe_spi_cons_l, roa_spi_cons_l, gm_spi_cons_l, eps_spi_cons_l, trade_liq_l)

# Set the liquidity constraint
price_spi_cons, pe_spi_cons, dividend_spi_cons, mktcap_spi_cons, beta_spi_cons, vol_spi_cons, roe_spi_con, roa_spi_con, gm_spi_cons, eps_spi_cons, trade_liq = liqudity_constraint(0.25)

# =============================================================================
# =============================================================================
# 4) Factor Construction
# =============================================================================
# =============================================================================

"""

In this section, we create eight different factors to build our portfolios.

"""

def run_factor_building(quantile):
    """
    This function builds eight different factors to build our portfolios:
        
        - Momentum (built with 12-month average returns)
        - Value (built with E/P)
        - Size (built with Market Cap)
        - Profitability (built with Gross Margin)
        - Low Beta (built with beta)
        - Low Volatility (built with the 12-month rolling volatility)
        - Dividend (built with dividend yield)
        - Quality Earning (built with EPS)
        
    In all these factors, we will take an EW position for all constituents with a 
    metrics below/above a given quantile (depending on the factors)
        

    Parameters
    ----------
    quantile : Float
        The quantile to build the factors.

    Returns
    -------
    returns_factors : DataFrame
        The returns of the factor.
    position_factors : DataFrame
        The weights of each constituents in a given factor.

    """
    
    global price_spi_cons, pe_spi_cons, dividend_spi_cons, mktcap_spi_cons, beta_spi_cons, vol_spi_cons, roe_spi_con, roa_spi_con, gm_spi_cons, eps_spi_cons, trade_liq
    
    """MOMENTUM (Price)"""
    returns_past12_mom = returns_spi_cons.rolling(12,closed='left').mean()*trade_liq #Include trade constraint
    #returns_past12_mom = returns_spi_cons.pct_change(12)*trade_liq
      
    position_mom = factor_building(returns_past12_mom, quantile)
    returns_mom = position_mom.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_mom))
    plt.title("Momentum")
    
    """VALUE"""
    position_value = factor_building(1/pe_spi_cons, quantile)
    returns_value = position_value.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_value))
    plt.title("Value")
    
    """SIZE (SMALL VS. BIG)"""
    position_size = factor_building(mktcap_spi_cons, 1-quantile, long_above_quantile=False)
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
    position_beta = factor_building(beta_spi_cons, 1-quantile, long_above_quantile=False)
    returns_beta = position_beta.mul(returns_spi_cons).sum(axis=1)
    
    plt.figure()
    plt.plot(cum_prod(returns_beta))
    plt.title("Beta")
    
    """VOLATILITY"""
    position_vol = factor_building(roll_vol_spi_cons, 1-quantile, long_above_quantile=False)
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
                                   "Dividend": returns_div, 'EPS': returns_eps}).dropna()['2001-01-01':]
    
    # Create a dictionary of factor weight
    position_factors = {"Momentum": position_mom, "Value": position_value,
                                   "Size": position_size, "Profitability": position_profit,
                                   "Beta": position_beta, "Volatility": position_vol,
                                   "Dividend": position_div, 'EPS': position_eps}
    
    return (returns_factors, position_factors)

# Collect the returns and weight allocation of factors
returns_factors, position_factors = run_factor_building(quantile = 0.5)

# Plo the correlation among factors
plt.figure(figsize=(7,5))
corr_factors = sns.heatmap(pd.concat([returns_factors[start_ptf:]], axis=1).corr(), annot=True)
plt.savefig('Plot/Summary/factors_corr.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

# =============================================================================
# =============================================================================
# 5) Creation of the Strategies
# =============================================================================
# =============================================================================

# =============================================================================
# 5.1) Strategy 1: Momentum of Factors
# =============================================================================

"""

This section with generate a portfolio of momentum of factors we created in the previous section.
We will also provide performances measures

"""

def run_momentum_factors(returns_factors, position_factors, quantile, combine_CW_weight, combine_CW, name):
    """
    
    This function will perform a momentum of factor in a similar fashion as the previous factor construction.
    It also compute the performances of the strategy. To minimize the tracking-error, this function can 
    also provide a combination with the benchmark to do so. This function will also be used for the sensitivity.
    
    Parameters
    ----------
    returns_factors : DataFrame
        The returns of factors.
    position_factors : DataFrame
        The equities allocation among a factor.
    quantile : Float
       Quantile used to create the momentum of factor.
    combine_CW_weight : Float
        Weight allocated to the benchmark.
    combine_CW : Bool
        True if we want to combine the portfolio with the benchmark. False otherwise
    name : Str
        Name of strategy (and settings).

    Returns
    -------
    weights_mom_factor : DataFrame
        Weights between factors in the strategy.
    returns_mom_factors : DataFrame
        Returns of the strategy.
    perf_mom_factors : Dataframe
        Performance metrics of the strategy.
    weights_spi_cons_mom_factor : DataFrame
        Weights allocated among constituents.

    """
    
    returns_factors_past12_mom = returns_factors.rolling(12, closed='left').mean().dropna()
    
    position_mom_factor  = factor_building(returns_factors_past12_mom, quantile)
    returns_mom_factors = position_mom_factor.mul(returns_factors).sum(axis=1)
    
    if combine_CW:
        
        returns_mom_factors = returns_mom_factors*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances ERC model
    perf_mom_factors = perf(returns_mom_factors[start_ptf:], cw_spi_index[start_ptf:], risk_free, name)
    # risk_mom_factors = risk_historical(returns_mom_factors[start_ptf:], 0.95, 12)
    # risk_mom_factors.plot(figsize=(7,5))
    
    weights_mom_factor = position_mom_factor.copy()
    
    # Create a DF of the total weights of each SPI constituents
    weights_spi_cons_mom_factor = (position_factors['Momentum'].mul(weights_mom_factor['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(weights_mom_factor['Value'], axis=0)
                                    + position_factors['Size'].mul(weights_mom_factor['Size'], axis=0)
                                    + position_factors['Profitability'].mul(weights_mom_factor['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(weights_mom_factor['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(weights_mom_factor['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(weights_mom_factor['Dividend'], axis=0)
                                    + position_factors['EPS'].mul(weights_mom_factor['EPS'], axis=0)).dropna()[start_ptf:]

    return (weights_mom_factor, returns_mom_factors, perf_mom_factors, weights_spi_cons_mom_factor)

"""No TE Monitor"""
run_mom_factors_noTE = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, 
                                            combine_CW_weight = 0, combine_CW = False, name = 'MF (No TE)')
return_mom_factors_noTE = run_mom_factors_noTE[1]

## Evolution of Weigths
run_mom_factors_noTE[0].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_momfactor_noTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor by Combining it with the CW Benchmark"""
run_mom_factors_combineCW = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, 
                                                 combine_CW_weight = 0.8, combine_CW = True, name = "80% MF, 20% CW")
return_mom_factors_combineCW = run_mom_factors_combineCW[1]

## Evolution of Weigths
run_mom_factors_combineCW[0].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_momfactor_combineCW_basis.png', dpi=400, bbox_inches='tight')

"""Merge Results"""
return_mom_factors = pd.DataFrame({'CW Benchmark': cw_spi_index})
return_mom_factors['MF (No TE)'] = run_mom_factors_noTE[1]
return_mom_factors['80% MF, 20% CW'] = run_mom_factors_combineCW[1]
return_mom_factors[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Basis/mom_factor_basis.png', dpi=400)

perf_mom_factors = pd.concat([perf_cwbenchmark, run_mom_factors_noTE[2], run_mom_factors_combineCW[2]], axis=1)
perf_mom_factors.to_latex('Output/Basis/mom_factor_basis.tex', column_format = 'lccc', multicolumn_format='c')

# =============================================================================
# 5.2) Strategy 2: ERC of Factors
# =============================================================================

"""

This section with generate a portfolio of equal-risk contribution among factors we created 
in the previous section. We will also provide performances measures.

"""

def build_erc(returns_factors, position_factors, TE_target, check_TE=True):
    """
    This function computes the optimal weights based on an equal-risk contribution
    among factors. 

    Parameters
    ----------
    returns_factors : DataFrame
        Returns of the factors.
    position_factors : Dataframe
        The equities allocation among a factor.
    TE_target : Float
        The TE to target.
    check_TE : Bool
        True if we want to minimize the TE in the optimization. False otherwise.

    Returns
    -------
    weights_factors_erc : DataFrame
        Portfolio weights among factors.

    """
    
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
                                                                                  + position_factors['EPS'].iloc[row-1].values * x[7]), 
                                                                                                 cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1].replace(np.nan,0))})
        
        else:
            
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})

        res_erc = minimize(criterion_erc, x0, args=(exp_returns_factors), bounds=Bounds, method='SLSQP', constraints=constraint_set)
        
        weights_factors_erc.iloc[row] = res_erc.x    
        
    return weights_factors_erc

def run_erc(returns_factors, position_factors, TE_target, TE_check, combine_CW_weight, combine_CW, name):
    """
    This function run the optimization ERC model. It also compute the performances of the strategy. 
    To minimize the tracking-error, this function can also provide a combination with the benchmark to do so. 
    This function will also be used for the sensitivity.

    Parameters
    ----------
    returns_factors : DataFrame
       Returns of the factors.
    position_factors : TYPE
        DESCRIPTION.
    TE_target : DataFrame
        The equities allocation among a factor.
    TE_check : Bool
        True if we want to minimize the TE in the optimization. False otherwise.
    combine_CW_weight : Float
        Weight allocated to the benchmark.
    combine_CW : Bool
        True if we want to combine the portfolio with the benchmark. False otherwise.
    name : Str
        Name of the stategy.

    Returns
    -------
    weights_factors_erc : DataFrame
        Portfolio weights among factors.
    erc_returns : DataFrame
        Returns of the strategy.
    perf_erc : DataFrame
        Performances of the strategy.
    weights_spi_cons_erc : Dataframe
        Equities allocation within the strategy.

    """
    
    weights_factors_erc = build_erc(returns_factors, position_factors, TE_target, TE_check)
    
    ## Compute the returns of the ERC model
    erc_returns = np.multiply(returns_factors, weights_factors_erc).sum(axis=1)
    
    if combine_CW:
        
        erc_returns = erc_returns*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances ERC model
    perf_erc = perf(erc_returns[start_ptf:], cw_spi_index[start_ptf:], risk_free, name)
    # risk_erc = risk_historical(erc_returns[start_ptf:], 0.95, 12)
    # risk_erc.plot(figsize=(7,5))
      
    # Create a DF of the total weights of each SPI constituents
    weights_spi_cons_erc = (position_factors['Momentum'].mul(weights_factors_erc['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(weights_factors_erc['Value'], axis=0)
                                    + position_factors['Size'].mul(weights_factors_erc['Size'], axis=0)
                                    + position_factors['Profitability'].mul(weights_factors_erc['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(weights_factors_erc['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(weights_factors_erc['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(weights_factors_erc['Dividend'], axis=0)
                                    + position_factors['EPS'].mul(weights_factors_erc['EPS'], axis=0)).dropna()[start_ptf:]
    
    return (weights_factors_erc, erc_returns, perf_erc, weights_spi_cons_erc)

"""No TE Monitor"""
run_erc_noTE = run_erc(returns_factors, position_factors, TE_target = 0, 
                       TE_check = False, combine_CW_weight = 0, combine_CW = False, 
                       name = "ERC (No TE)")
return_erc_noTE = run_erc_noTE[1]

## Evolution of Weigths
run_erc_noTE[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_erc_noTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor in Optimization"""
run_erc_checkTE = run_erc(returns_factors, position_factors, TE_target = 0.06, 
                          TE_check = True, combine_CW_weight = 0, combine_CW = False, 
                          name = " ERC (6% TE Target)")
returns_erc_checkTE = run_erc_checkTE[1]

## Evolution of Weigths
run_erc_checkTE[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_erc_checkTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor by Combining it with the CW Benchmark"""
run_erc_combineCW = run_erc(returns_factors, position_factors, TE_target = 0, 
                            TE_check = False, combine_CW_weight = 0.80, combine_CW = True, 
                            name = "80% ERC, 20% CW")
return_erc_combineCW = run_erc_combineCW[1]

## Evolution of Weigths
run_erc_combineCW[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_erc_combineCW_basis.png', dpi=400, bbox_inches='tight')

"""Merge Results"""
return_erc = pd.DataFrame({'CW Benchmark': cw_spi_index})
return_erc['ERC (No TE)'] = run_erc_noTE[1]
return_erc['ERC (6% TE Target)'] = run_erc_checkTE[1]
return_erc['80% ERC, 20% CW'] = run_erc_combineCW[1]

return_erc[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Basis/erc_basis.png', dpi=400)

perf_erc_basis = pd.concat([perf_cwbenchmark, run_erc_noTE[2], 
                       run_erc_checkTE[2], run_erc_combineCW[2]], axis=1)
perf_erc_basis.to_latex('Output/Basis/perf_erc_basis.tex', column_format = 'lcccc', multicolumn_format='c')


# =============================================================================
# 5.3) Strategy 3: Ridge Regression of Factors
# =============================================================================

"""

This section with generate a portfolio using a ridge regression among factors we created 
in the previous section. We will also provide performances measures.

"""

def build_ridge(returns_factors, position_factors, TE_target, check_TE=True):
    """
    This function computes the optimal weights based on a ridge regression
    among factors. 

    Parameters
    ----------
    returns_factors : DataFrame
        Returns of the factors.
    position_factors : Dataframe
        The equities allocation among a factor.
    TE_target : Float
        The TE to target.
    check_TE : Bool
        True if we want to minimize the TE in the optimization. False otherwise.

    Returns
    -------
    ridge_weights_factors : DataFrame
        Portfolio weights among factors.

    """
    
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
                                                                                  + position_factors['EPS'].iloc[row-1].values * x[7]), 
                                                                                                 cw_spi_cons.iloc[row-1].replace(np.nan,0).values, returns_spi_cons.iloc[:row-1].replace(np.nan,0))})
        
        else:
            
            constraint_set = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
        
        res_ridge = minimize(criterion_ridge, x0, args=(expected_return,varcov_matrix), bounds=bounds_ridge, method='SLSQP',constraints=constraint_set)
        ridge_weights_factors.iloc[row] = res_ridge.x
        
    return ridge_weights_factors

def run_ridge(returns_factors, position_factors, TE_target, TE_check, combine_CW_weight, combine_CW, name):
    """
    This function run the optimization ridge regressiion model. It also compute the performances of the strategy. 
    To minimize the tracking-error, this function can also provide a combination with the benchmark to do so. 
    This function will also be used for the sensitivity.

    Parameters
    ----------
    returns_factors : DataFrame
       Returns of the factors.
    position_factors : TYPE
        DESCRIPTION.
    TE_target : DataFrame
        The equities allocation among a factor.
    TE_check : Bool
        True if we want to minimize the TE in the optimization. False otherwise.
    combine_CW_weight : Float
        Weight allocated to the benchmark.
    combine_CW : Bool
        True if we want to combine the portfolio with the benchmark. False otherwise.
    name : Str
        Name of the stategy.

    Returns
    -------
    ridge_weights_factors : DataFrame
        Portfolio weights among factors.
    ridge_returns : DataFrame
        Returns of the strategy.
    perf_ridge : DataFrame
        Performances of the strategy.
    weights_spi_cons_ridge : Dataframe
        Equities allocation within the strategy.

    """

    ridge_weights_factors = build_ridge(returns_factors, position_factors, TE_target, TE_check)
    
    ## Compute the returns of ridge regression
    ridge_returns = np.multiply(returns_factors, ridge_weights_factors).sum(axis=1)  
    
    if combine_CW:
        
        ridge_returns = ridge_returns*combine_CW_weight + cw_spi_index*(1-combine_CW_weight)
    
    ## Performances Ridge Regression
    perf_ridge = perf(ridge_returns[start_ptf:], cw_spi_index[start_ptf:], risk_free, name)
    # risk_ridge = risk_historical(ridge_returns[start_ptf:], 0.95, 12)
    # risk_ridge.plot(figsize=(7,5))
            
    ## Create a DF of the total weights of each SPI constituents
    weights_spi_cons_ridge = (position_factors['Momentum'].mul(ridge_weights_factors['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(ridge_weights_factors['Value'], axis=0)
                                    + position_factors['Size'].mul(ridge_weights_factors['Size'], axis=0)
                                    + position_factors['Profitability'].mul(ridge_weights_factors['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(ridge_weights_factors['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(ridge_weights_factors['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(ridge_weights_factors['Dividend'], axis=0)
                                    + position_factors['EPS'].mul(ridge_weights_factors['EPS'], axis=0)).dropna()[start_ptf:]

    return (ridge_weights_factors, ridge_returns, perf_ridge, weights_spi_cons_ridge)

"""No TE Monitor"""
run_ridge_noTE = run_ridge(returns_factors, position_factors, TE_target = 0, 
                           TE_check = False, combine_CW_weight = 0, combine_CW = False, 
                           name = "Ridge (No TE)")
return_ridge_noTE = run_ridge_noTE[1]

## Evolution of Weigths
run_ridge_noTE[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_ridge_noTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor in Optimization"""
run_ridge_checkTE = run_ridge(returns_factors, position_factors, TE_target = 0.06, 
                              TE_check = True, combine_CW_weight = 0, combine_CW = False, 
                              name = "Ridge (6% TE Target)")
returns_ridge_checkTE = run_ridge_checkTE[1]

## Evolution of Weigths
run_ridge_checkTE[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_ridge_checkTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor by Combining it with the CW Benchmark"""
run_ridge_combineCW = run_ridge(returns_factors, position_factors, TE_target = 0, 
                                TE_check = False, combine_CW_weight = 0.80, combine_CW = True, 
                                name = "80% Ridge, 20% CW")
return_ridge_combineCW = run_ridge_combineCW[1]

## Evolution of Weigths
run_ridge_combineCW[0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_ridge_combineCW_basis.png', dpi=400, bbox_inches='tight')

"""Merge Results"""
return_ridge = pd.DataFrame({'CW Benchmark': cw_spi_index})
return_ridge['Ridge (No TE)'] = run_ridge_noTE[1]
return_ridge['Ridge (6% TE Target)'] = run_ridge_checkTE[1]
return_ridge['80% Ridge, 20% CW'] = run_ridge_combineCW[1]

return_ridge[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Basis/ridge_basis.png', dpi=400)

perf_ridge_basis = pd.concat([perf_cwbenchmark, run_ridge_noTE[2], 
                       run_ridge_checkTE[2], run_ridge_combineCW[2]], axis=1)
perf_ridge_basis.to_latex('Output/Basis/perf_ridge_basis.tex', column_format = 'lcccc', multicolumn_format='c')

# =============================================================================
# 5.4) Strategy 4: Parametrics using Macro Data (Factor Timing)
# =============================================================================

"""

This section with generate a portfolio using a parametric weights model among factors we created 
in the previous section, using macro data as explanatory variables. We will also provide performances measures.

"""


"""PARAMETRIC WEIGHTS WITH ALL MACRO VARIABLES"""
def build_parametrics(returns_factors, select_macro_data):
    """

    This function computes the optimal weights among factors based on a parametric weights model,
    using macro data as the explanatory variable to time the factors.
    
    Parameters
    ----------
    returns_factors : DataFrame
        Returns of the factors.
    select_macro_data : DataFrame
        Macro data to parametrize the weights.

    Returns
    -------
    conditional_weights_factors : DataFrame
        Factor allocation of the strategy.
    returns_factors_parametric : DataFrame
        Returns of the strategy.

    """
    
    returns_factors_parametric = returns_factors.iloc[:-1].copy()
    macro_variables_parametric = macro_data.loc['2001-01-01':, select_macro_data].copy() #keep as a dataframe and not series: macro_data.iloc[10:, 1:2].copy()
         
    shape = returns_factors_parametric.shape[1]*macro_variables_parametric.shape[1]
    
    risk_aversion = (perf_cwbenchmark.T['SR']/perf_cwbenchmark.T['Ann. STD (%)']).iloc[0]
    
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

def run_parametrics(returns_factors, position_factors, select_macro_data, combine_CW_weight, combine_CW, name):
    """
    This function run the parametric weights model with a given set of macro data. It also compute the performances of the strategy. 
    To minimize the tracking-error, this function can also provide a combination with the benchmark to do so. 
    This function will also be used for the sensitivity.   
    
    Parameters
    ----------
    returns_factors : DataFrame
       Returns of the factors.
    position_factors : DataFrame
        DESCRIPTION.
    select_macro_data : Str
        Select the set of macro data for the model.
    combine_CW_weight : Float
        Weight allocated to the benchmark.
    combine_CW : Bool
        True if we want to combine the portfolio with the benchmark. False otherwise.
    name : Str
        Name of the stategy.

    Returns
    -------
    conditional_weights_factors : DataFrame
        Portfolio weights among factors.
    parametric_returns : DataFrame
        Returns of the strategy.
    perf_parametric : DataFrame
        Performances of the strategy.
    weights_spi_cons_parametrics : Dataframe
        Equities allocation within the strategy.


    """
    
    conditional_weights_factors, returns_factors_parametric = build_parametrics(returns_factors, select_macro_data)
    
    parametric_returns = np.multiply(conditional_weights_factors,returns_factors_parametric).sum(axis=1)
    
    if combine_CW:
        
        parametric_returns = parametric_returns*combine_CW_weight + cw_spi_index*(1 - combine_CW_weight)
    
    ## Performances Parametrics
    perf_parametric = perf(parametric_returns[start_ptf:], cw_spi_index[start_ptf:], risk_free, name)
    # risk_parametric = risk_historical(parametric_returns[start_ptf:], 0.95, 12)
    # risk_parametric.plot(figsize=(7,5))
        
    ## Create a DF of the total weights of each SPI constituents
    weights_spi_cons_parametrics = (position_factors['Momentum'].mul(conditional_weights_factors['Momentum'], axis=0) 
                                    + position_factors['Value'].mul(conditional_weights_factors['Value'], axis=0)
                                    + position_factors['Size'].mul(conditional_weights_factors['Size'], axis=0)
                                    + position_factors['Profitability'].mul(conditional_weights_factors['Profitability'], axis=0)
                                    + position_factors['Beta'].mul(conditional_weights_factors['Beta'], axis=0)
                                    + position_factors['Volatility'].mul(conditional_weights_factors['Volatility'], axis=0)
                                    + position_factors['Dividend'].mul(conditional_weights_factors['Dividend'], axis=0)
                                    + position_factors['EPS'].mul(conditional_weights_factors['EPS'], axis=0)).dropna()
    
    return (conditional_weights_factors, parametric_returns, perf_parametric, weights_spi_cons_parametrics)

macro_variable_list = macro_data.columns.tolist()

"""No TE Monitor"""
run_parametrics_noTE_dict = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                             'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                             'All Macro Data': []}

for i in macro_variable_list:
    run_parametrics_noTE_dict[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], 
                                                   combine_CW_weight = 0, combine_CW = False, 
                                                   name = 'Parametrics (No TE)')

run_parametrics_noTE_dict['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, 
                                                              combine_CW_weight = 0, combine_CW = False, 
                                                              name = "Parametrics (No TE)")

## Evolution of Weigths
run_parametrics_noTE_dict['All Macro Data'][0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_parametrics_allmacro_noTE_basis.png', dpi=400, bbox_inches='tight')

run_parametrics_noTE_dict['VIX'][0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_parametrics_vix_noTE_basis.png', dpi=400, bbox_inches='tight')

"""TE Monitor by Combining it with the CW Benchmark"""
run_parametrics_combineCW_dict = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                             'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                             'All Macro Data': []}

for i in macro_variable_list:
    run_parametrics_combineCW_dict[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], 
                                                        combine_CW_weight = 0.8, combine_CW = True, 
                                                        name = '80% Parametrics, 20% CW')

run_parametrics_combineCW_dict['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, 
                                                                   combine_CW_weight = 0.8, combine_CW = True, 
                                                                   name = "80% Parametrics, 20% CW")

## Evolution of Weigths
run_parametrics_combineCW_dict['All Macro Data'][0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_parametrics_allmacro_combineCW_basis.png', dpi=400, bbox_inches='tight')

run_parametrics_combineCW_dict['VIX'][0][start_ptf:].plot(figsize=(10, 7))
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Plot/Weights/weights_parametrics_vix_combineCW_basis.png', dpi=400, bbox_inches='tight')

"""Merge Results"""
# VIX
return_parametrics_vix = pd.DataFrame({'CW Benchmark': cw_spi_index})
return_parametrics_vix['Parametrics (No TE)'] = run_parametrics_noTE_dict['VIX'][1]
return_parametrics_vix['80% Parametrics, 20% CW'] = run_parametrics_combineCW_dict['VIX'][1]
return_parametrics_vix[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Basis/parametrics_vix_basis.png', dpi=400)

perf_parametrics_vix = pd.concat([perf_cwbenchmark, run_parametrics_noTE_dict['VIX'][2], run_parametrics_combineCW_dict['VIX'][2]], axis=1)
perf_parametrics_vix.to_latex('Output/Basis/parametrics_vix_basis.tex', column_format = 'lccc', multicolumn_format='c')

# All Macro
return_parametrics_allmacro = pd.DataFrame({'CW Benchmark': cw_spi_index})
return_parametrics_allmacro['Parametrics (No TE)'] = run_parametrics_noTE_dict['All Macro Data'][1]
return_parametrics_allmacro['80% Parametrics, 20% CW'] = run_parametrics_combineCW_dict['All Macro Data'][1]
return_parametrics_allmacro[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Basis/parametrics_allmacro_basis.png', dpi=400)

return_parametrics_allmacro = pd.concat([perf_cwbenchmark, run_parametrics_noTE_dict['All Macro Data'][2], 
                                  run_parametrics_combineCW_dict['All Macro Data'][2]], axis=1)
return_parametrics_allmacro.to_latex('Output/Basis/parametrics_allmacro_basis.tex', column_format = 'lccc', multicolumn_format='c')

# All Factor Weights Merged
avg_weight_merged = pd.DataFrame({'Mom Factor': (run_mom_factors_noTE[0][start_ptf:].mean()*100).round(2),
                                  'ERC': (run_erc_noTE[0][start_ptf:].mean()*100).round(2),
                                  'Ridge': (run_ridge_noTE[0][start_ptf:].mean()*100).round(2),
                                  'Parametrics': (run_parametrics_noTE_dict['VIX'][0][start_ptf:].mean()*100).round(2)})
avg_weight_merged.to_latex('Output/Summary/avg_weights_merged.tex', column_format = 'lcccc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 6) Fama-French Factor Analysis
# =============================================================================
# =============================================================================

# =============================================================================
# 6.1) Creation of the FF Factors in the Swiss Market (SPI Constitutents)
# =============================================================================

"""

This section aims to provide a Fama-French analysis of our portfolios to determine its 
exposure to various factors. 

The factors have been created using a similar methodology described on the website of Kenneth R. French:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

"""

# Get the necessary data without constraints

price_spi_cons = spi[0] # Price of all constituents
pe_spi_cons = spi[1] # PE ratios for all constituents
mktcap_spi_cons = spi[3] # Market cap for all consituents
mb_spi_cons = spi[12] #Market-to-book ratio of all constituents
bm_spi_cons = 1/spi[12] #Book-to-market of all constituents
profit_spi_cons = spi[14] #Operating Profit Margin of all constituents

#Compute the returns
returns_spi_cons = (price_spi_cons/price_spi_cons.shift(1) - 1) #.replace(0, np.nan)
returns_spi_cons.loc['2000-01-01'] = 0
returns_spi_cons = returns_spi_cons.replace([np.inf, -np.inf], 0)

returns_past12 = returns_spi_cons.rolling(12,closed='left').mean()

#Compute the 12-months rolling variance
roll_vol_spi_cons = returns_spi_cons.rolling(12).std()

"""Market Factor"""
excess_return_market = returns_spi['SPI INDEX'] - libor1M_CHF['1M Libor CHF']/100

"""SMB Factor"""
def SMB_bm():
    """
    Small-minus-big factor based on book-to-market ratio.
    """
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
    """
    Small-minus-big factor based on operating profitability.

    """
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
    """
    Small-minus-big factor based on investment of the company.

    """
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
    """
    High-minus-low factor based on the book-to-market ratio

    """
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
    """
    Winner-minus-loser based on the past 12-month average returns.
    """
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
    """
    Robust-minus-weak factor based on the operating profitability. 

    """
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
    """
    Conservative-minus-aggressive factor based on the investments of the firms.

    """
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
    """
    Volatility factor based on the 12-month rolling volatility.

    """
    position_small = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=False, ew_position=False).replace(0, np.nan)
    position_small_high = factor_building((roll_vol_spi_cons*position_small), quantile=0.60, long_above_quantile=True, ew_position=False)
    position_small_low = factor_building((roll_vol_spi_cons*position_small), quantile=0.40, long_above_quantile=False, ew_position=False)
        
    position_big = factor_building(mktcap_spi_cons, quantile=0.5, long_above_quantile=True, ew_position=False).replace(0, np.nan)
    position_big_high = factor_building((roll_vol_spi_cons*position_big), quantile=0.60, long_above_quantile=True, ew_position=False)
    position_big_low = factor_building((roll_vol_spi_cons*position_big), quantile=0.40, long_above_quantile=False, ew_position=False)
    
    returns_small_high = (position_small_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_small_low = (position_small_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    returns_big_high = (position_big_high*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    returns_big_low = (position_big_low*returns_spi_cons).replace(0, np.nan).mean(axis=1)
    
    vol = 0.5*(returns_small_low + returns_big_low) - 0.5*(returns_small_high + returns_big_high)
    
    return vol

vol = VOL()

# =============================================================================
# 6.2) Regress a CAPM Regression
# =============================================================================

"""CAPM Model Analysis"""
capm_mktrf = pd.DataFrame({'MktRf': excess_return_market}).dropna()
capm_mktrf_constant = sm.add_constant(capm_mktrf[start_ptf:])

## No TE Monitor
capm_mom_factor_noTE = run_ff_regression(return_mom_factors_noTE, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_erc_noTE = run_ff_regression(return_erc_noTE, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_ridge_noTE = run_ff_regression(return_ridge_noTE, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_parametrics_noTE = run_ff_regression(run_parametrics_noTE_dict['VIX'][1], capm_mktrf_constant[:'2021-09-01'], libor1M_CHF['1M Libor CHF']/100)

## TE Monitor by Combining it with the CW Benchmark
capm_mom_factor_combineCW = run_ff_regression(return_mom_factors_combineCW, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_erc_combineCW = run_ff_regression(return_erc_combineCW, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_ridge_combineCW = run_ff_regression(return_ridge_combineCW, capm_mktrf_constant, libor1M_CHF['1M Libor CHF']/100)
capm_parametrics_combineCW = run_ff_regression(run_parametrics_combineCW_dict['VIX'][1], capm_mktrf_constant[:'2021-09-01'], libor1M_CHF['1M Libor CHF']/100)

# =============================================================================
# 6.3) Regress a 3 factors Fama-French 
# =============================================================================

"""Fama-French 3 Factor Model Analysis"""
ff3_merged = pd.DataFrame({'MktRf': excess_return_market, 'SMB': smb, 'HML': hml}).dropna()
ff3_merged_constant = sm.add_constant(ff3_merged[start_ptf:])

## No TE Monitor
ff3_mom_factor_noTE = run_ff_regression(return_mom_factors_noTE, ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_erc_noTE = run_ff_regression(return_erc_noTE, ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_ridge_noTE = run_ff_regression(return_ridge_noTE, ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_parametrics_noTE = run_ff_regression(run_parametrics_noTE_dict['VIX'][1], ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)

## TE Monitor by Combining it with the CW Benchmark
ff3_mom_factor_combineCW  = run_ff_regression(return_mom_factors_combineCW , ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_erc_combineCW  = run_ff_regression(return_erc_combineCW , ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_ridge_combineCW  = run_ff_regression(return_ridge_combineCW , ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff3_parametrics_combineCW  = run_ff_regression(run_parametrics_combineCW_dict['VIX'][1], ff3_merged_constant, libor1M_CHF['1M Libor CHF']/100)

# =============================================================================
# 6.4) Regress a multiple factors Fama-French
# =============================================================================

"""Fama-French Multiple Factor Model Analysis"""
ff_merged = pd.DataFrame({'MktRf': excess_return_market, 'SMB': smb, 'HML': hml, 'WML': wml, 'RMW': rmw, 'CMA': cma, 'VOL': vol}).dropna()
ff_merged_constant = sm.add_constant(ff_merged[start_ptf:])

## No TE Monitor
ff_mom_factor_noTE = run_ff_regression(return_mom_factors_noTE, ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_erc_noTE = run_ff_regression(return_erc_noTE, ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_ridge_noTE = run_ff_regression(return_ridge_noTE, ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_parametrics_noTE = run_ff_regression(run_parametrics_noTE_dict['VIX'][1], ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)

## TE Monitor by Combining it with the CW Benchmark
ff_mom_factor_combineCW  = run_ff_regression(return_mom_factors_combineCW , ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_erc_combineCW  = run_ff_regression(return_erc_combineCW , ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_ridge_combineCW  = run_ff_regression(return_ridge_combineCW , ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)
ff_parametrics_combineCW  = run_ff_regression(run_parametrics_combineCW_dict['VIX'][1], ff_merged_constant, libor1M_CHF['1M Libor CHF']/100)

# =============================================================================
# 6.5) Merge the Results
# =============================================================================

"""Momentum Factor"""
ff_merged_momfactor_noTE = pd.concat([capm_mom_factor_noTE, ff3_mom_factor_noTE, ff_mom_factor_noTE], axis=0).replace(np.nan, '-')
ff_merged_momfactor_noTE['R2'] = ff_merged_momfactor_noTE.pop('R2')
ff_merged_momfactor_noTE.to_latex('Output/FF/ff_merged_momfactor_noTE.tex', column_format = 'lccccccccc', multicolumn_format='c')

ff_merged_momfactor_combineCW = pd.concat([capm_mom_factor_combineCW, ff3_mom_factor_combineCW, ff_mom_factor_combineCW], axis=0).replace(np.nan, '-')
ff_merged_momfactor_combineCW['R2'] = ff_merged_momfactor_combineCW.pop('R2')
ff_merged_momfactor_combineCW.to_latex('Output/FF/ff_merged_momfactor_combinecw.tex')

"""ERC"""
ff_merged_erc_noTE = pd.concat([capm_erc_noTE, ff3_erc_noTE, ff_erc_noTE], axis=0).replace(np.nan, '-')
ff_merged_erc_noTE['R2'] = ff_merged_erc_noTE.pop('R2')
ff_merged_erc_noTE.to_latex('Output/FF/ff_merged_erc_noTE.tex')

ff_merged_erc_combineCW = pd.concat([capm_erc_combineCW, ff3_erc_combineCW, ff_erc_combineCW], axis=0).replace(np.nan, '-')
ff_merged_erc_combineCW['R2'] = ff_merged_erc_combineCW.pop('R2')
ff_merged_erc_combineCW.to_latex('Output/FF/ff_merged_erc_combinecw.tex')

"""Ridge"""
ff_merged_ridge_noTE = pd.concat([capm_ridge_noTE, ff3_ridge_noTE, ff_ridge_noTE], axis=0).replace(np.nan, '-')
ff_merged_ridge_noTE['R2'] = ff_merged_ridge_noTE.pop('R2')
ff_merged_ridge_noTE.to_latex('Output/FF/ff_merged_ridge_noTE.tex')

ff_merged_ridge_combineCW = pd.concat([capm_ridge_combineCW, ff3_ridge_combineCW, ff_ridge_combineCW], axis=0).replace(np.nan, '-')
ff_merged_ridge_combineCW['R2'] = ff_merged_ridge_combineCW.pop('R2')
ff_merged_ridge_combineCW.to_latex('Output/FF/ff_merged_ridge_combinecw.tex')

"""Parametrics"""
ff_merged_parametrics_noTE = pd.concat([capm_parametrics_noTE, ff3_parametrics_noTE, ff_parametrics_noTE], axis=0).replace(np.nan, '-')
ff_merged_parametrics_noTE['R2'] = ff_merged_parametrics_noTE.pop('R2')
ff_merged_parametrics_noTE.to_latex('Output/FF/ff_merged_parametrics_noTE.tex')

ff_merged_parametrics_combineCW = pd.concat([capm_parametrics_combineCW, ff3_parametrics_combineCW, ff_parametrics_combineCW], axis=0).replace(np.nan, '-')
ff_merged_parametrics_combineCW['R2'] = ff_merged_parametrics_combineCW.pop('R2')
ff_merged_parametrics_combineCW.to_latex('Output/FF/ff_merged_parametrics_combinecw.tex')

# =============================================================================
# =============================================================================
# 7) Sensitivity 
# =============================================================================
# =============================================================================

"""
In this section, we will perform a sensitivity analysis on 3 different parameters of our model
to check its robustness. We will perform it on 

    - The quantile chosen to set a liquidity constraint in our portfolio allocation
    - The quantile chosen to construct the eight different factors of section 4
    - The combination between the portfolio and benchmark to reduce the tracking-error.

"""

# =============================================================================
# 7.1) Liquidity Constraint Sensitivity
# =============================================================================

# Create list to collect the results of the sensitivity
list_return_mom_factors_noTE = [cw_spi_index]
list_return_mom_factors_combineCW = [cw_spi_index]
list_perf_mom_factors = []

list_return_erc_noTE = [cw_spi_index]
list_return_erc_combineCW = [cw_spi_index]
list_perf_erc = []

list_return_ridge_noTE = [cw_spi_index]
list_return_ridge_combineCW = [cw_spi_index]
list_perf_ridge = []

list_return_parametrics_noTE = [cw_spi_index]
list_return_parametrics_combineCW = [cw_spi_index]
list_perf_parametrics = []

# Run the sensitivity
for k in [0, 0.1, 0.3, 0.5]:
    price_spi_cons, pe_spi_cons, dividend_spi_cons, mktcap_spi_cons, beta_spi_cons, vol_spi_cons, roe_spi_con, roa_spi_con, gm_spi_cons, eps_spi_cons, trade_liq = liqudity_constraint(k)
    returns_factors, position_factors = run_factor_building(quantile = 0.5)
    
    ### Momentum of Factor ###
    """No TE Monitor"""
    run_mom_factors_noTE_temp = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, combine_CW_weight = 0, combine_CW = False, name = 'MF (No TE)')
    
    list_return_mom_factors_noTE.append(run_mom_factors_noTE_temp[1])
    list_perf_mom_factors.append(run_mom_factors_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_mom_factors_combineCW_temp = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, combine_CW_weight = 0.8, combine_CW = True, name = "80% MF - 20% CW")
    
    list_return_mom_factors_combineCW.append(run_mom_factors_combineCW_temp[1])
    list_perf_mom_factors.append(run_mom_factors_combineCW_temp[2].squeeze().tolist())
    
    ### ERC Regression ###
    """No TE Monitor"""
    run_erc_noTE_temp = run_erc(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "ERC (No TE)")
    
    list_return_erc_noTE.append(run_erc_noTE_temp[1])
    list_perf_erc.append(run_erc_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_erc_combineCW_temp = run_erc(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "80% ERC - 20% CW")
    
    list_return_erc_combineCW.append(run_erc_combineCW_temp[1])
    list_perf_erc.append(run_erc_combineCW_temp[2].squeeze().tolist())
    
    ### Ridge Regression ###
    """No TE Monitor"""
    run_ridge_noTE_temp = run_ridge(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "Ridge (No TE)")
    
    list_return_ridge_noTE.append(run_ridge_noTE_temp[1])
    list_perf_ridge.append(run_ridge_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_ridge_combineCW_temp = run_ridge(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "80% Ridge - 20% CW")
    
    list_return_ridge_combineCW.append(run_ridge_combineCW_temp[1])
    list_perf_ridge.append(run_ridge_combineCW_temp[2].squeeze().tolist())
    
    ### Parametrics Model ###
    """No TE Monitor"""
    run_parametrics_noTE_dict_temp = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                                  'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                                  'All Macro Data': []}
    
    # for i in macro_variable_list:
    #     run_parametrics_noTE_dict_temp[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], combine_CW_weight = 0, combine_CW = False, name = f'({i}, No TE Monitor)')
    
    # run_parametrics_noTE_dict_temp['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, combine_CW_weight = 0, combine_CW = False, name = "(All Macro Data, No TE Monitor)")
    
    # list_return_parametrics_noTE.append(run_parametrics_noTE_dict_temp['All Macro Data'][1])
    # list_perf_parametrics.append(run_parametrics_noTE_dict_temp['All Macro Data'][2].squeeze().tolist()) 
    
    run_parametrics_noTE_dict_temp['VIX'] = run_parametrics(returns_factors, position_factors, select_macro_data = ['VIX'], 
                                                       combine_CW_weight = 0, combine_CW = False, name = "Parametrics - VIX (No TE)")
    
    list_return_parametrics_noTE.append(run_parametrics_noTE_dict_temp['VIX'][1])
    list_perf_parametrics.append(run_parametrics_noTE_dict_temp['VIX'][2].squeeze().tolist()) 
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_parametrics_combineCW_dict_temp = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                                  'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                                  'All Macro Data': []}
    
    # for i in macro_variable_list:
    #     run_parametrics_combineCW_dict_temp[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], combine_CW_weight = 0.8, combine_CW = True, name = f'({i}, 80% Portfolio, 20% CW Benchmark)')
    
    # run_parametrics_combineCW_dict_temp['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, combine_CW_weight = 0.8, combine_CW = True, name = "(All Macro Data, 80% Portfolio, 20% CW Benchmark)")
    
    # list_return_parametrics_combineCW.append(run_parametrics_combineCW_dict_temp['All Macro Data'][1])
    # list_perf_parametrics.append(run_parametrics_combineCW_dict_temp['All Macro Data'][2].squeeze().tolist()) 
    
    run_parametrics_combineCW_dict_temp['VIX'] = run_parametrics(returns_factors, position_factors, select_macro_data = ['VIX'], combine_CW_weight = 0.8, combine_CW = True, name = "80% Parametrics (VIX), 20% CW")
    
    list_return_parametrics_combineCW.append(run_parametrics_combineCW_dict_temp['VIX'][1])
    list_perf_parametrics.append(run_parametrics_combineCW_dict_temp['VIX'][2].squeeze().tolist()) 
    
### Merge Data ###
index_liqsensitivity = ['CW Benchmark', '0% Qtl', '10% Qtl', '30% Qtl', '50% Qtl']
mux_liqsensitivity = pd.MultiIndex.from_product([['0% Qtl', '10% Qtl', '30% Qtl', '50% Qtl'], ['No TE','20% CW']])

"""Momentum of Factor"""
return_mom_factors_noTE_liqsentivity = pd.DataFrame(list_return_mom_factors_noTE, index_liqsensitivity).T
return_mom_factors_noTE_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_mom_factors_noTE_liqsentivity.png', dpi=400)

return_mom_factors_combineCW_liqsentivity = pd.DataFrame(list_return_mom_factors_combineCW, index_liqsensitivity).T
return_mom_factors_combineCW_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_mom_factors_combineCW_liqsentivity.png', dpi=400)

perf_mom_factors_liqsentivity = pd.DataFrame(list_perf_mom_factors , index=mux_liqsensitivity, columns=perf_cwbenchmark.index).T
perf_mom_factors_liqsentivity.to_latex('Output/Sensitivity/Liquidity/perf_mom_factors_liqsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""ERC"""
return_erc_noTE_liqsentivity = pd.DataFrame(list_return_erc_noTE, index_liqsensitivity).T
return_erc_noTE_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_erc_noTE_liqsentivity.png', dpi=400)

return_erc_combineCW_liqsentivity = pd.DataFrame(list_return_erc_combineCW, index_liqsensitivity).T
return_erc_combineCW_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_erc_combineCW_liqsentivity.png', dpi=400)

perf_erc_liqsentivity = pd.DataFrame(list_perf_erc, index=mux_liqsensitivity, columns=perf_cwbenchmark.index).T
perf_erc_liqsentivity.to_latex('Output/Sensitivity/Liquidity/perf_erc_liqsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""Ridge"""
return_ridge_noTE_liqsentivity = pd.DataFrame(list_return_ridge_noTE, index_liqsensitivity).T
return_ridge_noTE_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_ridge_noTE_liqsentivity.png', dpi=400)

return_ridge_combineCW_liqsentivity = pd.DataFrame(list_return_ridge_combineCW, index_liqsensitivity).T
return_ridge_combineCW_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_ridge_combineCW_liqsentivity.png', dpi=400)

perf_ridge_liqsentivity = pd.DataFrame(list_perf_ridge, index=mux_liqsensitivity, columns=perf_cwbenchmark.index).T
perf_ridge_liqsentivity.to_latex('Output/Sensitivity/Liquidity/perf_ridge_liqsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""Parametrics"""
return_parametrics_noTE_liqsentivity = pd.DataFrame(list_return_parametrics_noTE, index_liqsensitivity).T
return_parametrics_noTE_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_parametrics_noTE_liqsentivity.png', dpi=400)

return_parametrics_combineCW_liqsentivity = pd.DataFrame(list_return_parametrics_combineCW, index_liqsensitivity).T
return_parametrics_combineCW_liqsentivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/Liquidity/return_parametrics_combineCW_liqsentivity.png', dpi=400)

perf_parametrics_liqsentivity = pd.DataFrame(list_perf_parametrics, index=mux_liqsensitivity, columns=perf_cwbenchmark.index).T
perf_parametrics_liqsentivity.to_latex('Output/Sensitivity/Liquidity/perf_parametrics_liqsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

# =============================================================================
# 7.2) Factor Construction Quantile Sensitivity        
# =============================================================================

# Create list to collect the results of the sensitivity
list_return_mom_factors_noTE = [cw_spi_index]
list_return_mom_factors_combineCW = [cw_spi_index]
list_perf_mom_factors = []

list_return_erc_noTE = [cw_spi_index]
list_return_erc_combineCW = [cw_spi_index]
list_perf_erc = []

list_return_ridge_noTE = [cw_spi_index]
list_return_ridge_combineCW = [cw_spi_index]
list_perf_ridge = []

list_return_parametrics_noTE = [cw_spi_index]
list_return_parametrics_combineCW = [cw_spi_index]
list_perf_parametrics = []

# Run the sensitivity
for k in [0.2, 0.3, 0.4, 0.6]:
    price_spi_cons, pe_spi_cons, dividend_spi_cons, mktcap_spi_cons, beta_spi_cons, vol_spi_cons, roe_spi_con, roa_spi_con, gm_spi_cons, eps_spi_cons, trade_liq = liqudity_constraint(0.25)
    returns_factors, position_factors = run_factor_building(quantile = k)
    
    ### Momentum of Factor ###
    """No TE Monitor"""
    run_mom_factors_noTE_temp = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, combine_CW_weight = 0, combine_CW = False, name = 'MF (No TE)')
    
    list_return_mom_factors_noTE.append(run_mom_factors_noTE_temp[1])
    list_perf_mom_factors.append(run_mom_factors_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_mom_factors_combineCW_temp = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, combine_CW_weight = 0.8, combine_CW = True, name = "80% MF - 20% CW")
    
    list_return_mom_factors_combineCW.append(run_mom_factors_combineCW_temp[1])
    list_perf_mom_factors.append(run_mom_factors_combineCW_temp[2].squeeze().tolist())
    
    ### ERC Regression ###
    """No TE Monitor"""
    run_erc_noTE_temp = run_erc(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "ERC (No TE)")
    
    list_return_erc_noTE.append(run_erc_noTE_temp[1])
    list_perf_erc.append(run_erc_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_erc_combineCW_temp = run_erc(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "80% ERC - 20% CW")
    
    list_return_erc_combineCW.append(run_erc_combineCW_temp[1])
    list_perf_erc.append(run_erc_combineCW_temp[2].squeeze().tolist())
    
    ### Ridge Regression ###
    """No TE Monitor"""
    run_ridge_noTE_temp = run_ridge(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0, combine_CW = False, name = "Ridge (No TE)")
    
    list_return_ridge_noTE.append(run_ridge_noTE_temp[1])
    list_perf_ridge.append(run_ridge_noTE_temp[2].squeeze().tolist())
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_ridge_combineCW_temp = run_ridge(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = 0.80, combine_CW = True, name = "80% Ridge - 20% CW")
    
    list_return_ridge_combineCW.append(run_ridge_combineCW_temp[1])
    list_perf_ridge.append(run_ridge_combineCW_temp[2].squeeze().tolist())
    
    ### Parametrics Model ###
    """No TE Monitor"""
    run_parametrics_noTE_dict_temp = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                                  'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                                  'All Macro Data': []}
    
    # for i in macro_variable_list:
    #     run_parametrics_noTE_dict_temp[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], combine_CW_weight = 0, combine_CW = False, name = f'({i}, No TE Monitor)')
    
    # run_parametrics_noTE_dict_temp['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, combine_CW_weight = 0, combine_CW = False, name = "(All Macro Data, No TE Monitor)")
    
    # list_return_parametrics_noTE.append(run_parametrics_noTE_dict_temp['All Macro Data'][1])
    # list_perf_parametrics.append(run_parametrics_noTE_dict_temp['All Macro Data'][2].squeeze().tolist()) 
    
    run_parametrics_noTE_dict_temp['VIX'] = run_parametrics(returns_factors, position_factors, select_macro_data = ['VIX'], 
                                                       combine_CW_weight = 0, combine_CW = False, name = "Parametrics - VIX (No TE)")
    
    list_return_parametrics_noTE.append(run_parametrics_noTE_dict_temp['VIX'][1])
    list_perf_parametrics.append(run_parametrics_noTE_dict_temp['VIX'][2].squeeze().tolist()) 
    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_parametrics_combineCW_dict_temp = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                                  'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                                  'All Macro Data': []}
    
    # for i in macro_variable_list:
    #     run_parametrics_combineCW_dict_temp[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], combine_CW_weight = 0.8, combine_CW = True, name = f'({i}, 80% Portfolio, 20% CW Benchmark)')
    
    # run_parametrics_combineCW_dict_temp['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, combine_CW_weight = 0.8, combine_CW = True, name = "(All Macro Data, 80% Portfolio, 20% CW Benchmark)")
    
    # list_return_parametrics_combineCW.append(run_parametrics_combineCW_dict_temp['All Macro Data'][1])
    # list_perf_parametrics.append(run_parametrics_combineCW_dict_temp['All Macro Data'][2].squeeze().tolist()) 
    
    run_parametrics_combineCW_dict_temp['VIX'] = run_parametrics(returns_factors, position_factors, select_macro_data = ['VIX'], 
                                                                 combine_CW_weight = 0.8, combine_CW = True, name = "80% Parametrics (VIX), 20% CW")
    
    list_return_parametrics_combineCW.append(run_parametrics_combineCW_dict_temp['VIX'][1])
    list_perf_parametrics.append(run_parametrics_combineCW_dict_temp['VIX'][2].squeeze().tolist()) 
     
### Merge Data ###
index_fsensitivity = ['CW Benchmark', '20% Qtl', '30% Qtl', '40% Qtl', '60% Qtl']
mux_fsensitivity = pd.MultiIndex.from_product([['20% Qtl', '30% Qtl', '40% Qtl', '60% Qtl'], ['No TE','20% CW']])

"""Momentum of Factor"""
return_mom_factors_noTE_fsensitivity = pd.DataFrame(list_return_mom_factors_noTE, index_fsensitivity).T
return_mom_factors_noTE_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_mom_factors_noTE_fsentivity.png', dpi=400)

return_mom_factors_combineCW_fsensitivity = pd.DataFrame(list_return_mom_factors_combineCW, index_fsensitivity).T
return_mom_factors_combineCW_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_mom_factors_combineCW_fsentivity.png', dpi=400)

perf_mom_factors_fsensitivity = pd.DataFrame(list_perf_mom_factors, index=mux_fsensitivity, columns=perf_cwbenchmark.index).T
perf_mom_factors_fsensitivity.to_latex('Output/Sensitivity/FactorQuantile/perf_mom_factors_fsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""ERC"""
return_erc_noTE_fsensitivity = pd.DataFrame(list_return_erc_noTE, index_fsensitivity).T
return_erc_noTE_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_erc_noTE_fsentivity.png', dpi=400)

return_erc_combineCW_fsensitivity = pd.DataFrame(list_return_erc_combineCW, index_fsensitivity).T
return_erc_combineCW_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_erc_combineCW_fsentivity.png', dpi=400)

perf_erc_fsensitivity = pd.DataFrame(list_perf_erc, index=mux_fsensitivity, columns=perf_cwbenchmark.index).T
perf_erc_fsensitivity.to_latex('Output/Sensitivity/FactorQuantile/perf_erc_fsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""Ridge"""
return_ridge_noTE_fsensitivity = pd.DataFrame(list_return_ridge_noTE, index_fsensitivity).T
return_ridge_noTE_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_ridge_noTE_fsentivity.png', dpi=400)

return_ridge_combineCW_fsensitivity = pd.DataFrame(list_return_ridge_combineCW, index_fsensitivity).T
return_ridge_combineCW_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_ridge_combineCW_fsentivity.png', dpi=400)

perf_ridge_fsensitivity = pd.DataFrame(list_perf_ridge, index=mux_fsensitivity, columns=perf_cwbenchmark.index).T
perf_ridge_fsensitivity.to_latex('Output/Sensitivity/FactorQuantile/perf_ridge_fsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

"""Parametrics"""
return_parametrics_noTE_fsensitivity = pd.DataFrame(list_return_parametrics_noTE, index_fsensitivity).T
return_parametrics_noTE_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_parametrics_noTE_fsentivity.png', dpi=400)

return_parametrics_combineCW_fsensitivity = pd.DataFrame(list_return_parametrics_combineCW, index_fsensitivity).T
return_parametrics_combineCW_fsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/FactorQuantile/return_parametrics_combineCW_fsentivity.png', dpi=400)

perf_parametrics_fsensitivity = pd.DataFrame(list_perf_parametrics, index=mux_fsensitivity, columns=perf_cwbenchmark.index).T
perf_parametrics_fsensitivity.to_latex('Output/Sensitivity/FactorQuantile/perf_parametrics_fsentivity.tex', column_format = 'lcccccccc', multicolumn_format='c')

# =============================================================================
# 7.3) Combination with CW Benchmark Sensitivity (TE Reduction)
# =============================================================================

price_spi_cons, pe_spi_cons, dividend_spi_cons, mktcap_spi_cons, beta_spi_cons, vol_spi_cons, roe_spi_con, roa_spi_con, gm_spi_cons, eps_spi_cons, trade_liq = liqudity_constraint(0.25)
returns_factors, position_factors = run_factor_building(quantile = 0.5)

# Create list to collect the results of the sensitivity
mom_factor_return_cwsensitivity = pd.DataFrame({'CW Benchmark': cw_spi_index})
erc_return_cwsensitivity = pd.DataFrame({'CW Benchmark': cw_spi_index})
ridge_return_cwsensitivity = pd.DataFrame({'CW Benchmark': cw_spi_index})
parametrics_return_cwsensitivity = pd.DataFrame({'CW Benchmark': cw_spi_index})

list_perf_mom_factors = []
list_perf_erc = []
list_perf_ridge = []
list_perf_parametrics = []

# Run the sensitivity
for k in [0.5, 0.6, 0.7, 0.9]:
    ### Momentum of Factor ###    
    """TE Monitor by Combining it with the CW Benchmark"""
    run_mom_factors_combineCW_temp = run_momentum_factors(returns_factors, position_factors, quantile = 0.5, combine_CW_weight = k, combine_CW = True, name = f'{k*100}% Mom. Factor, {round((1-k)*100)}% CW')

    mom_factor_return_cwsensitivity[f'{k*100}% Mom. Factor, {round((1-k)*100)}% CW'] = run_mom_factors_combineCW_temp[1]
    list_perf_mom_factors.append(run_mom_factors_combineCW_temp[2].squeeze().tolist())
    
    ### ERC Regression ###
    """TE Monitor by Combining it with the CW Benchmark"""
    run_erc_combineCW_temp = run_erc(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = k, combine_CW = True, name = f'{k*100}% ERC, {round((1-k)*100)}% CW')

    erc_return_cwsensitivity[f'{k*100}% ERC, {round((1-k)*100)}% CW'] = run_erc_combineCW_temp[1]    
    list_perf_erc.append(run_erc_combineCW_temp[2].squeeze().tolist())
    
    ### Ridge Regression ### 
    """TE Monitor by Combining it with the CW Benchmark"""
    run_ridge_combineCW_temp = run_ridge(returns_factors, position_factors, TE_target = 0, TE_check = False, combine_CW_weight = k, combine_CW = True, name = f'{k*100}% Ridge, {round((1-k)*100)}% CW')

    ridge_return_cwsensitivity[f'{k*100}% Ridge, {round((1-k)*100)}% CW'] = run_ridge_combineCW_temp[1]    
    list_perf_ridge.append(run_ridge_combineCW_temp[2].squeeze().tolist())
    
    ### Parametrics Model ###
    """TE Monitor by Combining it with the CW Benchmark"""
    run_parametrics_combineCW_dict_temp = {'10y Bond Yield US': [], 'VIX': [], 'CPI US': [], 
                                  'TED Spread': [], '3M Libor US': [], '12M Libor US': [],
                                  'All Macro Data': []}
    
    # for i in macro_variable_list:
    #     run_parametrics_combineCW_dict[i] = run_parametrics(returns_factors, position_factors, select_macro_data = [i], combine_CW_weight = k, combine_CW = True, name = f'({i}, 80% Portfolio, 20% CW Benchmark)')
    
    # run_parametrics_combineCW_dict['All Macro Data'] = run_parametrics(returns_factors, position_factors, select_macro_data = macro_variable_list, combine_CW_weight = k, combine_CW = True, name = "(All Macro Data, 80% Portfolio, 20% CW Benchmark)")
    
    run_parametrics_combineCW_dict_temp['VIX'] = run_parametrics(returns_factors, position_factors, select_macro_data = ['VIX'], 
                                                            combine_CW_weight = k, combine_CW = True, name = f'{k*100}% Parametrics, {round((1-k)*100)}% CW')
    
    parametrics_return_cwsensitivity[f'{k*100}% Parametrics, {round((1-k)*100)}% CW'] = run_parametrics_combineCW_dict_temp['VIX'][1]    
    list_perf_parametrics.append(run_parametrics_combineCW_dict_temp['VIX'][2].squeeze().tolist())
    
### Merge Data ###
index_cwsensitivity = ['50% Ptf, 50% CW', '60% Ptf, 40% CW', '70% Ptf, 30% CW', '90% Ptf, 10% CW']

"""Momentum of Factor"""
mom_factor_return_cwsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/CW/return_mom_factor_cwsentivity.png', dpi=400)

perf_mom_factors_cwsentivity = pd.DataFrame(list_perf_mom_factors,index=index_cwsensitivity, columns=perf_cwbenchmark.index).T
perf_mom_factors_cwsentivity.to_latex('Output/Sensitivity/CW/perf_mom_factor_cwsentivity.tex', column_format = 'lcccc', multicolumn_format='c')

"""ERC"""
erc_return_cwsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/CW/return_erc_cwsentivity.png', dpi=400)

perf_erc_cwsentivity = pd.DataFrame(list_perf_erc,index=index_cwsensitivity, columns=perf_cwbenchmark.index).T
perf_erc_cwsentivity.to_latex('Output/Sensitivity/CW/perf_erc_cwsentivity.tex', column_format = 'lcccc', multicolumn_format='c')

"""Ridge"""
ridge_return_cwsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/CW/return_ridge_cwsentivity.png', dpi=400)

perf_ridge_cwsentivity = pd.DataFrame(list_perf_ridge,index=index_cwsensitivity, columns=perf_cwbenchmark.index).T
perf_ridge_cwsentivity.to_latex('Output/Sensitivity/CW/perf_ridge_cwsentivity.tex', column_format = 'lcccc', multicolumn_format='c')

"""Parametrics"""
parametrics_return_cwsensitivity[start_ptf:].apply(cum_prod).plot(figsize=(10,7), colormap = 'Set1')
plt.savefig('Plot/Sensitivity/CW/return_parametrics_cwsentivity.png', dpi=400)

perf_parametrics_cwsentivity = pd.DataFrame(list_perf_parametrics,index=index_cwsensitivity, columns=perf_cwbenchmark.index).T
perf_parametrics_cwsentivity.to_latex('Output/Sensitivity/CW/perf_parametrics_cwsentivity.tex', column_format = 'lcccc', multicolumn_format='c')

# =============================================================================
# =============================================================================
# 8) Collect all Data for the Dashboard
# =============================================================================
# =============================================================================

"""
This section aims to collect data and result to be included in our dashboard. The dashboard is 
divided into five different categories:
    
    - Overview
    - Defensive Portfolio (Ridge Regression)
    - Balanced Portfolio (Parametric Portfolio with VIX)
    - Dynamic Portfolio (Momentum of Factors)
    - News

"""

# =============================================================================
# 8.1) Overview
# =============================================================================

"""All basis portfolio returns"""
dash_returns_all = pd.DataFrame({'Benchmark': cw_spi_index[start_ptf:'2021-09-01']})
dash_returns_all['Dynamic Portfolio'] = run_mom_factors_noTE[1][start_ptf:'2021-09-01']
dash_returns_all['Defensive Portfolio'] = run_ridge_noTE[1][start_ptf:'2021-09-01']
dash_returns_all['Balanced Portfolio'] = run_parametrics_noTE_dict['VIX'][1][start_ptf:'2021-09-01']
dash_returns_all.index.name = 'Date'

dash_returns_all.apply(cum_prod).to_csv('dash-financial-report/data/returns_all_ptf.csv')

"""Performance Table"""
dash_perf_all_basis = pd.concat([perf_cwbenchmark.iloc[0:3, 0:1], run_ridge_noTE[2].iloc[0:3, 0:1], 
                                   run_parametrics_noTE_dict['VIX'][2].iloc[0:3, 0:1], run_mom_factors_noTE[2].iloc[0:3, 0:1]], axis=1)

dash_perf_all_label = [] 
dash_perf_all_label.insert(0, {'CW': 'Benchmark', 'Ridge (No TE)': 'Defensive', 'Parametrics (No TE)': 'Balanced', 'MF (No TE)': 'Dynamic'})

dash_perf_all_basis = pd.concat([pd.DataFrame(dash_perf_all_label, index={''}), dash_perf_all_basis])
dash_perf_all_basis.to_csv('dash-financial-report/data/perf_all_basis.csv')

# =============================================================================
# 8.2) Defensive Portfolio (Ridge Regression)
# =============================================================================

"""Performance Table"""
dash_perf_ridge_basis = pd.concat([perf_cwbenchmark, run_ridge_noTE[2], run_ridge_combineCW[2]], axis=1)

dash_perf_ridge_label = [] 
dash_perf_ridge_label.insert(0, {'CW': 'Benchmark', 'Ridge (No TE)': '100% Portfolio', '80% Ridge, 20% CW': '80% Portfolio, 20% Benchmark'})

dash_perf_ridge_basis = pd.concat([pd.DataFrame(dash_perf_ridge_label, index={''}), dash_perf_ridge_basis])
#dash_perf_ridge_basis.index.name = 'index'
dash_perf_ridge_basis.to_csv('dash-financial-report/data/perf_ridge_basis.csv')

"""Average Factor Weights"""
dash_avg_weights_ridge = run_ridge_noTE[0][start_ptf:].mean().round(3)*100
dash_avg_weights_ridge.to_csv('dash-financial-report/data/weights_ridge_basis.csv')

"""Cumulative Stock Returns"""
dash_return_ridge = return_ridge[start_ptf:].copy()
dash_return_ridge.drop('Ridge (6% TE Target)', axis=1, inplace=True)
dash_return_ridge.rename(columns={'CW Benchmark': 'Benchmark', 'Ridge (No TE)': '100% Portfolio', '80% Ridge, 20% CW': '80% Portfolio, 20% Benchmark'}, inplace=True)
dash_return_ridge.index.name = 'Date'
dash_return_ridge.apply(cum_prod).to_csv('dash-financial-report/data/returns_ridge.csv')

dash_avg_returns_ridge = pd.concat([avg_returns(dash_return_ridge['Benchmark']), 
                                    avg_returns(dash_return_ridge['100% Portfolio']),
                                    avg_returns(dash_return_ridge['80% Portfolio, 20% Benchmark'])], axis=0)

dash_avg_returns_ridge_label = [] 
dash_avg_returns_ridge_label.insert(0, {'1 Year': '1 Year', '3 Years': '3 Years', '5 Years': '5 Years', '10 Years': '10 Years', 'Since Inception': 'Since Inception'})

dash_avg_returns_ridge = pd.concat([pd.DataFrame(dash_avg_returns_ridge_label, index={''}), dash_avg_returns_ridge])
dash_avg_returns_ridge.to_csv('dash-financial-report/data/avg_returns_ridge.csv')

# =============================================================================
# 8.3) Balanced Portfolio (Parametric Portfolio with VIX)
# =============================================================================

"""Performance Table"""
dash_perf_parametrics_basis = pd.concat([perf_cwbenchmark, run_parametrics_noTE_dict['VIX'][2], run_parametrics_combineCW_dict['VIX'][2]], axis=1)

dash_perf_parametrics_label = [] 
dash_perf_parametrics_label.insert(0, {'CW': 'Benchmark', 'Parametrics (No TE)': '100% Portfolio', '80% Parametrics, 20%': '80% Portfolio, 20% Benchmark'})

dash_perf_parametrics_basis = pd.concat([pd.DataFrame(dash_perf_parametrics_label, index={''}), dash_perf_parametrics_basis])
dash_perf_parametrics_basis.to_csv('dash-financial-report/data/perf_parametrics_basis.csv')

"""Average Factor Weights"""
dash_avg_weights_parametrics = (run_parametrics_noTE_dict['VIX'][0][start_ptf:].mean()*100).round(2)
dash_avg_weights_parametrics.to_csv('dash-financial-report/data/weights_parametrics_basis.csv')

"""Cumulative Stock Returns"""
dash_return_parametrics = return_parametrics_vix[start_ptf:].copy()
dash_return_parametrics.rename(columns={'CW Benchmark': 'Benchmark', 'Parametrics (No TE)': '100% Portfolio', '80% Parametrics, 20% CW': '80% Portfolio, 20% Benchmark'}, inplace=True)
dash_return_parametrics.index.name = 'Date'
dash_return_parametrics.apply(cum_prod).to_csv('dash-financial-report/data/returns_parametrics.csv')

dash_avg_returns_parametrics  = pd.concat([avg_returns(dash_return_parametrics['Benchmark']), 
                                    avg_returns(dash_return_parametrics['100% Portfolio']),
                                    avg_returns(dash_return_parametrics['80% Portfolio, 20% Benchmark'])], axis=0)

dash_avg_returns_parametrics_label = [] 
dash_avg_returns_parametrics_label.insert(0, {'1 Year': '1 Year', '3 Years': '3 Years', '5 Years': '5 Years', '10 Years': '10 Years', 'Since Inception': 'Since Inception'})

dash_avg_returns_parametrics = pd.concat([pd.DataFrame(dash_avg_returns_parametrics_label, index={''}), dash_avg_returns_parametrics])
dash_avg_returns_parametrics.to_csv('dash-financial-report/data/avg_returns_parametrics.csv')

# =============================================================================
# 8.4) Dynamic Portfolio (Momentum of Factors)
# =============================================================================

"""Performance Table"""
dash_perf_mom_factors_basis = pd.concat([perf_cwbenchmark, run_mom_factors_noTE[2], run_mom_factors_combineCW[2]], axis=1)

dash_perf_mom_factors_label = [] 
dash_perf_mom_factors_label.insert(0, {'CW': 'Benchmark', 'MF (No TE)': '100% Portfolio', '80% MF, 20% CW': '80% Portfolio, 20% Benchmark'})

dash_perf_mom_factors_basis = pd.concat([pd.DataFrame(dash_perf_mom_factors_label, index={''}), dash_perf_mom_factors_basis])
dash_perf_mom_factors_basis.to_csv('dash-financial-report/data/perf_mom_factors_basis.csv')

"""Average Factor Weights"""
dash_avg_weights_mom_factors = (run_mom_factors_noTE[0][start_ptf:].mean()*100).round(2)
dash_avg_weights_mom_factors.to_csv('dash-financial-report/data/weights_mom_factors_basis.csv')

"""Cumulative Stock Returns"""
dash_return_mom_factors = return_mom_factors[start_ptf:].copy()
dash_return_mom_factors.rename(columns={'CW Benchmark': 'Benchmark', 'MF (No TE)': '100% Portfolio', '80% MF, 20% CW': '80% Portfolio, 20% Benchmark'}, inplace=True)
dash_return_mom_factors.index.name = 'Date'
dash_return_mom_factors.apply(cum_prod).to_csv('dash-financial-report/data/returns_mom_factors.csv')

dash_avg_returns_mom_factors = pd.concat([avg_returns(dash_return_mom_factors['Benchmark']), 
                                    avg_returns(dash_return_mom_factors['100% Portfolio']),
                                    avg_returns(dash_return_mom_factors['80% Portfolio, 20% Benchmark'])], axis=0)

dash_avg_returns_mom_factors_label = [] 
dash_avg_returns_mom_factors_label.insert(0, {'1 Year': '1 Year', '3 Years': '3 Years', '5 Years': '5 Years', '10 Years': '10 Years', 'Since Inception': 'Since Inception'})

dash_avg_returns_mom_factors = pd.concat([pd.DataFrame(dash_avg_returns_mom_factors_label, index={''}), dash_avg_returns_mom_factors])
dash_avg_returns_mom_factors.to_csv('dash-financial-report/data/avg_returns_mom_factors.csv')



