#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:28:28 2021

@author: sebastiengorgoni
"""

import os
import pandas as pd
import numpy as np
import re
import yfinance as yf
import datetime as dt

from import_data import get_spi

# =============================================================================
# Import Data
# =============================================================================

"""Swiss Performance Index"""
#Price
price_spi = get_spi()[0] 
#PE Ratio
pe_spi = get_spi()[1] 
#Dividend Yield
dividend_spi = get_spi()[2]
#Market Cap
mktcap_spi = get_spi()[3]
#Beta
beta_spi = get_spi()[4]
#Volatility
vol_spi = get_spi()[5]
#ROE
roe_spi = get_spi()[6]
#ROA
roa_spi = get_spi()[7]
#Gross Margin
gm_spi = get_spi()[8]

"""Macro Data"""
start_fin = dt.datetime(2000, 1, 1).date()
end_fin = dt.datetime(2020, 12, 1).date()
#VIX
vix = yf.download(tickers = "^VIX", start=start_fin, end=end_fin)

#10y Tbill Rate
tbill = yf.download(tickers = "^TNX", start=start_fin, end=end_fin)



