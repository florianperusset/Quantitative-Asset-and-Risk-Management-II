#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:10:29 2021

@author: sebastiengorgoni
"""

import os
import pandas as pd
import numpy as np
import re

df = pd.read_excel("Data/SPI_DATA_ALL.xlsx",sheet_name='Price')

df = df.drop(df.columns[2:4], axis=1)
df = df.drop(df.columns[0], axis=1)

for i in range(df.shape[0]):
    df.iloc[i,0] = df.iloc[i,0].split(' - ', 1)[0]
    df.iloc[i,0] = df.iloc[i,0].replace("'", "")
    df.iloc[i,0] = df.iloc[i,0].replace(".", " ")
    df.iloc[i,0] = df.iloc[i,0].replace("-", " ")
    df.iloc[i,0] = df.iloc[i,0].replace("+", " ")
    df.iloc[i,0] = re.sub(r"DEAD", "", df.iloc[i,0])
    #df.iloc[i,0] = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', df.iloc[i,0])

def import_spi(sheet):
    
    os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Quantitative Asset & Risk Management 2/Project")
    
    df = pd.read_excel("Data_SPI/SPI_DATA_ALL.xlsx",sheet_name=sheet)
    
    df = df.drop(df.columns[2:4], axis=1)
    df = df.drop(df.columns[0], axis=1)
    
    for i in range(df.shape[0]):
        df.iloc[i,0] = df.iloc[i,0].split(' - ', 1)[0]
        df.iloc[i,0] = df.iloc[i,0].replace("'", "")
        df.iloc[i,0] = df.iloc[i,0].replace(".", " ")
        df.iloc[i,0] = df.iloc[i,0].replace("-", " ")
        df.iloc[i,0] = df.iloc[i,0].replace("+", " ")
        df.iloc[i,0] = re.sub(r"DEAD", "", df.iloc[i,0])
        #df.iloc[i,0] = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', df.iloc[i,0])
    
    df.index = df['NAME']
    del df['NAME']
    
    df = df._convert(numeric=True)
    
    return df

def get_spi():
    
    """COLLECTING PRICES"""
    price_spi = import_spi('Price')
    price_spi = price_spi.T
    price_spi.index = pd.to_datetime(price_spi.index)
    
    """COLLECTING PE"""
    pe_spi = import_spi('PE')
    pe_spi = pe_spi.T
    
    """COLLECTING DIVIDEND YIELD"""
    dividend_spi = import_spi('Dividend Yield')
    dividend_spi = dividend_spi.T
    
    """COLLECTING MARKET CAP"""
    mktcap_spi = import_spi('Market Value')
    mktcap_spi = mktcap_spi.T
    
    """COLLECTING BETA"""
    beta_spi = import_spi('Beta')
    beta_spi = beta_spi.T
    
    """COLLECTING VOLATILITY"""
    vol_spi = import_spi('Volatility')
    vol_spi = vol_spi.T
    
    """COLLECTING ROE"""
    roe_spi = import_spi('ROE')
    roe_spi = roe_spi.T
    
    """COLLECTING ROA"""
    roa_spi = import_spi('ROA')
    roa_spi = roa_spi.T
    
    """COLLECTING GROSS MARGIN"""
    gm_spi = import_spi('Gross Margin')
    gm_spi = gm_spi.T
    
    return (price_spi, pe_spi, dividend_spi, mktcap_spi, beta_spi, vol_spi, roe_spi, roa_spi, gm_spi)
