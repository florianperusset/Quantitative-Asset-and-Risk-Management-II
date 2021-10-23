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

def import_spi(sheet):
    
    os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 5.1/Quantitative Asset & Risk Management 2/Project")
    
    df = pd.read_excel("Data/data_qarm2.xlsx",sheet_name=sheet)
    
    for i in range(df.shape[0]):
        df.iloc[i,0] = df.iloc[i,0].split(' - ', 1)[0]
        df.iloc[i,0] = df.iloc[i,0].replace("'", "")
        df.iloc[i,0] = df.iloc[i,0].replace(".", " ")
        df.iloc[i,0] = re.sub(r"DEAD", "", df.iloc[i,0])
        #df.iloc[i,0] = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', df.iloc[i,0])
    
    df = df.rename(columns={'Unnamed: 0':'NAME'})
    df.index = df['NAME']
    del df['NAME']
    
    df = df._convert(numeric=True)
    #df = df.fillna(0)
    
    return df

def get_spi():
    
    """COLLECTING PRICES"""
    price_old = import_spi('price old')
    price_new = import_spi('price new')
    
    price_spi = price_old.join(price_new)
    price_spi = price_spi.T
    #del price_spi['#ERROR']

    """COLLECTING PE"""
    pe_old = import_spi('pe old')
    pe_new = import_spi('pe new')
    
    pe_spi = pe_old.join(pe_new)
    pe_spi = pe_spi.T
    #del pe_spi['#ERROR']
    
    """COLLECTING DIVIDEND YIELD"""
    dividend_old = import_spi('dividend yield old')
    dividend_new = import_spi('dividend yield new')
    
    dividend_spi = dividend_old.join(pe_new)
    dividend_spi = dividend_spi.T
    #del dividend_spi['#ERROR']
    
    """COLLECTING DIVIDEND YIELD"""
    dividend_old = import_spi('dividend yield old')
    dividend_new = import_spi('dividend yield new')
    
    dividend_spi = dividend_old.join(dividend_new)
    dividend_spi = dividend_spi.T
    #del dividend_spi['#ERROR']
    
    """COLLECTING MARKET CAP"""
    mktcap_old = import_spi('market cap old')
    mktcap_new = import_spi('market cap new')
    
    mktcap_spi = mktcap_old.join(mktcap_new)
    mktcap_spi = mktcap_spi.T
    #del mktcap_spi['#ERROR']
    
    """COLLECTING BETA"""
    beta_old = import_spi('beta old')
    beta_new = import_spi('beta new')
    
    beta_spi = beta_old.join(beta_new)
    beta_spi = beta_spi.T
    #del beta_spi['#ERROR']
    
    """COLLECTING VOLATILITY"""
    vol_old = import_spi('volat old')
    vol_new = import_spi('volat new')
    
    vol_spi = vol_old.join(vol_new)
    vol_spi = vol_spi.T
    #del vol_spi['#ERROR']
    
    """COLLECTING ROE"""
    roe_old = import_spi('roe old')
    roe_new = import_spi('roe new')
    
    roe_spi = roe_old.join(roe_new)
    roe_spi = roe_spi.T
    #del roe_spi['#ERROR']
    
    """COLLECTING ROA"""
    roa_old = import_spi('roa old')
    roa_new = import_spi('roa new')
    
    roa_spi = roa_old.join(roa_new)
    roa_spi = roa_spi.T
    #del roa_spi['#ERROR']
    
    """COLLECTING GROSS MARGIN"""
    gm_old = import_spi('gross margin old')
    gm_new = import_spi('gross margin new')
    
    gm_spi = gm_old.join(gm_new)
    gm_spi = gm_spi.T
    #del gm_spi['#ERROR']
    
    return (price_spi, pe_spi, dividend_spi, mktcap_spi, beta_spi, vol_spi, roe_spi, roa_spi, gm_spi)
