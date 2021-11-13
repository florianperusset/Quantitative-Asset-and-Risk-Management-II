import os
import pandas as pd
import numpy as np
import re


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
    price_spi = import_spi('Price').T
    
    """COLLECTING PE"""
    pe_spi = import_spi('PE').T
    
    """COLLECTING DIVIDEND YIELD"""
    dividend_spi = import_spi('Dividend Yield').T
    
    """COLLECTING MARKET CAP"""
    mktcap_spi = import_spi('Market Value').T
    
    """COLLECTING BETA"""
    beta_spi = import_spi('Beta').T

    """COLLECTING VOLATILITY"""
    vol_spi = import_spi('Volatility').T
    
    """COLLECTING ROE"""
    roe_spi = import_spi('ROE').T
    
    """COLLECTING ROA"""
    roa_spi = import_spi('ROA').T
    
    """COLLECTING GROSS MARGIN"""
    gm_spi = import_spi('Gross Margin').T
    
    """COLLECTING EPS"""
    eps_spi = import_spi('EPS').T
    
    """COLLECTING VOLUME TRADED"""
    trade_spi = import_spi('Volume Trade').T
    
    return (price_spi, pe_spi, dividend_spi, mktcap_spi, beta_spi, vol_spi, roe_spi, roa_spi, gm_spi, eps_spi, trade_spi)


