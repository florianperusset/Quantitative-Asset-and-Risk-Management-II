"""
-----------------------------------------------------------------------
QUANTITATIVE ASSET & RISK MANAGEMENT II

HEC LAUSANNE - AUTUMN 2021

Title: Style Rotation on Swiss Long-Only Equity Factors

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

File Name: import_data.py
-----------------------------------------------------------------------

This is an external file for main.py which import the data from the excel file SPI_DATA_ALL.xlsx and
process it to create a usable dataframe. 

"""

import os
import pandas as pd
import numpy as np
import re


def import_spi(sheet):
    """
    This function import the data from the excel file SPI_DATA_ALL.xlsx
    and process it to create a usable dataframe.

    Parameters
    ----------
    sheet : String
        The name of the sheet wanted in the excel file.

    Returns
    -------
    df : DataFrame
        The dataframe of the wanted excel sheet, ready to use.

    """
    
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
    """
    This function will call the function import_spi() to process all
    sheets included in the excel file. 

    Returns
    -------
    price_spi : TYPE
        DESCRIPTION.
    pe_spi : TYPE
        DESCRIPTION.
    dividend_spi : TYPE
        DESCRIPTION.
    mktcap_spi : TYPE
        DESCRIPTION.
    beta_spi : TYPE
        DESCRIPTION.
    vol_spi : TYPE
        DESCRIPTION.
    roe_spi : TYPE
        DESCRIPTION.
    roa_spi : TYPE
        DESCRIPTION.
    gm_spi : TYPE
        DESCRIPTION.
    eps_spi : TYPE
        DESCRIPTION.
    trade_spi : TYPE
        DESCRIPTION.
    industry_spi : TYPE
        DESCRIPTION.
    mb_spi : TYPE
        DESCRIPTION.
    investment_spi : TYPE
        DESCRIPTION.
    profit_spi : TYPE
        DESCRIPTION.

    """
    
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
    
    """COLLECTING THE INDUSTRY TYPE"""
    industry_spi = import_spi('Industry')
    
    """COLLECTING MARKET-TO-BOOK RATIO"""
    mb_spi = import_spi('MB').T

    """COLLECTING OPERATING PROFIT MARGIN"""
    investment_spi = import_spi('Other Investment').T

    """COLLECTING OPERATING PROFIT MARGIN"""
    profit_spi = import_spi('Operating Profit Margin').T
    
    return (price_spi, pe_spi, dividend_spi, mktcap_spi, beta_spi, vol_spi, roe_spi, roa_spi, gm_spi, eps_spi, trade_spi, industry_spi, mb_spi, investment_spi, profit_spi)


