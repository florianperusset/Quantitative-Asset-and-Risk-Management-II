"""
-----------------------------------------------------------------------
QUANTITATIVE ASSET & RISK MANAGEMENT II

HEC LAUSANNE - AUTUMN 2021

Title: Style Rotation on Swiss Long-Only Equity Factors

Authors: Sebastien Gorgoni, Florian Perusset, Florian Vogt

File Name: optimization_criteria.py
-----------------------------------------------------------------------

This is an external file for main.py which create the portfolios using various optimization models.
We included the following optimization model:
    
    - Equal Risk Contribution (ERC)
    - Ridge Regression

"""

import pandas as pd
import numpy as np
import datetime as dt

def mcr(allocation, returns):
    """
    It computes the marginal risk contribution for a given allocation and a given set of returns.

    Parameters
    ----------
    allocation : Numpy
        The given asset allocation.
    returns : DataFrame
        matrix containing the time series of returns for each asset.

    Returns
    -------
    TYPE
       the marginal risk contribution.

    """
    portfolio = np.multiply(returns, allocation)
    portfolio_r = np.sum(portfolio,1) # sum across columns
    sigma_portfolio = np.std(portfolio_r)
    Sigma = np.cov(np.transpose(returns))
    return (Sigma @ np.transpose(allocation)) / sigma_portfolio

def criterion_erc(allocation, returns):
    """
    It computes the value of the ERC criterion for a given asset allocation and a given set of returns

    Parameters
    ----------
    allocation : Numpy
        The asset allocation.
    returns : DataFrame
        Matrix containing the time series of returns for each asset.

    Returns
    -------
    criterion : Float
        The value of the ERC criterion.

    """
    portfolio = np.multiply(returns, allocation)
    portfolio_r = np.sum(portfolio,1) # sum across columns
    sigma_portfolio = np.std(portfolio_r)
    indiv_erc = allocation * mcr(allocation,returns)
    criterion = (indiv_erc - sigma_portfolio/len(allocation)) ** 2
    criterion = np.sum(criterion) * 1000000000
    return criterion


def criterion_ridge(weights, expected_returns, varcov_matrix, lbda=20):
    """
    It computes the value of the criterion to be maximized in a ridge regression approach.

    Parameters
    ----------
    weights : Numpy
        Vector of weights, the one to make vary.
    expected_returns : DataFrame
        DESCRIPTION.
    varcov_matrix : Numpy
        DESCRIPTION.
    lbda : Int, optional
        Lambda of the model, The default is 20.

    Returns
    -------
    Numpy
        The value of the criterion for the given parameters.

    """
    r_bar = weights @ expected_returns
    sigma_square = weights @ varcov_matrix @ weights
    
    return -((r_bar/sigma_square) - lbda * (weights**2).sum())