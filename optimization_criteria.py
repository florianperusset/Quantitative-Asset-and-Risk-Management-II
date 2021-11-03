#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:39:27 2021

@author: Florian
"""

import os
import pandas as pd
import numpy as np
import datetime as dt



def mcr(allocation, returns):
    """
    Computes the marginal risk contribution for a given allocation and a given set of returns.
    
    Parameters:
        allocation: the given asset allocation
        r: matrix containing the time series of returns for each asset
    
    Returns:
        the marginal risk contribution
    """
    portfolio = np.multiply(returns, allocation)
    portfolio_r = np.sum(portfolio,1) # sum across columns
    sigma_portfolio = np.std(portfolio_r)
    Sigma = np.cov(np.transpose(returns))
    return (Sigma @ np.transpose(allocation)) / sigma_portfolio

###ERC Allocation###
def criterion_erc(allocation, returns):
    """
    Computes the value of the ERC criterion for a given asset allocation and a given set of returns
    
    Parameters:
        allocation: the asset allocation
        r: matrix containing the time series of returns for each asset
        
    Returns:
        the value of the ERC criterion
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
    Computes the value of the criterion to be maximized in a ridge regression approach
    
    Parameters:
        weights: vector of weights, the one to make vary
        expected_returns: vector of expected returns for each factor
        varcov_matrix: the variance-covariance matrix for each factor
        
    Returns:
        the value of the criterion for the given parameters
    """
    r_bar = weights @ expected_returns
    sigma_square = weights @ varcov_matrix @ weights
    
    return -((r_bar/sigma_square) - lbda * (weights**2).sum())




































