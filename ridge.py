#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:31:14 2021

@author: Florian
"""


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









































