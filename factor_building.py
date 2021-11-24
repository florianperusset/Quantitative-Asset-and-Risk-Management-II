#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:08:53 2021

@author: Florian
"""
import numpy as np

def factor_building(returns, quantile=0.5, long_above_quantile=True):
    quantile_factor = returns.quantile(q=quantile, axis=1)
    position_factor = returns.copy()
    
    if long_above_quantile:
        for i in position_factor.columns:
            position_factor.loc[returns[i] >= quantile_factor, i] = 1
            position_factor.loc[returns[i] < quantile, i] = 0
    else:
        for i in position_factor.columns:
            position_factor.loc[returns[i] >= quantile_factor, i] = 0
            position_factor.loc[returns[i] < quantile, i] = 1
    return position_factor.div(position_factor.sum(axis=1), axis=0).replace(np.nan,0)





















