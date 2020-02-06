#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:58:28 2019

@author: wenninger
"""
import numpy as np

def gaussian(x,xpeak,ypeak,ywidth):
    '''This function computes a gaussian (not necessarily peaking at 0).
    
    inputs
    ------
    x: 1D array
        The points to be evaluated.
    xpeak: float
        The x value where the peak is located. 
    ypeak: float
        The maximum of the gaussian.
    ywidth: float
        The width of the gaussian. This value corresponds to one sigma.
    
    returns
    -------
    2d np array
        The x values and the corresponding y values, which represent the gaussian.
    '''
    y = np.multiply(ypeak,np.exp(-np.divide(np.multiply(np.subtract(x,xpeak),np.subtract(x,xpeak)),ywidth*ywidth)))
    return np.vstack([x,y])