#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:11:53 2019

@author: wenninger
"""
import numpy as np
import matplotlib.pylab as plt
#def expandFunc(a):
#    a=np.hstack([a,a[-1]-a[0]+a[1:]])
#    a=np.hstack([a[0]-a[-1]+a[:-1],a])
#    for i in np.arange(3,19,2):
#        a=np.hstack([a,i*a[-1]-a[0]+a[2:]])
#        a=np.hstack([i*a[0]-a[-1]+a,a])
#    return a

def expandFuncFor(a,n):
    '''This functions expands an 1d array n times in negative and positive direction.
    The array is token and appended to both ends of the array, so that an the difference between the array data points is kept.
    If the array is even spaced, the function returns an even spaced array again. There is no point double or skipt where the array is expanded
    
    inputs
    ------
    a: 1d np array
        The array to be expanded
    n: int
        The number of expansion iterations.
        n = 1 returns the same array again.
        
    returns
    -------
    1d np array
        The array a expanded n times in positive and negative direction
    '''
    for i in np.arange(3,1+2*n,2):
        an=np.hstack([a,a[-1]-a[0]+a[1:]])
        an=np.hstack([a[0]-a[-1]+a[:-1],an])
        a = an
    return a

def expandFuncWhile(a,limit):
    '''This functions expands an 1d array in negative and positive direction so that the limit value is included in the expanded array.
    The array is token and appended to both ends of the array, so that an the difference between the array data points is kept.
    If the array is even spaced, the function returns an even spaced array again. There is no point double or skipt where the array is expanded
    
    inputs
    ------
    a: 1d np array
        The array to be expanded
    limit: float
        The maximum limit (in postivie and negative direction) which need to be included in the output array.
        
    returns
    -------
    1d np array
        The array a expanded n times in positive and negative direction
    '''
    i=3
    while -limit<a[0] or limit> a[-1] :
        an=np.hstack([a,a[-1]-a[0]+a[1:]])
        an=np.hstack([a[0]-a[-1]+a[:-1],an])
        a = an
        i=+2
    return a

#
#a=np.arange(-10,10)
#plt.plot(expandFuncFor(a,3))
#plt.plot(expandFuncWhile(a,2000))
#
#def plotdelta(a):
#    plt.plot(a[1:]-a[:-1])