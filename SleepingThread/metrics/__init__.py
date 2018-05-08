"""
Paste here useful metric and norm functions
"""

import numpy as np

def mean(y):
    """
    Input - y - array-like
    Return: mean of array
    """
    ylen = len(y)
    res = 0.0
    for i in xrange(ylen):
        res += y[i]

    res = res/ylen

    return res

def SStot(y):
    """
    Input - y - array-like
    Return: dispersion value
    """
    ymean = mean(y)
    ylen = len(y)
    res = 0.0
    for i in xrange(ylen):
        res += (y[i]-ymean)**2

    return res

def SSres(x,y):
    """
    Input - x,y - array-like
    Return: squared error
    """
    ylen = len(y)
    xlen = len(x)
    if xlen != ylen :
        print "Error: xlen!=ylen"
        return None

    res = 0.0
    for i in xrange(xlen):
        res += (x[i]-y[i])**2

    return res

def r_squared(x,y):
    """
    x,y - array-like
    y - target
    x - predicted
    """
    res = 0.0
    res = 1.0 - SSres(x,y)/SStot(y)

    return res

def sigma(x,y):
    """
    x,y - array-like
    Root mean squared error
    """
    res = (SSres(x,y)/len(y))**0.5

    return res

def dispersion(y):
    return SStot(y)

def mean_dispersion(y):
    return SStot(y)/len(y)

def rmse(x,y):
    """
    x,y - array-like
    return: root mean squared error
    """
    return sigma(x,y)

def se(x,y):
    """
    x,y - array-like
    return: squared error
    """
    return SSres(x,y)

def mse(x,y):
    """
    x,y - array-like
    return: mean squared error
    """
    return SSres(x,y)/len(y)

