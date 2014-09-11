'''
Created on Sep 10, 2014

@author: zwicker

Utility functions
'''

from __future__ import division

def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, total = 0, 0.
    for value in values:
        total += value
        n += 1
    return total/n if n > 0 else empty
    
