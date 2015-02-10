'''
Created on Jan 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np


def trim_nan(data):
    """ removes nan values from the ends of the array """
    for s in xrange(len(data)):
        if not np.isnan(data[s]):
            break
    for e in xrange(len(data) - 1, s, -1):
        if not np.isnan(data[e]):
            break
    else:
        return []
    return data[s:e + 1]
    


def moving_average(data, window=1):
    """ calculates a moving average with a given window along the first axis
    of the given data.
    """
    height = len(data)
    result = np.zeros_like(data) + np.nan
    size = 2*window + 1
    assert height >= size
    for pos in xrange(height):
        # determine the window
        if pos < window:
            rows = slice(0, size)
        elif pos > height - window:
            rows = slice(height - size, height)
        else:
            rows = slice(pos - window, pos + window + 1)
            
        # find indices where all values are valid
        cols = np.all(np.isfinite(data[rows, :]), axis=0)
        result[pos, cols] = data[rows, cols].mean(axis=0)
    return result
            
