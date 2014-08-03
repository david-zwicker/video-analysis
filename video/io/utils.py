'''
Created on Aug 2, 2014

@author: zwicker
'''

from __future__ import division

import numpy as np


def get_color_range(dtype):
    """
    determines the color depth of the numpy array `data`.
    If the dtype is an integer, the range that it can hold is returned.
    If dtype is an inexact number (a float point), zero and one is returned
    """
    if(np.issubdtype(dtype, np.integer)):
        info = np.iinfo(dtype)
        return info.min, info.max
        
    elif(np.issubdtype(dtype, np.floating)):
        return 0, 1
        
    else:
        raise ValueError('Unsupported data type `%r`' % dtype)
