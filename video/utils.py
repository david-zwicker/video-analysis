'''
Created on Aug 2, 2014

@author: zwicker
'''

from __future__ import division

import logging
import numpy as np
from matplotlib.colors import ColorConverter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warn('Package tqdm could not be imported and progress bars are '
                 'thus not available')




def verbose(mode=logging.DEBUG):
    """ determine whether debugging is requested """
    return logging.getLogger().getEffectiveLevel() <= mode


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


def safe_typecast(data, dtype):
    """
    truncates the data such that it fits within the supplied dtype.
    This function only supports integer datatypes so far.
    """
    info = np.iinfo(dtype)
    return np.clip(data, info.min, info.max).astype(dtype)
    

def display_progress(iterator, total=None):
    """
    displays a progress bar when iterating
    """
    if tqdm is not None and verbose(logging.INFO):
        return tqdm(iterator, total=total, leave=True)
    else:
        return iterator
    
    
def get_color(color):
    """
    function that returns a RGB color with channels ranging from 0..255.
    The matplotlib color notation is used.
    """
    
    if get_color.converter is None:
        get_color.converter = ColorConverter().to_rgb
        
    return (255*np.array(get_color.converter(color))).astype(int)

get_color.converter = None


    