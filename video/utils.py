'''
Created on Aug 2, 2014

@author: zwicker
'''

from __future__ import division

from collections import defaultdict
import logging
import os

import numpy as np
from matplotlib.colors import ColorConverter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warn('Package tqdm could not be imported and progress bars are '
                 'thus not available.')


def logging_level():
    """ returns the level of the current logger """
    return logging.getLogger().getEffectiveLevel()


def ensure_directory_exists(folder):
    """ creates a folder if it not already exists """
    try:
        os.makedirs(folder)
    except OSError:
        # assume that the directory already exists
        pass


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


def prepare_data_for_yaml(data):
    """ recursively converts all numpy types to their closest python equivalents """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, dict):
        return {k: prepare_data_for_yaml(v) for k, v in data.iteritems()}
    elif isinstance(data, (list, tuple)):
        return [prepare_data_for_yaml(v) for v in data]
    else:
        return data
   
    
def homogenize_arraylist(data):
    """ stores a list of arrays of different length in a single array.
    This is achieved by appending np.nan as necessary.
    """
    len_max = max(len(d) for d in data)
    result = np.empty((len(data), len_max) + data[0].shape[1:], dtype=data[0].dtype)
    result.fill(np.nan)
    for k, d in enumerate(data):
        result[k, :len(d), ...] = d
    return result


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
    if tqdm is not None and logging_level() <= logging.INFO:
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



class NestedDict(defaultdict):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key:
    
    d = NestedDict({'a': {'b': 1}})
    
    d['a/b']
    >>>> 1
    
    d['c/d'] = 2
    
    d
    >>>> {'a': {'b': 1}, 'c': {'d': 2}}
    """
    
    sep = '/'
    
    def __init__(self, data=None):
        super(NestedDict, self).__init__(NestedDict)
        if data is not None:
            self.from_dict(data)
    
    
    def __getitem__(self, key):
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            return super(NestedDict, self).__getitem__(parent)[rest]
        else:
            return super(NestedDict, self).__getitem__(key)
        
        
    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise KeyError('Keys have to be strings in NestedDict.')
        
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            super(NestedDict, self).__getitem__(parent)[rest] = value
        else:
            super(NestedDict, self).__setitem__(key, value)
    
    
    def __delitem__(self, key):
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            del super(NestedDict, self).__getitem__(parent)[rest]
        else:
            super(NestedDict, self).__delattr__(key)
           
            
#     def __repr__(self):
#         return dict.__repr__(self)


    def copy(self):
        res = NestedDict()
        for key, value in self.iteritems():
            if isinstance(value, dict):
                value = value.copy()
            res[key] = value
        return res


    def from_dict(self, data):
        for key, value in data.iteritems():
            if isinstance(value, dict):
                self[key] = NestedDict(value)
            else:
                self[key] = value

            
    def to_dict(self):
        res = {}
        for key, value in self.iteritems():
            if isinstance(value, NestedDict):
                value = value.to_dict()
            res[key] = value
        return res
        
    