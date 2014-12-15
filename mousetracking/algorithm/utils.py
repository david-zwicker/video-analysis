'''
Created on Sep 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Utility functions
'''

from __future__ import division

import contextlib
import logging
import os
import warnings

import numpy as np
from scipy import stats

from video.analysis.utils import cached_property
from collections import OrderedDict



def get_loglevel_from_name(name_or_int):
    """ converts a logging level name to the numeric representation """
    # see whether it is already an integer
    if isinstance(name_or_int, int):
        return name_or_int
    
    # convert it from the name
    level = logging.getLevelName(name_or_int.upper())
    if isinstance(level, int):
        return level
    else:
        raise ValueError('`%s` is not a valid logging level.' % name_or_int)



def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, total = 0, 0.
    for value in values:
        total += value
        n += 1
    return total/n if n > 0 else empty



@contextlib.contextmanager
def change_directory(path):
    """
    A context manager which changes the directory to the given
    path, and then changes it back to its previous value on exit.
    Stolen from http://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)



def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    Taken from http://code.activestate.com/recipes/391367-deprecated/
    """
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc



def unique_based_on_id(data):
    """ returns a list with only unique items, where the uniqueness
    is determined from the id of the items. This can be useful in the
    case where the items cannot be hashed and a set can thus not be used. """
    result, seen = [], set()
    for item in data:
        if id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result



def save_dict_to_csv(data, filename, first_columns=None, **kwargs):
    """ function that takes a dictionary of lists and saves it as a csv file """
    if first_columns is None:
        first_columns = []

    # sort the columns 
    sorted_index = {c: k for k, c in enumerate(sorted(data.keys()))}
    def column_key(col):
        """ helper function for sorting the columns in the given order """
        try:
            return first_columns.index(col)
        except ValueError:
            return len(first_columns) + sorted_index[col]
    sorted_keys = sorted(data.keys(), key=column_key)
        
    # create a data table and indicated potential units associated with the data
    # in the header
    table = OrderedDict()
    for key in sorted_keys:
        value = data[key]
        if hasattr(value, 'magnitude'):
            key += ' [%s]' % value.units
            value = value.magnitude
        elif len(value) > 0 and hasattr(value[0], 'magnitude'):
            assert len(set(str(item.units) for item in value)) == 1
            key += ' [%s]' % value[0].units
            value = [item.magnitude for item in value]
        table[key] = value

    # create a pandas data frame to save data to CSV
    import pandas as pd
    pd.DataFrame(table).to_csv(filename, **kwargs)



NORMAL_DISTRIBUTION_NORMALIZATION = 1/np.sqrt(2*np.pi)

class NormalDistribution(object):
    """ class representing normal distributions """ 
    
    def __init__(self, mean, var, count=None):
        """ normal distributions are described by their mean and variance.
        Additionally, count denotes how many observations were used to
        estimate the parameters. All values can also be numpy arrays to
        represent many distributions efficiently """ 
        self.mean = mean
        self.var = var
        self.count = count
        
        
    def copy(self):
        return self.__class__(self.mean, self.var, self.count)
        
        
    @cached_property
    def std(self):
        """ return standard deviation """
        return np.sqrt(self.var)
        
    
    def pdf(self, value, mask=None):
        """ return probability density function at value """
        if mask is None:
            mean = self.mean
            var = self.var
            std = self.std
        else:
            mean = self.mean[mask]
            var = self.var[mask]
            std = self.std[mask]
        
        return NORMAL_DISTRIBUTION_NORMALIZATION/std \
                *np.exp(-0.5*(value - mean)**2/var)
                
                
    def add_observation(self, value):
        """ add an observed value and adjust mean and variance of the
        distribution. This returns a new distribution and only works if
        count was set """
        if self.count is None:
            return self.copy()
        else:
            M2 = self.var*(self.count - 1)
            count = self.count + 1
            delta = value - self.mean
            mean = self.mean + delta/count
            M2 = M2 + delta*(value - mean)
            return NormalDistribution(mean, M2/(count - 1), count)
                
                
    def distance(self, other, kind='kullback-leibler'):
        """ return the distance between two normal distributions """
        if kind == 'kullback-leibler':
            dist = 0.5*(np.log(other.var/self.var) 
                        + (self.var + (self.mean - self.mean)**2)/other.var 
                        - 1)
            
        elif kind == 'bhattacharyya':
            var_ratio = self.var/other.var
            term1 = np.log(0.25*(var_ratio + 1/var_ratio + 2))
            term2 = (self.mean - other.mean)**2/(self.var + other.var)
            dist = 0.25*(term1 + term2)
            
        elif kind == 'hellinger':
            dist_b = self.distance(other, kind='bhattacharyya')
            dist = np.sqrt(1 - np.exp(-dist_b))
            
        else:
            raise ValueError('Unknown distance `%s`' % kind)
        
        return dist
    
    
    def welch_test(self, other):
        """ performs Welch's t-test of two normal distributions """
        # calculate the degrees of freedom
        s1, s2 = self.var/self.count, other.var/other.count
        nu1, nu2 = self.count - 1, other.count - 1
        dof = (s1 + s2)**2/(s1**2/nu1 + s2**2/nu2)

        # calculate the Welch t-value
        t = (self.mean - other.mean)/np.sqrt(s1 + s2)
        
        # calculate the probability using the Student's T distribution 
        prob = stats.t.sf(np.abs(t), dof) * 2
        return prob
    
    
    def overlap(self, other, common_variance=True):
        """ estimates the amount of overlap between two distributions """
        if common_variance:
            if self.count is None:
                if other.count is None: # neither is sampled
                    S = np.sqrt(0.5*(self.var + other.var))
                else: # other is sampled
                    S = self.std
            else: 
                if other.count is None:  # self is sampled
                    S = other.std
                else: # both are sampled
                    expr = (self.count - 1)*self.var + (other.count - 1)*other.var
                    S = np.sqrt(expr/(self.count + other.count - 2))

            delta = np.abs(self.mean - other.mean)/S
            return 2*stats.norm.cdf(-0.5*delta)

        else:
            # here, we would have to integrate numerically
            raise NotImplementedError
        
        