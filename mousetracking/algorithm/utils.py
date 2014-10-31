'''
Created on Sep 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Utility functions
'''

from __future__ import division

import contextlib
import logging
import os

import numpy as np
from scipy import stats

from video.analysis.utils import cached_property



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
    

