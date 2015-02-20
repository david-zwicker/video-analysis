'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains math functions
'''

from __future__ import division


import numpy as np
from scipy import stats, interpolate

from .cache import cached_property



def trim_nan(data, left=True, right=True):
    """ removes nan values from the either end of the array `data`.
    `left` and `right` determine whether these ends of the array are processed.
    The default is to process both ends.
    If data has more than one dimension, the reduction is done along the first
    dimension if any of entry along the other dimension is nan.
    """
    if left:
        # trim left side
        for s in xrange(len(data)):
            if not np.any(np.isnan(data[s])):
                break
    else:
        # don't trim the left side
        s = 0
        
    if right:
        # trim right side
        for e in xrange(len(data) - 1, s, -1):
            if not np.any(np.isnan(data[e])):
                # trim right side
                return data[s:e + 1]
        # array is all nan
        return []
    
    else:
        # don't trim the right side
        return data[s:]
    


def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, total = 0, 0
    for value in values:
        total += value
        n += 1
    return total/n if n > 0 else empty



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
            



class Interpolate_1D_Extrapolated(interpolate.interp1d):
    """ extend the interpolate class from scipy to return boundary values for
    values beyond the support.
    Here, we return the value at the boundary for all points beyond it.
    """
    
    def __call__(self, x):
        if x < self.x[0]:
            return self.y[0]
        elif x > self.x[-1]:
            return self.y[-1]
        else:
            return super(Interpolate_1D_Extrapolated, self).__call__(x)
            
            

def round_to_even(value):
    """ rounds the value to the nearest even integer """
    return 2*int(value/2 + 0.5)



def round_to_odd(value):
    """ rounds the value to the nearest odd integer """
    return 2*int(value/2) + 1



def get_number_range(dtype):
    """
    determines the minimal and maximal value a certain number type can hold
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise ValueError('Unsupported data type `%r`' % dtype)

    return info.min, info.max
        

    
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



def is_equidistant(data):
    """ checks whether the 1d array given by `data` is equidistant """
    if len(data) < 2:
        return True
    diff = np.diff(data)
    return np.allclose(diff, diff.mean())


    
def contiguous_true_regions(condition):
    """ Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index
    Taken from http://stackoverflow.com/a/4495197/932593
    """
    if len(condition) == 0:
        return []
    
    # Find the indices of changes in "condition"
    d = np.diff(condition.astype(int))
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx



def contiguous_int_regions_iter(data):
    """ Finds contiguous regions of the integer array "data". Regions that
    have falsey (0 or False) values are not returned.
    Returns three values (value, start, end), denoting the value and the pairs
    of indices and indicating the start index and end index of the region.
    """
    data = np.asarray(data, int)
    
    # Find the indices of changes in "data"
    d = np.diff(data)
    idx, = d.nonzero() 

    last_k = 0
    for k in idx:
        yield data[k], last_k, k + 1
        last_k = k + 1

    # yield last data point
    if len(data) > 0:
        yield data[-1], last_k, data.size



def safe_typecast(data, dtype):
    """
    truncates the data such that it fits within the supplied dtype.
    This function only supports integer data types so far.
    """
    info = np.iinfo(dtype)
    return np.clip(data, info.min, info.max).astype(dtype)
    
    

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
        
        