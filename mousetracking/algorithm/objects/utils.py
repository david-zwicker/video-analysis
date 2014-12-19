'''
Created on Sep 5, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy.interpolate import interp1d


def rtrim_nan(data):
    """ removes nan values from the end of the array """
    for k in xrange(len(data) - 1, -1, -1):
        if not np.any(np.isnan(data[k])):
            break
    else:
        return []
    return data[:k + 1]
    


class Interpolate_1D_Extrapolated(interp1d):
    """ extend the interpolate class from scipy to return boundary values for
    values beyond the support """
    
    def __call__(self, x):
        if x < self.x[0]:
            return self.y[0]
        elif x > self.x[-1]:
            return self.y[-1]
        else:
            return super(Interpolate_1D_Extrapolated, self).__call__(x)
            
            

