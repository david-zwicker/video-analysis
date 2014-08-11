'''
Created on Aug 11, 2014

@author: zwicker

contains functions that are useful for curve analysis
'''

import itertools
import numpy as np
from ._rdp import rdp as simplify_curve # make it available under current scope


def curve_length(points):
    """ returns the total arc length of a curve definded by a number of points """
    return np.sum(np.linalg.norm(p2 - p1)
                  for p1, p2 in itertools.izip(points, points[1:]))


def make_cruve_equidistantly(points, spacing):
    """ returns a new parametrization of the same curve where points have been
    choosen equidistantly. The originial curve may be slightly modified """
    
    # walk along and pick points equidistantly
    profile_length = curve_length(points)
    dx = profile_length/np.round(profile_length/spacing)
    dist = 0
    result = [points[0]]
    for p1, p2 in itertools.izip(points[:-1], points[1:]):
        # determine the distance between the last two points 
        dp = np.linalg.norm(p2 - p1)
        # add points to the result list
        while dist + dp > dx:
            p1 = p1 + (dx - dist)/dp*(p2 - p1)
            result.append(p1.copy())
            dp = np.linalg.norm(p2 - p1)
            dist = 0
        
        # add the remaining distance 
        dist += dp
        
    # add the last point if necessary
    if dist > dx/2:
        result.append(points[-1])
        
    return result

