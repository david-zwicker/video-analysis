'''
Created on Aug 11, 2014

@author: zwicker

contains functions that are useful for curve analysis
'''

from __future__ import division

import itertools
import math
import numpy as np

import shapely.geometry as geometry

# make simplify_curve available under current scope 
from lib.simplify_polygon_rdp import rdp as simplify_curve # @UnusedImport


def point_distance(p1, p2):
    """ calculates the distance between point p1 and p2 """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def curve_length(points):
    """ returns the total arc length of a curve defined by a number of points """
    if len(points) < 2:
        return 0
    else:
        return np.sum(point_distance(p1, p2)
                      for p1, p2 in itertools.izip(points, points[1:]))


def make_curve_equidistant(points, spacing=None, count=None):
    """ returns a new parameterization of the same curve where points have been
    chosen equidistantly. The original curve may be slightly modified """
    points = np.asarray(points, np.double)
    
    if spacing is not None:
        # walk along and pick points with given spacing
        profile_length = curve_length(points)
        if profile_length == 0:
            return points
        
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
        if dist > 1e-8:
            result.append(points[-1])
            
    elif count is not None:
        # get arc length of support points
        s = np.cumsum([point_distance(p1, p2)
                       for p1, p2 in itertools.izip(points, points[1:])])
        s = np.insert(s, 0, 0) # prepend element for first point
        # divide arc length equidistantly
        sp = np.linspace(s[0], s[-1], count)
        # interpolate points
        result = np.transpose((np.interp(sp, s, points[:, 0]),
                               np.interp(sp, s, points[:, 1])))
    
    else:
        raise ValueError('Either spacing or counts of points must be provided.')
        
    return result


def get_projection_point(line, point):
    """ determines the point on the line closest to `point` """
    point = geometry.Point(point)
    point = line.interpolate(line.project(point))
    return (point.x, point.y)

