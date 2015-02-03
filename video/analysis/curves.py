'''
Created on Aug 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

contains functions that are useful for curve analysis
'''

from __future__ import division

import itertools
import math
import numpy as np
from scipy import interpolate

from shapely import geometry

# make simplify_curve available under current scope 
from external.simplify_polygon_rdp import rdp as simplify_curve # @UnusedImport



def point_distance(p1, p2):
    """ calculates the distance between point p1 and p2 """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])



def angle_between_points(p1, p2, p3):
    """ calculates the angle at p2 of the line given by the three points """ 
    ps = np.array([p1, p2, p3])
    d12 = ps[1] - ps[0]
    d23 = ps[2] - ps[1]
    # use dot product to get the angle
    denom = np.linalg.norm(d12)*np.linalg.norm(d23)
    if denom == 0:
        angle = np.nan
    else:
        arg = np.dot(d12, d23)/denom
        try:
            angle = math.acos(arg)
        except ValueError:
            # raised, when argument is not in [-1, 1]
            # => we just extrapolate the value at the boundary
            angle = 0 if arg > 0 else math.pi
    return angle



def translate_points(points, xoff, yoff):
    """ translate points by a certain offset """
    return [(p[0] + xoff, p[1] + yoff) for p in points]



def curve_length(points):
    """ returns the total arc length of a curve defined by a number of points """
    if len(points) < 2:
        return 0
    else:
        return sum(math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                   for p1, p2 in itertools.izip(points, points[1:]))
    # Note that a vectorized numpy version using np.diff and np.hypot or using 
    # np.linalg.norm is considerably slower for the typical short lists that
    # are encountered here.



def make_curve_equidistant(points, spacing=None, count=None):
    """ returns a new parameterization of the same curve where points have been
    chosen equidistantly. The original curve may be slightly modified """
    points = np.asarray(points, np.double)
    
    if spacing is not None:
        # walk along and pick points with given spacing
        profile_length = curve_length(points)
        if profile_length < spacing:
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
            
    else:
        if count is None:
            count = len(points)
            
        # get arc length of support points
        s = np.cumsum([point_distance(p1, p2)
                       for p1, p2 in itertools.izip(points, points[1:])])
        s = np.insert(s, 0, 0) # prepend element for first point
        # divide arc length equidistantly
        sp = np.linspace(s[0], s[-1], count)
        # interpolate points
        result = np.transpose((np.interp(sp, s, points[:, 0]),
                               np.interp(sp, s, points[:, 1])))
        
    return result



def get_projection_point(line, point):
    """ determines the point on the line closest to `point` """
    point = geometry.Point(point)
    line = geometry.LineString(line)
    point = line.interpolate(line.project(point))
    return (point.x, point.y)



def average_normalized_functions(profiles):
    """ averages functions defined on the interval [0, 1] """
    len_max = max(len(ps) for ps in profiles)
    xs = np.linspace(0, 1, len_max)
    ys = np.mean([np.interp(xs, ps[:, 0], ps[:, 1])
                  for ps in profiles], axis=0)
    return np.c_[xs, ys]

    

def smooth_curve(points, smoothing=10, degree=3, derivative=0):
    """ smooth a curve by interpolating the points
    `smoothing` determines the smoothness of the curve.  This value can be used
        to control the trade-off between closeness and smoothness of fit.
        Larger values means more smoothing while smaller values indicate less
        smoothing. The resulting, smoothed yi fulfill
            sum((y - yi)**2, axis=0) <= smoothing*len(points)
    `degree` determines the degree of the splines used
    `derivative` determines the order of the derivative 
    """
    u = np.linspace(0, 1, len(points))
    try:
        # do spline fitting to smooth the line
        tck, _ = interpolate.splprep(np.transpose(points), u=u, k=degree,
                                     s=smoothing*len(points))
    except ValueError:
        pass
    else:
        # interpolate the line
        points = interpolate.splev(u, tck, der=derivative)
        points = zip(*points) #< transpose list
    
    return np.asarray(points)
    
    
