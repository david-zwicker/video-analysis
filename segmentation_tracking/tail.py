'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import copy

import numpy as np
from scipy.spatial import distance
from shapely import geometry
import cv2

from data_structures.cache import cached_property
from video.analysis import curves, regions
from video.analysis.active_contour import ActiveContour

from video import debug  # @UnusedImport



class Tail(object):
    """ class representing a single mouse tail in a single frame.
    Every tail is defined by its contour.
    """
    
    line_names = ['ventr', 'dors']
    
    
    def __init__(self, contour):
        self.contour = contour
        
        
    def __repr__(self):
        return '%s(pos=(%d, %d), %d contour points)' % \
                (self.__class__.__name__, self.center[0], self.center[1],
                 len(self.contour))
    
        
    @property
    def contour(self):
        return self._contour.copy()
    
    @contour.setter
    def contour(self, points):
        points = regions.regularize_contour_points(points)
        points = curves.make_curve_equidistant(points, spacing=20)

        # sometimes the polygon has to be regularized again
        if not geometry.Polygon(points).is_valid:
            points = regions.regularize_contour_points(points)
                    
        if geometry.LinearRing(points).is_ccw:
            self._contour = np.array(points[::-1])
        else:
            self._contour = np.array(points)
        
        self._cache = {} #< clear cache
        
        
    def update_contour(self, points):
        """ updates the contour, keeping the identity of the end points,
        ventral line, and the measurement lines intact """
        tail_prev = copy.deepcopy(self)
        self.contour = points
        # update important features of the tail in reference to the previous
        self.get_endpoint_indices(tail_prev)
        self.update_sides(tail_prev)
        self.update_ventral_side(tail_prev)
        

    @cached_property
    def outline(self):
        return geometry.LinearRing(self._contour)
    
    @cached_property
    def polygon(self):
        return geometry.Polygon(self._contour)

    @cached_property
    def center(self):
        return self.polygon.centroid.coords[0]

    @cached_property
    def bounds(self):
        return np.array(self.outline.bounds, np.int)
    
    @cached_property
    def area(self):
        return self.polygon.area
    
    @cached_property
    def mask(self):
        """ return a binary mask large enough to hold the tails image and an
        offset the determines the position of the mask in global coordinates """
        x_min, y_min, x_max, y_max = self.bounds
        shape = (y_max - y_min) + 3, (x_max - x_min) + 3
        offset =  (x_min - 1, y_min - 1)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [self._contour.astype(np.int)], 1,
                     offset=(-offset[0], -offset[1]))
        return mask, offset
    
    
    def get_endpoint_indices(self, tail_prev=None):
        """ locate the end points as contour points with maximal distance 
        The posterior point is returned first.
        """
        # get the points which are farthest away from each other
        dist = distance.squareform(distance.pdist(self._contour))
        indices = np.unravel_index(np.argmax(dist), dist.shape)
        
        if tail_prev is None:
            # determine the mass of tissue to determine posterior end
            mass = []
            for k in indices:
                p = geometry.Point(self._contour[k]).buffer(500)
                mass.append(self.polygon.intersection(p).area)
                
            # determine posterior end point by measuring the surrounding
            if mass[1] < mass[0]:
                indices = indices[::-1]
                
        else:
            # sort end points according to previous frame
            prev_p, prev_a = tail_prev.endpoints
            this_1 = self._contour[indices[0]]
            this_2 = self._contour[indices[1]]
            dist1 = curves.point_distance(this_1, prev_p) + \
                    curves.point_distance(this_2, prev_a)
            dist2 = curves.point_distance(this_1, prev_a) + \
                    curves.point_distance(this_2, prev_p)
            if dist2 < dist1:
                indices = indices[::-1]

        # save indices in cache
        self._cache['endpoint_indices'] = indices
        return indices        
    
    
    @cached_property
    def endpoint_indices(self):
        """ locate the end points as contour points with maximal distance 
        The posterior point is returned first.
        """
        return self.get_endpoint_indices()
        
        
    @cached_property
    def endpoints(self):
        """ returns the posterior and the anterior end point """
        j, k = self.endpoint_indices
        return self._contour[j], self._contour[k]
    
    
    def _sort_sides(self, sides, first_line):
        """ sorts sides such that the first line in `sides` is closest to
        `first_line` """

    
    def determine_sides(self, line_ref='ventral'):
        """ determine the sides of the tail """
        # get the two sides
        k1, k2 = self.endpoint_indices
        if k2 > k1:
            sides = [self._contour[k1:k2 + 1],
                     np.r_[self._contour[k2:], self._contour[:k1 + 1]]]
        else:
            sides = [self._contour[k2:k1 + 1][::-1],
                     np.r_[self._contour[k1:], self._contour[:k2 + 1]][::-1, :]]
            
        # determine how to sort them
        if 'ventral' == line_ref:
            line_ref = self.ventral_side
            
        if line_ref is not None:
            # sort lines such that reference line comes first
            first_line = geometry.LineString(line_ref)
            dists = [np.mean([first_line.distance(geometry.Point(p))
                              for p in side])
                     for side in sides]
            if dists[0] > dists[1]:
                sides = sides[1], sides[0]
        return sides
    
        
    def update_sides(self, tail_prev):
        """ determines the side of the tails and align them with an earlier
        shape """
        # get the sides and make sure they agree with the previous order
        self._cache['sides'] = self.determine_sides(line_ref=tail_prev.sides[0])
    
    
    @property
    def sides(self):
        if 'sides' not in self._cache:
            self._cache['sides'] = self.determine_sides()
        return self._cache['sides']
            
        
    def determine_ventral_side(self):
        """ determines the ventral side from the curvature of the tail """
        # define a line connecting both end points
        k1, k2 = self.endpoint_indices
        line = geometry.LineString([self._contour[k1], self._contour[k2]])
        
        # cut the shape using this line and return the largest part
        parts = self.polygon.difference(line.buffer(0.1))
        if isinstance(parts, geometry.MultiPolygon):
            areas = [part.area for part in parts]
            polygon = parts[np.argmax(areas)].buffer(0.1)
        else:
            polygon = parts
            
        # measure the fraction of points that lie in the polygon
        fs = []
        sides = self.determine_sides(line_ref=None)
        for c in sides:
            mp = geometry.MultiPoint(c)
            frac = len(mp.intersection(polygon))/len(mp)
            fs.append(frac)

        return sides[np.argmax(fs)]
    
    
    def update_ventral_side(self, tail_prev):
        """ determines the ventral side by comparing to an earlier shape """
        # get average distance of these two lines to the previous dorsal line
        line_prev = geometry.LineString(tail_prev.ventral_side)
        dists = [np.mean([line_prev.distance(geometry.Point(p))
                          for p in c])
                 for c in self.sides]
        
        self._cache['ventral_side'] = self.sides[np.argmin(dists)]

        
    @property
    def ventral_side(self):
        """ returns the points along the ventral side """
        if 'ventral_side' not in self._cache:
            self._cache['ventral_side'] = self.determine_ventral_side()
        return self._cache['ventral_side']
    
    
    @cached_property
    def centerline(self):
        """ determine the center line of the tail """
        mask, offset = self.mask
        dist_map = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 5)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=5,
                           closed_loop=False,
                           alpha=0, #< line length is constraint by beta
                           beta=1e6,
                           gamma=1e-2)
        ac.max_iterations = 500
        ac.set_potential(dist_map)
        
        # find centerline starting from the ventral_side
        points = curves.translate_points(self.ventral_side,
                                         -offset[0], -offset[1])
        points = curves.make_curve_equidistant(points, spacing=50)
        # use the active contour algorithm
        points = ac.find_contour(points)
        # translate points back into global coordinate system
        points = curves.translate_points(points, offset[0], offset[1])
        
        # orient the centerline such that it starts at the posterior end
        dist1 = curves.point_distance(points[0], self.endpoints[0])
        dist2 = curves.point_distance(points[-1], self.endpoints[0])
        if dist1 > dist2:
            points = points[::-1]
        
        return points
    
    
     
