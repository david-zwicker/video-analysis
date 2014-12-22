'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import copy

import numpy as np
from scipy import interpolate
from scipy.spatial import distance
from shapely import geometry
import cv2

from data_structures.cache import cached_property
from video.analysis import curves
from video.analysis.active_contour import ActiveContour



class Tail(object):
    
    spline_smoothing = 20
    
    
    def __init__(self, contour):
        self.contour = contour
        
        
    @property
    def contour(self):
        return self._contour
    
    @contour.setter
    def contour(self, points):
        points = curves.make_curve_equidistant(points, spacing=20)
        if geometry.LinearRing(points).is_ccw:
            self._contour = np.array(points[::-1])
        else:
            self._contour = np.array(points)
        self._cache = {} #< clear cache
        
        
    def update_contour(self, points):
        """ updates the contour, keeping the identity of the ventral line and
        the measurement lines intact """
        tail_prev = copy.deepcopy(self)
        self.contour = points
        self.update_sides(tail_prev)
        self.update_ventral_side(tail_prev)
        

    @cached_property
    def outline(self):
        return geometry.LinearRing(self.contour)
    
    @cached_property
    def polygon(self):
        return geometry.Polygon(self.contour)


    @cached_property
    def bounds(self):
        return np.array(self.outline.bounds, np.int)
    
    
    @cached_property
    def mask(self):
        x_min, y_min, x_max, y_max = self.bounds
        shape = (y_max - y_min) + 3, (x_max - x_min) + 3
        offset =  (-x_min + 1, -y_min + 1)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [self.contour.astype(np.int)], 255,
                     offset=offset)
        return mask, offset
    
    
    @cached_property
    def endpoint_indices(self):
        """ locate the end points as contour points with maximal distance 
        The posterior point is returned first.
        """
        # get the points which are farthest away from each other
        dist = distance.squareform(distance.pdist(self.contour))
        indices = np.unravel_index(np.argmax(dist), dist.shape)
        
        # determine the surrounding mass of tissue to determine posterior end
        # TODO: We might have to determine the posterior end from previous
        # tails, too
        mass = []
        for k in indices:
            p = geometry.Point(self.contour[k]).buffer(500)
            mass.append(self.polygon.intersection(p).area)
            
        # determine posterior end point by measuring the surrounding
        if mass[1] > mass[0]:
            return indices
        else:
            return indices[::-1]
        
        
    @cached_property
    def endpoints(self):
        j, k = self.endpoint_indices
        return self.contour[j], self.contour[k]
    
    
    def determine_sides(self):
        """ determine the sides of the tail """
        k1, k2 = self.endpoint_indices
        if k2 > k1:
            ps = [self.contour[k1:k2 + 1],
                  np.r_[self.contour[k2:], self.contour[:k1 + 1]]]
        else:
            ps = [self.contour[k2:k1 + 1][::-1],
                  np.r_[self.contour[k1:], self.contour[:k2 + 1]][::-1, :]]
        return ps
    
    
    def update_sides(self, tail_prev):
        """ determines the side of the tails and align them with an earlier
        shape """
        # get the sides
        sides = self.determine_sides()
        first_side = geometry.LineString(tail_prev.sides[0])
        
        # compare the distance to the previous sides
        dists = [np.mean([first_side.distance(geometry.Point(p))
                          for p in side])
                 for side in self.sides]
        if dists[0] > dists[1]:
            sides = sides[1], sides[0]
        
        self._cache['sides'] = sides
    
    
    @property
    def sides(self):
        if 'sides' not in self._cache:
            self._cache['sides'] = self.determine_sides()
        return self._cache['sides']
            
        
    def determine_ventral_side(self):
        """ determines the ventral side from the curvature of the tail """
        # define a line connecting both end points
        k1, k2 = self.endpoint_indices
        line = geometry.LineString([self.contour[k1], self.contour[k2]])
        
        # cut the shape using this line and return the largest part
        parts = self.polygon.difference(line.buffer(0.1))
        areas = [part.area for part in parts]
        polygon = parts[np.argmax(areas)].buffer(0.1)
        
        # measure the fraction of points that lie in the polygon
        fs = []
        for c in self.sides:
            mp = geometry.MultiPoint(c)
            frac = len(mp.intersection(polygon))/len(mp)
            fs.append(frac)

        return self.sides[np.argmax(fs)]
    
    
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
        points = curves.translate_points(self.ventral_side, *offset)
        points = curves.make_curve_equidistant(points, spacing=50)
        # use the active contour algorithm
        points = ac.find_contour(points)
        # translate points back into global coordinate system
        points = curves.translate_points(points, -offset[0], -offset[1])
        
        # orient the centerline such that it starts at the posterior end
        dist1 = curves.point_distance(points[0], self.endpoints[0])
        dist2 = curves.point_distance(points[-1], self.endpoints[0])
        if dist1 > dist2:
            points = points[::-1]
        
        return points
    
    
    @cached_property
    def measurement_lines(self):
        """ determines the measurement line, we should use the central line
        and the ventral line to determine where to measure """
        centerline = self.centerline
        result = []
        for side in self.sides:
            # find the line between the centerline and the ventral line
            points = []
            for p in centerline:
                pp = curves.get_projection_point(side, p)
                points.append((0.5*(p[0] + pp[0]), 0.5*(p[1] + pp[1])))
                
            # do spline fitting
            smoothing = self.spline_smoothing*len(points)
            tck, _ = interpolate.splprep(np.transpose(points),
                                         k=2, s=smoothing)
            
            points = interpolate.splev(np.linspace(-0.5, .8, 100), tck)
            points = zip(*points) #< transpose list
    
            # restrict centerline to object
            cline = geometry.LineString(points).intersection(self.polygon)
            
            # pick longest line if there are many due to strange geometries
            if isinstance(cline, geometry.MultiLineString):
                cline = cline[np.argmax([l.length for l in cline])]
                
            result.append(np.array(cline.coords))
            
        return result    
