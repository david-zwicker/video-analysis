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

from utils.cache import cached_property
from video.analysis import curves, regions, shapes
from video.analysis.active_contour import ActiveContour

from video import debug  # @UnusedImport



class Tail(shapes.Polygon):
    """ class representing a single mouse tail in a single frame.
    Every tail is defined by its contour.
    """
    
    endpoint_mass_radius = 500 #< radius for identifying the end points
    contour_spacing = 20       #< spacing of the contour points 
    centerline_spacing = 50    #< spacing of centerline points
    
    # parameters for the active snake algorithm for finding the centerline
    centerline_blur_radius = 5
    centerline_bending_stiffness = 1e6
    centerline_adaptation_rate = 1e-2
    centerline_max_iterations = 500
    
    line_names = ['Side A', 'Side B']
    
    
    def __init__(self, contour, extra_data=None):
        """ initialize a tail with its contour and optional extra data """
        super(Tail, self).__init__(contour)
        self.persistence_data = {}
        if extra_data is None:
            self.data = {}
        else:
            self.data = extra_data
    
        
    def __repr__(self):
        return '%s(pos=(%d, %d), %d contour points)' % \
                (self.__class__.__name__, self.centroid[0], self.centroid[1],
                 len(self.contour))
    
    
    def copy(self):
        obj = self.__class__(self.contour.copy(), self.extra_data.copy())
        obj.persistence_data = self.persistence_data.copy()
        return obj
    
    
    @shapes.Polygon.contour.setter
    def contour(self, points):
        """ sets the contour of the tail performing some sanity tests """
        # do a first regularization
        points = regions.regularize_contour_points(points)
        spacing = self.contour_spacing
        # make the contour line equidistant
        points = curves.make_curve_equidistant(points, spacing=spacing)
        # regularize again, just to be sure
        points = regions.regularize_contour_points(points)
        # call parent setter
        shapes.Polygon.contour.fset(self, points)


    @classmethod
    def create_similar(cls, contour, tail):
        """ creates a tail described by `contour` that has a similar orientation
        to the given `tail` """
        obj = cls(contour)
        obj.determine_endpoint_indices(tail) 
        obj.match_side_order(tail)
        return obj
        
        
    def update_contour(self, points):
        """ updates the contour, keeping the identity of the end points,
        ventral line, and the measurement lines intact """
        tail_prev = copy.deepcopy(self)
        self.contour = points
        # update important features of the tail in reference to the previous
        self.determine_endpoint_indices(tail_prev)
        self.match_side_order(tail_prev)
        

    @cached_property
    def mask(self):
        """ return a binary mask large enough to hold the tails image and an
        offset the determines the position of the mask in global coordinates """
        return self.get_mask(margin=5, ret_offset=True)
    
    
    def determine_endpoint_indices(self, tail_prev=None):
        """ locate the end points as contour points with maximal distance 
        The posterior point is returned first.
        If `tail_prev` is given, the end points are oriented in the same way as
        in the previous tail, otherwise they are assigned automatically.
        """
        # get the points which are farthest away from each other
        dist = distance.squareform(distance.pdist(self.contour))
        indices = np.unravel_index(np.argmax(dist), dist.shape)
        
        if tail_prev is None:
            # determine the mass of tissue to determine posterior end
            mass = []
            for k in indices:
                radius = self.endpoint_mass_radius
                endzone = geometry.Point(self.contour[k]).buffer(radius)
                poly = self.polygon.buffer(0) #< clean the polygon
                mass.append(poly.intersection(endzone).area)
                
            # determine posterior end point by measuring the surrounding
            if mass[1] < mass[0]:
                indices = indices[::-1]
               
        else:
            # sort end points according to previous frame
            prev_p, prev_a = tail_prev.endpoints
            this_1 = self.contour[indices[0]]
            this_2 = self.contour[indices[1]]
            dist1 = curves.point_distance(this_1, prev_p) + \
                    curves.point_distance(this_2, prev_a)
            dist2 = curves.point_distance(this_1, prev_a) + \
                    curves.point_distance(this_2, prev_p)
            if dist2 < dist1:
                indices = indices[::-1]

        # save indices in cache
        self.persistence_data['endpoint_indices'] = indices
        return indices        
    
    
    @property
    def endpoint_indices(self):
        """ locate the end points as contour points with maximal distance 
        The posterior point is returned first.
        """
        indices = self.persistence_data.get('endpoint_indices', None)
        if indices is None:
            indices = self.determine_endpoint_indices()
        return indices
        
        
    @cached_property
    def endpoints(self):
        """ returns the posterior and the anterior end point """
        j, k = self.endpoint_indices
        return self.contour[j], self.contour[k]
    
    
    def _sort_sides(self, sides, first_line):
        """ sorts sides such that the first line in `sides` is closest to
        `first_line` """


    def determine_side_order(self, sides):
        """ determine which of the sides is the ventral side based on geometric
        properties and return indices such that the ventral side comes first """ 
        # define a line connecting both end points
        k1, k2 = self.endpoint_indices
        line = geometry.LineString([self.contour[k1], self.contour[k2]])
        
        # cut the shape using this line and return the largest part
        parts = self.polygon.difference(line.buffer(0.1))
        if isinstance(parts, geometry.MultiPolygon):
            areas = [part.area for part in parts]
            polygon = parts[np.argmax(areas)].buffer(0.1)
        else:
            polygon = parts
            
        # measure the fraction of points that lie in the polygon
        fs = []
        for c in sides:
            mp = geometry.MultiPoint(c)
            intersection = mp.intersection(polygon)
            if isinstance(intersection, geometry.Point):
                frac = 1/len(mp)
            else:
                frac = len(intersection)/len(mp)
            fs.append(frac)
            
        # return the order in which the sides should be put
        if np.argmax(fs) == 0:
            return (0, 1)
        else:
            return (1, 0) 
    
    
    def get_sides(self):
        """ determine the sides of the tail and return them in arbitrary order
        """
        # get the two sides
        k1, k2 = self.endpoint_indices
        if k2 > k1:
            sides = [self.contour[k1:k2 + 1],
                     np.r_[self.contour[k2:], self.contour[:k1 + 1]]]
        else:
            sides = [self.contour[k2:k1 + 1][::-1],
                     np.r_[self.contour[k1:], self.contour[:k2 + 1]][::-1, :]]
            
        return sides
    
        
    def match_side_order(self, tail_prev):
        """ determines the side of the tails and align them with an earlier
        shape """
        # get the two sides
        sides = self.get_sides() 
        
        # get the reference line to determine the order of the sides
        line_ref = tail_prev.sides[0]
        
        # sort lines such that reference line comes first
        first_line = geometry.LineString(line_ref)
        dists = [np.mean([first_line.distance(geometry.Point(p))
                          for p in side])
                 for side in sides]
        if dists[0] < dists[1]:
            order = (0, 1)
        else:
            order = (1, 0)

        # save the order of the sides to be able to recover them after pickling            
        self.persistence_data['side_order'] = order
        
        # get the sides and make sure they agree with the previous order
        self._cache['sides'] = sides[order[0]], sides[order[1]]
    
    
    @cached_property
    def sides(self):
        """ return the two sides of the tail """
        sides = self.get_sides()
        order = self.persistence_data.get('side_order', None)
        if order is None:
            order = self.determine_side_order(sides)
            self.persistence_data['side_order'] = order
        return sides[order[0]], sides[order[1]]
            
        
    @property
    def ventral_side(self):
        """ returns the ventral side """
        return self.sides[0]
    
    
    @property
    def dorsal_side(self):
        """ returns the ventral side """
        return self.sides[1]
    
    
    @cached_property
    def centerline(self):
        """ determine the center line of the tail """
        mask, offset = self.mask
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=self.centerline_blur_radius,
                           closed_loop=False,
                           alpha=0, #< line length is constraint by beta
                           beta=self.centerline_bending_stiffness,
                           gamma=self.centerline_adaptation_rate)
        ac.max_iterations = self.centerline_max_iterations
        ac.set_potential(dist_map)
        
        # find centerline starting from the ventral_side
        points = curves.translate_points(self.ventral_side,
                                         -offset[0], -offset[1])
        spacing = self.centerline_spacing
        points = curves.make_curve_equidistant(points, spacing=spacing)
        # use the active contour algorithm
        points = ac.find_contour(points)
        points = curves.make_curve_equidistant(points, spacing=spacing)
        # translate points back into global coordinate system
        points = curves.translate_points(points, offset[0], offset[1])
        
        # orient the centerline such that it starts at the posterior end
        dist1 = curves.point_distance(points[0], self.endpoints[0])
        dist2 = curves.point_distance(points[-1], self.endpoints[0])
        if dist1 > dist2:
            points = points[::-1]
        
        return points
    
    
     
