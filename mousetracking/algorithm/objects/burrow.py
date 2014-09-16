'''
Created on Aug 28, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Holds classes describing mouse burrows
'''

from __future__ import division

import itertools

import numpy as np
import cv2
import shapely
import shapely.geometry as geometry

from .utils import LazyHDFCollection, rtrim_nan
from video.analysis import curves, regions
from video.analysis.utils import cached_property

from ..debug import *  # @UnusedWildImport


# monkey patch shapely.geometry to get compatibility with older shapely versions
if not hasattr(geometry, 'LinearRing'):
    geometry.LinearRing = geometry.polygon.LinearRing


class Burrow(object):
    """ represents a single burrow.
    Note that the outline are always given in clockwise direction
    """
    
    storage_class = LazyHDFCollection
    array_columns = ('Outline X', 'Outline Y', 
                     'Burrow length + Centerline X',
                     'Flag if burrow was refined + Centerline Y')
    
    # parameters influencing how the centerline is determined
    curvature_radius_max = 50
    centerline_segment_length = 25
    ground_point_distance = 10
    
    
    def __init__(self, outline, centerline=None, length=None, refined=False):
        """ initialize the structure using line on its outline """
        if centerline is not None and length is None:
            length = curves.curve_length(centerline)

        self.outline = outline
        self.centerline = centerline
        self.length = length
        self.refined = refined


    def copy(self):
        return Burrow(self.outline.copy())

        
    def __repr__(self):
        polygon = self.polygon
        center = polygon.centroid
        return ('Burrow(center=(%d, %d), area=%s, points=%d)' %
                (center.x, center.y, polygon.area, len(self._outline)))


    @property
    def outline(self):
        return self._outline

    
    @outline.setter
    def outline(self, point_list):
        """ sets a new outline """
        point_list = np.asarray(point_list, np.double)
        # make sure that the outline is given in clockwise direction
        if geometry.LinearRing(point_list).is_ccw:
            point_list = point_list[::-1]
        self._outline = point_list

        # reset cache
        self.centerline = None
        self.length = None
        self.refined = False
        self._cache = {}
        
        
    @cached_property
    def polygon(self):
        """ return the polygon of the burrow outline """
        return geometry.Polygon(np.asarray(self.outline, np.double))
    
    
    @cached_property
    def area(self):
        """ return the area of the burrow shape """
        return self.polygon.area
    
    
    @property
    def is_valid(self):
        return len(self._outline) > 3
    
    
    @cached_property
    def eccentricity(self):
        """ return the eccentricity of the burrow shape
        The eccentricity will be between 0 and 1, corresponding to a circle
        and a straight line, respectively.
        """
        m = cv2.moments(np.asarray(self.outline, np.uint8))
        a, b, c = m['mu20'], -m['mu11'], m['mu02']
        e1 = (a + c) + np.sqrt(4*b**2 + (a - c)**2)
        e2 = (a + c) - np.sqrt(4*b**2 + (a - c)**2)
        if e1 == 0:
            return 0
        else:
            return np.sqrt(1 - e2/e1)
    
                
    def contains(self, point):
        """ returns True if the point is inside the burrow """
        return self.polygon.contains(geometry.Point(point))
    
    
    def intersects(self, polygon):
        """ returns True if polygon intersects the burrow """
        try:
            return not self.polygon.intersection(polygon).is_empty
        except shapely.geos.TopologicalError:
            return False
    
    
    def simplify_outline(self, tolerance=0.1):
        """ simplifies the outline """
        outline = geometry.LineString(self.outline)
        tolerance = tolerance*outline.length
        outline = outline.simplify(tolerance, preserve_topology=True)
        self.outline = np.array(outline.coords, np.double)
    
    
    def get_bounding_rect(self, margin=0):
        """ returns the bounding rectangle of the burrow """
        bounds = self.polygon.bounds
        bound_rect = regions.corners_to_rect(bounds[:2], bounds[2:])
        return regions.expand_rectangle(bound_rect, margin)
    
    
    def extend_outline(self, extension_polygon, simplify_threshold):
        """ extends the outline of the burrow to also enclose the object given
        by polygon """
        # get the union of the burrow and the extension
        burrow = self.polygon.union(extension_polygon)
        
        # determine the outline of the union
        outline = regions.get_enclosing_outline(burrow)
        
        outline = outline.simplify(simplify_threshold*outline.length)

        self.outline = np.asarray(outline, np.int32)

    
    def get_centerline(self, ground):
        """ determine the centerline, given the outline and the ground profile.
        The ground profile is used to determine the burrow exit. """
        
        if self.centerline is not None:
            return self.centerline
        
        # get the ground line 
        ground_line = ground.linestring
        
        # reparameterize the burrow outline to locate the burrow exit reliably
        outline = curves.make_curve_equidistant(self.outline, 10)
        outline = np.asarray(outline, np.double)

        # calculate the distance of each outline point to the ground
        dist = np.array([ground_line.distance(geometry.Point(p)) for p in outline])
        
        # get points at the burrow exit (close to the ground profile)
        indices = (dist < self.ground_point_distance)
        if np.any(indices):
            p_exit = outline[indices, :].mean(axis=0)
        else:
            p_exit = outline[np.argmin(dist)]
        p_exit = curves.get_projection_point(ground_line, p_exit)
            
        # get the two ground points closest to the exit point
        dist = np.linalg.norm(ground.line - p_exit, axis=1)
        k1 = np.argmin(dist)
        dist[k1] = np.inf
        k2 = np.argmin(dist)
        p1, p2 = ground.line[k1], ground.line[k2]
        # ensure that p1 is left of p2
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        
        # send out rays perpendicular to the ground profile
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + np.pi/2
        point_anchor = (p_exit[0] + 5*np.cos(angle), p_exit[1] + 5*np.sin(angle))
        outline_poly = geometry.LinearRing(self.outline)
        
        # calculate the angle each segment is allowed to deviate from the 
        # previous one based on the maximal radius of curvature
        ratio = self.centerline_segment_length/self.curvature_radius_max
        angle_max = np.arccos(1 - 0.5*ratio**2)
        segment_length = self.centerline_segment_length
        
        centerline = [p_exit]
        while True:
            # find the next point along the burrow
            point_max, dist_max, angle = regions.get_farthest_ray_intersection(
                point_anchor,
                np.linspace(angle - angle_max, angle + angle_max, 16),
                outline_poly)
                # this also sets the angle for the next iteration

            # abort if the search was not successful
            if point_max is None:
                break
                
            # get the length of the longest ray
            if dist_max > segment_length:
                # continue shooting out rays
                point_anchor = (point_anchor[0] + segment_length*np.cos(angle),
                                point_anchor[1] + segment_length*np.sin(angle))
                centerline.append(point_anchor)
            else:
                # we've hit the end of the burrow
                centerline.append(point_max)
                break
                    
        # save results                    
        self.centerline = centerline
        return centerline
            
        
    def get_length(self, ground):
        """ calculates the centerline and returns its length """
        if self.length is None:
            centerline = self.get_centerline(ground)
            self.length = curves.curve_length(centerline)
        return self.length
            
    
    def to_array(self):
        """ converts the internal representation to a single array """
        # collect the data for the first two columns
        data1 = np.asarray(self.outline, np.double)

        # collect the data for the last two columns
        data_attr = np.array([[self.length, self.refined]], np.double)
        spacer = np.array([[np.nan, np.nan]])
        if self.centerline is None:
            data_centerline = np.array([])
        else:
            data_centerline = self.centerline
        data2 = np.concatenate((data_attr, spacer, data_centerline))
        
        # save the data in a convenient array
        l1, l2 = len(data1), len(data2)
        data_res = np.empty((max(l1, l2), 4), np.double)
        data_res.fill(np.nan)
        data_res[:l1, :2] = data1
        data_res[:l2, 2:] = data2 
        return data_res
        

    @classmethod
    def create_from_array(cls, data):
        """ creates a burrow track from a single array """
        # load the data from the respective places
        data = np.asarray(data)
        data_attr = data[0, 2:]
        data_outline = rtrim_nan(data[:, :2])
        data_centerline = rtrim_nan(data[2:, 2:])
        if len(data_centerline) == 0:
            data_centerline = None
            
        # create the object
        return cls(outline=data_outline, centerline=data_centerline,
                   length=data_attr[0], refined=bool(data_attr[1]))
        
        
        
class BurrowTrack(object):
    """ class that stores the evolution of a single burrow over time.
    """
    array_columns = ('Time',) + Burrow.array_columns
    
    def __init__(self, time=None, burrow=None):
        self.times = [] if time is None else [time]
        self.burrows = [] if burrow is None else [burrow]
        
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'BurrowTrack([])'
        elif len(self.times) == 1:
            return 'BurrowTrack(time=%d)' % (self.times[0])
        else:
            return 'BurrowTrack(times=%d..%d)' % (self.times[0], self.times[-1])
        
        
    def __len__(self):
        return len(self.times)
    
    
    @property
    def last(self):
        """ return the last burrow in the track """
        return self.burrows[-1]
    
    
    @property
    def track_start(self): return self.times[0]
    @property
    def track_end(self): return self.times[-1]
    
    
    def get_burrow(self, time, ret_next_change=False):
        """ returns the burrow at a specific time.
        If ret_next_change is True, we also return the frame where the burrow
        changes next. """
        if not self.times[0] <= time <= self.times[-1]:
            raise IndexError
        
        idx = np.argmin(np.abs(np.asarray(self.times) - time))
        burrow = self.burrows[idx]
        if ret_next_change:
            if idx == len(self.times):
                return burrow, time + 1
            else:
                return burrow, (self.times[idx] + self.times[idx + 1])/2
        else:
            return burrow
    
    
    def append(self, time, burrow):
        """ append a new burrow with a time code """
        self.times.append(time)
        self.burrows.append(burrow)
    
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        res = []
        for time, burrow in itertools.izip(self.times, self.burrows):
            burrow_data = burrow.to_array()
            time_array = np.zeros((len(burrow_data), 1), np.int32) + time
            res.append(np.hstack((time_array, burrow_data)))
        if res:
            return np.vstack(res)
        else:
            return []
        
        
    @classmethod
    def create_from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        burrow_track = cls()
        burrow_data = None
        time_cur = -1
        for d in data:
            if d[0] != time_cur:
                if burrow_data is not None:
                    burrow_track.append(time_cur, Burrow.create_from_array(burrow_data))
                time_cur = d[0]
                burrow_data = [d[1:]]
            else:
                burrow_data.append(d[1:])

        if burrow_data is not None:
            burrow_track.append(time_cur, Burrow.create_from_array(burrow_data))

        return burrow_track
    
 
    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        if key in hdf_file:
            del hdf_file[key]
        hdf_file.create_dataset(key, data=self.to_array(), track_times=True)
        hdf_file[key].attrs['column_names'] = self.array_columns
        hdf_file[key].attrs['remark'] = (
            'Each burrow is represented by its outline saved as a list of points '
            'of the format (Time, X, Y, a, b), where all points with the same '
            'Time belong to the same burrow. The last two values (a, b) contain '
            'additional data, e.g. the burrow length and the coordinates of the '
            'centerline.'
        ) 


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.create_from_array(hdf_file[key])
   
   

class BurrowTrackList(list):
    """ class that stores instances of BurrowTrack in a list """
    item_class = BurrowTrack
    storage_class = LazyHDFCollection

    def get_burrows(self, frame_id, ret_next_change=False):
        """ returns a list of all burrows active in a given frame.
        If ret_next_change is True, the number of the frame where something
        will change in the returned burrows is also returned. This can be
        useful while iterating over all frames. """
        result = []
        for burrow_track in self:
            try:
                res = burrow_track.get_burrow(frame_id, ret_next_change)
            except IndexError:
                continue
            else:
                result.append(res)
                
        if ret_next_change:
            burrows = [res[0] for res in result]

            if burrows:
                next_change = min(res[1] for res in result)
            else:
                next_change = frame_id + 1
            return burrows, next_change
        else:
            return result

