'''
Created on Aug 28, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Holds classes describing mouse burrows
'''

from __future__ import division

import copy
import itertools

import numpy as np
from scipy import cluster 
import shapely
from shapely import geometry

from utils.cache import cached_property
from utils.data_structures import LazyHDFCollection
from utils.math import trim_nan
from video.analysis import curves, regions, shapes

from video import debug  # @UnusedImport


# monkey patch shapely.geometry to get compatibility with older shapely versions
if not hasattr(geometry, 'LinearRing'):
    geometry.LinearRing = geometry.polygon.LinearRing



def bool_nan(number):
    """ returns the boolean value of the number.
    Returns False if number is not-a-number"""
    if np.isnan(number):
        return False
    else:
        return bool(number)



class EndPoint(shapes.Point):
    """ class that is used to collect the end points of burrows """
    
    def __init__(self, x, y, is_exit=False):
        super(EndPoint, self).__init__(x, y)
        self.is_exit = is_exit

    def __repr__(self):
        return ("%s(x=%g, y=%g, is_exit=%r)" % 
                (self.__class__.__name__, self.x, self.y, self.is_exit))
        


class Burrow(shapes.Polygon):
    """ represents a single burrow.
    Note that the contour are always given in clockwise direction
    """
    
    # parameters that influence burrow detection and measurement
    default_parameters = {'ground_point_distance': 20,
                          'centerline_segment_length': 10,
                          'centerline_skip_length': 10,
                          'width': 20}
    
    # class controlling the storage of this class
    storage_class = LazyHDFCollection
    # extra information for the storage
    array_columns = ('Outline X', 'Outline Y',
                     'Burrow length + Centerline X',
                     'Flag if burrow was refined + Centerline Y')
    
    
    def __init__(self, contour, centerline=None, length=None,
                 refined=False, endpoints=None, parameters=None):
        """ initialize the structure using line on its contour """
        if len(contour) < 3:
            raise ValueError("Burrow contour must be defined by at least "
                             "three points.")
        
        super(Burrow, self).__init__(contour)
            
        self.refined = refined
        self._centerline = None
        self._endpoints = None

        self.parameters = self.default_parameters.copy()
        if parameters is not None:
            self.parameters.update(parameters)

        if endpoints is not None:
            self.endpoints = endpoints
            
        if centerline is not None:
            self.centerline = centerline

        if length is not None:
            self.length = length
        else:
            self.length = -1


    def copy(self):
        """ return a copy of this object """
        return Burrow(copy.copy(self.contour), copy.copy(self.centerline),
                      self.length, self.refined, copy.copy(self._endpoints),
                      self.parameters)

        
    def __repr__(self):
        center = self.polygon.centroid
        flags = ['center=(%d, %d)' % (center.x, center.y),
                 'points=%d' % len(self._contour)]
        if self.length:
            flags.append('length=%d' % self.length)
        else:
            flags.append('area=%d' % self.polygon.area)
        if self.refined:
            flags.append('refined')
        if self._endpoints is not None:
            flags.append('%d endpoints' % len(self._endpoints))
            
        return ('Burrow(%s)' % (', '.join(flags)))


    @property
    def endpoints(self):
        if self._endpoints is None:
            self.get_endpoints()
        return self._endpoints
    
    @endpoints.setter
    def endpoints(self, value):
        self._endpoints = value
        

    def estimate_endpoints(self, p_start=None):
        """ find the farthest two points in the burrow mask """
        mask, offset = self.get_mask(margin=1, dtype=np.int32,
                                     ret_offset=True)
        # find a point in the mask
        if p_start is None:
            xs, ys = np.nonzero(mask)
            p_start = EndPoint(xs[0], ys[0], is_exit=False)
            dist_prev = 0
        else:
            p_start.translate(-offset[0], -offset[1])
            dist_prev = np.inf # we only need one iteration

        # iterate until the farthest pair of points is found
        while True:
            # make distance map starting from point p_start
            distance_map = mask.copy()
            regions.make_distance_map(distance_map,
                                      start_points=(p_start.coords,))
            # find point farthest point away from p_start
            idx_max = np.unravel_index(distance_map.argmax(),
                                       distance_map.shape)
            dist = distance_map[idx_max]
            p_end = EndPoint(idx_max[1], idx_max[0], is_exit=False)
            if dist <= dist_prev:
                break
            dist_prev = dist
            # take farthest point as new start point
            p_start = p_end
            
        # return the two points
        return [p.translate(*offset) for p in (p_start, p_end)]


    def get_exit_regions(self, ground_line):
        """ returns parts of the outline that are associated with exits """
        # determine burrow points close to the ground
        g_line = ground_line.linestring
        dist_max = self.parameters['ground_point_distance']
        exitpoints = [point for point in self.contour
                      if g_line.distance(geometry.Point(point)) < dist_max]

        if len(exitpoints) == 0:
            exit_regions = []
        elif len(exitpoints) == 1:
            exit_regions = [exitpoints]
        else:
            # check whether points are clustered around other points
            exitpoints = np.array(exitpoints)
    
            # cluster the points to detect multiple connections 
            # this is important when a burrow has multiple exits to the ground
            dist_max = self.parameters['width']
            data = cluster.hierarchy.fclusterdata(exitpoints, dist_max,
                                                  method='single', 
                                                  criterion='distance')
            
            # find the exit points
            exit_regions = [exitpoints[data == cluster_id, :]
                            for cluster_id in np.unique(data)]
            
        return exit_regions


    def get_endpoints(self, ground_line=None):
        """ estimate burrow exit points """
        
        if ground_line:
            # find exits, which are close to the ground line
            exit_regions = self.get_exit_regions(ground_line)
            
            # find the exit points
            exits, exit_size = [], []
            for exit_points in exit_regions:
                xm, ym = exit_points.mean(axis=0)
                dist = np.hypot(exit_points[:, 0] - xm, exit_points[:, 1] - ym)
                exits.append(exit_points[np.argmin(dist), :])
                exit_size.append(len(exit_points))
    
            exits = np.array(exits)
            exit_size = np.array(exit_size)
    
            if len(exits) > 0:
                # sorted points by their size
                endpoints = exits[np.argsort(-exit_size), :]
                # convert the end points to proper objects
                endpoints = [EndPoint(x, y, is_exit=True) for x, y in endpoints]
            else:
                endpoints = []
    
        else:
            endpoints = []

        # fill in end points that have not yet been found            
        if len(endpoints) == 0:
            endpoints = self.estimate_endpoints()
        elif len(endpoints) == 1:
            endpoints = self.estimate_endpoints(endpoints[0])
            
        # store the end points in the cache and also return them
        self._endpoints = endpoints
        self._centerline = None #< delete cache of the centerline 
        return endpoints


    @property
    def centerline(self):
        """ retrieve the centerline """
        if self._centerline is None:
            self.determine_centerline()
        return self._centerline
    
    @centerline.setter
    def centerline(self, points):
        """ set the new centerline """
        if points is None:
            self._centerline = None
        else:
            self._centerline = np.array(points, np.double)
            self.length = curves.curve_length(self._centerline)
        self._cache = {}


    def determine_centerline(self):
        """ determines the centerline """
        spacing = self.parameters['centerline_segment_length']
        skip_length = self.parameters['centerline_skip_length']

        endpoints = [p.coords for p in self.endpoints]
        centerline = self.get_centerline_smoothed(spacing=spacing,
                                                  skip_length=skip_length,
                                                  endpoints=endpoints)
        self._centerline = np.asarray(centerline)
        self.length = curves.curve_length(self._centerline)


    def get_morphological_graph(self):
        """ determines the morphological graph of the burrow, which also takes
        into account the exit points """
        # determine the morphological graph of the polygon
        graph = super(Burrow, self).get_morphological_graph()
        # this graph does not include the exits or other end points
        
        # add all the end points to the graph
        for endpoint in self.endpoints:
            graph.add_and_connect_node_point(endpoint.coords)
        
        return graph


    def merge(self, other):
        """ merge this burrow with another one """
        polygon = self.polygon.union(other.polygon)
        self.contour = regions.get_enclosing_outline(polygon)
        
        # set the centerline to the longest of the two
        if other.length > self.length:
            self.centerline = other.centerline


    @cached_property
    def linestring(self):
        if self.centerline is None or len(self.centerline) < 2:
            return geometry.LineString()
        else:
            return geometry.LineString(self.centerline)
    
    
    @property
    def entry_point(self):
        if self.centerline is None:
            return None
        else:
            return self.centerline[0]
    
    
    @property
    def end_point(self):
        if self.centerline is None:
            return None
        else:
            return self.centerline[-1]
    
    
    @property
    def is_valid(self):
        return len(self._contour) > 3
    
    
    def intersects(self, burrow_or_shape):
        """ returns True if polygon intersects the burrow """
        if isinstance(burrow_or_shape, Burrow):
            burrow_or_shape = burrow_or_shape.polygon
        try:
            return self.polygon.intersects(burrow_or_shape)
        except (shapely.geos.TopologicalError, shapely.geos.PredicateError):
            return False
    
    
    def simplify_outline(self, tolerance=0.1):
        """ simplifies the contour """
        outline = geometry.LineString(self.contour)
        tolerance *= outline.length
        outline = outline.simplify(tolerance, preserve_topology=True)
        self.contour = outline.coords
    
    
    def get_bounding_rect(self, margin=0):
        """ returns the bounding rectangle of the burrow """
        burrow_points = self.contour
        if self._centerline is not None:
            burrow_points = np.vstack((burrow_points, self._centerline))
        bounds = geometry.MultiPoint(burrow_points).bounds
        bound_rect = regions.corners_to_rect(bounds[:2], bounds[2:])
        if margin:
            bound_rect = regions.expand_rectangle(bound_rect, margin)
        return np.asarray(bound_rect, np.int)
    
    
    def extend_outline(self, extension_polygon, simplify_threshold):
        """ extends the contour of the burrow to also enclose the object given
        by polygon """
        # get the union of the burrow and the extension
        burrow = self.polygon.union(extension_polygon)
        
        # determine the contour of the union
        outline = regions.get_enclosing_outline(burrow)
        
        outline = outline.simplify(simplify_threshold*outline.length)

        self.contour = np.asarray(outline, np.int32)
        
            
    def to_array(self):
        """ converts the internal representation to a single array """
        # collect the data for the first two columns
        if self.contour is None:
            data1 = np.zeros((0, 2), np.double)
        else:
            data1 = np.asarray(self.contour, np.double)

        # collect the data for the last two columns
        data2 = np.array([[self.length, self.refined]], np.double)
        if self.centerline is not None:
            data2 = np.concatenate((data2, self.centerline))
        
        # save the data in a convenient array
        l1, l2 = len(data1), len(data2)
        data_res = np.empty((max(l1, l2), 4), np.double)
        data_res.fill(np.nan)
        data_res[:l1, :2] = data1
        data_res[:l2, 2:] = data2 
        return data_res
        

    @classmethod
    def from_array(cls, data):
        """ creates a burrow track from a single array """
        # load the data from the respective places
        data = np.asarray(data)
        data_attr = data[:2, 2:]
        data_contour = trim_nan(data[:, :2], left=False)
        data_centerline = trim_nan(data[2:, 2:], left=False)
        if len(data_centerline) == 0:
            data_centerline = None
            
        # create the object
        return cls(contour=data_contour,
                   centerline=data_centerline,
                   length=data_attr[0, 0],
                   refined=bool_nan(data_attr[0, 1]))
    
        # data_attr[1, 0] was previously used to indicate `two_exits` flag
        
        
        
class BurrowTrack(object):
    """ class that stores the evolution of a single burrow over time.
    """
    column_names = ('Time',) + Burrow.array_columns
    
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
    
    
    def append(self, time, burrow):
        """ append a new burrow with a time code """
        self.times.append(time)
        self.burrows.append(burrow)
        
        
    def __delitem__(self, key):
        """ deletes a certain burrow from the track list """
        del self.times[key]
        del self.burrows[key]
        
    
    @property
    def first(self):
        """ return the first burrow in the track """
        return self.burrows[0]
    
    @property
    def last(self):
        """ return the last burrow in the track """
        return self.burrows[-1]
    
    
    @property
    def track_start(self): return self.times[0]
    @property
    def track_end(self): return self.times[-1]
    
    
    def get_max_length(self):    
        """ returns the maximal length of the burrow """
        return max(burrow.length for burrow in self.burrows)

    
    def get_burrow_index(self, time):
        """ returns the index of the burrow at a specific time. """
        if not self.times[0] <= time <= self.times[-1]:
            raise IndexError
        
        return np.argmin(np.abs(np.asarray(self.times) - time))
    
    
    def get_burrow(self, time, ret_next_change=False):
        """ returns the burrow at a specific time.
        If ret_next_change is True, we also return the frame where the burrow
        changes next. """
        idx = self.get_burrow_index(time)
        burrow = self.burrows[idx]
        if ret_next_change:
            if idx >= len(self.times) - 1:
                return burrow, time + 1
            else:
                return burrow, (self.times[idx] + self.times[idx + 1])/2
        else:
            return burrow
    
    
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
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        burrow_track = cls()
        burrow_data = None
        time_cur = -1
        for d in data:
            if d[0] != time_cur:
                if burrow_data is not None:
                    burrow_track.append(time_cur, Burrow.from_array(burrow_data))
                time_cur = d[0]
                burrow_data = [d[1:]]
            else:
                burrow_data.append(d[1:])

        if burrow_data is not None:
            burrow_track.append(time_cur, Burrow.from_array(burrow_data))

        return burrow_track
    
 
    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        if key in hdf_file:
            del hdf_file[key]
        hdf_file.create_dataset(key, data=self.to_array(), track_times=True)


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.from_array(hdf_file[key])
   
   

class BurrowTrackList(list):
    """ class that stores instances of BurrowTrack in a list """
    item_class = BurrowTrack
    storage_class = LazyHDFCollection
    hdf_attributes = {'column_names': BurrowTrack.column_names,
                      'remark':
                        'Each burrow is represented by its contour saved as a '
                        'list of points of the format (Time, X, Y, a, b), '
                        'where all points with the same Time belong to the same '
                        'burrow. The last two values (a, b) contain additional '
                        'data, e.g. the burrow length and the coordinates of '
                        'the centerline.'} 


    def create_track(self, frame_id, burrow):
        """ creates a new burrow track and appends it to the list """
        self.append(BurrowTrack(frame_id, burrow))
        

    def find_burrows(self, frame_id, ret_next_change=False):
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

