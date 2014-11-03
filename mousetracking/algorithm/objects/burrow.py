'''
Created on Aug 28, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Holds classes describing mouse burrows
'''

from __future__ import division

import itertools

import numpy as np
import cv2
from shapely import geometry, geos

from .utils import LazyHDFCollection, rtrim_nan
from video.analysis import curves, regions
from video.analysis.utils import cached_property

from ..debug import *  # @UnusedWildImport


# monkey patch shapely.geometry to get compatibility with older shapely versions
if not hasattr(geometry, 'LinearRing'):
    geometry.LinearRing = geometry.polygon.LinearRing


class Burrow(object):
    """ represents a single burrow.
    Note that the outline are always given in clockwise direction, which is
    ensured automatically
    """
    
    storage_class = LazyHDFCollection
    array_columns = ('Outline X', 'Outline Y',
                     'Burrow length + Centerline X',
                     'Flag if burrow was refined + Centerline Y')
    
    
    def __init__(self, outline, centerline=None, length=None,
                 refined=False, exit_count=1):
        """ initialize the structure using line on its outline """
        if len(outline) < 3:
            raise ValueError("Burrow outline must be defined by at least "
                             "three points.")
        self.outline = outline
        self.centerline = centerline
        self.refined = refined
        self.exit_count = exit_count

        if length is not None and np.isfinite(length):
            self.length = length


    def copy(self):
        return Burrow(self.outline.copy())

        
    def __repr__(self):
        polygon = self.polygon
        center = polygon.centroid
        return ('Burrow(center=(%d, %d), area=%s, points=%d)' %
                (center.x, center.y, polygon.area, len(self._outline)))


    def __eq__(self, other):
        return np.all(self.outline == other.outline)

    def __ne__(self, other):
        return np.any(self.outline != other.outline)


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
        
        
    @property
    def centerline(self):
        return self._centerline
    
    @centerline.setter
    def centerline(self, point_list):
        """ sets a new centerline """
        if point_list is None:
            self._centerline = None
            self.length = None
        else:
            self._centerline = np.asarray(point_list, np.double)
            self.length = curves.curve_length(self._centerline)
        

    @cached_property
    def outline_ring(self):
        """ return the linear ring of the burrow outline """
        return geometry.asLinearRing(self.outline)
    
        
    @cached_property
    def polygon(self):
        """ return the polygon of the burrow outline """
        return geometry.Polygon(self.outline)
    
    
    @cached_property
    def position(self):
        return self.polygon.representative_point()
    
    
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
    
    
    def intersects(self, burrow):
        """ returns True if polygon intersects the burrow """
        try:
            # We'd like to use the simple form
            #    return self.polygon.intersects(polygon)
            # but that does not work due a shapely error
            poly1 = self.polygon
            poly2 = burrow.polygon
            intersection = poly1.intersection(poly2)
            return not intersection.is_empty
        except geos.TopologicalError:
            return False
    
    
    def simplify_outline(self, tolerance=0.1):
        """ simplifies the outline """
        outline = geometry.asLineString(self.outline)
        tolerance = tolerance*outline.length
        outline = outline.simplify(tolerance, preserve_topology=True)
        self.outline = outline.coords
    
    
    def get_bounding_rect(self, margin=0):
        """ returns the bounding rectangle of the burrow """
        bounds = self.polygon.bounds
        bound_rect = regions.corners_to_rect(bounds[:2], bounds[2:])
        bound_rect = regions.expand_rectangle(bound_rect, margin)
        return np.asarray(bound_rect, np.int)
    
    
    def extend_outline(self, extension_polygon, simplify_threshold):
        """ extends the outline of the burrow to also enclose the object given
        by polygon """
        # get the union of the burrow and the extension
        burrow = self.polygon.union(extension_polygon)
        
        # determine the outline of the union
        outline = regions.get_enclosing_outline(burrow)
        
        outline = outline.simplify(simplify_threshold*outline.length)

        self.outline = np.asarray(outline, np.int32)
 
   
    def get_mask(self, margin=0, dtype=np.uint8, ret_shift=False):
        """ builds a mask of the burrow """
        # prepare the array to store the mask into
        rect = self.get_bounding_rect(margin=margin)
        mask = np.zeros((rect[3], rect[2]), dtype)

        # draw the burrow into the mask
        outline = np.asarray(self.outline, np.int)
        offset = (-rect[0], -rect[1])
        cv2.fillPoly(mask, [outline], color=1, offset=offset)
        
        if ret_shift:
            return mask, (-offset[0], -offset[1])
        else:
            return mask
            
    
    def to_array(self):
        """ converts the internal representation to a single array """
        # collect the data for the first two columns
        data1 = np.asarray(self.outline, np.double)

        # collect the data for the last two columns
        data2 = np.array([[self.length, self.refined],
                          [np.nan, np.nan]], np.double)
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
    column_names = ('Time',) + Burrow.array_columns
    
    def __init__(self, time=None, burrow=None):
        self.times = [] if time is None else [time]
        self.burrows = [] if burrow is None else [burrow]
        self.active = True
        
        
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
            if idx >= len(self.times) - 1:
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


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.create_from_array(hdf_file[key])
   
   

class BurrowTrackList(list):
    """ class that stores instances of BurrowTrack in a list """
    item_class = BurrowTrack
    storage_class = LazyHDFCollection
    hdf_attributes = {'column_names': BurrowTrack.column_names,
                      'remark':
                        'Each burrow is represented by its outline saved as a '
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

