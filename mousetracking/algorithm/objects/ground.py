'''
Created on Aug 28, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Holds classes useful for describing and fitting the ground profile
'''

from __future__ import division

import itertools

import numpy as np
from scipy import ndimage
from shapely import geometry

from .utils import LazyHDFValue, Interpolate_1D_Extrapolated
from video.analysis import curves
from video.analysis.utils import cached_property


class GroundProfile(object):
    """ class representing a single ground profile """
    
    def __init__(self, points):
        self.points = points
        
 
    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, value):
        self._points = np.asarray(value, np.double)
        self._cache = {} #< reset the cache of the cached_property
        
        
    def __repr__(self):
        return ('GroundProfile(num_points=%d, length=%g)'
                % (len(self), self.length))
    
    
    def __len__(self):
        return len(self._points)
        
        
    @cached_property
    def length(self):
        """ returns the length of the profile """
        return curves.curve_length(self.points)
    

    @cached_property
    def linestring(self):
        """ returns a shapely line string corresponding to the ground """
        return geometry.LineString(self.points)
    
    
    def get_polygon_points(self, depth, left=None, right=None):
        """ returns the points of a polygon representing the ground with a
        given `depth`, reaching from `left` to `right` """
        # build left side of the polygon
        if left is None:
            ground_points = [(self.points[0, 0], depth)]
            idx_left = 0
        else:
            ground_points = [(left, depth),
                             (left, self.points[0, 1])]
            idx_left = np.nonzero(self.points[:, 0] > left)[0][0]
            
        if right is None:
            ground_points += self.points[idx_left:, :].tolist()
            ground_points.append((self.points[-1, 0], depth))
        else:
            idx_right = np.nonzero(self.points[:, 0] < right)[0][-1]
            ground_points += self.points[idx_left:idx_right, :].tolist()
            ground_points.append((right, self.points[-1, 1]))
            ground_points.append((right, depth))

        return ground_points
            
                
    def get_polygon(self, depth, left=None, right=None):
        """ returns a polygon representing the ground with a given `depth` """
        points = self.get_polygon_points(depth, left, right)
        return geometry.Polygon(points)
    
                
    def make_equidistant(self, **kwargs):
        """ makes the ground profile equidistant """
        self.points = curves.make_curve_equidistant(self.points, **kwargs)


    @cached_property
    def midline(self):
        """ returns the average y-value along the profile """
        return np.mean(self.points[:, 1])


    @cached_property
    def interpolator(self):
        return Interpolate_1D_Extrapolated(self._points[:, 0],
                                           self._points[:, 1],
                                           copy=False)
    
    
    def get_y(self, x, nearest_neighbor=False):
        """ returns the y-value of the profile at a given x-position.
        This function interpolates between points and extrapolates beyond the
        edge points. """
        if nearest_neighbor:
            idx = np.argmin(np.abs(self.points[:, 0] - x))
            return self.points[idx, 1]
        else:
            return self.interpolator(x)
        
        
    def get_distance(self, (x, y), signed=False):
        """ calculates the (signed) distance of a point to the ground line. If
        the distance is signed, points above the ground are associated with 
        negative distances """
        dist = self.linestring.distance(geometry.Point(x, y))
        if signed and self.above_ground((x, y)):
            dist *= -1
        return dist

        
    def above_ground(self, (x, y)):
        """ returns True if the point is above the ground """
        # Note that the y axis points down
        return self.get_y(x) > y
   


class GroundProfileList(object):
    """ organizes a list of ground profiles """
    hdf_attributes = {'column_names': ('Time', 'Position X', 'Position Y')}
    storage_class = LazyHDFValue
   
    def __init__(self):
        self.times = []
        self.grounds = []
    
    
    def __len__(self):
        return len(self.times)


    def append(self, time, ground):
        self.times.append(time)
        self.grounds.append(ground)
   

    def append_from_array(self, data):
        """ append a single ground profile with data taken from an array """
        if data:
            data = np.asarray(data)
            time = data[0, 0]
            ground = GroundProfile(data[:, 1:])
            self.append(time, ground)

   
    def to_array(self):
        """ convert the data stored in the object to an array """
        results = []
        for time, ground in itertools.izip(self.times, self.grounds):
            time_array = np.zeros((len(ground), 1), np.int32) + time
            results.append(np.hstack((time_array, ground.points)))

        if results:
            return np.concatenate(results)
        else:
            return []
    
    
    @classmethod
    def create_from_array(cls, value):
        """ create the object from a supplied array """
        result = cls()
        index, obj_data = None, None
        # iterate over the data and create objects from it
        for line in value:
            if line[0] == index:
                # append object to the current track
                obj_data.append(line)
            else:
                # save the track and start a new one
                result.append_from_array(obj_data)
                obj_data = [line]
                index = line[0]
        
        result.append_from_array(obj_data)
            
        return result
   


class GroundProfileTrack(object):
    """ class holding the ground profile information for the entire video.
    For efficient data storage the ground profiles are re-parameterized
    to have the same number of support line.
    """
    
    hdf_attributes = {'row_names': ('Time 1', 'Time 2', '...'),
                      'column_names': ('Time', 'Point 1', 'Point 2', '...'),
                      'depth_names': ('X Coordinate', 'Y Coordinate')}
    storage_class = LazyHDFValue

    
    def __init__(self, times, profiles):
        # store information in numpy arrays 
        self.times = np.asarray(times, np.int)
        self.profiles = np.asarray(profiles, np.double)
        # profiles is a 3D array: len(times) x num_points x 2
        self._cache = {}
        
        
    def __len__(self):
        return len(self.times)
    
    
    def __repr__(self):
        return '%s(frames=%d, line=%d)' % (self.__class__.__name__,
                                           self.profiles.shape[0],
                                           self.profiles.shape[1])
    
    
    def smooth(self, sigma):
        """ smoothes the profiles in time using a Gaussian window of given sigma """
        # convolve each point with a gaussian filter
        self.profiles = ndimage.filters.gaussian_filter1d(self.profiles,
                                                          sigma,  #< std of the filter
                                                          axis=0, #< time axis
                                                          mode='nearest')


    def get_ground_profile(self, frame_id):
        """ returns the ground object for a certain frame """
        # for simplicity, find the index which is closest to the data we have
        idx = np.argmin(np.abs(self.times - frame_id))
        if self._cache.get('ground_idx') != idx:
            self._cache['ground'] = GroundProfile(self.profiles[idx, :, :])
            self._cache['ground_idx'] = idx
        return self._cache['ground']


    def get_groundline(self, frame_id):
        """ returns the ground line for a certain frame """
        idx = np.argmin(np.abs(self.times - frame_id))
        return self.profiles[idx, :, :]


    def get_profile(self, frame_id):
        """ returns the ground object for a certain frame """
        # for simplicity, find the index which is closest to the data we have
        return GroundProfile(self.get_groundline(frame_id))

    
    def to_array(self):
        """ collect the data in a single array """
        t2 = np.vstack((self.times, self.times)).T
        return np.concatenate((t2.reshape((-1, 1, 2)), self.profiles), axis=1)
        
        
    @classmethod
    def create_from_ground_profile_list(cls, ground_profiles):
        # determine the maximal number of points in a single ground
        num_points = max(len(ground) for ground in ground_profiles.grounds)

        # iterate through all profiles and convert them to have equal number of line
        # and store the data
        times = ground_profiles.times
        profiles = [curves.make_curve_equidistant(ground.points, count=num_points)
                    for ground in ground_profiles.grounds]
        
        # store information in numpy arrays 
        # profiles is a 3D array: len(times) x num_points x 2
        return cls(times=times, profiles=profiles)        
        
        
    @classmethod
    def create_from_array(cls, data):
        """ collect the data in a single array """
        return cls(times=data[:, 0, 0], profiles=data[:, 1:, :])
    
     
    