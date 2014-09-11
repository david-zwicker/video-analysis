'''
Created on Aug 28, 2014

@author: zwicker

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
    
    def __init__(self, line):
        self._line = np.asarray(line)
        self._cache = {}
        
    @property
    def line(self):
        return self._line
    
    @line.setter
    def line(self, value):
        self._line = np.asarray(value)
        self._cache = {}
        
    def __repr__(self):
        return 'GroundProfile(%d line)' % (len(self.line))
    
    def __len__(self):
        return len(self.line)
        
    @cached_property
    def length(self):
        """ returns the length of the profile """
        return curves.curve_length(self.line)
    
    @cached_property
    def linestring(self):
        """ returns a shapely line string corresponding to the ground """
        return geometry.LineString(np.array(self.line, np.double))
    
    def make_equidistant(self, **kwargs):
        """ makes the ground profile equidistant """
        self.line = curves.make_curve_equidistant(self.line, **kwargs)
   
    def get_y(self, x):
        if 'interpolator' in self._cache:
            interpolator = self._cache['interpolator']
        else:
            interpolator = Interpolate_1D_Extrapolated(self._line[:, 0], self._line[:, 1])
           
        return interpolator(x)
   


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
            results.append(np.hstack((time_array, ground.line)))

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


    def get_ground(self, frame_id):
        """ returns the ground object for a certain frame """
        # for simplicity, find the index which is closest to the data we have
        idx = np.argmin(np.abs(self.times - frame_id))
        if self._cache.get('ground_idx') != idx:
            self._cache['ground'] = GroundProfile(self.profiles[idx, :, :])
            self._cache['ground_idx'] = idx
        return self._cache['ground']


    def get_profile(self, frame_id):
        """ returns the ground line for a certain frame """
        # for simplicity, find the index which is closest to the data we have
        idx = np.argmin(np.abs(self.times - frame_id))
        return self.profiles[idx, :, :]

    
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
        profiles = [curves.make_curve_equidistant(ground.line, count=num_points)
                    for ground in ground_profiles.grounds]
        
        # store information in numpy arrays 
        # profiles is a 3D array: len(times) x num_points x 2
        return cls(times=times, profiles=profiles)        
        
        
    @classmethod
    def create_from_array(cls, data):
        """ collect the data in a single array """
        return cls(times=data[:, 0, 0], profiles=data[:, 1:, :])


    
class RidgeProfile(object):
    """ represents a ridge profile to compare it against an image in fitting """
    
    def __init__(self, size, profile_width=1):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.size = size
        self.ys, self.xs = np.ogrid[-size:size+1, -size:size+1]
        self.width = profile_width
        self.image = None
        
        
    def set_data(self, image, angle):
        """ sets initial data used for fitting
        image denotes the data we compare the model to
        angle defines the direction perpendicular to the profile 
        """
        
        self.image = image - image.mean()
        self.image_std = image.std()
        self._sina = np.sin(angle)
        self._cosa = np.cos(angle)
        
        
    def get_difference(self, distance):
        """ calculates the difference between image and model, when the 
        model is moved by a certain distance in its normal direction """ 
        # determine center point
        px =  distance*self._cosa
        py = -distance*self._sina
        
        # determine the distance from the ridge line
        dist = (self.ys - py)*self._sina - (self.xs - px)*self._cosa
        
        # apply sigmoidal function
        model = np.tanh(dist/self.width)
     
        return np.ravel(self.image - 1.5*self.image_std*model)

