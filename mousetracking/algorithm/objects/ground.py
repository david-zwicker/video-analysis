'''
Created on Aug 28, 2014

@author: zwicker

Holds classes useful for describing and fitting the ground profile
'''

from __future__ import division

import numpy as np
from scipy import ndimage

from .utils import cached_property, LazyHDFValue
from video.analysis import curves


class GroundProfile(object):
    """ class representing a single ground profile
    FIXME: This class should not know about time """

    hdf_attributes = {'column_names': ('Position X', 'Position Y')}
    storage_class = LazyHDFValue
    
    def __init__(self, time, points):
        self.time = time
        self.points = points
        
    def __repr__(self):
        return 'GroundProfile(time=%d, %d points)' % (self.time, len(self.points))
    
    def __len__(self):
        return len(self.points)
        
    @cached_property
    def length(self):
        """ returns the length of the profile """
        return curves.curve_length(self.points)
    
    def make_equidistant(self, num_points):
        """ makes the ground profile equidistant """
        self.points = curves.make_curve_equidistant(self.points, count=num_points)
        
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        time_array = np.zeros((len(self.points), 1), np.int32) + self.time
        return np.hstack((time_array, self.points))

    @classmethod
    def create_from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        data = np.asarray(data)
        return cls(data[0, 0], data[:, 1:])
   
   

class GroundProfileList(list):
    """ organizes a list of ground profiles """
    hdf_attributes = {'column_names': ('Time', 'Position X', 'Position Y')}
    storage_class = LazyHDFValue
   
    def to_array(self):
        if len(self) == 0:
            return []
        else:
            return np.concatenate([obj.to_array() for obj in self])
    
    
    @classmethod
    def create_from_array(cls, value):
        result = cls()
        index, obj_data = None, None
        # iterate over the data and create objects from it
        for line in value:
            if line[0] == index:
                # append object to the current track
                obj_data.append(line)
            else:
                # save the track and start a new one
                if obj_data:
                    result.append(GroundProfile.create_from_array(obj_data))
                obj_data = [line]
                index = line[0]
        
        if obj_data:
            result.append(GroundProfile.create_from_array(obj_data))
            
        return result
   


class GroundProfileTrack(object):
    """ class holding the ground profile information for the entire video.
    For efficient data storage the ground profiles are re-parameterized
    to have the same number of support points.
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
        
        
    def __len__(self):
        return len(self.times)
    
    
    def __repr__(self):
        return '%s(frames=%d, points=%d)' % (self.__class__.__name__,
                                             self.profiles.shape[0],
                                             self.profiles.shape[1])
    
    
    def smooth(self, sigma):
        """ smoothes the profiles in time using a Gaussian window of given sigma """
        # convolve each point with a gaussian filter
        self.profiles = ndimage.filters.gaussian_filter1d(self.profiles,
                                                          sigma,  #< std of the filter
                                                          axis=0, #< time axis
                                                          mode='nearest')


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
        # determine the maximal number of points
        num_points = max(len(profile) for profile in ground_profiles)

        # iterate through all profiles and convert them to have equal number of points
        # and store the data
        times, profiles = [], []
        for profile in ground_profiles:
            points = curves.make_curve_equidistant(profile.points, count=num_points)
            times.append(profile.time)
            profiles.append(points)

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

