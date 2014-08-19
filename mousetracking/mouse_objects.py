'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import itertools

import numpy as np
import cv2

from video.analysis.curves import point_distance
import debug



class Object(object):
    """ represents a single object by its position and size """
    __slots__ = ['pos', 'size', 'label'] #< save some memory
    
    def __init__(self, pos, size, label=None):
        self.pos = (int(pos[0]), int(pos[1]))
        self.size = size
        self.label = label



class ObjectTrack(object):
    """ represents a time course of objects """
    # TODO: hold everything inside lists, not list of objects
    # TODO: speed up by keeping track of velocity vectors
    
    array_columns = ['Frame ID', 'Position X', 'Position Y', 'Object Area']
    index_columns = 0
    
    def __init__(self, time=None, obj=None, moving_window=20, moving_threshold=10):
        self.times = [] if time is None else [time]
        self.objects = [] if obj is None else [obj]
        self.moving_window = moving_window
        self.moving_threshold = moving_threshold*moving_window
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(time=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(time=%d..%d)' % (self.times[0], self.times[-1])
        
    def __len__(self):
        return len(self.times)
    
    @property
    def last_pos(self):
        """ return the last position of the object """
        return self.objects[-1].pos
    
    @property
    def last_size(self):
        """ return the last size of the object """
        return self.objects[-1].size
    
    def predict_pos(self):
        """ predict the position in the next frame.
        It turned out that setting the current position is the best predictor.
        This is because mice are often stationary (especially in complicated
        tracking situations, like inside burrows). Additionally, when mice
        throw out dirt, there are frames, where dirt + mouse are considered 
        being one object, which moves the center of mass in direction of the
        dirt. If in the next frame two objects are found, than it is likely
        that the dirt would be seen as the mouse, if we'd predict the position
        based on the continuation of the previous movement
        """
        return self.objects[-1].pos
        
    def get_track(self):
        """ return a list of positions over time """
        return [obj.pos for obj in self.objects]

    def append(self, time, obj):
        """ append a new object with a time code """
        self.times.append(time)
        self.objects.append(obj)
        
    def is_moving(self):
        """ return if the object has moved in the last frames """
        pos = self.objects[-1].pos
        dist = sum(point_distance(pos, obj.pos)
                   for obj in self.objects[-self.moving_window:])
        return dist > self.moving_threshold
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        return np.array([(time, obj.pos[0], obj.pos[1], obj.size)
                         for time, obj in itertools.izip(self.times, self.objects)],
                        dtype=np.int32)

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        res = cls()
        res.times = [d[0] for d in data]
        res.objects = [Object(pos=(d[1], d[2]), size=d[3]) for d in data]
        return res



class GroundProfile(object):
    """ dummy class representing a single ground profile at a certain point
    in time """
    
    array_columns = ['Time', 'Position X', 'Position Y']
    index_columns = 1
    
    def __init__(self, time, points):
        self.time = time
        self.points = points
        
    def __repr__(self):
        return 'GroundProfile(time=%d, %d points)' % (self.time, len(self.points))
        
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        time_array = np.zeros((len(self.points), 1), np.int32) + self.time
        return np.hstack((time_array, self.points))

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        data = np.asarray(data)
        return cls(data[0, 0], data[1:, :])
   
   
   
class Burrow(object):
    """ represents a single burrow to compare it against an image in fitting """
    
    array_columns = ['Time', 'Position X', 'Position Y']
    index_columns = 0 #< there could be multiple burrows at each time point
    # Hence, time can not be used as an index
    
    def __init__(self, outline, time=None, image=None):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.outline = outline
        self.image = image
        self.time = time

        
    def __len__(self):
        return len(self.outline)
        
        
    def get_centerline(self):
        raise NotImplementedError
    
        
    def adjust_outline(self, deviations):
        """ adjust the current outline by moving points perpendicular by
        a distance given by `deviations` """
        pass
    
        
    def get_difference(self, deviations):
        """ calculates the difference between image and model, when the 
        model is moved by a certain distance in its normal direction """
        raise NotImplementedError 
        
        dist = 1
           
        # apply sigmoidal function
        model = np.tanh(dist/self.width)
     
        return np.ravel(self.image_mean + 1.5*self.image_std*model - self.image)
    
    
    def show_image(self, mark_points):
        # draw burrow
        image = self.image.copy()
        cv2.drawContours(image, np.array([self.outline], np.int32), -1, 255, 1)
        for k, p in enumerate(self.outline):
            color = 255 if mark_points[k] else 128
            cv2.circle(image, (int(p[0]), int(p[1])), 3, color, thickness=-1)
        debug.show_image(image)
    
    
    def to_array(self):
        """ converts the internal representation to a single array """
        self.outline = np.asarray(self.outline)
        time_array = np.zeros((len(self.outline), 1), np.int32) + self.time
        return np.hstack((time_array, self.outline))


    @classmethod
    def from_array(cls, data):
        data = np.asarray(data)
        return cls(outline=data[1:, :], time=data[0, 0])
        
    
    
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


    