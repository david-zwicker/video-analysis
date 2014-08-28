'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import itertools

import numpy as np

from video.analysis import curves

from ..debug import *



class MovingObject(object):
    """ represents a single object by its position and size """
    __slots__ = ['pos', 'size', 'label'] #< save some memory
    
    def __init__(self, pos, size, label=None):
        self.pos = (int(pos[0]), int(pos[1]))
        self.size = size
        self.label = label
        
    def __repr__(self):
        if self.label:
            return 'MovingObject((%d, %d), %d, %s)' % (self.pos + (self.size, self.label))
        else:
            return 'MovingObject((%d, %d), %d)' % (self.pos + (self.size,))



class ObjectTrack(object):
    """ represents a time course of objects """
    # TODO: hold everything inside lists, not list of objects
    # TODO: speed up by keeping track of velocity vectors
    
    array_columns = ['Time', 'Position X', 'Position Y', 'Object Area']
    
    def __init__(self, time=None, obj=None, moving_window=20, moving_threshold=10):
        self.times = [] if time is None else [time]
        self.objects = [] if obj is None else [obj]
        self.moving_window = moving_window
        self.moving_threshold = moving_threshold*moving_window
        
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(span=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(span=%d..%d)' % (self.times[0], self.times[-1])
        
        
    def __len__(self): return len(self.times)
    def __getitem__(self, *args): return self.objects.__getitem__(*args)
    
    
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
        dist = sum(curves.point_distance(pos, obj.pos)
                   for obj in self.objects[-self.moving_window:])
        return dist > self.moving_threshold
    
    
    def is_concurrent(self, other):
        """ returns True if the other ObjectTrack overlaps with the current one """
        s0, s1 = self.time[0], self.time[-1]
        o0, o1 = other[0], other[-1]
        return (s0 <= o1 and o0 <= s1)
    
    
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
        res.objects = [MovingObject(pos=(d[1], d[2]), size=d[3]) for d in data]
        return res


    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        hdf_file.create_dataset(key, data=self.to_array())
        hdf_file[key].attrs['column_names'] = self.array_columns


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.from_array(hdf_file[key])
        


   
    