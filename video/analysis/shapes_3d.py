'''
Created on Nov 5, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np



class Cuboid(object):
    """ class that represents a cuboid in n dimensions """
    
    def __init__(self, pos, size):
        self.pos = np.asarray(pos)
        self.size = np.asarray(size)
        assert len(self.pos) == len(self.size)
        
    @classmethod
    def from_points(cls, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return cls(np.minimum(p1, p2), np.abs(p1 - p2))
    
    @classmethod
    def from_centerpoint(cls, centerpoint, size):
        centerpoint = np.asarray(centerpoint)
        size = np.asarray(size)
        return cls(centerpoint - size/2, size)
    
    def copy(self):
        return self.__class__(self.pos, self.size)
        
    def __repr__(self):
        return "%s(pos=%s, size=%s)" % (self.__class__.__name__, self.pos,
                                        self.size)
            
    def set_corners(self, p1, p2):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        self.pos = np.minimum(p1, p2)
        self.size = np.abs(p1 - p2)

    @property
    def bounds(self):
        return [(p, p + s) for p, s in zip(self.pos, self.size)]
            
    @property
    def corners(self):
        return self.pos, self.pos + self.size
    @corners.setter
    def corners(self, ps):
        self.set_corners(ps[0], ps[1])

    @property
    def dimension(self):
        return len(self.pos)
        
    @property
    def slices(self):
        return [slice(int(p), int(p + s)) for p, s in zip(self.pos, self.size)]

    @property
    def centroid(self):
        return [p + s/2 for p, s in zip(self.pos, self.size)]
    
    @property
    def volume(self):
        return np.prod(self.size)
    

    def translate(self, distance=0, inplace=True):
        """ translates the cuboid by a certain distance in all directions """
        distance = np.asarray(distance)
        if inplace:
            self.pos += distance
            return self
        else:
            return self.__class__(self.pos + distance, self.size)
    
            
    def buffer(self, amount=0, inplace=True):
        """ dilate the cuboid by a certain amount in all directions """
        amount = np.asarray(amount)
        if inplace:
            self.pos -= amount
            self.size += 2*amount
            return self
        else:
            return self.__class__(self.pos - amount, self.size + 2*amount)
    

    def scale(self, factor=1, inplace=True):
        """ scale the cuboid by a certain amount in all directions """
        factor = np.asarray(factor)
        if inplace:
            self.pos *= factor
            self.size *= factor
            return self
        else:
            return self.__class__(self.pos * factor, self.size * factor)
    

