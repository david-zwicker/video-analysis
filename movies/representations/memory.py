'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling. Concrete implementations are collected in the
backend subpackage.
'''

from __future__ import division

from .base import MovieBase

class MemoryMovie(MovieBase):
    
    def __init__(self, data, fps=25):
        self.data = data
        
        frame_count = data.shape[0]
        size = data.shape[1:3]
        
        super(MemoryMovie, self).__init__(size=size, frame_count=frame_count, fps=fps)
        
        
    def get_frame_raw(self, index):
        frame = self.data[index, :, :, :]
        self._frame_pos = index + 1
        return frame
    

    def get_next_frame_raw(self):
        """ returns the next frame """

        # this also sets the internal pointer to the next frame
        frame = self.get_frame_raw(self._frame_pos)
        return frame


