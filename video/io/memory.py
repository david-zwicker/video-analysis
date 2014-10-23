'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This package provides class definitions for describing videos
that are stored in memory using numpt arrays
'''

from __future__ import division

import numpy as np

from .base import VideoBase


class VideoMemory(VideoBase):
    """
    class which holds all the video data in memory.
    We allow for direct manipulation of the data attribute in order to make
    reading and writing data convenient.
    """
    
    write_access = True
    
    def __init__(self, data, fps=25, copy_data=True):
        
        # only copy the _data if requested or required
        self.data = np.array(data, copy=copy_data)
        
        # remove the color dimension if it is single
        if self.data.ndim > 3 and self.data.shape[3] == 1:
            self.data = np.squeeze(self.data, 3) 

        # read important information
        frame_count = data.shape[0]
        size = (data.shape[2], data.shape[1])
        if data.ndim == 3:
            is_color = False
        elif data.shape[3] == 3:
            is_color = True
        else:
            raise ValueError('The last dimension of the data must be either 1 or 3.')
        
        super(VideoMemory, self).__init__(size=size, frame_count=frame_count,
                                          fps=fps, is_color=is_color)
        
        
    def get_frame(self, index):
        return self.data[index]


    def __getitem__(self, key):
        return self.data[key]  
        

    def __setitem__(self, key, value):
        """ writes video data to the frame or slice given in key """
        # delegate the writing to the data directly
        self.data[key] = value  
        
        
        
class VideoMemoryBuffer(VideoBase):
    """ class which receives frames from another video and holds them in memory
    until they are consumed. This class thus acts as a video buffer """
    pass

