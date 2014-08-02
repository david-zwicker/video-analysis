'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for describing videos
that are stored in memory using numpt arrays
'''

from __future__ import division

import numpy as np

from .base import VideoBase


class VideoMemory(VideoBase):
    """ class which holds all the video _data in memory """ 
    
    def __init__(self, data, fps=25, copy_data=False):
        
        # only copy the _data if requested or required
        self._data = np.array(data, copy=copy_data, ndmin=4) 

        # read important information
        frame_count = data.shape[0]
        size = data.shape[1:3]
        if data.shape[3] == 1:
            is_color = False
        elif data.shape[3] == 3:
            is_color = True
        else:
            raise ValueError('The last dimension of the _data must be either 1 or 3.')
        
        super(VideoMemory, self).__init__(size=size, frame_count=frame_count,
                                          fps=fps, is_color=is_color)
        
        
    def get_frame(self, index):
        return self._data[index]

