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
    """ class which holds all the video data in memory """ 
    
    def __init__(self, data, fps=25, copy_data=False):
        
        # only copy the data if requested or required
        self.data = np.array(data, copy=copy_data, ndmin=3) 

        # read important information
        frame_count = data.shape[0]
        size = data.shape[1:3]
        is_color = (data.ndim == 4)
        
        super(VideoMemory, self).__init__(size=size, frame_count=frame_count, fps=fps, is_color=is_color)
        
        
    def get_frame(self, index):
        try:
            frame = self.data[index]
        except IndexError:
            raise StopIteration
        self._frame_pos = index + 1
        return frame

