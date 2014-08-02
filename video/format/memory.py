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
        # copy data if requested
        if copy_data:
            self.data = data[:]
        else:
            self.data = data

        # read important information
        frame_count = data.shape[0]
        size = data.shape[1:3]
        
        super(VideoMemory, self).__init__(size=size, frame_count=frame_count, fps=fps)
        
        
    def get_frame(self, index):
        frame = self.data[index, :, :, :]
        self._frame_pos = index + 1
        return frame

