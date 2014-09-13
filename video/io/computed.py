'''
Created on Aug 5, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from .base import VideoBase
from video.utils import safe_typecast


class VideoGaussianNoise(VideoBase):
    """ class that creates Gaussian noise for each frame """
    
    def __init__(self, frame_count, size, mean=0, std=1, fps=None, is_color=False, dtype=None):
        
        self.mean = mean
        self.std = std
        self.dtype = dtype
        
        super(VideoGaussianNoise, self).__init__(size=size, frame_count=frame_count,
                                                 fps=fps, is_color=is_color)
        
        self._frame_shape = self.shape[1:]
        
        
    def get_frame(self, index):
        if index >= self.frame_count:
            raise IndexError
        else:
            frame = self.mean + self.std*np.random.randn(*self._frame_shape)
            if self.dtype is None:
                return frame
            else:
                return safe_typecast(frame, self.dtype)
        