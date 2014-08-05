'''
Created on Aug 5, 2014

@author: zwicker
'''

import numpy as np

from .base import VideoBase


class VideoGaussianNoise(VideoBase):
    """ class that creates Gaussian noise for each frame """
    
    def __init__(self, frame_count, size, mean=0, std=1, fps=None, is_color=False):
        
        self.mean = mean
        self.std = std
        
        super(VideoGaussianNoise, self).__init__(size=size, frame_count=frame_count,
                                                 fps=fps, is_color=is_color)
        
        self._frame_shape = self.shape[1:]
        
        
    def get_frame(self, index):
        if index >= self.frame_count:
            raise IndexError
        else:
            return self.mean + self.std*np.random.randn(*self._frame_shape)
        