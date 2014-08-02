'''
Created on Jul 31, 2014

@author: zwicker

This package contains functions which modify video.
Typically, these functions take a single movie, process it, and return a MovieMemory
'''

from __future__ import division

import numpy as np

from .format import VideoMemory


def frame_differences(movie):
    """ subtracts consecutive frames and returns a new movie with one frame less """
    
    # build empty array to store the data to
    shape = [movie.frame_count - 1]
    shape.extend(movie.size)
    shape.append(3)
    data = np.empty(shape)
    
    last_frame = movie.get_frame(0)
    for k, frame in enumerate(movie):
        data[k, :, :, :] = frame - last_frame
        last_frame = frame
        
    return VideoMemory(data, fps=movie.fps)