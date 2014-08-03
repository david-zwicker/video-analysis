'''
Created on Aug 2, 2014

@author: zwicker
'''

from __future__ import division

import logging
import numpy as np

from .base import VideoFilterBase


class VideoSlice(VideoFilterBase):
    """ iterates only over part of the frames """
    
    def __init__(self, source, start=0, stop=None, step=1):
        
        if step == 0:
            raise ValueError('step argument must not be zero.')
        
        self._start = start
        self._stop = self.source.frame_count if stop is None else stop 
        self._step = step
            
        # calculate the number of frames to be expected
        frame_count = int(np.ceil((self._stop - self._start)/self._step))
        
        # correct the size, since we are going to crop the movie
        super(VideoSlice, self).__init__(source, frame_count=frame_count)

        logging.debug('Created video slice (%d, %d, %d) of length %d.' % 
                      (self._start, self._stop, self._step, frame_count)) 
        if step < 0:
            logging.warn('Reversing a video can slow down the processing significantly.')
        
        
    def set_frame_pos(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)
        self._source.set_frame_pos(self._start + index*self._step)
        
        
    def get_frame(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)

        return self._source.get_frame(self._start + index*self._step)
        
        
    def next(self):
        # check whether we reached the end
        if self.get_frame_pos() >= self.frame_count:
            raise StopIteration
        
        if self._step == 1:
            # return the next frame in question
            frame = self._source.next()
        else:
            # return the specific frame in question
            frame = self.get_frame(self._frame_pos)

        # advance to the next frame
        self._frame_pos += 1
        return frame
    