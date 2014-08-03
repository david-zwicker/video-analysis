'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory.

Some filters allow to access the underlying frames at random using the get_frame method

'''

from __future__ import division

import logging
import numpy as np

from .io.base import VideoFilterBase
from .io.utils import get_color_range


class FilterNormalize(VideoFilterBase):
    """ normalizes a color range to the interval 0..1 """
    
    def __init__(self, source, vmin=0, vmax=1, dtype=None):
        """
        warning:
        vmin must not be smaller than the smallest value source can hold.
        Otherwise wrapping can occur. The same thing holds for vmax, which
        must not be larger than the maximum value in the color channels.
        """
        
        # interval From which to convert 
        self._fmin = vmin
        self._fmax = vmax
        
        # interval To which to convert
        self._dtype = dtype
        self._tmin = None
        self._alpha = None
        
        super(FilterNormalize, self).__init__(source)
        logging.debug('Created filter for normalizing range [%g..%g]', vmin, vmax)


    def _filter_frame(self, frame):
        
        # ensure that we decided on a dtype
        if self._dtype is None:
            self._dtype = frame.dtype
            
        # ensure that we know the bounds of this dtype
        if self._tmin is None:
            self._tmin, tmax = get_color_range(self._dtype)
            self._alpha = (tmax - self._tmin)/(self._fmax - self._fmin)
            
            # some safety checks on the first run:
            fmin, fmax = get_color_range(frame.dtype)
            if self._fmin < fmin:
                logging.warn('Lower normalization bound is below what the format can hold.')
            if self._fmax > fmax:
                logging.warn('Upper normalization bound is above what the format can hold.')

        # clip the data before converting
        np.clip(frame, self._fmin, self._fmax, out=frame)

        # do the conversion from [fmin, fmax] to [tmin, tmax]
        frame = (frame - self._fmin)*self._alpha + self._tmin
        
        # cast the data to the right type
        return frame.astype(self._dtype)



class FilterCrop(VideoFilterBase):
    """ crops the video to the given rect=(top, left, height, width) """
    
    def __init__(self, source, rect):
        """ initialized the filter that crops to the given rect=(top, left, height, width) """
        
        def _check_number(value, max_value):
            """ helper function checking the bounds of the rectangle """
            
            # convert to integer by interpreting float values as fractions
            value = int(value*max_value if -1 < value < 1 else value)
            
            # interpret negative numbers as counting from opposite boundary
            if value < 0:
                value += max_value
                
            # check whether the value is within bounds
            if not 0 <= value < max_value:
                raise IndexError('Cropping rectangle reaches out of frame.')
            
            return value
            
        # interpret float values as fractions
        self.rect = [
            _check_number(rect[0], source.size[0]),
            _check_number(rect[1], source.size[1]),
            _check_number(rect[2], source.size[0]),
            _check_number(rect[3], source.size[1]),
        ]
        
        size = (self.rect[2], self.rect[3])
        
        # correct the size, since we are going to crop the movie
        super(FilterCrop, self).__init__(source, size=size)

        logging.debug('Created filter for cropping to rectangle %s', self.rect)
        
       
    def _filter_frame(self, frame):
        r = self.rect
        return frame[r[0]:r[0] + r[2], r[1]:r[1] + r[3]]



class FilterMonochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='normal'):
        self.mode = mode.lower()
        super(FilterMonochrome, self).__init__(source, is_color=False)

        logging.debug('Created filter for converting video to monochrome with method `%s`', mode)

    def _filter_frame(self, frame):
        if self.mode == 'normal':
            return np.mean(frame, axis=2).astype(frame.dtype)
        elif self.mode == 'r':
            return frame[:, :, 0]
        elif self.mode == 'g':
            return frame[:, :, 1]
        elif self.mode == 'b':
            return frame[:, :, 2]
        else:
            raise ValueError('Unsupported conversion method to monochrome: %s' % self.mode)
    
    

class FilterTimeDifference(VideoFilterBase):
    """
    returns the differences between consecutive frames.
    This filter is best used by just iterating over it. Retrieving individual
    frame differences can be a bit slow, since two frames have to be loaded.
    """ 
    
    def __init__(self, source, dtype=np.int16):
        """
        dtype contains the dtype that is used to calculate the difference.
        If dtype is None, no type casting is done.
        """
        
        self._dtype = dtype
        
        # correct the frame count since we are going to return differences
        super(FilterTimeDifference, self).__init__(source, frame_count=source.frame_count-1)

        logging.debug('Created filter for calculating differences between consecutive frames.')
    
    
    def set_frame_pos(self, index):
        # set the underlying movie to requested position 
        self._source.set_frame_pos(index)
        # advance one frame and save it in the previous frame structure
        self.prev_frame = self._source.next()
    
      
    def get_frame(self, index):
        this_frame = self._source.get_frame(index + 1)
        # cast into different dtype if requested
        if self._dtype is not None:
            this_frame = this_frame.astype(self._dtype) 
        return this_frame - self._source.get_frame(index) 
    
    
    def next(self):
        # get this frame ...
        this_frame = self._source.next()
        # ... cast into different dtype if requested ...
        if self._dtype is not None:
            this_frame = this_frame.astype(self._dtype) 
        # .. and subtract from it the previous one
        diff = this_frame - self.prev_frame
        
        # this frame will be the previous frame of the next one
        self.prev_frame = this_frame
        
        return diff
  
        
