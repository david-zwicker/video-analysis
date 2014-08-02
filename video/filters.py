'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory.

Some filters allow to access the underlying frames at random using the get_frame method

'''

import logging
import numpy as np

from .formats.base import VideoFilterBase


#===============================================================================
# FILTERS THAT ALLOW SEEKING IN THE MATERIAL
#===============================================================================


class Crop(VideoFilterBase):
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
        super(Crop, self).__init__(source, size=size)

        logging.debug('Created filter for cropping to rectangle %s', self.rect)
        
       
    def _filter_frame(self, frame):
        r = self.rect
        return frame[r[0]:r[0] + r[2], r[1]:r[1] + r[3]]



class Monochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='normal'):
        self.mode = mode.lower()
        super(Monochrome, self).__init__(source, is_color=False)

        logging.debug('Created filter for converting video to monochrome with method `%s`', mode)

    def _filter_frame(self, frame):
        if self.mode == 'normal':
            return np.mean(frame, axis=2)
        elif self.mode == 'r':
            return frame[:, :, 0]
        elif self.mode == 'g':
            return frame[:, :, 1]
        elif self.mode == 'b':
            return frame[:, :, 2]
        else:
            raise ValueError('Unsupported conversion method to monochrome: %s' % self.mode)
    

#===============================================================================
# FILTERS THAT CAN ONLY BE USED AS ITERATORS
#===============================================================================


class TimeDifference(VideoFilterBase):
    """
    returns the differences between consecutive frames.
    Here, frames cannot be accessed directly, but one can only iterate over the video
    """ 
    
    def __init__(self, source):
        # correct the frame count since we are going to return differences
        super(TimeDifference, self).__init__(source, frame_count=source.frame_count-1)
        # store the first frame, because we always need a previous frame
        self.prev_frame = self._source.next()

        logging.debug('Created filter for calculating differences between consecutive frames.')
    
    def set_frame_pos(self, index):
        raise ValueError('Iterators do not allow to seek a specific position')
      
    def get_frame(self, index):
        raise ValueError('Iterators do not allow to seek a specific position')
    
    def next(self):
        # get this frame and subtract from it the previous one
        this_frame = self._source.next()
        diff = this_frame - self.prev_frame
        
        # this frame will be the previous frame of the next one
        self.prev_frame = this_frame
        
        return diff
  
        
