'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory
'''

import logging
import numpy as np

from format.base import VideoBase


class VideoIteratorBase(VideoBase):
    """ class which does not hold its own data, but is more like a view """
     
    def __init__(self, source, size=None, frame_count=None, fps=None, is_color=None):
        # store an iterator of the source video
        self.source = iter(source)
        
        # determine properties of the video
        size = source.size if size is None else size
        frame_count = source.frame_count if frame_count is None else frame_count
        fps = source.fps if fps is None else fps
        is_color = source.is_color if is_color is None else is_color
        
        # initialize the base video
        super(VideoIteratorBase, self).__init__(
            size=size, frame_count=frame_count, fps=fps, is_color=is_color
        )
    
    def set_frame_pos(self, index):
        raise ValueError('Iterators do not allow to seek a specific position')
    
    def __iter__(self):
        return self
    
    def next(self):
        raise NotImplementedError
    


class Crop(VideoIteratorBase):
    """ crops the video to the given rect=(left, top, right, bottom) """
    
    def __init__(self, source, rect):
        
        # interpret float values as fractions
        self.rect = [
            int(rect[0]*source.size[0] if 0 < rect[0] < 1 else rect[0]),
            int(rect[1]*source.size[1] if 0 < rect[1] < 1 else rect[1]),
            int(rect[2]*source.size[0] if 0 < rect[2] < 1 else rect[2]),
            int(rect[3]*source.size[1] if 0 < rect[3] < 1 else rect[3]),
        ]
        size = (self.rect[2] - self.rect[0], self.rect[3] - self.rect[1])
        
        logging.debug('The cropping indices are `%s`', self.rect)
        
        # correct the size, since we are going to crop the movie
        super(Crop, self).__init__(source, size=size)
                
    def next(self):
        r = self.rect
        return self.source.next()[r[0]:r[2], r[1]:r[3]]



class TimeDifference(VideoIteratorBase):
    """ returns the differences between consecutive frames """ 
    
    def __init__(self, source):
        # correct the frame count since we are going to return differences
        super(TimeDifference, self).__init__(source, frame_count=source.frame_count-1)
        # store the first frame, because we always need a previous frame
        self.prev_frame = self.source.next()
                
    def next(self):
        # get this frame and subtract from it the previous one
        this_frame = self.source.next()
        diff = this_frame - self.prev_frame
        
        # this frame will be the previous frame of the next one
        self.prev_frame = this_frame
        
        return diff
  
        

class Monochrome(VideoIteratorBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='normal'):
        self.mode = mode.lower()
        super(Monochrome, self).__init__(source, is_color=False)

    def next(self):
        frame = self.source.next()
        if self.mode == 'normal':
            return np.mean(frame, axis=2)
        elif self.mode == 'r':
            return frame[:, :, 0]
        elif self.mode == 'g':
            return frame[:, :, 1]
        elif self.mode == 'b':
            return frame[:, :, 2]
        else:
            raise ValueError('Unsupported conversion to monochrome: %s' % self.mode)

