'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory.

Some filters allow to access the underlying frames at random using the get_frame method

'''

import logging
import numpy as np

from formats.base import VideoBase

class VideoFilterBase(VideoBase):
    """ class which does not hold its own data, but is more like a view """
     
    def __init__(self, source, size=None, frame_count=None, fps=None, is_color=None):
        # store an iterator of the source video
        self._source = iter(source)
        
        # determine properties of the video
        size = source.size if size is None else size
        frame_count = source.frame_count if frame_count is None else frame_count
        fps = source.fps if fps is None else fps
        is_color = source.is_color if is_color is None else is_color
        
        # initialize the base video
        super(VideoFilterBase, self).__init__(
            size=size, frame_count=frame_count, fps=fps, is_color=is_color
        )
        
    def _filter_frame(self, frame):
        """ returns the frame with a filter applied """
        raise NotImplementedError
    
    def __iter__(self):
        self._source.set_frame_pos(0)
        return self
    
    def get_frame(self, index):
        return self._filter_frame(self._source.get_frame(index))
                
    def next(self):
        return self._filter_frame(self._source.next())
    


#===============================================================================
# FILTERS THAT ALLOW SEEKING IN THE MATERIAL
#===============================================================================


class TimeSlice(VideoFilterBase):
    """ iterates only over part of the frames """
    
    def __init__(self, source, start=0, stop=-1, step=1):
        
        # interpret negative indices as counting from the end of the video
        if start < 0:
            self._start = source.frame_count + start
        else:
            self._start = start
            
        if stop < 0:
            self._stop = source.frame_count + stop
        else:
            self._stop = stop
            
        self._step = step
            
        # calculate the number of frames to be expected
        frame_count = int(np.ceil((self._stop - self._start)/self._step)) 

        # correct the size, since we are going to crop the movie
        super(TimeSlice, self).__init__(source, frame_count=frame_count)

        
    def set_frame_pos(self, index):
        frame_index = self._start + index*self._step
        if not self._start <= frame_index < self._end:
            raise IndexError 
        self._source.set_frame_pos(frame_index)
        
        
    def get_frame(self, index):
        frame_index = self._start + index*self._step
        if not self._start <= frame_index < self._end:
            raise IndexError 
        self._source.get_frame(frame_index)
        
        
    def next(self):
        if self._step == 1:
            # check whether we are already beyond our end
            if self.get_frame_pos() >= self._end:
                raise StopIteration
        else:
            # advance to the next frame using step
            try:
                self.set_frame_pos(self._frame_pos)
            except IndexError:
                raise StopIteration

        # return the next frame in question
        return self._source.next()



class Crop(VideoFilterBase):
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
        
        logging.debug('The cropping indices are %s', self.rect)
        
        # correct the size, since we are going to crop the movie
        super(Crop, self).__init__(source, size=size)
       
    def _filter_frame(self, frame):
        r = self.rect
        return frame[r[0]:r[2], r[1]:r[3]]



class Monochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='normal'):
        self.mode = mode.lower()
        super(Monochrome, self).__init__(source, is_color=False)

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
            raise ValueError('Unsupported conversion to monochrome: %s' % self.mode)
    

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
  
        
