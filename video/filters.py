'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory
'''

from format.base import VideoBase


class VideoIteratorBase(VideoBase):
    """ class which does not hold its own data, but is more like a view """
     
    def __init__(self, source):
        self.source = iter(source)
        super(VideoIteratorBase, self).__init__(
            size=source.size, frame_count=source.frame_count,
            fps=source.fps
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
        self.rect = rect
        super(Crop, self).__init__(source)
                
    def next(self):
        r = self.rect
        return self.source.next()[r[0]:r[2], r[1]:r[3], :]


class TimeDifference(VideoIteratorBase):
    """ returns the differences between consecutive frames """ 
    
    def __init__(self, source, rect):
        self.rect = rect
        super(TimeDifference, self).__init__(source)
        
        self.frame_count -= 1
        self.last_frame = self.source.next()
                
    def next(self):
        this_frame = self.source.next()
        diff = this_frame - self.last_frame
        self.last_frame = this_frame
        
        return diff
        
        
class NormalizeBrightness(VideoIteratorBase):
    """
    adjusts individual frames such that their brightness corresponds to
    the initial frame
    """ 
    raise NotImplementedError

