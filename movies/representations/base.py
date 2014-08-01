'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling. Concrete implementations are collected in the
backend subpackage.
'''

from __future__ import division

class MovieBase(object):
    """
    Base class for movies.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=25):
        
        # store number of frames
        self.frame_count = frame_count
        
        # store the dimensions of the movie as width x height in pixel
        self.real_size = size
        self.fps = fps
        
        self.crop = None # rectangle for cropping the movie
        
        # internal pointer to the current frame - might not be used by subclasses
        self._frame_pos = 0
    
    
    @property
    def size(self):
        """ Returns the movie size, taking potential cropping into account """
        if self.crop is None:
            return self.real_size
        else:
            return self.crop[2:]
    
   
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return self._frame_pos


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if 0 <= index < self.frame_count:
            self._frame_pos = index
        else:
            raise ValueError('Seeking to frame %d was not possible.' % index)
   
   
    def __iter__(self):
        # rewind the movie
        self.set_frame_pos(0)
        return self
          
   
    def _process_frame(self, frame):
        """ processes the raw data of a frame if necessary """
        
        # crop frame if requested
        if self.crop is not None:
            frame = frame[self.crop[0]:self.crop[2], self.crop[1]:self.crop[3], :]
            
        return frame  
    
    
    def get_next_frame_raw(self):
        raise NotImplementedError

    def next(self):
        """ returns the next frame """
        return self._process_frame(self.get_next_frame_raw())
   
   
    def get_frame_raw(self, index):
        raise NotImplementedError
    
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        return self._process_frame(self.get_frame_raw(index))
