'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base classes to define an interface and common
functions for video handling.
'''

from __future__ import division

import glob
import numpy as np


class VideoBase(object):
    """
    Base class for videos.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=25, is_color=True):
        """
        size stores the dimensions of the video
        frame_count stores the number of frames
        fps are the frames per second
        is_color indicates whether the video is in color or monochrome
        colordepth indicates how many colors are stored per chanel
        """
        
        # store information about the video
        self.frame_count = frame_count
        self.size = size
        self.fps = fps
        self.is_color = is_color
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
    def __len__(self):
        return self.frame_count
    
    
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        return (
            self.frame_count,
            self.size[0],
            self.size[1],
            3 if self.is_color else 1
        )
    
    
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return self._frame_pos


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if 0 <= index < self.frame_count:
            self._frame_pos = index
        else:
            raise IndexError('Seeking to frame %d was not possible.' % index)

      
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        raise NotImplementedError


    def __iter__(self):
        """ initializes the iterator """
        # rewind the movie
        self.set_frame_pos(0)
        return self

          
    def next(self):
        """
        returns the next frame while iterating over a movie.
        This is a generic function, which just retrieves the next image based
        on the internal frame_pos. Subclasses may overwrite this with more
        efficient implementations (i.e. for streaming)
        """
        # retrieve current frame
        try:
            frame = self.get_frame(self._frame_pos)
        except IndexError:
            raise StopIteration

        # set the internal pointer to the next frame
        self._frame_pos += 1
        return frame


    def __getitem__(self, key):
        """ returns a single frame or a video corresponding to a slice """ 
        if isinstance(key, slice):
            # prevent circular import by lazy importing
            from .time_slice import VideoSlice
            return VideoSlice(self, *key.indices(self.frame_count))
        
        elif isinstance(key, int):
            return self.get_frame(key)
        
        else:
            raise TypeError("Invalid key for indexing")
        

    #===========================================================================
    # CONTROL THE DATA STREAM OF THE MOVIE
    #===========================================================================
    
    def copy(self, dtype=np.uint8):
        """
        Creates a copy of the current video and returns a VideoMemory instance
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        # copy the data into a numpy array
        data = np.empty(self.shape, dtype)
        for k, val in enumerate(self):
            data[k, ...] = val
        
        # construct the memory object without copying the data
        return VideoMemory(data, fps=self.fps, copy_data=False)
    
    


class VideoImageStackBase(VideoBase):
    """ abstract base class that represents a movie stored as individual frame images """
    
    def __init__(self, filename_scheme, fps=None):
        # find all the files belonging to this stack
        self.filenames = sorted(glob.glob(filename_scheme))
        frame_count = len(self.filenames)
        
        # load the first frame to get information
        frame = self.get_frame(0)
        size = frame.shape[:2]
        if frame.shape[3] == 1:
            is_color = False
        elif frame.shape[3] == 3:
            is_color = True
        else:
            raise ValueError('The last dimension of the data must be either 1 or 3.')
                
        super(VideoImageStackBase, self).__init__(size=size, frame_count=frame_count,
                                                  fps=fps, is_color=is_color)
        
        

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

    def set_frame_pos(self, index):
        self._source.set_frame_pos(0)
    
    def get_frame(self, index):
        return self._filter_frame(self._source.get_frame(index))
                
    def next(self):
        return self._filter_frame(self._source.next())
    
