'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base classes to define an interface and common
functions for video handling.
'''

from __future__ import division

import glob
import logging
import numpy as np


class VideoBase(object):
    """
    Base class for videos.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """

    write_access = False  
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=25, is_color=True):
        """
        size stores the dimensions of the video
        frame_count stores the number of frames
        fps are the frames per second
        is_color indicates whether the video is in color or monochrome
        colordepth indicates how many colors are stored per channel

        TODO: Add flag to movie that indicates whether its writeable or not
        """
        
        # store information about the video
        self.frame_count = frame_count
        self.size = size
        self.fps = fps
        self.is_color = is_color
        
        self._is_iterating = False
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    
    def __str__(self):
        return "%s(size=%s, frame_count=%s, fps=%s, is_color=%s)" % (
                self.__class__.__name__, self.size, self.frame_count,
                self.fps, self.is_color
            )
    
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

    
    def _start_iterating(self):
        """ internal function called when we finished iterating """
        if self._is_iterating:
            raise RuntimeError("Videos cannot be iterated over multiple times "
                               "simultaneously. If you need to do this, make a "
                               "copy of the video before iterating.")
        
        # rewind the movie
        self.set_frame_pos(0)
        self._is_iterating = True
    

    def _end_iterating(self):
        """ internal function called when we finished iterating """
        self._is_iterating = False


    def __iter__(self):
        """ initializes the iterator """
        self._start_iterating()
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
            # stop iterating
            self._end_iterating()            
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
        
        
    def __setitem__(self, key, value):
        """ writes video data to the frame or slice given in key """
        raise ValueError("Writing to this video stream is prohibited.")  
        

    #===========================================================================
    # CONTROL THE DATA STREAM OF THE MOVIE
    #===========================================================================
    
    def copy(self, dtype=np.uint8):
        """
        Creates a copy of the current video and returns a VideoMemory instance.
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        logging.debug('Copy a video stream and store it in memory')
        
        # copy the data into a numpy array
        data = np.empty(self.shape, dtype)
        for k, val in enumerate(self):
            data[k, ...] = np.atleast_3d(val)
            
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
        self._source = source
        
        # determine properties of the video
        size = source.size if size is None else size
        frame_count = source.frame_count if frame_count is None else frame_count
        fps = source.fps if fps is None else fps
        is_color = source.is_color if is_color is None else is_color
        
        # initialize the base video
        super(VideoFilterBase, self).__init__(
            size=size, frame_count=frame_count, fps=fps, is_color=is_color
        )
        
        
    def __str__(self):
        """ delegate the string function to actual source """
        return str(self._source) + ' +' + self.__class__.__name__        

    
    def _start_iterating(self):
        """ internal function called when we starting iterating """
        self._source._start_iterating()
        super(VideoFilterBase, self)._start_iterating()


    def _end_iterating(self):
        """ internal function called when we finished iterating """
        super(VideoFilterBase, self)._end_iterating()
        self._source._end_iterating()

        
    def _filter_frame(self, frame):
        """ returns the frame with a filter applied """
        raise NotImplementedError


    def set_frame_pos(self, index):
        self._source.set_frame_pos(index)
        super(VideoFilterBase, self).set_frame_pos(index)
    
    
    def get_frame(self, index):
        return self._filter_frame(self._source.get_frame(index))
          
                
    def next(self):
        try:
            return self._filter_frame(self._source.next())
        except StopIteration:
            self._end_iterating()
            raise
    
