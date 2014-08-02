'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling.
'''

from __future__ import division

import os
import numpy as np

# dictionary that maps standard file endings to fourcc codes
# more codes can be found at http://www.fourcc.org/codecs.php
VIDEO_FORMATS = {
    '.xvid': 'XVID',
    '.mov': 'SVQ3',   # standard quicktime codec
    '.mpeg': 'FMP4'   # mpeg 4 variant 
}


class VideoBase(object):
    """
    Base class for video.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=25):
        
        # store number of frames
        self.frame_count = frame_count
        
        # store the dimensions of the movie as width x height in pixel
        self.size = size
        self.fps = fps
        
        # internal pointer to the current frame - might not be used by subclasses
        self._frame_pos = 0
    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return self._frame_pos


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if 0 <= index < self.frame_count:
            self._frame_pos = index
        else:
            raise ValueError('Seeking to frame %d was not possible.' % index)
      
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        raise NotImplementedError

    def __iter__(self):
        """ initializes the iterator """
        # rewind the movie
        self.set_frame_pos(0)
        return self
          
    def next(self):
        """ returns the next frame """
        # this also sets the internal pointer to the next frame
        return self.get_frame(self._frame_pos)

    #===========================================================================
    # WRITE OUT MOVIES
    #===========================================================================
    
    def copy(self):
        """
        Creates a copy of the current video and returns a VideoMemory instance
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        # determine the shape of the required array
        shape = [self.frame_count]
        shape.extend(self.size)
        shape.append(3)
        
        # copy the data into a numpy array
        data = np.empty(shape)
        for k, val in enumerate(self):
            data[k, ...] = val
        
        # construct the copy
        return VideoMemory(data, fps=self.fps)
    
    
    def save(self, filename, video_format=None):
        """
        Saves the video to the file indicated by filename.
        video_format must be a fourcc code from http://www.fourcc.org/codecs.php
            If video_format is None, the code is determined from the filename extension.
        """
        
        # use OpenCV to save the video
        import cv2
        import cv2.cv as cv
        
        if video_format is None:
            # detect format from file ending
            file_ext = os.path.splitext(filename)[1].lower()
            try:
                video_format = VIDEO_FORMATS[file_ext]
            except KeyError:
                raise ValueError('Video format `%s` is unsupported.' % video_format) 
        
        # get the code defining the video format
        fourcc = cv.FOURCC(*video_format)
        out = cv2.VideoWriter(filename, fourcc, self.fps, self.size)

        for frame in self:
            out.write(frame)
            
        out.release()