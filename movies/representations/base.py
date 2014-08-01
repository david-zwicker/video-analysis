'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling. Concrete implementations are collected in the
backend subpackage.
'''

from __future__ import division

import os

# dictionary that maps standard file endings to fourcc codes
# more codes can be found at http://www.fourcc.org/codecs.php
MOVIE_FORMATS = {
    '.xvid': 'XVID',
    '.mov': 'SVQ3',   # standard quicktime codec
    '.mpeg': 'FMP4'   # mpeg 4 variant 
}


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
    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
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


    #===========================================================================
    # WRITE OUT MOVIES
    #===========================================================================
    
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
            file_ext = os.path.splitext(filename)[1].tolower()
            try:
                video_format = MOVIE_FORMATS[file_ext]
            except KeyError:
                raise ValueError('Video format `%s` is unsupported.' % video_format) 
        
        # get the code defining the video format
        fourcc = cv.FOURCC(*video_format)
        out = cv2.VideoWriter(filename, fourcc, self.fps, self.size)

        for frame in self:
            out.write(frame)
            
        out.release()