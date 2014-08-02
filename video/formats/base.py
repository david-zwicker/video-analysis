'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base class to define an interface and common
functions for video handling.
'''

from __future__ import division

import glob
import os
import platform
import numpy as np
import logging

# dictionary that maps standard file endings to fourcc codes
# more codes can be found at http://www.fourcc.org/codecs.php
if platform.system() == 'Darwin':
    VIDEO_FORMATS = {
        '.xvid': 'XVID',
        '.mov': 'mp4v',   # standard quicktime codec - tested
        '.mpeg': 'FMP4',  # mpeg 4 variant
        '.avi': 'IYUV',   # uncompressed avi - tested
    }
else:
    VIDEO_FORMATS = {
        '.xvid': 'XVID',
        '.mov': 'mp4v', #'SVQ3',   # standard quicktime codec
        '.mpeg': 'FMP4',  # mpeg 4 variant
        '.avi': 'IYUV',   # uncompressed avi 
    }


class VideoBase(object):
    """
    Base class for videos.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=25, is_color=True):
        
        # store number of frames
        self.frame_count = frame_count
        
        # store the dimensions of the movie as width x height in pixel
        self.size = size
        self.fps = fps
        self.is_color = is_color
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
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
        # retrieve current frame
        try:
            frame = self.get_frame(self._frame_pos)
        except IndexError:
            raise StopIteration

        # set the internal pointer to the next frame
        self._frame_pos += 1
        return frame

    #===========================================================================
    # CONTROL THE DATA STREAM OF THE MOVIE
    #===========================================================================
    
    def copy(self):
        """
        Creates a copy of the current video and returns a VideoMemory instance
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        # copy the data into a numpy array
        data = np.empty(self.shape)
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
        
        if video_format is None:
            # detect format from file ending
            file_ext = os.path.splitext(filename)[1].lower()
            try:
                video_format = VIDEO_FORMATS[file_ext]
            except KeyError:
                raise ValueError('Video format `%s` is unsupported.' % video_format) 
        
        # get the code defining the video format
        logging.info('Start writing video with format `%s`', video_format)
        fourcc = cv2.cv.FOURCC(*video_format)
        out = cv2.VideoWriter(filename, fourcc=fourcc, fps=self.fps,
                              frameSize=self.size, isColor=self.is_color)

        # write out all individual frames
        for frame in self:
            out.write(np.asarray(frame, np.uint8))
            
        out.release()
        logging.info('Wrote video to file `%s`', filename)



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
        