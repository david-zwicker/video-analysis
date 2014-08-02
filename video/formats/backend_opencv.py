'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for referencing a single video file.
The video is loaded using OpenCV.
'''

from __future__ import division

import logging

import cv2
import cv2.cv as cv # still necessary for some constants

from .base import VideoBase, VideoImageStackBase


class VideoOpenCV(VideoBase):
    """
    Class handling a single _movie file using opencv
    """ 
    
    def __init__(self, filename):
        # load the _movie
        self.filename = filename
        
        logging.debug('Loading video `%s` using OpenCV', filename)
        self._movie = cv2.VideoCapture(filename)
        # this call doesn't fail if the file could not be found, but returns
        # an empty video instead.
        
        # determine _movie properties
        size = (int(self._movie.get(cv.CV_CAP_PROP_FRAME_HEIGHT)),
                int(self._movie.get(cv.CV_CAP_PROP_FRAME_WIDTH)))
        frame_count = int(self._movie.get(cv.CV_CAP_PROP_FRAME_COUNT))
        fps = self._movie.get(cv.CV_CAP_PROP_FPS)

        if frame_count == 0:
            raise IOError('There were problems loading the video.')
        
        # rewind _movie
        self.set_frame_pos(0)
        
        super(VideoOpenCV, self).__init__(size=size, frame_count=frame_count, fps=fps, is_color=True)
                
        
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return int(self._movie.get(cv.CV_CAP_PROP_POS_FRAMES))


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if not self._movie.set(cv.CV_CAP_PROP_POS_FRAMES, index):
            raise IndexError('Seeking to frame %d was not possible.' % index)


    def next(self):
        """ returns the next frame """
        # get the next frame, which automatically increments the internal frame index
        ret, frame = self._movie.read()
        if ret:
            return frame
        else:
            # reading the data failed for whatever reason
            raise StopIteration

    
    def get_frame(self, index):
        """
        returns a specific frame identified by its index.
        Note that this sets the internal frame index of the video and this function
        should thus not be used while iterating over the video.
        """ 
        self.set_frame_pos(index)
        ret, frame = self._movie.read()
        if ret:
            return frame
        else:
            # reading the data failed for whatever reason
            raise IndexError
            
                    
    def __del__(self):
        self._movie.release()


class VideoImageStackOpenCV(VideoImageStackBase):
    """ class that loads a stack of images using opencv """
    
    def get_frame(self, index):
        return cv2.imread(self.filenames[index])
