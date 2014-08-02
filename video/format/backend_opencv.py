'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for referencing a single video file.
The video is loaded using OpenCV.
'''

from __future__ import division

import cv2
import cv2.cv as cv # still necessary for some constants

from .base import VideoBase


class VideoOpenCV(VideoBase):
    """
    Class handling a single movie file using opencv
    """ 
    
    def __init__(self, filename):
        # load the movie
        self.filename = filename
        self.movie = cv2.VideoCapture(filename)
        
        # determine movie properties
        size = (int(self.movie.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self.movie.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(self.movie.get(cv.CV_CAP_PROP_FRAME_COUNT))
        fps = self.movie.get(cv.CV_CAP_PROP_FPS)
        
        # rewind movie
        self.set_frame_pos(0)
        
        super(VideoOpenCV, self).__init__(size=size, frame_count=frame_count, fps=fps)
                
        
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return int(self.movie.get(cv.CV_CAP_PROP_POS_FRAMES))


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if not self.movie.set(cv.CV_CAP_PROP_POS_FRAMES, index):
            raise ValueError('Seeking to frame %d was not possible.' % index)


    def next(self):
        """ returns the next frame """
        ret, frame = self.movie.read()
        if ret:
            return frame
        else:
            raise StopIteration

    
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        self.set_frame_pos(index)
        return self.next()
            
                    
    def __del__(self):
        self.movie.release()

