'''
Created on Jul 31, 2014

@author: zwicker

This package provides class definitions for referencing a single video file.
The video is loaded using OpenCV.
'''

from __future__ import division

import os
import platform
import logging

import cv2
import cv2.cv as cv # still necessary for some constants

from .base import VideoBase, VideoImageStackBase


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
        
        # get the next frame, which also increments the internal frame index
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



def show_video_opencv(video):
    """ shows a video using opencv """
    for frame in video:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()



def write_video_opencv(video, filename, video_format=None):
    """
    Saves the video to the file indicated by filename.
    video_format must be a fourcc code from http://www.fourcc.org/codecs.php
        If video_format is None, the code is determined from the filename extension.
    """
    
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
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=video.fps,
                          frameSize=video.size, isColor=video.is_color)

    # write out all individual frames
    for frame in video:
        # convert the data to uint8 before writing it out
        out.write(cv2.convertScaleAbs(frame))
        
    out.release()
    logging.info('Wrote video to file `%s`', filename)