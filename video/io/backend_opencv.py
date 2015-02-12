'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This package provides class definitions for referencing a single video file.
The video is loaded using OpenCV.
'''

from __future__ import division

import os.path
import platform
import logging

import cv2
import cv2.cv as cv # still necessary for some constants

from .base import VideoBase, VideoImageStackBase

logger = logging.getLogger('video.io')


# dictionary that maps standard file endings to fourcc codes
# more codes can be found at http://www.fourcc.org/codecs.php
if platform.system() == 'Darwin':
    CODECS = {
        '.xvid': 'XVID',
        '.mov': 'mp4v',   # standard quicktime codec - tested
        '.mpeg': 'FMP4',  # mpeg 4 variant
        '.avi': 'IYUV',   # uncompressed avi - tested
    }
else:
    CODECS = {
        '.xvid': 'XVID',
        '.mov': 'mp4v', #'SVQ3',   # standard quicktime codec
        '.mpeg': 'FMP4',  # mpeg 4 variant
        '.avi': 'IYUV',   # uncompressed avi 
    }



class VideoOpenCV(VideoBase):
    """
    Class handling a single movie file using OpenCV
    """ 
    
    def __init__(self, filename):
        # load the _movie
        self.filename = os.path.expanduser(filename)
        
        self._movie = cv2.VideoCapture(self.filename)
        # this call doesn't fail if the file could not be found, but returns
        # an empty video instead. We thus fail later by checking the video length
        
        # determine _movie properties
        size = (int(self._movie.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self._movie.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(self._movie.get(cv.CV_CAP_PROP_FRAME_COUNT))
        fps = self._movie.get(cv.CV_CAP_PROP_FPS)
        self.pix_fmt = 'bgr' #< seems to be OpenCV default

        if frame_count == 0:
            raise IOError('There were problems loading the video.')
        
        # rewind _movie
        self.set_frame_pos(0)
        
        super(VideoOpenCV, self).__init__(size=size, frame_count=frame_count,
                                          fps=fps, is_color=True)

        logger.debug('Initialized video `%s` with %d frames using OpenCV',
                     filename, frame_count)
    
        
    @property
    def closed(self):
        return self._movie is None
        
        
    def open(self):
        """ Opens the video file """
        logger.debug('Open video `%s`' % self.filename)
        self.close() #< make sure that the previous video is closed
        self._movie = cv2.VideoCapture(self.filename)
        
        
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return int(self._movie.get(cv.CV_CAP_PROP_POS_FRAMES))


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        frame_pos = self.get_frame_pos()
        
        if index < frame_pos and not self.seekable:
            # reopen the video to be able to seek forward
            self.open()
            frame_pos = self.get_frame_pos()

        if index > frame_pos:
            # OpenCV seeking is not exact to the frame
            # => we seek about 1 sec before the frame ...
            if index > frame_pos + self.fps:
                self._movie.set(cv.CV_CAP_PROP_POS_FRAMES, index - self.fps)
            
            # ... and iterate through the remaining frames
            for _ in xrange(self.get_frame_pos(), index):
                self._movie.grab()
                
            # just double check that we are at the right frame
            if self.get_frame_pos() != index:
                raise IndexError('Seeking to frame %d was not possible. The '
                                 'video is at frame %d.' % (index, self.get_frame_pos()))


    def get_next_frame(self):
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
        Note that this sets the internal frame index of the video and this
        function should thus not be used while iterating over the video.
        """
        self.set_frame_pos(index)
        
        # get the next frame, which also increments the internal frame index
        ret, frame = self._movie.read()
        
        if ret:
            return frame
        else:
            # reading the data failed for whatever reason
            raise IndexError('OpenCV could not read frame.')
            
                    
    def close(self):
        if self._movie:
            self._movie.release()
        self._movie = None

                    
    def __enter__(self):
        return self
    
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()
    
                    
    def __del__(self):
        self.close()
        


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



class VideoWriterOpenCV(object):
    def __init__(self, filename, size, fps, is_color=True, codec=None,
                 **kwargs):
        """
        Saves the video to the file indicated by filename.
        codec must be a fourcc code from http://www.fourcc.org/codecs.php
            If codec is None, the code is determined from the filename extension.
        """
        self.filename = os.path.expanduser(filename)
        self.size = size
        self.is_color = is_color
        self.frames_written = 0

        if codec is None:
            # detect format from file ending
            file_ext = os.path.splitext(filename)[1].lower()
            try:
                codec = CODECS[file_ext]
            except KeyError:
                raise ValueError('Video format `%s` is unsupported.' % codec) 
        
        # get the code defining the video format
        fourcc = cv2.cv.FOURCC(*codec)
        self._writer = cv2.VideoWriter(self.filename, fourcc=fourcc, fps=fps,
                                       frameSize=(size[1], size[0]),
                                       isColor=is_color)

        logger.info('Start writing video `%s` with codec `%s`',
                    self.filename, codec)
                
        
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        shape = (self.size[1], self.size[0])
        if self.is_color:
            shape += (3,)
        return shape


    def write_frame(self, frame):
        self._writer.write(cv2.convertScaleAbs(frame))
        self.frames_written += 1   
        
        
    def close(self):
        self._writer.release()
        logger.info('Wrote video to file `%s`', self.filename)
    
    
    def __enter__(self):
        return self
    
        
    def __exit__(self, e_type, e_value, e_traceback):
        self.close()        
    
    
    def __del__(self):
        self.close()
    
