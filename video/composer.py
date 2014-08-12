'''
Created on Aug 6, 2014

@author: zwicker
'''

import numpy as np
import cv2

from .utils import get_color
from .io.file import VideoFileWriter


class VideoComposer(VideoFileWriter):
    """ A class that can be used to compose a video frame by frame.
    Additional elements like geometric objects can be added to each frame
    """
    
    def __init__(self, filename, size, fps, is_color, **kwargs):
        
        self.frame = None
        
        super(VideoComposer, self).__init__(filename, size, fps, is_color, **kwargs)


    def set_frame(self, frame):
        # write the current frame
        if self.frame is not None:
            self.write_frame(self.frame)
        
        # set current frame
        if self.is_color and frame.ndim == 2:
            self.frame = frame[:, :, None]*np.ones((1, 1, 3), np.uint8)
        else:
            self.frame = frame.copy()
        
    
    def add_image(self, image, mask=None, alpha=1):
        """ adds an image to the frame
        FIXME: alpha does not seem to work!
        """
        frame = self.frame
        
        # check image dimensions
        if frame.shape[:2] != image.shape[:2]:
            raise ValueError('The two images to be added must have the same size')
        
        # check color properties
        if frame.ndim == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.cv.CV_GRAY2RGB)
        elif frame.ndim == 2 and image.ndim == 3:
            raise ValueError('Cannot add a color image to a monochrome one')
        
        if alpha != 1:
            image = (alpha*image).astype(np.uint8)
        
        if mask is None:
            cv2.add(frame, image, frame)
        else:
            cv2.add(frame, image, frame, mask=mask.astype(np.uint8))
        
        
    def blend_image(self, image, weight=0.5, mask=None):
        """ overlay image with weight """
        frame = self.frame
        
        # check image dimensions
        if frame.shape[:2] != image.shape[:2]:
            raise ValueError('The two images to be added must have the same size')
        
        # check color properties
        if frame.ndim == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.cv.CV_GRAY2RGB)
        elif frame.ndim == 2 and image.ndim == 3:
            raise ValueError('Cannot add a color image to a monochrome one')

        # TODO: this creates an extra copy of the frame, which might not be necessary
        result = cv2.addWeighted(frame, 1 - weight, image, weight, gamma=0)

        if mask is not None:
            result[~mask] = frame[~mask]
            
        self.frame = result        
        
        
    def add_contour(self, mask, color='w', thickness=1):
        """ adds the contours of a mask.
        Note that this function modifies the mask 
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.frame, contours, -1,  get_color(color), thickness=thickness)
    
    
    def add_polygon(self, points, color='w', is_closed=True, width=1):
        """ adds a polygon to the frame """
        points = np.asarray(points, np.int)               
        cv2.polylines(self.frame, [points], isClosed=is_closed,
                      color=get_color(color), thickness=width)
        
    
    def add_rectangle(self, rect, color='w', width=1):
        """ add a rect=(top, left, height, width) to the frame """
        cv2.rectangle(self.frame, rect[0, 1], rect[:2] + rect[2:],
                      get_color(color), width)
        
        
    def add_circle(self, pos, radius=2, color='w', thickness=-1):
        """ add a circle to the frame.
        thickness=-1 denotes a filled circle 
        """
        pos = (int(pos[0]), int(pos[1]))
        cv2.circle(self.frame, pos, radius, get_color(color), thickness=thickness)
        
    
    def add_points(self, points, radius, color):
        """ adds a sequence of points to the frame """
        c = get_color(color)
        for p in points:
            cv2.circle(self.frame, (int(p[0]), int(p[1])), radius, c, thickness=-1)
        
        
    def __del__(self):
        # write the last frame
        
        self.write_frame(self.frame)
        # close the video writer
        self.close()
        
    
        
class VideoComposerListener(VideoComposer):
    """ A class that can be used to compose a video frame by frame.
    This class automatically listens to another video and captures the newest
    frame from it. Additional elements like geometric objects can then be added
    to each frame. This is useful to annotate a copy of a video.
    """
    
    
    def __init__(self, filename, background, is_color=None, **kwargs):
        
        self._background = background
        background.register_listener(self.set_frame)
        
        super(VideoComposerListener, self).__init__(filename, background.size,
                                                    background.fps, is_color, **kwargs)
        