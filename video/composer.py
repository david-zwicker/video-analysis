'''
Created on Aug 6, 2014

@author: zwicker
'''

import numpy as np
import cv2

from .utils import get_color
from .io.file import VideoFileWriter


class VideoComposer(VideoFileWriter):
    """ A class that takes a background video and can add additional geometric
    structures on top of this background. This makes it useful for debug output,
    hence the name """
    
    def __init__(self, filename, background, is_color=None, **kwargs):
        self._background = iter(background)
        self._frame_pos = -1
        self._frame = None#next(self._background).copy()
        
        if is_color is None:
            is_color = background.is_color
        
        super(VideoComposer, self).__init__(filename, background.size,
                                            background.fps, is_color, **kwargs)


    def advance(self, index=None):
        """ advances to the frame index.
        Advance to the next frame if index is None.
        """
        # advance the video to the requested frame
        while self._frame_pos < index:
            # write the current frame
            if self._frame is not None:
                self.write_frame(self._frame)

            # retrieve the next frame
            self._frame_pos += 1
            self._frame = next(self._background).copy()

        
    def get_frame(self, index):
        """ returns the frame at position index to be drawn onto """
        if index == self._frame_pos:
            # just return the current frame to allow it to be manipulated
            pass
        
        elif index > self._frame_pos:
            self.advance(index)
            
        else:
            raise RuntimeError('The debug video went out of sync.')

        return self._frame
        
    
    def add_image(self, index, image, mask=None, alpha=1):
        """ adds an image to the frame
        FIXME: alpha does not seem to work!
        """
        
        frame = self.get_frame(index)

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
        
        
    def blend_image(self, index, image, weight=0.5):
        """ overlay image with weight """
        frame = self.get_frame(index)
        cv2.addWeighted(frame, 1 - weight, image, weight, 0)
        
    
    def add_rectangle(self, index, rect, color='w', width=1):
        """ add a rect=(top, left, height, width) to the frame """
        frame = self.get_frame(index)
        cv2.rectangle(frame, rect[0, 1], rect[:2] + rect[2:], get_color(color), width)
        
        
    def add_circle(self, index, pos, radius=2, color='w'):
        """ add a circle to the frame """
        frame = self.get_frame(index)
        pos = (int(pos[0]), int(pos[1]))
        cv2.circle(frame, pos, radius, get_color(color), thickness=-1)
        # thickness = -1 denotes a filled circle
        
        
    def __del__(self):
        # write the last frame
        
        self.write_frame(self._frame)
        # close the video writer
        self.close()