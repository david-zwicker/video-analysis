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
        
        self._background = background
        background.register_listener(self.set_frame)
        
        self.frame = None
        
        if is_color is None:
            is_color = background.is_color
        
        super(VideoComposer, self).__init__(filename, background.size,
                                            background.fps, is_color, **kwargs)

    def set_frame(self, frame):
        # write the current frame
        if self.frame is not None:
            self.write_frame(self.frame)
        
        # set current frame
        self.frame = frame.copy()


#     def advance(self, index=None):
#         """ advances to the frame index.
#         Advance to the next frame if index is None.
#         """
#         # advance the video to the requested frame
#         while self._frame_pos < index:
#             # write the current frame
#             if self._frame is not None:
#                 self.write_frame(self._frame)
# 
#             # retrieve the next frame
#             self._frame_pos += 1
#             self._frame = next(self._background).copy()
# 
#         
#     def get_frame(self, index):
#         """ returns the frame at position index to be drawn onto """
#         if index == self._frame_pos:
#             # just return the current frame to allow it to be manipulated
#             pass
#         
#         elif index > self._frame_pos:
#             self.advance(index)
#             
#         else:
#             raise RuntimeError('The debug video went out of sync.')
# 
#         return self._frame
        
    
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
        
        
    def __del__(self):
        # write the last frame
        
        self.write_frame(self.frame)
        # close the video writer
        self.close()
        