'''
Created on Aug 6, 2014

@author: zwicker
'''

import cv2
from matplotlib.colors import ColorConverter

COLOR_CONVERTER = ColorConverter().to_rgb


from .io.file import VideoFileWriter

class VideoComposer(VideoFileWriter):
    """ A class that takes a background video and can add additional geometric
    structures on top of this background. This makes it useful for debug output,
    hence the name """
    
    def __init__(self, filename, background, **kwargs):
        self._background = iter(background)
        self._frame_pos = 0
        self._frame = next(self._background).copy()
        
        super(VideoComposer, self).__init__(filename, **kwargs)

        
    def get_frame(self, index):
        """ returns the frame at position index to be drawn onto """
        if index == self._frame_pos:
            # just return the current frame to allow it to be manipulated
            return self._frame
        
        elif index > self._frame_pos:
            
            # advance the video to the requested frame
            while self._frame_pos < index:
                # write the current frame
                self.write_frame(self._frame)

                # retrieve the next frame
                self._frame_pos += 1
                self._frame = next(self._background).copy()
            
            return self._frame
        
        else:
            raise RuntimeError('The debug video went out of sync.')
        
    
    def add_image(self, index, image, mask=None):
        """ adds an image to the frame """
        
        frame = self.get_frame(index)
        
        if mask is None:
            self._frame = cv2.add(frame, image)            
        else:
            self._frame[mask] = cv2.add(frame[mask], image[mask])
            
        pass
        
        
    def blend_image(self, index, image, weight=0.5):
        """ overlay image with weight """
        frame = self.get_frame(index)
        self._frame = cv2.addWeighted(frame, 1 - weight, image, weight, 0)
        
    
    def add_rectangle(self, index, rect, color='w', width=1):
        """ add a rect=(top, left, height, width) to the frame """
        frame = self.get_frame(index)
        self._frame = cv2.rectangle(frame, rect[0, 1], rect[:2] + rect[2:],
                                    COLOR_CONVERTER(color), width)
        
        
    def add_circle(self, index, pos, radius=2, color='w'):
        """ add a circle to the frame """
        frame = self.get_frame(index)
        self._frame = cv2.circle(frame, pos, radius, COLOR_CONVERTER(color),
                                 thickness=-1)
        # thickness = -1 denotes a filled circle
        
        
    def __del__(self):
        # write the last frame
        self.write_frame(self._frame)
        # close the video writer
        self.close()