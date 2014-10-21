'''
Created on Aug 6, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import cv2

from .utils import get_color, contiguous_regions
from .analysis.regions import rect_to_corners
from .io.file import VideoFileWriter


def skip_if_no_output(func):
    """ decorator which only calls the function if the current frame will
    be written to the file """
    def func_wrapper(self, *args, **kwargs):
        if self.output_this_frame:
            return func(self, *args, **kwargs)
    return func_wrapper


CHANNEL_NAMES = {0: 0, 'r': 0, 'red': 0,
                 1: 1, 'g': 1, 'green': 1,
                 2: 2, 'b': 2, 'blue': 2}


class VideoComposer(VideoFileWriter):
    """ A class that can be used to compose a video frame by frame.
    Additional elements like geometric objects can be added to each frame
    """
    
    def __init__(self, filename, size, fps, is_color, output_period=1, **kwargs):
        """
        Initializes a video file writer with additional functionality to annotate
        videos. The first arguments (size, fps, is_color) are directly passed on
        to the VideoFileWriter.
        The additional argument `output_period` determines how often frames are
        actually written. `output_period=10` for instance only outputs every
        tenth frame to the file.
        """
        self.frame = None
        self.next_frame = -1
        self.output_period = output_period
        
        super(VideoComposer, self).__init__(filename, size, fps, is_color, **kwargs)


    @property
    def output_this_frame(self):
        """ determines whether the current frame should be written to the video """
        return (self.next_frame % self.output_period) == 0


    def set_frame(self, frame, copy=True):
        """ set the current frame from an image """
        self.next_frame += 1
        if self.output_this_frame:
            if self.frame is None:
                # first frame => initialize
                if self.is_color and frame.ndim == 2:
                    self.frame = frame[:, :, None]*np.ones((1, 1, 3), np.uint8)
                elif not self.is_color and frame.ndim == 3:
                    raise ValueError('Cannot copy a color image into a monochrome video.')
                else:
                    self.frame = frame.copy()
                
            else:
                # write the last frame
                self.write_frame(self.frame)
            
                # set current frame
                if self.is_color and frame.ndim == 2:
                    # set all three color channels
                    # Here, explicit iteration is faster than numpy broadcasting
                    for c in xrange(3): 
                        self.frame[:, :, c] = frame
                elif copy:
                    self.frame[:] = frame[:]
                else:
                    self.frame = frame

        
    @skip_if_no_output
    def highlight_mask(self, mask, channel='all', strength=128):
        """ highlights the non-zero entries of a mask in the current frame """
        if channel is None or channel == 'all':
            if self.is_color:
                channel = slice(0, 3)
            else:
                channel = 0
        elif self.is_color:
            try:
                channel = CHANNEL_NAMES[channel]
            except KeyError:
                raise ValueError('Unknown value `%s` for channel.' % channel)
        else:
            raise ValueError('Highlighting a specific channel is only '
                             'supported for color videos.')

        factor = (255 - strength)/255
        self.frame[mask, channel] = strength + factor*self.frame[mask, channel]
        

    @skip_if_no_output
    def add_image(self, image, mask=None):
        """ adds an image to the frame """
        frame = self.frame
        
        # check image dimensions
        if frame.shape[:2] != image.shape[:2]:
            raise ValueError('The two images to be added must have the same size')
        
        # check color properties
        if frame.ndim == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.cv.CV_GRAY2RGB)
        elif frame.ndim == 2 and image.ndim == 3:
            raise ValueError('Cannot add a color image to a monochrome one')
        
        if mask is None:
            cv2.add(frame, image, frame)
        else:
            cv2.add(frame, image, frame, mask=mask.astype(np.uint8))
            
    
    @skip_if_no_output
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

        result = cv2.addWeighted(frame, 1 - weight, image, weight, gamma=0)

        if mask is not None:
            result[~mask] = frame[~mask]
            
        self.frame = result        
        
        
    @skip_if_no_output
    def add_contour(self, mask_or_contour, color='w', thickness=1, copy=True):
        """ adds the contours of a mask.
        Note that this function modifies the mask, unless copy=True
        """
        if np.any(s == 1 for s in mask_or_contour.shape[:2]):
            # given value is a list of contour points
            contours = [mask_or_contour]
        else:
            # given value is a mask 
            if copy:
                mask_or_contour = mask_or_contour.copy()
            contours, _ = cv2.findContours(mask_or_contour,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
            
        cv2.drawContours(self.frame, contours, -1,
                         get_color(color), thickness=int(thickness))
    
    
    @skip_if_no_output
    def add_line(self, points, color='w', is_closed=True, mark_points=False, width=1):
        """ adds a polygon to the frame """
        if len(points) == 0:
            return
        
        points = np.asarray(points, np.int32)
        
        # find the regions where the points are finite
        # Here, we compare to 0 to capture nans in the int32 array        
        indices = contiguous_regions(points[:, 0] > 0)
        
        for start, end in indices:
            # add the line
            cv2.polylines(self.frame, [points[start:end, :]],
                          isClosed=is_closed, color=get_color(color),
                          thickness=int(width))
            # mark the anchor points if requested
            if mark_points:
                for p in points[start:end, :]:
                    self.add_circle(p, 2*width, color, thickness=-1)

        
    
    @skip_if_no_output
    def add_rectangle(self, rect, color='w', width=1):
        """ add a rect=(left, top, width, height) to the frame """
        cv2.rectangle(self.frame, *rect_to_corners(rect),
                      color=get_color(color), thickness=int(width))
        
        
    @skip_if_no_output
    def add_circle(self, pos, radius=2, color='w', thickness=-1):
        """ add a circle to the frame.
        thickness=-1 denotes a filled circle 
        """
        try:
            pos = (int(pos[0]), int(pos[1]))
            cv2.circle(self.frame, pos, int(radius), get_color(color), thickness=int(thickness))
        except (ValueError, OverflowError):
            pass
        
    
    @skip_if_no_output
    def add_points(self, points, radius=1, color='w'):
        """ adds a sequence of points to the frame """
        c = get_color(color)
        for p in points:
            try:
                cv2.circle(self.frame, (int(p[0]), int(p[1])), radius, c, thickness=-1)
            except OverflowError:
                # happens with negative coordinates 
                pass
        
    
    @skip_if_no_output
    def add_text(self, text, pos, color='w', size=1, anchor='bottom'):
        """ adds text to the video.
        pos denotes the bottom left corner of the text
        """
        
        if anchor == 'top':
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                           fontScale=size, thickness=1)
            pos = (pos[0], pos[1] + text_size[1])
            
        cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=size, color=get_color(color), thickness=1)
        
        
    def close(self):
        # write the last frame
        if self.frame is not None:
            self.write_frame(self.frame)
            self.frame = None
                
        # close the video writer
        super(VideoComposer, self).close()
        
    
        
class VideoComposerListener(VideoComposer):
    """ A class that can be used to compose a video frame by frame.
    This class automatically listens to another video and captures the newest
    frame from it. Additional elements like geometric objects can then be added
    to each frame. This is useful to annotate a copy of a video.
    """
    
    
    def __init__(self, filename, background_video, is_color=None, **kwargs):
        self.background_video = background_video
        self.background_video.register_listener(self.set_frame)
        
        super(VideoComposerListener, self).__init__(filename,
                                                    self.background_video.size,
                                                    self.background_video.fps,
                                                    is_color, **kwargs)
        
        
    def close(self):
        try:
            self.background.unregister_listener(self.set_frame)
        except (AttributeError, ValueError):
            # apparently, the listener is already removed 
            pass
        super(VideoComposerListener, self).close()
