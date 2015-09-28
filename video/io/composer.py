'''
Created on Aug 6, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import cv2
from matplotlib.colors import ColorConverter

from utils.math import contiguous_true_regions
from .file import VideoFileWriter
from ..analysis.regions import rect_to_corners


    
def get_color(color):
    """
    function that returns a RGB color with channels ranging from 0..255.
    The matplotlib color notation is used.
    """
    
    if get_color.parser is None:
        get_color.parser = ColorConverter().to_rgb
        
    return [int(255*c) for c in get_color.parser(color)]

get_color.parser = None



def skip_if_no_output(func):
    """ decorator which only calls the function if the current _frame will
    be written to the file """
    def func_wrapper(self, *args, **kwargs):
        if self.output_this_frame:
            return func(self, *args, **kwargs)
    return func_wrapper


CHANNEL_NAMES = {0: 0, 'r': 0, 'red': 0,
                 1: 1, 'g': 1, 'green': 1,
                 2: 2, 'b': 2, 'blue': 2}


class VideoComposer(VideoFileWriter):
    """ A class that can be used to compose a video _frame by _frame.
    Additional elements like geometric objects can be added to each _frame
    """
    
    def __init__(self, filename, size, fps, is_color, output_period=1, 
                 zoom_factor=1, **kwargs):
        """
        Initializes a video file writer with additional functionality to annotate
        videos. The first arguments (size, fps, is_color) are directly passed on
        to the VideoFileWriter.
        `output_period` determines how often frames are actually written.
            `output_period=10` for instance only outputs every tenth _frame to
            the file.
        `zoom_factor` determines how much the output will be scaled down. The
            width of the new image is given by dividing the width of the
            original image by the zoom_factor.
        """
        self._frame = None
        self.next_frame = -1
        self.output_period = output_period
        self.zoom_factor = zoom_factor
        target_size = (int(size[0]/zoom_factor), int(size[1]/zoom_factor))
        
        super(VideoComposer, self).__init__(filename, target_size, fps,
                                            is_color, **kwargs)


    @property
    def output_this_frame(self):
        """ determines whether the current _frame should be written to the video """
        return (self.next_frame % self.output_period) == 0


    def set_frame(self, frame, copy=True):
        """ set the current _frame from an image """
        self.next_frame += 1
        
        if self.output_this_frame:
            # scale current frame if necessary 
            if self.zoom_factor != 1:
                frame = cv2.resize(frame, self.size)
                copy = False #< copy already happened

            if self._frame is None:
                # first frame => initialize the video 
                if self.is_color and frame.ndim == 2:
                    # copy the monochrome _frame into the color video
                    self._frame = np.repeat(frame[:, :, None], 3, axis=2)
                elif not self.is_color and frame.ndim == 3:
                    raise ValueError('Cannot copy a color image into a '
                                     'monochrome video.')
                elif copy:
                    self._frame = frame.copy()
                else:
                    self._frame = frame
                
            else:
                # had a previous _frame => write the last frame
                self.write_frame(self._frame)

                # set current frame
                if self.is_color and frame.ndim == 2:
                    # set all three color channels
                    # Here, explicit iteration is faster than numpy broadcasting
                    for c in xrange(3): 
                        self._frame[:, :, c] = frame
                elif copy:
                    self._frame[:] = frame
                else:
                    self._frame = frame

        
    @skip_if_no_output
    def highlight_mask(self, mask, channel='all', strength=128):
        """ highlights the non-zero entries of a mask in the current _frame """
        # determine which color channel to use
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

        # scale mask if necessary
        if self.zoom_factor != 1:
            mask = cv2.resize(mask.astype(np.uint8), self.size).astype(np.bool)
        
        # mark the mask area in the image    
        factor = (255 - strength)/255
        self._frame[mask, channel] = strength + factor*self._frame[mask, channel]
        

    def _prepare_images(self, image, mask=None):
        """ scale image if necessary """
        if self.zoom_factor != 1:
            image = cv2.resize(image, self.size)
            if mask:
                mask = cv2.resize(mask.astype(np.uint8),
                                  self.size).astype(np.bool)
        return image, mask        


    @skip_if_no_output
    def add_image(self, image, mask=None):
        """ adds an image to the _frame """
        frame = self._frame
        image, mask = self._prepare_images(image, mask)
        
        # check image dimensions
        if frame.shape[:2] != image.shape[:2]:
            raise ValueError('The two images to be added must have the same size')
        
        # check color properties
        if frame.ndim == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 2 and image.ndim == 3:
            raise ValueError('Cannot add a color image to a monochrome one')
        
        if mask is None:
            cv2.add(frame, image, frame)
        else:
            cv2.add(frame, image, frame, mask=mask.astype(np.uint8))
            
    
    @skip_if_no_output
    def blend_image(self, image, weight=0.5, mask=None):
        """ overlay image with weight """
        frame = self._frame
        image, mask = self._prepare_images(image, mask)
        
        # check image dimensions
        if frame.shape[:2] != image.shape[:2]:
            raise ValueError('The two images to be added must have the same size')
        
        # check color properties
        if frame.ndim == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 2 and image.ndim == 3:
            raise ValueError('Cannot add a color image to a monochrome one')

        result = cv2.addWeighted(frame, 1 - weight, image, weight, gamma=0)

        if mask is not None:
            result[~mask] = frame[~mask]
            
        self._frame = result        
        
        
    @skip_if_no_output
    def add_contour(self, mask_or_contour, color='w', thickness=1, copy=True):
        """ adds the contours of a mask.
        Note that this function modifies the mask, unless copy=True
        """
        if any(s == 1 for s in mask_or_contour.shape[:2]):
            # given value is a list of contour points
            contours = [mask_or_contour]
        else:
            # given value is a mask 
            if copy:
                mask_or_contour = mask_or_contour.copy()
            contours = cv2.findContours(mask_or_contour, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[1]
            
        if self.zoom_factor != 1:
            contours = np.asarray(contours, np.double) / self.zoom_factor
            contours = contours.astype(np.int)
            thickness = np.ceil(thickness / self.zoom_factor)
            
        cv2.drawContours(self._frame, contours, -1,
                         get_color(color), thickness=int(thickness))
    
    
    @skip_if_no_output
    def add_line(self, points, color='w', is_closed=True, mark_points=False,
                 width=1):
        """ adds a polygon to the _frame """
        if len(points) == 0:
            return
        
        points = np.asarray(points)
        
        # find the regions where the points are finite
        # Here, we compare to 0 to capture nans in the int32 array 
        indices = contiguous_true_regions(points[:, 0] > 0)
        
        for start, end in indices:
            # add the line
            line_points = (points[start:end, :]/self.zoom_factor).astype(np.int)
            thickness = int(np.ceil(width / self.zoom_factor))
            cv2.polylines(self._frame, [line_points],
                          isClosed=is_closed, color=get_color(color),
                          thickness=thickness)
            # mark the anchor points if requested
            if mark_points:
                for p in points[start:end, :]:
                    self.add_circle(p, 2*width, color, thickness=-1)

        
    
    @skip_if_no_output
    def add_rectangle(self, rect, color='w', width=1):
        """ add a rect=(left, top, width, height) to the _frame """
        if self.zoom_factor != 1:
            rect = np.asarray(rect) / self.zoom_factor
            thickness = int(np.ceil(width / self.zoom_factor))
        else:
            thickness = int(width)
        
        # extract the corners of the rectangle
        try:    
            corners = rect.corners
        except AttributeError:
            corners = rect_to_corners(rect)
            
        corners = [(int(p[0]), int(p[1])) for p in corners]
            
        # draw the rectangle
        cv2.rectangle(self._frame, *corners, color=get_color(color),
                      thickness=thickness)
        
        
    @skip_if_no_output
    def add_circle(self, pos, radius=2, color='w', thickness=-1):
        """ add a circle to the _frame.
        thickness=-1 denotes a filled circle 
        """
        try:
            pos = (int(pos[0]/self.zoom_factor), int(pos[1]/self.zoom_factor))
            radius = int(np.ceil(radius / self.zoom_factor))
            if thickness > 0:
                thickness = int(np.ceil(thickness / self.zoom_factor))
            cv2.circle(self._frame, pos, radius, get_color(color),
                       thickness=thickness)
        except (ValueError, OverflowError):
            pass
        
    
    @skip_if_no_output
    def add_points(self, points, radius=1, color='w'):
        """ adds a sequence of points to the _frame """
        for p in points:
            self.add_circle(p, radius, color, thickness=-1)
        
    
    @skip_if_no_output
    def add_text(self, text, pos, color='w', size=1, anchor='bottom',
                 font=cv2.FONT_HERSHEY_COMPLEX_SMALL):
        """ adds text to the video.
        `pos` determines the position of the anchor of the text
        `anchor` can be a string containing (left, center, right) for
            horizontal placement and (upper, middle, lower) for the vertical one
        """
        if self.zoom_factor != 1:
            pos = [int(pos[0]/self.zoom_factor), int(pos[1]/self.zoom_factor)]
            size /= self.zoom_factor
        else:
            pos = [int(pos[0]), int(pos[1])]
    
        # determine text size to allow flexible positioning
        text_size, _ = cv2.getTextSize(text, font, fontScale=size, thickness=1)

        # determine horizontal position of text
        if 'right' in anchor:
            pos[0] = pos[0] - text_size[0]
        elif 'center' in anchor:
            pos[0] = pos[0] - text_size[0]//2
            
        # determine vertical position of text
        if 'upper' in anchor or 'top' in anchor:
            pos[1] = pos[1] + text_size[1]
        elif 'middle' in anchor:
            pos[1] = pos[1] + text_size[1]//2
            
        # place text
        cv2.putText(self._frame, text, tuple(pos), font, fontScale=size,
                    color=get_color(color), thickness=1)
        
        
    def close(self):
        # write the last _frame
        if self._frame is not None:
            self.write_frame(self._frame)
            self._frame = None
                
        # close the video writer
        super(VideoComposer, self).close()
        
    
        
class VideoComposerListener(VideoComposer):
    """ A class that can be used to compose a video _frame by _frame.
    This class automatically listens to another video and captures the newest
    _frame from it. Additional elements like geometric objects can then be added
    to each _frame. This is useful to annotate a copy of a video.
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
