'''
Created on Aug 1, 2014

@author: zwicker

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory.

Some filters allow to access the underlying frames at random using the get_frame method

'''

from __future__ import division

import logging
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV was not found. Functions requiring OpenCV will not work.")


from .io.base import VideoFilterBase
from video.utils import get_color_range
  

class FilterFunction(VideoFilterBase):
    """ smoothes every frame """
    
    def __init__(self, source, function):
        
        self._function = function
        
        super(FilterFunction, self).__init__(source)
        
        logging.debug('Created filter applying a function to every frame')


    def _process_frame(self, frame):
        # process the current frame 
        frame = self._function(frame)
        # pass it to the parent function
        return super(FilterFunction, self)._process_frame(frame)
    
    
    
class FilterNormalize(VideoFilterBase):
    """ normalizes a color range to the interval 0..1 """
    
    def __init__(self, source, vmin=0, vmax=1, dtype=None):
        """
        warning:
        vmin must not be smaller than the smallest value source can hold.
        Otherwise wrapping can occur. The same thing holds for vmax, which
        must not be larger than the maximum value in the color channels.
        """
        
        # interval From which to convert 
        self._fmin = vmin
        self._fmax = vmax
        
        # interval To which to convert
        self._dtype = dtype
        self._tmin = None
        self._alpha = None
        
        super(FilterNormalize, self).__init__(source)
        logging.debug('Created filter for normalizing range [%g..%g]', vmin, vmax)


    def _process_frame(self, frame):
        
        # ensure that we decided on a dtype
        if self._dtype is None:
            self._dtype = frame.dtype
            
        # ensure that we know the bounds of this dtype
        if self._tmin is None:
            self._tmin, tmax = get_color_range(self._dtype)
            self._alpha = (tmax - self._tmin)/(self._fmax - self._fmin)
            
            # some safety checks on the first run:
            fmin, fmax = get_color_range(frame.dtype)
            if self._fmin < fmin:
                logging.warn('Lower normalization bound is below what the format can hold.')
            if self._fmax > fmax:
                logging.warn('Upper normalization bound is above what the format can hold.')

        # clip the data before converting
        np.clip(frame, self._fmin, self._fmax, out=frame)

        # do the conversion from [fmin, fmax] to [tmin, tmax]
        frame = (frame - self._fmin)*self._alpha + self._tmin
        
        # cast the data to the right type
        frame = frame.astype(self._dtype)
        
        # pass the frame to the parent function
        return super(FilterNormalize, self)._process_frame(frame)



class FilterCrop(VideoFilterBase):
    """ crops the video to the given rect=(top, left, height, width) """
    
    def __init__(self, source, rect):
        """ initialized the filter that crops to the given rect=(top, left, height, width) """
        
        def _check_number(value, max_value):
            """ helper function checking the bounds of the rectangle """
            
            # convert to integer by interpreting float values as fractions
            value = int(value*max_value if -1 < value < 1 else value)
            
            # interpret negative numbers as counting from opposite boundary
            if value < 0:
                value += max_value
                
            # check whether the value is within bounds
            if not 0 <= value < max_value:
                raise IndexError('Cropping rectangle reaches out of frame.')
            
            return value
            
        # interpret float values as fractions
        rect = [
            _check_number(rect[0], source.size[0]),
            _check_number(rect[1], source.size[1]),
            _check_number(rect[2], source.size[0]),
            _check_number(rect[3], source.size[1]),
        ]
        
        # save the rectangle
        self.rect = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]) 
        
        # correct the size, since we are going to crop the movie
        super(FilterCrop, self).__init__(source, size=(rect[2], rect[3]))

        logging.debug('Created filter for cropping to rectangle %s', self.rect)
        
       
    def _process_frame(self, frame):
        r = self.rect
        frame = frame[r[0]:r[2], r[1]:r[3]]
        # pass the frame to the parent function
        return super(FilterCrop, self)._process_frame(frame)



class FilterMonochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='normal'):
        self.mode = mode.lower()
        super(FilterMonochrome, self).__init__(source, is_color=False)

        logging.debug('Created filter for converting video to monochrome with method `%s`', mode)

    def _process_frame(self, frame):
        """
        reduces a single frame from color to monochrome, but keeps the
        extra dimension in the data
        """
        if self.mode == 'normal':
            frame = np.mean(frame, axis=2).astype(frame.dtype)
        elif self.mode == 'r':
            frame = frame[:, :, 0]
        elif self.mode == 'g':
            frame = frame[:, :, 1]
        elif self.mode == 'b':
            frame = frame[:, :, 2]
        else:
            raise ValueError('Unsupported conversion method to monochrome: %s' % self.mode)
    
        # pass the frame to the parent function
        return super(FilterMonochrome, self)._process_frame(frame)
    


class FilterFeatures(VideoFilterBase):
    """ detects features and draws them onto the image """
    
    def _process_frame(self, frame):
        frame = frame.copy()
        
        corners = cv2.goodFeaturesToTrack(frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)
        
        for i in corners:
            x,y = i.ravel()
            cv2.circle(frame, (x,y), 5, 255, -1)
            
        # pass the frame to the parent function
        return super(FilterFeatures, self)._process_frame(frame)
    
    

class FilterBackground(VideoFilterBase):
    """ filter that filters out the background of a video """
    
    def __init__(self, source, timescale=128):
        self._background = None
        self.timescale = timescale
        super(FilterBackground, self).__init__(source)
    
    
    def _process_frame(self, frame):
        # adapt the current background model
        # Here, the cut-off sets the relaxation time scale
        #diff_max = 128/self.background_history
        #self._background += np.clip(diff, -diff_max, diff_max)
        self._background = np.minimum(self._background, frame)

        # pass the frame to the parent function
        return super(FilterBackground, self)._process_frame(self._background)
    
    
    
class FilterMoveTowards(VideoFilterBase):
    """
    Implementation copied from
    http://www.aforgenet.com/framework/docs/html/f7f4ad2c-fb50-a76b-38fc-26355e0eafcf.htm
    """
    def __init__(self, step):
        raise NotImplementedError # this filter might not be useful after all
        self._prev_frame = None
        self.step = step
        
        
    def set_frame_pos(self, index):
        super(FilterMoveTowards, self).get_frame(index)
        # set the cache to the right value
        if index == 0:
            self._prev_frame = None
        else:
            self._prev_frame = self.get_frame(index - 1)
        
    
    def get_frame(self, index):
        raise NotImplementedError
        
        
    def _process_frame(self, frame):
        
        if self._prev_frame is None:
            self._prev_frame = frame
        else:
            diff = self._prev_frame - frame
            frame = self._prev_frame + \
                np.min(np.abs(diff), self.step)*np.sign(diff)

        # pass the frame to the parent function
        return super(FilterMoveTowards, self)._process_frame(frame)
    
    
    
#===============================================================================
# FILTERS THAT ANALYZE CONSECUTIVE FRAMES
#===============================================================================



class FilterDiffBase(VideoFilterBase):
    """
    Base class for filtering a video based on comparing consecutive frames.
    """ 
    
    def __init__(self, source):
        """
        dtype contains the dtype that is used to calculate the difference.
        If dtype is None, no type casting is done.
        """
        
        self._prev_frame = None
        
        # correct the frame count since we are going to return differences
        super(FilterDiffBase, self).__init__(source, frame_count=source.frame_count-1)
    
    
    def set_frame_pos(self, index):
        # set the underlying movie to requested position 
        self._source.set_frame_pos(index)
        # advance one frame and save it in the previous frame structure
        self._prev_frame = self._source.next()
    
    
    def _compare_frames(self, this_frame, prev_frame):
        raise NotImplementedError
    
      
    def get_frame(self, index):
        return self._compare_frames(self._source.get_frame(index + 1),
                                    self._source.get_frame(index)) 
    
    
    def next(self):
        # get this frame and evaluate it
        this_frame = self._source.next()
        result = self._compare_frames(this_frame, self._prev_frame)

        # this frame will be the previous frame of the next one
        self._prev_frame = this_frame
        
        return result



class FilterTimeDifference(FilterDiffBase):
    """
    returns the differences between consecutive frames.
    This filter is best used by just iterating over it. Retrieving individual
    frame differences can be a bit slow, since two frames have to be loaded.
    """ 
    
    def __init__(self, source, dtype=np.int16):
        """
        dtype contains the dtype that is used to calculate the difference.
        If dtype is None, no type casting is done.
        """
        
        self._dtype = dtype
        
        # correct the frame count since we are going to return differences
        super(FilterTimeDifference, self).__init__(source)

        logging.debug('Created filter for calculating differences between consecutive frames.')

      
    def _compare_frames(self, this_frame, prev_frame):
        # cast into different dtype if requested
        if self._dtype is not None:
            this_frame = this_frame.astype(self._dtype) 
        return this_frame - prev_frame
    
    

class FilterOpticalFlow(FilterDiffBase):
    """
    calculates the flow of consecutive frames 
    """
    
    def __init__(self, *args, **kwargs):
        super(FilterOpticalFlow, self).__init__(*args, **kwargs)
    
        
    def _compare_frames(self, this_frame, prev_frame):
        flow = cv2.calcOpticalFlowFarneback(prev_frame, this_frame,
                                            pyr_scale=0.5, levels=3,
                                            winsize=2, iterations=3, poly_n=5,
                                            poly_sigma=1.2, flags=0)
    
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return mag
    
