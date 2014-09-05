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
from .analysis.regions import rect_to_slices
from .utils import get_color_range

logger = logging.getLogger('video')
  

# translation dictionary for color channels
COLOR_CHANNELS = {'blue':  0, 'b': 0, 0: 0,
                  'green': 1, 'g': 1, 1: 1,
                  'red':   2, 'r': 2, 2: 2}
    
  

class FilterFunction(VideoFilterBase):
    """ smoothes every frame """
    
    def __init__(self, source, function):
        
        self._function = function
        
        super(FilterFunction, self).__init__(source)
        
        logger.debug('Created filter applying a function to every frame')


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
        logger.debug('Created filter for normalizing range [%g..%g]', vmin, vmax)


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
                logger.warn('Lower normalization bound is below what the format can hold.')
            if self._fmax > fmax:
                logger.warn('Upper normalization bound is above what the format can hold.')

        # clip the data before converting
        np.clip(frame, self._fmin, self._fmax, out=frame)

        # do the conversion from [fmin, fmax] to [tmin, tmax]
        frame = (frame - self._fmin)*self._alpha + self._tmin
        
        # cast the data to the right type
        frame = frame.astype(self._dtype)
        
        # pass the frame to the parent function
        return super(FilterNormalize, self)._process_frame(frame)



def _check_coordinate(value, max_value):
    """ helper function checking the bounds of the rectangle """
    if -1 < value < 1:
        # convert to integer by interpreting float values as fractions
        value = int(value*max_value)
    
    # interpret negative numbers as counting from opposite boundary
    if value < 0:
        value += max_value
        
    # check whether the value is within bounds
    if not 0 <= value < max_value:
        raise IndexError('Coordinate %d of points is out of bounds.')
    
    return value
  
    

class FilterCrop(VideoFilterBase):
    """ crops the video to the given rect=(top, left, height, width) """

    def __init__(self, source, rect=None, region='', color_channel=None):
        """
        initialized the filter that crops to the given rect=(left, top, width, height)
        Alternative, the class understands the special strings 'lower', 'upper', 'left',
        and 'right, which can be given in the region parameter. 
        If color_channel is given, it is assumed that the input video is a color
        video and only the specified color channel is returned, thus turning
        the video into a monochrome one
        """
        source_width, source_height = source.size
        
        if rect is not None:
            # interpret float values as fractions of width/height
            left =   _check_coordinate(rect[0], source_width)
            top =    _check_coordinate(rect[1], source_height)
            width =  _check_coordinate(rect[2], source_width)
            height = _check_coordinate(rect[3], source_height)
                    
        else:
            # construct the rect from the given string
            region = region.lower()
            left, top = 0, 0
            width, height = source_width, source_height
            if 'left' in region:
                width //= 2 
            elif 'right' in region:
                width //= 2
                left = source_width - width 
            
            if 'upper' in region:
                height //= 2
            elif 'lower' in region:
                height //= 2
                top = source_height - height
        
        # contract with parent crop filters, if they exist 
        while isinstance(source, FilterCrop):
            logger.debug('Combine this crop filter with the parent crop filter.')
            left += source.rect[0]
            top += source.rect[1]
            if source.color_channel is not None:
                color_channel = source.color_channel
            source = source._source
                     
        # extract color information
        self.color_channel = COLOR_CHANNELS.get(color_channel, color_channel)
        is_color = None if color_channel is None else False

        # create the rectangle and store it 
        self.rect = (left, top, width, height)
        self.slices = rect_to_slices(self.rect)

        # correct the size, since we are going to crop the movie
        super(FilterCrop, self).__init__(source, size=self.rect[2:], is_color=is_color)
        logger.debug('Created filter for cropping to rectangle %s', self.rect)
        
       
    def _process_frame(self, frame):
        if self.color_channel is None:
            # extract the given rectangle 
            frame = frame[self.slices]

        else:
            # extract the given rectangle and get the color channel 
            frame = frame[self.slices[0], self.slices[1], self.color_channel]

        # pass the frame to the parent function
        return super(FilterCrop, self)._process_frame(frame)



class FilterMonochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='mean'):
        self.mode = COLOR_CHANNELS.get(mode.lower(), mode.lower())
        super(FilterMonochrome, self).__init__(source, is_color=False)

        logger.debug('Created filter for converting video to monochrome with method `%s`', mode)


    def _process_frame(self, frame):
        """
        reduces a single frame from color to monochrome, but keeps the
        extra dimension in the data
        """
        try:
            if self.mode == 'mean':
                frame = np.mean(frame, axis=2).astype(frame.dtype)
            else:
                frame = frame[:, :, self.mode]
        except ValueError:
            raise ValueError('Unsupported conversion method to monochrome: %s' % self.mode)
    
        # pass the frame to the parent function
        return super(FilterMonochrome, self)._process_frame(frame)
    
    

class FilterBlur(VideoFilterBase):
    """ returns the mouse video with a Gaussian blur filter """

    def __init__(self, source, sigma=3):
        self.sigma = sigma
        super(FilterBlur, self).__init__(source)

        logger.debug('Created filter blurring the video with radius %g', sigma)

        
    def _process_frame(self, frame):
        """
        blurs a single frame
        """
        return cv2.GaussianBlur(frame.astype(np.uint8), (0, 0), self.sigma)



class FilterFeatures(VideoFilterBase):
    """ detects features and draws them onto the image """
    
    def _process_frame(self, frame):
        frame = frame.copy()
        
        corners = cv2.goodFeaturesToTrack(frame, maxCorners=100,
                                          qualityLevel=0.01,minDistance=10)
   
        for i in corners:
            x,y = i.ravel()
            cv2.circle(frame, (x,y), 5, 255, -1)
            
        # pass the frame to the parent function
        return super(FilterFeatures, self)._process_frame(frame)
    
    

class FilterBackground(VideoFilterBase):
    """ filter that filters out the background of a video """
    
    def __init__(self, source, timescale=128):
        self.background = None
        self.timescale = timescale
        super(FilterBackground, self).__init__(source)
    
    
    def _process_frame(self, frame):
        # adapt the current background model
        # Here, the cut-off sets the relaxation time scale
        #diff_max = 128/self.background_history
        #self.background += np.clip(diff, -diff_max, diff_max)
        self.background = np.minimum(self.background, frame)

        # pass the frame to the parent function
        return super(FilterBackground, self)._process_frame(self.background)
    
    
    
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

        logger.debug('Created filter for calculating differences between consecutive frames.')

      
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
    
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return mag
    
