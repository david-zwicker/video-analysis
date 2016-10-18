'''
Created on Aug 1, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Filter are iterators that take a video as an input and return a special Video
that can be iterated over, but that doesn't store its data in memory.

Some filters allow to access the underlying frames at random using the
get_frame method

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
from utils.math import get_number_range

logger = logging.getLogger('video')

# translation dictionary for color channels
COLOR_CHANNELS = {'blue':  0, 'b': 0, 0: 0,
                  'green': 1, 'g': 1, 1: 1,
                  'red':   2, 'r': 2, 2: 2}
    
    

def get_color_range(dtype):
    """
    determines the color depth of the numpy array `data`.
    If the dtype is an integer, the range that it can hold is returned.
    If dtype is an inexact number (a float point), zero and one is returned
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.floating):
        return 0, 1        
    else:
        raise ValueError('Unsupported data type `%r`' % dtype)


    
  

class FilterFunction(VideoFilterBase):
    """ applies function to every frame """
    
    def __init__(self, source, function):
        
        self._function = function
        
        super(FilterFunction, self).__init__(source)
        
        logger.debug('Created filter applying a function to every _frame')


    def _process_frame(self, frame):
        # process the current _frame 
        frame = self._function(frame)
        # pass it to the parent function
        return super(FilterFunction, self)._process_frame(frame)

    
    
class FilterNormalize(VideoFilterBase):
    """ normalizes a color range to the interval 0..1 """
    
    def __init__(self, source, vmin=None, vmax=None, dtype=None):
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
        logger.debug('Created filter for normalizing range [%g..%g]',
                     vmin, vmax)


    def _process_frame(self, frame):
        
        # get dtype and bounds from first _frame if they were not specified
        if self._dtype is None:
            self._dtype = frame.dtype
        if self._fmin is None:
            self._fmin = frame.min()
        if self._fmax is None:
            self._fmax = frame.max()
            
        # ensure that we know the bounds of this dtype
        if self._tmin is None:
            self._tmin, tmax = get_color_range(self._dtype)
            self._alpha = (tmax - self._tmin)/(self._fmax - self._fmin)
            
            # some safety checks on the first run:
            fmin, fmax = get_number_range(frame.dtype)
            if self._fmin < fmin:
                logger.warn('Lower normalization bound is below what the '
                            'format can hold.')
            if self._fmax > fmax:
                logger.warn('Upper normalization bound is above what the '
                            'format can hold.')

        # clip the data before converting
        np.clip(frame, self._fmin, self._fmax, out=frame)

        # do the conversion from [fmin, fmax] to [tmin, tmax]
        frame = (frame - self._fmin)*self._alpha + self._tmin
        
        # cast the data to the right type
        frame = frame.astype(self._dtype)
        
        # pass the _frame to the parent function
        return super(FilterNormalize, self)._process_frame(frame)



def _check_coordinate(value, max_value):
    """ helper function checking the bounds of the rectangle """
    if -1 < value < 1:
        # convert to integer by interpreting float values as fractions
        value = int(value * max_value)
    
    # interpret negative numbers as counting from opposite boundary
    if value < 0:
        value += max_value
        
    # check whether the value is within bounds
    if not 0 <= value < max_value:
        raise IndexError('Coordinate %d is out of bounds [0, %d].'
                         % (value, max_value))
    
    return value
  
    

class FilterCrop(VideoFilterBase):
    """ crops the video to the given region """

    def __init__(self, source, rect=None, region='', color_channel=None,
                 size_alignment=1):
        """
        initialized the filter that crops the video to the specified rectangle.
        
        The rectangle can be either given directly by supplying
        rect=(left, top, width, height) or a region can be specified in the
        `region` parameter, which can take the following values : 'lower',
        'upper', 'left', and 'right or combinations thereof. If both `rect` and
        `region` are supplied, the `region` is discarded.
         
        If color_channel is given, it is assumed that the input video is a color
        video and only the specified color channel is returned, thus turning
        the video into a monochrome one.
        
        `size_alignment` can be given to force the width and the height to be a
            multiple of the given integer. This might be useful to force the
            size to be an even number, which some video codecs require. The 
            default value is 1 and the width and the height is thus any integer.
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
            logger.debug('Combine this crop filter with the parent one.')
            left += source.rect[0]
            top += source.rect[1]
            if source.color_channel is not None:
                color_channel = source.color_channel
            source = source._source
                     
        # extract color information
        self.color_channel = COLOR_CHANNELS.get(color_channel, color_channel)
        is_color = None if color_channel is None else False

        # enforce alignment
        if size_alignment != 1:
            # we use round to make sure we pick the size that is closest to the
            # specified one
            width = int(round(width / size_alignment) * size_alignment)
            height = int(round(height / size_alignment) * size_alignment)

        # create the rectangle and store it 
        self.rect = (left, top, width, height)
        self.slices = rect_to_slices(self.rect)

        # correct the size, since we are going to crop the movie
        super(FilterCrop, self).__init__(source, size=self.rect[2:],
                                         is_color=is_color)
        logger.debug('Created filter for cropping to rectangle %s', self.rect)
        
       
    def _process_frame(self, frame):
        if self.color_channel is None:
            # extract the given rectangle 
            frame = frame[self.slices]

        else:
            # extract the given rectangle and get the color channel
            frame = frame[self.slices[0], self.slices[1], self.color_channel]

        # pass the _frame to the parent function
        return super(FilterCrop, self)._process_frame(frame)



class FilterResize(VideoFilterBase):
    """ resizes the video to a new size """

    def __init__(self, source, size=None, interpolation='auto',
                 even_dimensions=False):
        """
        initialized the filter that crops to the given size=(width, height)
        If size is a single value it is interpreted as a factor of the input
        video size.
            `interpolation` chooses the interpolation used for the resizing
            `even_dimensions` is a flag that determines whether the image
                dimensions are enforced to be even numbers
        """
        # determine target size
        if hasattr(size, '__iter__'):
            width, height = size
        else:
            width = int(source.size[0] * size)
            height = int(source.size[1] * size)
        if even_dimensions:
            width += (width % 2)
            height += (height % 2)
            
        # set interpolation method
        if (width, height) == source.size:
            self.interpolation = None
        elif interpolation == 'auto':
            if width*height < source.size[0]*source.size[1]:
                # image size is decreased
                self.interpolation = cv2.INTER_AREA
            else:
                # image size is increased
                self.interpolation = cv2.INTER_CUBIC                
        elif interpolation == 'nearest':
            self.interpolation = cv2.INTER_NEAREST
        elif interpolation == 'linear':
            self.interpolation = cv2.INTER_LINEAR
        elif interpolation == 'area':
            self.interpolation = cv2.INTER_AREA
        elif interpolation == 'cubic':
            self.interpolation = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            self.interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError('Unknown interpolation method: %s', interpolation)
             
        # contract with parent crop filters, if they exist 
        while isinstance(source, FilterResize):
            logger.debug('Combine this resize filter with the parent one.')
            source = source._source

        # correct the size, since we are going to crop the movie
        super(FilterResize, self).__init__(source, size=(width, height))
        logger.debug('Created filter for resizing to size %dx%d', width, height)
        
       
    def _process_frame(self, frame):
        # resize the frame if necessary
        if self.interpolation:
            frame = cv2.resize(frame, self.size,
                               interpolation=self.interpolation)

        # pass the frame to the parent function
        return super(FilterResize, self)._process_frame(frame)
    


class FilterRotate(VideoFilterBase):
    """ returns the video rotated in counter-clockwise direction """
    
    def __init__(self, source, angle=0):
        """ rotate the video by angle in counter-clockwise direction """
        angle = angle % 360
        
        if angle == 0 or angle == 180:
            size = source.size
        elif angle == 90 or angle == 270:
            size = (source.size[1], source.size[0])
        else: 
            raise ValueError('angle must be from [0, 90, 180, 270] but was %s'
                             % angle)
        self.angle = angle
        
        # correct the size, since we are going to crop the movie
        super(FilterRotate, self).__init__(source, size=size)
        
        
    def _process_frame(self, frame):
        # rotate the array
        frame = np.rot90(frame, self.angle // 90)

        # pass the frame to the parent function
        return super(FilterRotate, self)._process_frame(frame)



class FilterMonochrome(VideoFilterBase):
    """ returns the video as monochrome """
    
    def __init__(self, source, mode='mean'):
        self.mode = COLOR_CHANNELS.get(mode.lower(), mode.lower())
        super(FilterMonochrome, self).__init__(source, is_color=False)

        logger.debug('Created filter for converting video to monochrome with '
                     'method `%s`', mode)


    def _process_frame(self, frame):
        """
        reduces a single _frame from color to monochrome, but keeps the
        extra dimension in the data
        """
        try:
            if self.mode == 'mean':
                frame = np.mean(frame, axis=2).astype(frame.dtype)
            else:
                frame = frame[:, :, self.mode]
        except ValueError:
            raise ValueError('Unsupported conversion method to monochrome: %s'
                             % self.mode)
    
        # pass the frame to the parent function
        return super(FilterMonochrome, self)._process_frame(frame)
    
    

class FilterBlur(VideoFilterBase):
    """ returns the video with a Gaussian blur filter """

    def __init__(self, source, sigma=3):
        self.sigma = sigma
        super(FilterBlur, self).__init__(source)

        logger.debug('Created filter blurring the video with radius %g', sigma)

        
    def _process_frame(self, frame):
        """
        blurs a single _frame
        """
        return cv2.GaussianBlur(frame.astype(np.uint8), (0, 0), self.sigma)


    
class FilterReplicate(VideoFilterBase):
    """ replicates the video `count` times """
    
    def __init__(self, source, count=1):
        
        self.count = count
            
        # calculate the number of frames to be expected
        frame_count = source.frame_count * count

        # correct the size, since we are going to crop the movie
        super(FilterReplicate, self).__init__(source, frame_count=frame_count)

    
    def set_frame_pos(self, index):
        if index < 0:
            index += self.frame_count

        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)

        self._source.set_frame_pos(index % self._source.frame_count)
        self._frame_pos = index
        
        
    def get_next_frame(self):
        if self.get_frame_pos() % self._source.frame_count == 0:
            # rewind source video
            self._source.set_frame_pos(0)
        
        frame = self._source.get_next_frame()

        # advance to the next _frame
        self._frame_pos += 1
        return frame



class FilterDropFrames(VideoFilterBase):
    """ removes frames to accelerate the video """
    
    def __init__(self, source, compression=1):
        """ `source` is the source video and `compression` sets the compression
        factor """
        
        self._compression = compression 
        fps = source.fps / self._compression
            
        # calculate the number of frames to be expected
        frame_count = int((source.frame_count - 1) / self._compression) + 1

        # correct the duration and the fps
        super(FilterDropFrames, self).__init__(source, frame_count=frame_count, 
                                               fps=fps)

        logger.debug('Created filter to change frame rate from %g to %g '
                     '(compression factor: %g)' %
                     (source.fps, self.fps, self._compression))

    
    def _source_index(self, index):
        """ calculates the index in the source video corresponding to the given
        `index` in the current video """
        return int(index * self._compression)
    
    
    def set_frame_pos(self, index):
        if index < 0:
            index += self.frame_count
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)
        
        self._source.set_frame_pos(self._source_index(index))
        self._frame_pos = index

        
    def get_frame(self, index):
        if index < 0:
            index += self.frame_count
        frame = self._source[self._source_index(index)]
        self._frame_pos = index + 1
        return frame

        
    def get_next_frame(self):
        frame = self._source[self._source_index(self._frame_pos)]
        self._frame_pos += 1
        return frame

    
    
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
        
        # correct the _frame count since we are going to return differences
        super(FilterDiffBase, self).__init__(source,
                                             frame_count=source.frame_count - 1)
    
    
    def set_frame_pos(self, index):
        if index < 0:
            index += self.frame_count
        # set the underlying movie to requested position 
        self._source.set_frame_pos(index)
        # advance one _frame and save it in the previous _frame structure
        self._prev_frame = self._source.next()
    
    
    def _compare_frames(self, this_frame, prev_frame):
        raise NotImplementedError
    
      
    def get_frame(self, index):
        if index < 0:
            index += self.frame_count
        return self._compare_frames(self._source.get_frame(index + 1),
                                    self._source.get_frame(index)) 
    
    
    def next(self):
        # get this _frame and evaluate it
        this_frame = self._source.next()
        result = self._compare_frames(this_frame, self._prev_frame)

        # this _frame will be the previous _frame of the next one
        self._prev_frame = this_frame
        
        return result



class FilterTimeDifference(FilterDiffBase):
    """
    returns the differences between consecutive frames.
    This filter is best used by just iterating over it. Retrieving individual
    _frame differences can be a bit slow, since two frames have to be loaded.
    """ 
    
    def __init__(self, source, dtype=np.int16):
        """
        dtype contains the dtype that is used to calculate the difference.
        If dtype is None, no type casting is done.
        """
        
        self._dtype = dtype
        
        # correct the _frame count since we are going to return differences
        super(FilterTimeDifference, self).__init__(source)

        logger.debug('Created filter for calculating differences between '
                     'consecutive frames.')

      
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
    
