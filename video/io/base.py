'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base classes to define an interface and common
functions for video handling.
'''

from __future__ import division

import glob
import logging
import numpy as np

from video.utils import display_progress


class VideoBase(object):
    """
    Base class for videos.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    """

    write_access = False  
    
    def __init__(self, size=(0, 0), frame_count=-1, fps=None, is_color=True):
        """
        size stores the dimensions of the video
        frame_count stores the number of frames
        fps are the frames per second
        is_color indicates whether the video is in color or monochrome
        """
        
        # store information about the video
        self.frame_count = frame_count
        self.size = size
        self.fps = fps if fps is not None else 25
        self.is_color = is_color
        
        self._is_iterating = False
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    
    def __str__(self):
        return "%s(size=%s, frame_count=%s, fps=%s, is_color=%s)" % (
                self.__class__.__name__, self.size, self.frame_count,
                self.fps, self.is_color
            )
    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
    def __len__(self):
        return self.frame_count
    
    
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        shape = (self.frame_count, self.size[0], self.size[1])
        if self.is_color:
            shape += (3,)
        return shape
    
    
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return self._frame_pos


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if 0 <= index < self.frame_count:
            self._frame_pos = index
        else:
            raise IndexError('Seeking to frame %d was not possible.' % index)

      
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        raise NotImplementedError

    
    def _start_iterating(self):
        """ internal function called when we start iterating """
        if self._is_iterating:
            raise RuntimeError("Videos cannot be iterated over multiple times "
                               "simultaneously. If you need to do this, use a "
                               "VideoFork or make a copy of the video before "
                               "iterating.")
        
        # rewind the movie
        self.set_frame_pos(0)
        self._is_iterating = True
    

    def _end_iterating(self):
        """ internal function called when we finished iterating """
        self._is_iterating = False


    def __iter__(self):
        """ initializes the iterator """
        self._start_iterating()
        return self

          
    def next(self):
        """
        returns the next frame while iterating over a movie.
        This is a generic function, which just retrieves the next image based
        on the internal frame_pos. Subclasses may overwrite this with more
        efficient implementations (i.e. for streaming)
        """
        # retrieve current frame
        try:
            frame = self.get_frame(self._frame_pos)
        except IndexError:
            # stop iterating
            self._end_iterating()            
            raise StopIteration

        # set the internal pointer to the next frame
        self._frame_pos += 1
        return frame


    def __getitem__(self, key):
        """ returns a single frame or a video corresponding to a slice """ 
        if isinstance(key, slice):
            return VideoSlice(self, *key.indices(self.frame_count))
        
        elif isinstance(key, int):
            return self.get_frame(key)
        
        else:
            raise TypeError("Invalid key for indexing")
        
        
    def __setitem__(self, key, value):
        """ writes video data to the frame or slice given in key """
        raise ValueError("Writing to this video stream is prohibited.")  
        

    #===========================================================================
    # CONTROL THE DATA STREAM OF THE MOVIE
    #===========================================================================
    
    def copy(self, dtype=np.uint8):
        """
        Creates a copy of the current video and returns a VideoMemory instance.
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        logging.debug('Copy a video stream and store it in memory')
        
        # copy the data into a numpy array
        data = np.empty(self.shape, dtype)
        for k, val in enumerate(display_progress(self)):
            data[k, ...] = val
            
        # construct the memory object without copying the data
        return VideoMemory(data, fps=self.fps, copy_data=False)
    
    

class VideoImageStackBase(VideoBase):
    """ abstract base class that represents a movie stored as individual frame images """
    
    def __init__(self, filename_scheme, fps=None):
        # find all the files belonging to this stack
        self.filenames = sorted(glob.glob(filename_scheme))
        frame_count = len(self.filenames)
        
        # load the first frame to get information on color
        frame = self.get_frame(0)
        size = frame.shape[:2]
        if frame.ndim == 2 or frame.shape[2] == 1:
            is_color = False
        elif frame.shape[2] == 3:
            is_color = True
        else:
            raise ValueError('The last dimension of the data must be either 1 or 3.')
                
        super(VideoImageStackBase, self).__init__(size=size, frame_count=frame_count,
                                                  fps=fps, is_color=is_color)
        
        

class VideoFilterBase(VideoBase):
    """
    class which applies a filter function to each frame of a video.
    This class does not hold its own data, but is more like a view in numpy.
    """
     
    def __init__(self, source, size=None, frame_count=None, fps=None, is_color=None):
        # store an iterator of the source video
        self._source = source
        self._source_iter = None
        
        # determine properties of the video
        size = source.size if size is None else size
        frame_count = source.frame_count if frame_count is None else frame_count
        fps = source.fps if fps is None else fps
        is_color = source.is_color if is_color is None else is_color
        
        # initialize the base video
        super(VideoFilterBase, self).__init__(
            size=size, frame_count=frame_count, fps=fps, is_color=is_color
        )
        
        
    def __str__(self):
        """ delegate the string function to actual source """
        return str(self._source) + ' +' + self.__class__.__name__        

    
    def _start_iterating(self):
        """ internal function called when we starting iterating """
        self._source_iter = iter(self._source)
        super(VideoFilterBase, self)._start_iterating()


    def _end_iterating(self):
        """ internal function called when we finished iterating """
        #self._source._end_iterating() # should not be necessary
        self._source_iter = None
        super(VideoFilterBase, self)._end_iterating()

        
    def _filter_frame(self, frame):
        """ returns the frame with a filter applied """
        raise NotImplementedError


    def set_frame_pos(self, index):
        self._source.set_frame_pos(index)
        super(VideoFilterBase, self).set_frame_pos(index)
    
    
    def get_frame(self, index):
        return self._filter_frame(self._source.get_frame(index))
          
          
    def __iter__(self):
        self._start_iterating()
        return self
    
                
    def next(self):
        try:
            return self._filter_frame(self._source_iter.next())
        except StopIteration:
            self._end_iterating()
            raise
    


class VideoSlice(VideoFilterBase):
    """ Video that iterates only over a part of the frames """
    
    def __init__(self, source, start=0, stop=None, step=1):
        
        if step == 0:
            raise ValueError('step argument must not be zero.')
        
        self._start = start
        self._stop = self.source.frame_count if stop is None else stop 
        self._step = step
            
        # calculate the number of frames to be expected
        frame_count = int(np.ceil((self._stop - self._start)/self._step))
        
        # correct the size, since we are going to crop the movie
        super(VideoSlice, self).__init__(source, frame_count=frame_count)

        logging.debug('Created video slice [%d:%d:%d] of length %d.' % 
                      (self._start, self._stop, self._step, frame_count)) 
        if step < 0:
            logging.warn('Reversing a video can slow down the processing significantly.')
        
        
    def set_frame_pos(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)
        self._source.set_frame_pos(self._start + index*self._step)
        self._frame_pos = index
        
        
    def get_frame(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)

        return self._source.get_frame(self._start + index*self._step)
        
        
    def next(self):
        # check whether we reached the end
        if self.get_frame_pos() >= self.frame_count:
            self._source._end_iterating()
            self._end_iterating()
            raise StopIteration
        
        if self._step == 1:
            # return the next frame in question
            frame = self._source.next()
        else:
            # return the specific frame in question
            frame = self.get_frame(self._frame_pos)

        # advance to the next frame
        self._frame_pos += 1
        return frame
    
    
    
class _VideoForkChild(VideoBase):
    """ internal class representing the child of a VideoFork """
    
    def __init__(self, video_fork):
        """ initializes a private video fork child """
        self._parent = video_fork
        super(_VideoForkChild, self).__init__(size=video_fork.size,
                                              frame_count=video_fork.frame_count,
                                              fps=video_fork.fps,
                                              is_color=video_fork.is_color)
        
        
    def get_frame(self, index):
        """
        this function is called by the next method of the iterator.
        Instead of iterating the source video itself, it asks the parent for
        the current frame. Depending on the state of the other fork children,
        the parent may return a cached frame or retrieve a new one
        """
        frame = self._parent.get_frame(index)

        # check whether we exhausted the iterator        
        if frame is StopIteration:
            self._end_iterating()
            raise StopIteration
        
        return frame


    
class VideoFork(VideoFilterBase):
    """
    Class that distributes frames to multiple filters.
    This class can be used as an iterator multiple times. However, the user is
    responsible to keep these iterators in synchrony, i.e. all iterators must
    digest frames at the same rate. This can be useful, if multiple, different
    filters are to be applied to the same video independently.
    
        video_fork = VideoFork(video)
        video_1 = Filter1(video_fork)
        video_2 = Filter2(video_fork)
        
        for frame_1, frame_2 in itertools.izip(video_1, video_2):
            compare_frames(frame_1, frame_2)
    """
    
    def __init__(self, source):
        """ initialize the video fork """
        self._iterators = []
        self._frame = None
        self._frame_index = -1
        super(VideoFork, self).__init__(source)
        
        logging.debug('Created video fork.')

    
    def set_frame_pos(self, index):
        """ set the position pointer for the video fork and all children """
        super(VideoFork, self).set_frame_pos(index)
        
        # synchronize all the children
        for iterator in self._iterators:
            iterator.set_frame_pos(index)

        self._frame = None
        self._frame_index = index - 1
    
    
    def get_frame(self, index):
        """ 
        returns the frame with a specific index.
        If the frame is in the cache it is directly returned.
        If the frame is the next in line it is retrieved from the source.
        In any other case a RuntimeError is raised, since it is assumed
        that this function is only used to iterate sequentially.
        """
        if index == self._frame_index:
            # just return the cached frame
            pass
        
        elif index == self._frame_index + 1:
            # retrieve the next frame from the source video
            
            # save the index of the frame, which we are about to get
            self._frame_index = self.get_frame_pos()
            
            # get the frame and store it in the cache
            try:
                self._frame = self._source_iter.next()
            except StopIteration:
                self._end_iterating()
                self._frame = StopIteration
                
            self._frame_pos += 1
            
        else:
            raise RuntimeError('The children of the video fork ran out of sync. '
                               'The parent process is at frame %d, while one child '
                               'requested frame %d' % (self._frame_index, index))
        
        return self._frame
        
        
    def next(self):
        raise RuntimeError('VideoFork cannot be directly iterated over')
    
        
    def _end_iterating(self):
        """
        ends the iteration and removes all the fork children.
        Note that the children may still retrieve the last frame.
        """
        # remove all the iterators
        self._iterators = []
        logging.debug('Finished iterating and unregistered all children of '
                      'of the video fork.')
        super(VideoFork, self)._end_iterating()


    def __iter__(self):
        """
        returns a new fork child, which can then be used for iteration.
        """
        # the iteration of this class is initialized when the first child is iterated over
        if len(self._iterators) == 0:
            self._start_iterating()

        # create and register the child iterator
        child = _VideoForkChild(self)
        self._iterators.append(child)
        logging.debug('Registered child %d for video fork.', 
                      len(self._iterators))
        
        # return the child
        return iter(child)
    
        