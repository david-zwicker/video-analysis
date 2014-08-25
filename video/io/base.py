'''
Created on Jul 31, 2014

@author: zwicker

Package provides an abstract base classes to define an interface and common
functions for video handling.

All video classes support the iterator interface. For convenience, the methods
_start_iterating and _end_iterating are called to initialize and finalize the
iteration. Iteration is initialized implicitly when iter(video) is called.
Conversely, the iterator is finalized when its exhausted (at which point the
StopIteration exception is also raised). Additionally, an iteration may be
aborted by the user by calling the abort method.
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
        size stores the dimensions of the video (width x height)
        frame_count stores the number of frames
        fps are the frames per second
        is_color indicates whether the video is in color or monochrome
        """
        
        # check arguments for consistency
        if len(size) != 2:
            raise ValueError('Videos must have two spatial dimensions.') 
        
        # store information about the video
        self.frame_count = frame_count
        self.size = size
        self.fps = fps if fps is not None else 25
        self.is_color = is_color
        
        # flag that tells whether the video is currently been iterated over
        self._is_iterating = False
        # a list of listeners, which will be notified, when this video advances 
        self._listeners = []
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    
    def __str__(self):
        """ returns a string representation with important properties of the video """
        result = "%s(size=%s, frame_count=%s, fps=%s, is_color=%s)" % (
                    self.__class__.__name__, self.size, self.frame_count,
                    self.fps, self.is_color
                )
        if len(self._listeners) == 1:
            result += '[1 listener]'
        elif len(self._listeners) > 0:
            result += '[%d listeners]' % len(self._listeners)
        return result

    
    #===========================================================================
    # DATA ACCESS
    #===========================================================================
    
    
    def __len__(self):
        return self.frame_count
    
    
    @property
    def shape(self):
        """ returns the shape of the data describing the movie """
        shape = (self.frame_count, self.size[1], self.size[0])
        if self.is_color:
            shape += (3,)
        return shape
    
    
    @property
    def video_format(self):
        """ return a dictionary specifying properties of the video """
        return {'size': self.size,
                'frame_count': self.frame_count,
                'fps': self.fps,
                'is_color': self.is_color}
    
    
    def register_listener(self, listener_callback):
        """ registers a listener function, which will be called if this video is advanced """
        self._listeners.append(listener_callback)
    
    
    def unregister_listener(self, listener_callback):
        """ unregisters a listener function """
        self._listeners.remove(listener_callback)
    
    
    def get_frame_pos(self):
        """ returns the 0-based index of the next frame """
        return self._frame_pos


    def set_frame_pos(self, index):
        """ sets the 0-based index of the next frame """
        if 0 <= index < self.frame_count:
            self._frame_pos = index
        else:
            raise IndexError('Seeking to frame %d was not possible.' % index)


    def _process_frame(self, frame):
        """ returns the frame with a filter applied """
        # notify potential observers
        for observer in self._listeners:
            observer(frame)
        return frame

      
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


    def abort_iteration(self):
        """ stop the current iteration """
        self._end_iterating()


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
    
    
    def close(self):
        """ close the video and free acquired resources """
        pass


    def __getitem__(self, key):
        """ returns a single frame or a video corresponding to a slice """ 
        if isinstance(key, slice):
            return VideoSlice(self, *key.indices(self.frame_count))
        
        elif isinstance(key, int):
            return self.get_frame(key)
        
        else:
            raise TypeError("Invalid key `%r` for indexing" % key)
        
        
    def __setitem__(self, key, value):
        """ writes video data to the frame or slice given in key """
        raise ValueError("Writing to this video stream is prohibited.")  
        
    
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
        size = (frame.shape[1], frame.shape[0])
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
        
        # check the status of the source video
        if source._is_iterating:
            raise RuntimeError('Cannot put a filter on a video that is already iterating.')
        
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
        result = str(self._source) + ' +' + self.__class__.__name__        
        if len(self._listeners) == 1:
            result += '[1 listener]'
        elif len(self._listeners) > 0:
            result += '[%d listeners]' % len(self._listeners)
        return result

    
    def _start_iterating(self):
        """ internal function called when we starting iterating """
        self._source_iter = iter(self._source)
        super(VideoFilterBase, self)._start_iterating()


    def _end_iterating(self):
        """ internal function called when we finished iterating
        If propagate is True, this signal is propagated to the source file.
        This can be important if a filter in the filter chain decides to end
        the iteration (i.e. the VideoSlice class).  Under normal
        circumstances, the video source should end the iteration
        """
        # end the iteration of the current class
        self._source_iter = None
        super(VideoFilterBase, self)._end_iterating()

    
    def abort_iteration(self):
        """ stop the current iteration """
        self._source.abort_iteration()
        super(VideoFilterBase, self).abort_iteration()
        
        
    def set_frame_pos(self, index):
        self._source.set_frame_pos(index)
        # this recursive function call is actually not necessary when rewinding
        # the video before iterating, because we call
        # self._source._start_iterating() elsewhere. However, set_frame_pos
        # is suppose to only change the position of the video, when it is
        # actually different from the current position and the extra call
        # to set_frame_pos should thus not cause any performance issues. 
        
        super(VideoFilterBase, self).set_frame_pos(index)
    
    
    def get_frame(self, index):
        return self._process_frame(self._source.get_frame(index))
          
          
    def __iter__(self):
        self._start_iterating()
        return self
    
                
    def next(self):
        try:
            return self._process_frame(self._source_iter.next())
        except StopIteration:
            self._end_iterating()
            raise
    
    
    def close(self, propagate=True):
        """ closes a video and releases all resources it holds """
        if propagate and isinstance(self._source, VideoFilterBase):
            self._source.close(propagate=True)
        else:
            self._source.close()
            
        

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
            # propagate ending the iteration through the filter chain
            self.abort_iteration()
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
    
    
    
class _VideoForkClient(VideoBase):
    """ internal class representing the client of a VideoFork """
    
    def __init__(self, video_fork):
        """ initializes a private video fork client """
        self._parent = video_fork
        super(_VideoForkClient, self).__init__(**video_fork.video_format)
        
        
    def get_frame(self, index):
        """
        this function is called by the next method of the iterator.
        Instead of iterating the source video itself, it asks the parent for
        the current frame. Depending on the state of the other fork clients,
        the parent may return a cached frame or retrieve a new one
        """
        frame = self._parent.get_next_frame(index)

        # check whether we exhausted the iterator        
        if frame is StopIteration:
            self._end_iterating()
            raise StopIteration
        
        return frame


    def abort_iteration(self):
        self._parent.abort_iteration()
        super(_VideoForkClient, self).abort_iteration()
    
    
    def close(self):
        """ ask video fork to send SystemExit to all clients """
        self._parent.abort_iteration()
    


class SynchronizationError(RuntimeError):
    pass


    
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
            
    VideoFork has a simple test whether all clients are synchronized, which is
    based on the number of times a frame has been requested.
    """
    
    def __init__(self, source, synchronized=True, client_count=None):
        """ initialize the video fork
        If synchronized is True, all clients must be iterated together
        """
        self.synchronized = synchronized
        self._client_count = client_count
        self._clients = []
        self._frame = None
        self._frame_index = -1
        self._retrieve_count = np.inf #< how often was the current frame retrieved?
        self.status = 'normal'
        super(VideoFork, self).__init__(source)
        
        logging.debug('Created video fork.')


    @property
    def client_count(self):
        if self._client_count is None:
            return len(self._clients)
        else:
            return self._client_count


    def set_frame_pos(self, index):
        """ set the position pointer for the video fork and all clients """
        super(VideoFork, self).set_frame_pos(index)
        
        # synchronize all the clients
        for client in self._clients:
            client.set_frame_pos(index)

        self._frame = None
        self._frame_index = index - 1
    
    
    def get_frame(self, index):
        """ 
        returns the frame with a specific index.
        """
        return self._source.get_frame(index)
        
     
    def get_next_frame(self, index):
        """ 
        returns the next frame for a VideoForkClient.
        If the frame is in the cache it is directly returned.
        If the frame is the next in line it is retrieved from the source.
        In any other case a RuntimeError is raised, since it is assumed
        that this function is only used to iterate sequentially.
        """
        if self.status == 'aborting':
            raise SystemExit('Another client of the VideoFork requested to '
                             'abort the iteration.')
        
        if index == self._frame_index:
            # increase the counter and return the cached frame
            self._retrieve_count += 1

        elif not self._is_iterating:
            raise RuntimeError('VideoFork is not iterating')
        
        elif index == self._frame_index + 1:
            # retrieve the next frame from the source video iterator
            
            # check whether the other clients are synchronized
            if self.synchronized and self._retrieve_count < self.client_count:
                raise SynchronizationError('The other clients have not yet read '
                                           'the previous frame.')
            
            # save the index of the frame, which we are about to get
            self._frame_index = self.get_frame_pos()
            
            # get the frame and store it in the cache
            try:
                self._frame = self._source_iter.next()
            except StopIteration:
                self._end_iterating()
                self._frame = StopIteration
                
            self._retrieve_count = 1
            self._frame_pos += 1
            
        else:
            raise SynchronizationError('The clients of the video fork ran out of sync. '
                                       'The parent process is at frame %d, while one client '
                                       'requested frame %d' % (self._frame_index, index))
        
        return self._frame    
     
        
    def next(self):
        raise RuntimeError('VideoFork cannot be directly iterated over')
    
                
    def _end_iterating(self, propagate=False):
        """
        ends the iteration and removes all the fork clients.
        Note that the clients may still retrieve the last frame.
        """
        # remove all the iterators
        self._clients = []
        logging.debug('Finished iterating and unregistered all clients of '
                      'the video fork.')
        
    
    def abort_iteration(self):
        """ send SystemExit to all other clients """
        logging.info('The video fork is aborting by sending a SystemExit '
                     'signal to clients.')
        self.status = 'aborting'
        super(VideoFork, self).abort_iteration()


    def __iter__(self):
        """
        returns a new fork client, which can then be used for iteration.
        """
        # the iteration of this class is initialized when the first client is iterated over
        if len(self._clients) == 0:
            self._start_iterating()
        
        if self._client_count is not None and len(self._clients) >= self._client_count:
            raise ValueError('We already registered %d clients.' % self._client_count)

        # create and register the client iterator
        client = _VideoForkClient(self)
        self._clients.append(client)
        logging.debug('Registered client %d for video fork.', 
                      len(self._clients))
        
        self._retrieve_count = np.inf
        
        return iter(client)
    