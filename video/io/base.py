'''
Created on Jul 31, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Package provides an abstract base classes to define an interface and common
functions for video handling.
'''

from __future__ import division

import glob
import logging
import numpy as np

from utils.misc import display_progress

logger = logging.getLogger('video.io')

# custom error classes
class NotSeekableError(RuntimeError): pass
class SynchronizationError(RuntimeError): pass



class VideoBase(object):
    """
    Base class for videos.
    Every movie has an internal counter `frame_pos` stating which frame would
    be processed next.
    
    Furthermore, each video holds a list of callback functions called observers.
    These functions are called when the video is advanced to the next frame
    and can be used to observe the current progress or show the current frame. 
    """

    write_access = False
    seekable = False
    
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
        
        # a list of listeners, which will be notified, when this video advances 
        self._listeners = []
        
        # internal pointer to the next frame to be loaded when iterating
        # over the video
        self._frame_pos = 0
    
    
    def get_property_list(self):
        """ returns a list of properties in a way which is useful for printing """
        return ('size=(%d, %d)' % self.size,
                'frame_count=%s' % self.frame_count,
                'fps=%s' % self.fps,
                'is_color=%s' % self.is_color)
        
    
    def __str__(self):
        """ returns a string representation with important properties of the video """
        result = "%s(%s)" % (self.__class__.__name__,
                             ', '.join(self.get_property_list()))
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
    def width(self):
        """ returns the width of the video """
        return self.size[0]

    
    @property
    def height(self):
        """ returns the height of the video """
        return self.size[1]

    
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
        if self.seekable:
            # if the video is seekable, we can just set the new position
            if 0 <= index < self.frame_count:
                self._frame_pos = index
            else:
                raise IndexError('Seeking to frame %d was not possible.' % index)
            
        elif index >= self.get_frame_pos():
            # video is not seekable => we can skip some frames to fast forward
            for _ in xrange(self.get_frame_pos(), index):
                self.get_next_frame()
                
        else:
            raise NotSeekableError('Cannot seek to frame %d, because the video '
                                   'is already at frame %d' % 
                                   (index, self.get_frame_pos()))


    def rewind(self):
        """ rewind video to first frame """
        self.set_frame_pos(0) 


    def _process_frame(self, frame):
        """ returns the frame with a filter applied """
        # notify potential observers
        # This has to be done here and cannot be implemented as a filter, since
        # a filter would have to be iterated over directly. 
        for observer in self._listeners:
            observer(frame)
        return frame

      
    def get_frame(self, index):
        """ returns a specific frame identified by its index """ 
        raise NotImplementedError


    def abort_iteration(self):
        """ stop the current iteration """
        pass


    def __iter__(self):
        """ initializes the iterator """
        return VideoIterator(self)

          
    def get_next_frame(self):
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
        
    
    def copy(self, dtype=np.uint8, disp=True):
        """
        Creates a copy of the current video and returns a VideoMemory instance.
        """
        # prevent circular import by lazy importing
        from .memory import VideoMemory
        
        logger.debug('Copy a video stream and store it in memory')
        
        # copy the data into a numpy array
        data = np.empty(self.shape, dtype)
        
        if disp:
            iterator = display_progress(self)
        else:
            iterator = self
        
        for k, val in enumerate(iterator):
            data[k, ...] = val
            
        # construct the memory object without copying the data
        return VideoMemory(data, fps=self.fps, copy_data=False)
    
    
    
class VideoIterator(object):
    """ simple class implementing the iterator interface for videos """
    def __init__(self, video):
        self._video = video
        self._video.rewind() #< rewind video before iterating over it
        
    def next(self):
        return self._video.get_next_frame()
    


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
        
        # store an iterator of the source video
        self._source = source
        
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


    @property
    def seekable(self):
        """ flag indicating whether the video is seekable or not """
        return self._source.seekable

    
    def abort_iteration(self):
        """ stop the current iteration """
        self._source.abort_iteration()
        
        
    def set_frame_pos(self, index):
        self._source.set_frame_pos(index)
        self._frame_pos = index


    def get_frame_pos(self):
        return self._source.get_frame_pos()

    
    def get_frame(self, index):
        frame = self._source.get_frame(index)
        self._frame_pos = index
        return self._process_frame(frame)
          
                
    def get_next_frame(self):
        frame = self._source.get_next_frame()
        self._frame_pos += 1
        return self._process_frame(frame) 
    
    
    def close(self, propagate=True):
        """ closes a video and releases all resources it holds """
        if propagate and isinstance(self._source, VideoFilterBase):
            self._source.close(propagate=True)
        else:
            self._source.close()
            
        

class VideoSlice(VideoFilterBase):
    """ Video that iterates only over a part of the frames """
    
    def __init__(self, source, start=0, stop=None, step=1):
        
        # determine the start position
        if start >= 0:
            self._start = start
        else:
            self._start = self.source.frame_count + start #< negative start
            
        # determine the stop position
        if stop is None:
            self._stop = self.source.frame_count
        elif stop >= 0:
            self._stop = stop
        else:
            self._stop = self.source.frame_count + stop #< negative stop

        # determine the step size (stride)
        if step == 0:
            raise ValueError('step argument must not be zero.')
        self._step = step
            
        # calculate the number of frames to be expected
        frame_count = int(np.ceil((self._stop - self._start) / self._step))

        # seek the source video to the start position
        source.set_frame_pos(start)
        
        # set the new frame count 
        super(VideoSlice, self).__init__(source, frame_count=frame_count)

        logger.debug('Created video slice [%d:%d%s] of length %d.' % 
                     (self._start, self._stop, 
                      '' if self._step == 1 else ':%d' % self._step,
                      frame_count))
        if step < 0:
            logger.warn('Reversing a video can slow down the processing '
                        'significantly.')
        
        
    def set_frame_pos(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)

        self._source.set_frame_pos(self._start + index*self._step)
        self._frame_pos = index


    def get_frame_pos(self):
        return self._frame_pos
        
        
    def get_frame(self, index):
        if not 0 <= index < self.frame_count:
            raise IndexError('Cannot access frame %d.' % index)

        return self._source.get_frame(self._start + index*self._step)
        
        
    def get_next_frame(self):
        # check whether we reached the end
        if self.get_frame_pos() >= self.frame_count:
            # propagate ending the iteration through the filter chain
            self.abort_iteration()
            raise StopIteration
        
        if self._step == 1:
            # return the next frame in question
            frame = self._source.get_next_frame()
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
        
        
        
    def get_next_frame(self):
        """
        this function is called by the next method of the iterator.
        Instead of iterating the source video itself, it asks the parent for
        the current frame. Depending on the state of the other fork clients,
        the parent may return a cached frame or retrieve a new one
        """
        frame = self._parent.get_frame(self._frame_pos)
        self._frame_pos += 1

        # check whether we exhausted the iterator        
        if frame is StopIteration:
            raise StopIteration
        
        return frame


    def abort_iteration(self):
        self._parent.abort_iteration()
        super(_VideoForkClient, self).abort_iteration()
    
    
    def close(self):
        """ ask video fork to send SystemExit to all clients """
        self._parent.abort_iteration()
    


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
        self.state = 'normal'
        super(VideoFork, self).__init__(source)
        
        logger.debug('Created video fork.')


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
        returns the next frame for a VideoForkClient.
        If the frame is in the cache it is directly returned.
        If the frame is the next in line it is retrieved from the source.
        In any other case a RuntimeError is raised, since it is assumed
        that this function is only used to iterate sequentially.
        """
        if self.state == 'aborting':
            raise SystemExit('Another client of the VideoFork requested to '
                             'abort the iteration.')

        if index == self._frame_index:
            # increase the counter and return the cached frame
            self._retrieve_count += 1

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
                self._frame = self.get_next_frame()
            except StopIteration:
                self._frame = StopIteration
                
            self._retrieve_count = 1
            self._frame_pos += 1
            
        else:
            raise SynchronizationError('The clients of the video fork ran out of sync. '
                                       'The parent process is at frame %d, while one client '
                                       'requested frame %d' % (self._frame_index, index))
        
        return self._frame    
     
     
    def clear(self):
        """
        ends the iteration and removes all the fork clients.
        Note that the clients may still retrieve the last frame.
        """
        # remove all the iterators
        self._clients = []
        logger.debug('Finished iterating and unregistered all clients of '
                     'the video fork.')
        
    
    def abort_iteration(self):
        """ send SystemExit to all other clients """
        logger.info('The video fork is aborting by sending a SystemExit '
                    'signal to clients.')
        self.state = 'aborting'
        super(VideoFork, self).abort_iteration()


    def __iter__(self):
        raise RuntimeError('Cannot iterate over a VideoFork. Use the '
                           'get_client() method to get an iterable client.')


    def get_client(self):
        """
        returns a new fork client, which can then be used for iteration.
        """
        if self._client_count is not None and len(self._clients) >= self._client_count:
            raise ValueError('We already registered %d clients.' % self._client_count)

        # create and register the client iterator
        client = _VideoForkClient(self)
        self._clients.append(client)
        logger.debug('Registered client %d for video fork.', 
                     len(self._clients))
        
        self._retrieve_count = np.inf
        
        return client
    