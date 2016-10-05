'''
Created on Aug 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Establishes a video pipe, which can be used to transfer video frames between 
different processes.
The module provides the class VideoPipe, which controls the pipe and supplies
a VideoPipeReceiver instance as video_pipe.receiver, which acts as a video for the
child process.
The synchronization and communication between these classes is handled using
normal pipes, while frames are transported using the sharedmem package.
'''

from __future__ import division

import logging
import multiprocessing as mp
import time

import numpy as np
import sharedmem

from .base import VideoBase, VideoFilterBase, SynchronizationError
from .file import VideoFile
from utils.concurrency import WorkerThread

logger = logging.getLogger('video.io')


class VideoPipeError(RuntimeError): pass

START_TIME = time.time()

class VideoPipeReceiver(VideoBase):
    """ class that receives frames from a VideoPipe.
    This class usually needs not be instantiated directly, but is returned by
    video_pipe.receiver
    """
    
    def __init__(self, pipe, frame_buffer, video_format=None, name=''):
        # initialize the VideoBase
        if video_format is None:
            video_format = {}
        super(VideoPipeReceiver, self).__init__(**video_format)
        
        # set extra arguments of the video receiver
        self.pipe = pipe
        self.frame_buffer = frame_buffer
        self.name = name
        self.name_repr = '' if self.name is None else ' `%s`' % self.name
        
    
    def send_command(self, command, wait_for_reply=True):
        """ send a command to the associated VideoPipe """
        if not self.pipe.closed:
            logger.debug('Send command `%s`.', command)
            self.pipe.send(command)
            # wait for the sender to acknowledge the command
            if wait_for_reply and self.pipe.recv() != command + '_OK':
                raise VideoPipeError('Command `%s` failed' % command)
        
                    
    def abort_iteration(self):
        logger.debug('Receiver%s aborts iteration.', self.name_repr)
        self.send_command('abort_iteration', wait_for_reply=False)
        super(VideoPipeReceiver, self).abort_iteration()
                    
                    
    def wait_for_frame(self, index=None):
        """ request a _frame from the sender """
        # abort iteration if the pipe has been closed
        if self.pipe.closed:
            self.abort_iteration()
            raise SystemExit
        
        # send the request
        if index is None:
            self.pipe.send('next_frame')
        else:
            self.pipe.send('specific_frame')
            self.pipe.send(index)
        # wait for reply
        reply = self.pipe.recv()
        
        # handle the reply
        if reply == 'frame_ready':
            # set the internal pointer to the next _frame
            self._frame_pos += 1
            # return the _frame
            return self.frame_buffer
        
        elif reply == StopIteration:
            # signal that the iterator is exhausted
            raise StopIteration
        
        elif reply == 'abort_iteration':
            # signal that the iteration was aborted
            self.abort_iteration()
            raise SystemExit
        
        else:
            raise VideoPipeError('Unknown reply `%s`', reply)

        
    def get_next_frame(self):
        """ request the next _frame from the sender """
        return self.wait_for_frame()


    def get_frame(self, index):
        """ request a specific _frame from the sender """
        if index < 0:
            index += self.frame_count
        return self.wait_for_frame(index)

        
    def close(self):
        self.send_command('finished', wait_for_reply=False)
        self.pipe.close()
        logger.debug('Receiver%s closed itself.', self.name_repr)
        
        

class VideoPipeSender(VideoFilterBase):
    """ class that can be used to transport video frames between processes.
    Internally, the class uses sharedmem to share memory among different
    processes. This class has an event loop which handles commands entering
    via a normal pipe from the VideoPipeReceiver. This construct can be used
    to read a video in the current process and work on it in a different
    process.
    
    If read_ahead is True, the next _frame is already read before it is
    requested.
    """
    
    poll_frequency = 200 #< Frequency of polling pipe in Hz 
    
    def __init__(self, video, pipe, frame_buffer, name=None, read_ahead=False):
        super(VideoPipeSender, self).__init__(video)
        self.pipe = pipe
        self.frame_next = None
        self.name = name
        self.read_ahead = read_ahead
        self.running = True
        self._waiting_for_frame = False
        self._waiting_for_read_ahead = False
        self.frame_buffer = frame_buffer
        

    def try_reading_ahead(self):
        """ tries to retrieve a _frame from the video and copy it to the shared
        frame_buffer """
        try:
            # get the next _frame
            self.frame_next = self.get_next_frame()
            
        except SynchronizationError:
            self._waiting_for_read_ahead = True
            
        except StopIteration:
            # we reached the end of the video and the iteration should stop
            self.frame_next = StopIteration
            
        else:
            self._waiting_for_read_ahead = False
            

    def try_getting_frame(self, index=None):
        """ tries to retrieve a _frame from the video and copy it to the shared
        frame_buffer """
        try:
            # get the next _frame
            if index is None or index == self.get_frame_pos():
                self.frame_buffer[:] = self.get_next_frame()
            else:
                self.frame_buffer[:] = self.get_frame(index)
            
        except SynchronizationError:
            # _frame is not ready yet, wait another round
            self._waiting_for_frame = True
            
        except StopIteration:
            # we reached the end of the video and the iteration should stop
            self._waiting_for_frame = False
            self.pipe.send(StopIteration)
            
        else:
            # _frame is ready and was copied to the shared frame_buffer
            self._waiting_for_frame = False
            # notify the receiver that the _frame is ready
            self.pipe.send('frame_ready')
            if self.read_ahead and index is None:
                self.try_reading_ahead()


    def abort_iteration(self):
        """ abort the iteration and notify the receiver """
        if not self.pipe.closed:
            self.pipe.send('abort_iteration')
        self.running = False
        super(VideoPipeSender, self).abort_iteration()


    def load_next_frame(self):
        """ tries loading the next _frame, either directly from the video
        or from the buffer that has been filled by a read-ahead process """
        if self.read_ahead:
            if self._waiting_for_frame:
                # this should not happen, since the event loop only finishes
                # when the _frame was successfully read
                raise VideoPipeError('Frame was not properly read in advance.')

            elif self.frame_next is None:
                # _frame is not buffered
                # => read it directly and notify receiver
                self.try_getting_frame()
                
            elif self.frame_next is StopIteration:
                # we reached the end of the iteration
                self.pipe.send(StopIteration)
                
            else:
                # copy _frame to the right buffer
                self.frame_buffer[:] = self.frame_next
                self.frame_next = None
                # tell receiver that the _frame is ready
                self.pipe.send('frame_ready')
                # try getting the next _frame, which flags self._waiting_for_frame
                # and thus starts the event loop
                self._waiting_for_read_ahead = True
            
        else:
            self.try_getting_frame()


    def handle_command(self, command):
        """ handles commands received from the VideoPipeReceiver """
        if command == 'next_frame':
            # receiver requests the next _frame
            self.load_next_frame()
            
        elif command == 'abort_iteration':
            # receiver reached the end of the iteration
            self.abort_iteration()
            
        elif command == 'specific_frame':
            # receiver requests a specific _frame
            frame_id = self.pipe.recv()
            logger.debug('Specific _frame %d was requested from sender.',
                         frame_id)
            self.try_getting_frame(index=frame_id)
            
        elif command == 'finished':
            # the receiver wants to terminate the video pipe
            if not self.pipe.closed:
                self.pipe.send('finished_OK')
            self.pipe.close()
            logger.debug('Sender%s closed itself.',
                         '' if self.name is None else ' `%s`' % self.name)
            self.running = False
            
        else:
            raise VideoPipeError('Unknown command `%s`', command)

            
    def check(self):
        """ handles a command if one has been sent """
        # see whether we are waiting for a _frame
        if self._waiting_for_frame:
            self.try_getting_frame()
            
        if self._waiting_for_read_ahead:
            self.try_reading_ahead()
            
        # otherwise check the pipe for new commands
        if not self.pipe.closed and self.pipe.poll():
            command = self.pipe.recv()
            self.handle_command(command)
                
        return self.running


    def start(self):
        """ starts the event loop which handles commands until the receiver
        is finished """
        try:
            while self.running and not self.pipe.closed:
                # wait for a command of the receiver            
                command = self.pipe.recv()
                self.handle_command(command)
                
                # wait for frames to be retrieved
                while self.running and (self._waiting_for_frame 
                                        or self._waiting_for_read_ahead):
                    
                    time.sleep(1/self.poll_frequency)
                    if self._waiting_for_frame:
                        self.try_getting_frame()
                        
                    if self._waiting_for_read_ahead:
                        self.try_reading_ahead()

        except (KeyboardInterrupt, SystemExit):
            self.abort_iteration()
            
            

def create_video_pipe(video, name=None, read_ahead=False):
    """ creates the two ends of a video pipe.
    
    The typical use case is
    
    def worker_process(self, video):
        ''' worker process processing a video '''
        expensive_function(video)
        
    if __name__ == '__main__':
        # load a video file
        video = VideoFile('test.mov')
        # create the video pipe
        sender, receiver = create_video_pipe(video)
        # create the worker process
        proc = multiprocessing.Process(target=worker_process,
                                       args=(receiver,))
        proc.start()
        sender.start()   
    
    """
    # create the pipe used for communication
    pipe_sender, pipe_receiver = mp.Pipe(duplex=True)
    # create the buffer in memory that is used for passing frames
    frame_buffer = sharedmem.empty(video.shape[1:], np.uint8)
    
    # create the two ends of the video pipe
    sender = VideoPipeSender(video, pipe_sender, frame_buffer,
                             name, read_ahead)
    receiver = VideoPipeReceiver(pipe_receiver, frame_buffer,
                                 video.video_format, name)
    
    return sender, receiver

            
            
class VideoReaderProcess(mp.Process):
    """ Process that reads a video and returns it using a video pipe """
    def __init__(self, filename, video_class=VideoFile):
        super(VideoReaderProcess, self).__init__()
        self.daemon = True
        self.running = False
        self.filename = filename
        self.video_class = video_class
        
        # create the pipe used for communication
        self.pipe_sender, pipe_receiver = mp.Pipe(duplex=True)
        
        video = self.video_class(self.filename)
        # create the buffer in memory that is used for passing frames
        self.frame_buffer = sharedmem.empty(video.shape[1:], np.uint8)
        self.receiver =  VideoPipeReceiver(pipe_receiver, self.frame_buffer, video.video_format)
        video.close()

        
    def run(self):
        logger.debug('Started process %d to read video' % self.pid)
        video = self.video_class(self.filename)
        video_sender = VideoPipeSender(video, self.pipe_sender, self.frame_buffer)
        self.running = True
        video_sender.start()


    def terminate(self):
        self.video_pipe.abort_iteration()
        
        

def video_reader_process(filename, video_class=VideoFile):
    """ reads the given filename in a separate process.
    The given video_class is used to open the file. """
    proc = VideoReaderProcess(filename, video_class)
    proc.start()
    return proc.receiver


  
class VideoPreprocessor(object):
    """ class that reads video in a separate thread and apply additional
    functions using additional threads.
    
    Example: Given a `video` and a function `blur_frame` that takes an image
    and returns a blurred one, the class can be used as follows 
    
    video_processor = VideoPreprocessor(video, {'blur': blur_frame})
    for data in video_processor:
        frame_raw = data['raw']
        frame_blurred = data['blur']
    
    Importantly, the function used for preprocessing should release the python
    global interpreter lock (GIL) most of the time such that multiple threads
    can be run concurrently.
    """
    
    def __init__(self, video, functions, preprocess=None, use_threads=True):
        """ initializes the preprocessor
        `video` is the video to be iterated over
        `functions` is a dictionary of functions that should be applied while
            iterating
        `preprocess` can be a function that will be applied to the frame before
            anything is returned
        """
        if 'raw' in functions:
            raise KeyError('The key `raw` is reserved for the raw _frame and '
                           'may not be used for functions.')
        
        self.length = len(video)
        self.video_iter = iter(video)
        self.functions = functions
        self.preprocess = preprocess
        
        # initialize internal structures
        self._frame = None
        
        # initialize the background workers
        self._worker_next_frame = WorkerThread(self._get_next_frame,
                                               use_threads=use_threads)
        self._workers = {name: WorkerThread(func, use_threads=use_threads)
                         for name, func in self.functions.iteritems()}
        
        self._init_next_processing(self._get_next_frame())

#         
        
    def __len__(self):
        return self.length
        

    def _get_next_frame(self):
        """ get the next frame and preprocess it if necessary """
        try:
            frame = self.video_iter.next()
        except StopIteration:
            frame = None
        else:
            if self.preprocess:
                frame = self.preprocess(frame)
        return frame


    def _init_next_processing(self, frame_next):
        """ prepare the next processed frame in the background
        `frame_next` is the raw data of this _frame
        """
        self._frame = frame_next
        # ask all workers to process this frame
        for worker in self._workers.itervalues():
            worker.put(frame_next)
        # ask for the next frame
        self._worker_next_frame.put()

    
    def __iter__(self):
        return self

    
    def next(self):
        """ grab the raw and processed data of the next frame """
        # check whether there is data available
        if self._frame is None:
            raise StopIteration
                 
        # grab all results for the current _frame
        result = {name: worker.get()
                  for name, worker in self._workers.iteritems()}
        # store information about the current frame
        result['raw'] = self._frame

        # grab the next frame 
        frame_next = self._worker_next_frame.get()
        if frame_next is None:
            # stop the iteration in the next step. We still have to exit from
            # this function since we have results to return
            self._frame = None
        else:
            # start fetching the result for this next frame
            self._init_next_processing(frame_next)

        # while this is underway, return the current results
        return result
    
    