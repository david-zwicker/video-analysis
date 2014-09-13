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


logger = logging.getLogger('video.io')


class VideoPipeError(RuntimeError): pass



class VideoPipeReceiver(VideoBase):
    """ class that receives frames from a VideoPipe.
    This class usually needs not be instantiated directly, but is returned by
    video_pipe.receiver
    """
    
    def __init__(self, pipe, frame_buffer, video_format, name=''):
        super(VideoPipeReceiver, self).__init__(**video_format)
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
                    
        
    def get_next_frame(self):
        """ request the next frame from the sender """
        # send the request and wait for a reply
        if self.pipe.closed:
            # abort iteration if the pipe has been closed
            reply = 'abort_iteration'
        else:
            self.pipe.send('next_frame')
            reply = self.pipe.recv()
            
        # handle the reply
        if reply == 'frame_ready':
            # set the internal pointer to the next frame
            self._frame_pos += 1
            # return the frame
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


    def get_frame(self, index):
        """ request a specific frame from the sender """
        # send the request and wait for a reply
        if self.pipe.closed:
            # abort iteration if the pipe has been closed
            reply = 'abort_iteration'
        else:
            self.pipe.send('specific_frame')
            self.pipe.send(index)
            reply = self.pipe.recv()
            
        if reply == 'frame_ready':
            # return the frame
            return self.frame_buffer

        elif reply == 'abort_iteration':
            # signal that the iteration was aborted
            self.abort_iteration()
            raise SystemExit
        
        else:
            raise VideoPipeError('Unknown reply `%s`', reply)

        
    def close(self):
        self.send_command('finished', wait_for_reply=False)
        self.pipe.close()
        logger.debug('Receiver%s closed itself.', self.name_repr)
        
        

class VideoPipe(VideoFilterBase):
    """ class that can be used to transport video frames between processes.
    Internally, the class uses sharedmem to share memory among different
    processes.
    The typical use case is
    
    def worker_process(self, video):
        ''' worker process processing a video '''
        expensive_function(video)
        
    if __name__ == '__main__':
        # load a video file
        video = VideoFile('test.mov')
        # create the video pipe
        video_pipe = VideoPipe(video)
        # create the worker process
        proc = multiprocessing.Process(target=worker_process,
                                       args=(video_pipe.receiver,))
        proc.start()    
    """
    
    def __init__(self, video, name=None):
        super(VideoPipe, self).__init__(video)
        self.pipe, pipe_receiver = mp.Pipe(duplex=True)
        self.frame_buffer = sharedmem.empty(video.shape[1:], np.uint8)
        self.name = name
        self.running = True
        self._waiting_for_frame = False
        
        self.receiver = VideoPipeReceiver(pipe_receiver, self.frame_buffer,
                                      self.video_format, self.name)


    def try_getting_frame(self, index=None):
        """ tries to retrieve a frame from the video and copy it to the shared
        buffer """
        try:
            # get the next frame
            if index is None or index == self.get_frame_pos():
                self.frame_buffer[:] = self.get_next_frame()
            else:
                self.frame_buffer[:] = self.get_frame(index)
            
        except SynchronizationError:
            # frame is not ready yet, wait another round
            self._waiting_for_frame = True
            
        except StopIteration:
            # we reached the end of the video and the iteration should stop
            self._waiting_for_frame = False
            self.pipe.send(StopIteration)
            
        else:
            # frame is ready and was copied to the shared buffer
            self._waiting_for_frame = False
            self.pipe.send('frame_ready')


    def abort_iteration(self):
        if not self.pipe.closed:
            self.pipe.send('abort_iteration')
        self.running = False
        super(VideoPipe, self).abort_iteration()


    def handle_command(self, command):
        """ handles commands received from the VideoPipeReceiver """ 
        if command == 'next_frame':
            # receiver requests the next frame
            self.try_getting_frame()
            
        elif command == 'abort_iteration':
            # receiver reached the end of the iteration
            self.abort_iteration()
            
        elif command == 'specific_frame':
            # receiver requests a specific frame
            frame_id = self.pipe.recv()
            logger.debug('Specific frame %d was requested from sender.', frame_id)
            self.try_getting_frame(frame_id)
            
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
        # see whether we are waiting for a frame
        if self._waiting_for_frame:
            self.try_getting_frame()
            
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
                
                # don't proceed before we got the next frame
                while self._waiting_for_frame:
                    self.try_getting_frame()
                    time.sleep(0.01)

        except (KeyboardInterrupt, SystemExit):
            self.abort_iteration()
            