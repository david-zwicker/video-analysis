'''
Created on Aug 22, 2014

@author: zwicker

Establishes a video pipe, which can be used to send video frames between 
different processes.
The module provides two classes, which are used to send and receive videos,
respectively. Here, the synchronization and communication between these classes
is handled using normal pipes, while frames are transported using the
sharedmem package.

TODO: Allow receiver to request specific frame
'''

from __future__ import division

import logging
import multiprocessing
import time

import numpy as np
import sharedmem

from .base import VideoBase, VideoFilterBase, SynchronizationError



class VideoPipeError(RuntimeError):
    pass



class VideoSender(VideoFilterBase):
    """ class that can send video frames to a VideoReceiver.
    This class uses a sharedmem to transport the data. """
    
    
    def __init__(self, video, name=None):
        super(VideoSender, self).__init__(video)
        self.pipe, self.pipe_receiver = multiprocessing.Pipe(duplex=True)
        self.frame_buffer = sharedmem.empty(video.shape[1:], np.uint8)
        self.name = name
        self.running = True
        self._waiting_for_frame = False


    def try_getting_frame(self, index=None):
        """ tries to retrieve a frame from the video and copy it to the shared
        buffer """
        try:
            # get the next frame
            if index is None:
                self.frame_buffer[:] = next(self)
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


    def handle_command(self, command):
        """ handles commands received from the VideoReceiver """ 
        if command == 'next_frame':
            # receiver requests the next frame
            self.try_getting_frame()
            
        elif command == 'start_iterating':
            # receiver initializes the iterating
            self._start_iterating()
            self.pipe.send('start_iterating_OK')
            
        elif command == 'end_iterating':
            # receiver reached the end of the iteration
            self._end_iterating(propagate=True)
            self.pipe.send('end_iterating_OK')
            
        elif command == 'specific_frame':
            # receiver requests a specific frame
            frame_id = self.pipe.recv()
            logging.debug('Specific frame %d was requested from sender.', frame_id)
            self.try_getting_frame(frame_id)
            
        elif command == 'finished':
            # the receiver wants to terminate the video pipe
            self._end_iterating(propagate=True)
            self.pipe.send('finished_OK')
            self.pipe.close()
            logging.debug('Sender%s closed itself.',
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
        if self.pipe.poll():
            command = self.pipe.recv()
            self.handle_command(command)
                
        return self.running


    def start(self):
        """ starts the event loop which handles commands until the receiver
        is finished """
        while self.running:
            # wait for a command of the receiver
            command = self.pipe.recv()
            self.handle_command(command)
            
            # don't proceed before we got the next frame
            while self._waiting_for_frame:
                self.try_getting_frame()
                time.sleep(0.01)
            


class VideoReceiver(VideoBase):
    """ class that receives frames from a VideoSender """
    
    def __init__(self, sender):
        super(VideoReceiver, self).__init__(**sender.video_format)
        self.pipe = sender.pipe_receiver
        self.frame_buffer = sender.frame_buffer
        self.name = sender.name
        
    
    def send_command(self, command):
        """ send a command to the associated VideoSender """
        logging.debug('Send command `%s`.', command)
        self.pipe.send(command)
        # wait for the sender to acknowledge the command
        if self.pipe.recv() != command + '_OK':
            raise VideoPipeError('Command `%s` failed' % command)
        
        
    def _start_iterating(self):
        logging.debug('Receiver%s starts iteration.',
                      '' if self.name is None else ' `%s`' % self.name)
        self.send_command('start_iterating')
        super(VideoReceiver, self)._start_iterating()
        
                
    def _end_iterating(self):
        logging.debug('Receiver%s ends iteration.',
                      '' if self.name is None else ' `%s`' % self.name)
        self.send_command('end_iterating')
        super(VideoReceiver, self)._end_iterating()
                    
        
    def next(self):
        """ request the next frame from the sender """
        # send the request
        self.pipe.send('next_frame')
        
        # wait for the reply
        command = self.pipe.recv()
        if command == 'frame_ready':
            return self.frame_buffer
        
        elif command == StopIteration:
            raise StopIteration
        
        else:
            raise VideoPipeError('Unknown reply `%s`', command)


    def get_frame(self, index):
        """ request a specific frame from the sender """
        # send the request
        self.pipe.send('specific_frame')
        self.pipe.send(index)

        # wait for the reply
        command = self.pipe.recv()
        if command == 'frame_ready':
            return self.frame_buffer
        else:
            raise VideoPipeError('Unknown reply `%s`', command)

        
    def close(self):
        self.send_command('finished')
        self.pipe.close()
        logging.debug('Receiver%s closed itself.',
                      '' if self.name is None else ' `%s`' % self.name)
        