'''
Created on Aug 22, 2014

@author: zwicker

Establishes a video pipe, which can be used to send video frames between 
different processes.
The module provides two classes, which are used to send and receive videos,
respectively. Here, the synchronization and communication between these classes
is handled using normal pipes, while frames are transported using the
sharedmem package.
'''

from __future__ import division

import logging
import multiprocessing

import numpy as np
import sharedmem

from .base import VideoBase, VideoFilterBase


class VideoPipeError(RuntimeError):
    pass


class VideoSender(VideoFilterBase):
    def __init__(self, video, pipe):
        super(VideoSender, self).__init__(video)
        self.pipe = pipe
        self.frame_buffer = sharedmem.empty(video.shape[1:], np.uint8)

    def get_pipe(self):
        return self.pipe_receiver
    
    
    def handle_command(self, command):
        running = True
        if command == 'next_frame':
            try:
                self.frame_buffer[:] = next(self)
            except StopIteration:
                self.pipe.send(StopIteration)
            else:
                self.pipe.send('frame_ready')
            
        elif command == 'start_iterating':
            self._start_iterating()
            self.pipe.send('start_iterating_OK')
            
        elif command == 'end_iterating':
            self._end_iterating()
            self.pipe.send('end_iterating_OK')
            
        elif command == 'finished':
            self._end_iterating()
            self.pipe.send('finished_OK')
            self.pipe.close()
            running = False
            
        return running # return whether we still need to run
    
            
    def check_stats(self):
        if self.pipe.poll():
            command = self.pipe.recv()
            return self.handle_command(command)
                
        return True # return True if the process is not finished


    def start(self):
        running = True
        while running:
            command = self.pipe.recv()
            running = self.handle_command(command)
            


class VideoReceiver(VideoBase):
    """ class that receives frames from a VideoSender """
    def __init__(self, pipe, video_format, frame_buffer):
        super(VideoReceiver, self).__init__(**video_format)
        self.pipe = pipe
        self.frame_buffer = frame_buffer
        
    
    def send_command(self, command):
        self.pipe.send(command)
        if self.pipe.recv() != command + '_OK':
            raise VideoPipeError('Command `%s` failed' % command)
        
        
    def _start_iterating(self):
        logging.debug('Send command to start iteration')
        self.send_command('start_iterating')
        super(VideoReceiver, self)._start_iterating()
        
                
    def _end_iterating(self):
        logging.debug('Send command to end iteration')
        self.send_command('end_iterating')
        super(VideoReceiver, self)._end_iterating()
                    
        
    def next(self):
        self.pipe.send('next_frame')
        command = self.pipe.recv()
        if command == 'frame_ready':
            return self.frame_buffer
        
        elif command == StopIteration:
            raise StopIteration

        
    def close(self):
        self.send_command('finished')
        self.pipe.close()
        logging.debug('Receiver closed itself')
        

        
def get_video_pipe(video):
    pipe_sender, pipe_receiver = multiprocessing.Pipe(duplex=True)
    sender = VideoSender(video, pipe_sender)
    receiver = VideoReceiver(pipe_receiver, sender.video_format, sender.frame_buffer)
    return sender, receiver