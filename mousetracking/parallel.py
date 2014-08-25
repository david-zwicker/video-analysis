'''
Created on Aug 22, 2014

@author: zwicker
'''


import multiprocessing
import time

from .simple import scan_video
from video.io.base import VideoFork
from video.io.pipe import VideoSender, VideoReceiver
from video.filters import FilterCrop


QUADRANTS = {'UL': 'upper left',
             'DL': 'lower left',
             'UR': 'upper right',
             'DR': 'lower right'}
  


def scan_video_in_process(sender, **kwargs):
    """ helper function to scan a video in separate process """
    # create the video receiver communicating with the host process
    video = VideoReceiver(sender)
    # scan the video 
    scan_video(video, sender.name, **kwargs)
    # tell the host process that we are finished
    video.close()
            


def scan_video_quadrants(video, parameters=None, **kwargs):
    """ Takes a video and scans all four quadrants in parallel.
    Here, the video is read in one process, split into four video streams
    and analyzed in four separate processes """
    
    # create a fork, such that the data can be analyzed by multiple consumers
    video_fork = VideoFork(video, synchronized=True, client_count=len(QUADRANTS))
    
    senders = []
    for name, crop in QUADRANTS.iteritems():
        # crop the video to the right region
        video_crop = FilterCrop(video_fork, region=crop, color_channel=1)
        # construct the video sender 
        sender = VideoSender(video_crop, name=name)
        # launch a new process, where the receiver is going to live 
        proc = multiprocessing.Process(target=scan_video_in_process,
                                       args=(sender,), kwargs=kwargs)

        proc.start()
        senders.append(sender)
    
    # start the main loop where we check all senders periodically
    running = True
    while running:
        running = False
        for sender in senders:
            if sender.check():
                running = True 

        # let the CPU rest a little
        time.sleep(0.001)
        
        
    