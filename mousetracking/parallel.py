'''
Created on Aug 22, 2014

@author: zwicker
'''


import multiprocessing
import time

from .simple import scan_video
from video.io.base import VideoFork
from video.io.pipe import get_video_pipe
from video.filters import FilterCrop


QUADRANTS = {'UL': 'upper left',
             'DL': 'lower left',
             'UR': 'upper right',
             'DR': 'lower right'}
  


def scan_video_in_process(video, name, **kwargs):
    scan_video(video, name, **kwargs)
    video.close()
            


def scan_video_quadrants(video, parameters=None, debug_output=None, **kwargs):
    
    # create a fork, such that the data can be analyzed by multiple consumers
    video_fork = VideoFork(video, synchronized=True)
    
    senders = []
    for name, crop in QUADRANTS:
        # crop the video to the right region
        video_crop = FilterCrop(video_fork, region=crop, color_channel=1)
        
        # create the video pipe to transport the data between different processes
        sender, receiver = get_video_pipe(video_crop, name=name)
        
        # launch a new process, where the receiver is going to live 
        proc = multiprocessing.Process(target=scan_video,
                                       args=(receiver, name), kwargs=kwargs)
        proc.start()

        senders.append(sender)
    
    # start the main loop where we check all senders periodically
    while any(sender.check() for sender in senders):
        # let the CPU rest a little
        time.sleep(0.001)
        
        
    