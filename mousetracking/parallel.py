'''
Created on Aug 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains convenience functions for scanning multiple
mouse videos in parallel.
'''


import multiprocessing as mp
import time

from .simple import scan_video
from video.io.base import VideoFork
from video.io.pipe import create_video_pipe
from video.filters import FilterCrop


# dictionary defining the four quadrants
QUADRANTS = {'UL': 'upper left',
             'DL': 'lower left',
             'UR': 'upper right',
             'DR': 'lower right'}
  

def get_window_pos(location, video_size):
    """ calculate the window position given a location string and the size
    of the total video """
    width, height = video_size[0]//2, video_size[1]//2    
    if location == 'upper left':
        return 0, 0
    elif location == 'lower left':
        return 0, height
    elif location == 'upper right':
        return width, 0
    elif location == 'lower right':
        return width, height
    else:
        raise ValueError('Unknown location `%s`' % location)



def scan_video_quadrants(video, parameters=None, **kwargs):
    """ Takes a video and scans all four quadrants in parallel.
    Here, the video is read in one process, split into four video streams
    and analyzed in four separate processes
    Additional parameters include a dictionary 'parameters'
    """
    
    if parameters is None:
        parameters = {}
    
    # make sure that scan_video does not crop the video, since we already do
    # it in this process (see below)
    kwargs['crop_video'] = False
    
    # create a fork, such that the data can be analyzed by multiple consumers
    video_fork = VideoFork(video, synchronized=True, client_count=len(QUADRANTS))
    
    senders = []
    for name, crop in QUADRANTS.iteritems():
        # save the cropping rectangle for further analysis later
        parameters['video/cropping_rect'] = crop
        parameters['debug/window_position'] = get_window_pos(crop, video_fork.size)
        kwargs['parameters'] = parameters
                
        # crop the video to the right region
        video_crop = FilterCrop(video_fork.get_client(), region=crop,
                                color_channel=1)
        # construct the video sender
        sender, receiver = create_video_pipe(video_crop, name=name)
        # launch a new process, where the receiver is going to live 
        proc = mp.Process(target=scan_video, args=(name, receiver),
                          kwargs=kwargs)

        proc.start()
        senders.append(sender)
    
    try:
        # start the main loop where we check all senders periodically
        while any(sender.running for sender in senders):
            # check if any senders are running
            for sender in senders:
                sender.check()
    
            # let the CPU rest a little
            time.sleep(0.001)
        
    except (KeyboardInterrupt, SystemExit):
        # try to interrupt the system cleanly
        for sender in senders:
            sender.abort_iteration()
        
    