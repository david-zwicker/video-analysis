'''
Created on Aug 4, 2014

@author: zwicker
'''

import numpy as np

from video.utils import display_progress


def reduce_video(video, function, initial_value=None):
    """ applies function to consecutive frames """
    result = initial_value
    for frame in display_progress(video):
        if result is None:
            result = frame
        else:
            result = function(frame, result)
    return result

    
        
def measure_mean(video):
    """
    measures the mean of each movie pixel over time
    """
    mean = np.zeros(video.shape[1:])
 
    for n, frame in enumerate(display_progress(video)):
        mean = mean*n/(n + 1) + frame/(n + 1)
 
    return mean
        
        
        
def measure_mean_std(video):
    """
    measures the mean and the standard deviation of each movie pixel over time
    Uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
    """
    mean = np.zeros(video.shape[1:])
    M2 = np.zeros(video.shape[1:])
    
    for n, frame in enumerate(display_progress(video)):
        delta = frame - mean
        mean = mean + delta/(n + 1)
        M2 = M2 + delta*(frame - mean)
 
    if (n < 2):
        return frame, 0

    return mean, np.sqrt(M2/n)
        

