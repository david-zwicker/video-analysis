'''
Created on Aug 4, 2014

@author: zwicker
'''

import operator
import numpy as np
import cv2

from .filters import FilterFunction
from video.utils import display_progress


def find_bounding_rect(mask):
    """ finds the rectangle, which bounds a white region in a mask.
    The rectangle is returned as [top, left, bottom, right]
    Currently, this function only works reliably for connected regions 
    """

    # find top boundary
    top = 0
    while not np.any(mask[top, :]):
        top += 1
    
    # find bottom boundary
    bottom = top + 1
    try:
        while np.any(mask[bottom, :]):
            bottom += 1
    except IndexError:
        bottom = mask.shape[0] - 1

    # find left boundary
    left = 0
    while not np.any(mask[:, left]):
        left += 1
    
    # find right boundary
    try:
        right = left + 1
        while np.any(mask[:, right]):
            right += 1
    except IndexError:
        right = mask.shape[1] - 1
    
    return np.array([top, left, bottom, right])


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
        
        
def remove_background(video):
    """
    function which does a two-pass run to remove the background from a video
    """
    
    # apply the subtract background filter
    fgbg = cv2.BackgroundSubtractorMOG2(history=500, varThreshold=8., bShadowDetection=False)    
    def remove_background(frame):
        return fgbg.apply(frame)    
    video_noback = FilterFunction(video, remove_background)

    # determine objects that have been found in background and should not be there
    initial_value = np.zeros(video_noback.size, np.int)
    false_background = reduce_video(video_noback, operator.add, initial_value)
    false_background = (false_background > 255*video_noback.frame_count/2)
    
    # remove the wrong background from the image and perform some morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def correct_background(frame):
        frame[false_background] = 0
        cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        return frame
    return FilterFunction(video_noback, correct_background)
