'''
Created on Aug 4, 2014

@author: zwicker
'''

import operator
import numpy as np
import cv2

from .filters import FilterFunction

def reduce_video(video, function, initial_value=None):
    """ applies function to consecutive frames """
    result = initial_value
    for frame in video:
        if result is None:
            result = frame
        else:
            result = function(frame, result)
    return result
        
        
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