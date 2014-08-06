'''
Created on Aug 5, 2014

@author: zwicker
'''

import os
import logging
import numpy as np
import scipy.ndimage.measurements as scipy_measurements
import cv2

from video.io import VideoFileStack, VideoFork
from video.filters import FilterMonochrome, FilterFunction, FilterCrop
from video.analysis import measure_mean_std
from video.utils import display_progress
from video.composer import VideoComposer


class MouseMovie(object):
    """
    analyzes the mouse movie
    """
    
    mouse_size_threshold_pixel = 5

    def __init__(self, folder, frames=None, crop=None, debug_video=False):
        
        self.folder = folder
        
        # initialize video
        self.video = VideoFileStack(os.path.join(folder, 'raw_video/*'))
        
        # restrict the analysis to an interval of frames
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
            
        # restrict the analysis to a region
        if crop is not None:
            self.video = FilterCrop(self.video, crop)
            
        # check whether a debug video is requested
        if debug_video:
            # fork the video such that we can analyze it and use it as a background
            # for the debug video
            video_fork = VideoFork(self.video)
            self.video = video_fork
            
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', 'video.mov')
            self.debug_video = VideoComposer(debug_file, background=video_fork)
            
        else:
            self.debug_video = None


    def get_blurred_video(self, sigma=3):
        """ returns the mouse video with a monochrome and a blur filter """
        
        # convert to monochrome if necessary
        if self.video.is_color:
            video = FilterMonochrome(self.video)
        else:
            video = self.video
    
        # blur the video
        def blur(frame):
            """ helper function applying the blur """
            return cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=sigma)
        
        return FilterFunction(video, blur)


    def get_movie_statistics(self):
        """ returns the mean and std over the entire movie """
        
        cache_file = os.path.join(self.folder, 'statistics', 'mean_std.npz')
        
        try:
            # try to load the data from the cache
            data = np.load(cache_file)
            video_mean = data['mean']
            video_std = data['std']
            logging.info('Loaded mean and standard deviation from cache')
            
        except:
            # measure mean and standard deviation of the blurred video
            video = self.get_blurred_video()
        
            video_mean, video_std = measure_mean_std(video)
            np.savez(cache_file, mean=video_mean, std=video_std)
            
        return video_mean, video_std
    
    
    def find_mouse(self):
        """
        finds the mouse trajectory
        TODO: The mouse is the fast moving object
        TODO: Use the information that the mouse is brighter than the background
        TODO: Use that the mouse is of a certain size
            This can be used to carry a box around which moves only, if the "feature"
            runs out of the box this is useful since the movement can usually only be
            detected at the front/back of the mouse. Take into account maximal mouse speed here.
        """
        
        # get the video statistics
        video_mean, video_std = self.get_movie_statistics()
        
        # work with a blurred video
        video = self.get_blurred_video()        
        
        # prepare the kernels for morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        def find_features(frame):
            """ helper function for removing the background """
            
            # calculate the difference to the current background model
            diff = frame - find_features.background
            
            # adapt the current background model
            # Here, the cut-off sets the relaxation time scale
            find_features.background += np.clip(diff, -1, 1)
            
            # find features (i.e. the mouse) by evaluating changes and intensity 
            features = (
                (np.abs(diff) > 2.5*video_std.mean()) # mouse is moving
                & (diff > 0) # and mouse is brighter than background
            )
            
            # convert the binary image to the normal output
            features = 255*features.astype(np.uint8)
            
            # perform morphological closing to remove noise
            features = cv2.morphologyEx(features, cv2.MORPH_CLOSE, kernel_close)
            
            # perform morphological opening to combined feature patches that are close 
            features = cv2.morphologyEx(features, cv2.MORPH_OPEN, kernel_open)
            return features

        # initialize the background
        find_features.background = np.array(video[0], np.double)
        
        # apply the the feature finder, i.e. the mouse
        video_features = FilterFunction(video, find_features)
        
        # iterate over all frames and find the mouse
        mouse_trajectory = []
        for index, frame in enumerate(display_progress(video_features)):

            if self.debug_video:
                self.debug_video.add_image(index, frame, frame == 255)
        
            mouse_data = None
            
            # find all objects in the frame
            labels, num_features = scipy_measurements.label(frame)
            
            # find the largest object (which should be the mouse)
            sizes = scipy_measurements.sum(labels, labels, xrange(num_features))
            if len(sizes) > 0:
                mouse_label = np.argmax(sizes)
                
                # TODO: update internal mouse model
                
                mouse_size = sizes[mouse_label]
                
                if mouse_size > self.mouse_size_threshold_pixel:
                    # mouse position is roughly given by center of mass
                    mouse_pos = scipy_measurements.center_of_mass(labels, labels, mouse_label)
                    mouse_data = mouse_pos + (mouse_size,) 
    
                    if self.debug_video:
                        self.debug_video.add_circle(index, mouse_pos[::-1], 5, 'r')
                
            mouse_trajectory.append(mouse_data)

            if self.debug_video:
                self.debug_video.advance(index)
                

        return mouse_data