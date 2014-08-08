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

import debug


class MouseMovie(object):
    """
    analyzes the mouse movie
    """
    
    # radius of the mouse model in pixel
    mouse_size = 30
    # maximal speed in pixel per frame
    mouse_max_speed = 20 

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
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', 'video.mov')
            self.debug_video = VideoComposer(debug_file, background=self.video)
            
        else:
            self.debug_video = None


    def get_blurred_video(self, sigma=3):
        """ returns the mouse video with a monochrome and a blur filter """
        
        # add a monochrome filter if necessary
        if self.video.is_color:
            video = FilterMonochrome(self.video)
        else:
            video = self.video
    
        def blur(frame):
            """ helper function applying the blur """
            return cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=sigma)
        
        # return the video with a blur filter
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
    
    
    def _find_best_template_position(self, image, template, start_pos, max_deviation=None):
        """
        moves a template around until it has the largest overlap with image
        start_pos refers to the the center of the template inside the image
        """
        
        shape = template.shape
        pos_max = (image.shape[0] - shape[0], image.shape[1] - shape[1])
        start_pos = (start_pos[0] - shape[0]//2, start_pos[1] - shape[1]//2)
        points_to_check = [start_pos]
        seen = set()
        best_overlap = -np.inf
        best_pos = None 
        
        # iterate over all possible points
        while points_to_check:
            
            # get the next position to check, which refers to the top left image of the template
            pos = points_to_check.pop()
            seen.add(pos)
            
            # calculate the current overlap
            image_roi = image[pos[0]:pos[0] + shape[0], pos[1]:pos[1] + shape[1]]
            overlap = (image_roi*template).sum()
            
            # compare it to the previously seen one
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = pos
                
                # append all points around the current one
                for p in ((pos[0] - 1, pos[1]), (pos[0], pos[1] - 1),
                          (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1)):
                    if (not p in seen 
                        and 0 <= p[0] < pos_max[0] and 0 <= p[1] < pos_max[1]
                        and (max_deviation is None or 
                             np.hypot(p[0] - start_pos[0], p[1] - start_pos[1]) < max_deviation)
                        ):
                        points_to_check.append(p)
                
        return (best_pos[0] + shape[0]//2, best_pos[1] + shape[1]//2)
            
    
    def _find_mouse_in_binary_image(self, binary_image):
        """ finds the mouse in a binary image """
        
        # perform morphological closing to combined feature patches that are close 
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.mouse_size, self.mouse_size))
        moving_toward = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)

        # find all distinct features and label them
        labels, num_features = scipy_measurements.label(moving_toward)

        # find the largest object (which should be the mouse)
        mouse_label = np.argmax(
            scipy_measurements.sum(labels, labels, index=range(1, num_features + 1))
        )
        
        # determine mouse_pos by setting it to the center of mass
        return [
            int(p)
            for p in scipy_measurements.center_of_mass(labels, labels, mouse_label + 1)
        ]
    
    
    def find_mouse(self):
        """
        finds the mouse trajectory
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
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        def find_features(frame):
            """ helper function for removing the background """
            
            # calculate the difference to the current background model
            diff = frame - find_features.background
            
            # adapt the current background model
            # Here, the cut-off sets the relaxation time scale
            find_features.background += np.clip(diff, -1, 1)
            
            # find movement
            moving = (np.abs(diff) > 2.5*video_std.mean())
            
            # find features (i.e. the mouse) by evaluating changes and intensity 
            moving_toward = moving * (diff > 0)

            # convert the binary image to the normal output
            moving_toward = 255*moving_toward.astype(np.uint8)

            # perform morphological opening to remove noise
            moving_toward = cv2.morphologyEx(moving_toward, cv2.MORPH_OPEN, kernel_open)                        
            
            return moving_toward#, moving_away

        # initialize the background model
        find_features.background = np.array(video[0], np.double)
        
        # build a filter for finding the mouse position
        x, y = np.ogrid[-2*self.mouse_size:2*self.mouse_size + 1,
                        -2*self.mouse_size:2*self.mouse_size + 1]
        r = np.sqrt(x**2 + y**2)

        mouse_template = (
            # inner circle of ones
            (r <= self.mouse_size).astype(float)
            # + outer region that falls off
            + (np.cos((r - self.mouse_size)*np.pi/self.mouse_size) + 1)/2
              * ((self.mouse_size < r) & (r <= 2*self.mouse_size))
        )
#         debug.show_image(mouse_template)
        
        # iterate over all frames and find the mouse
        mouse_pos = None
        mouse_trajectory = []
        for frame in display_progress(video):

            # find features that indicate that the mouse moved
            moving_toward = find_features(frame)

            if moving_toward.sum() > 0:
                
                # find the position of the mouse
                if mouse_pos is None:
                    mouse_pos = self._find_mouse_in_binary_image(moving_toward)
                    
                else:
                    # adapt old mouse_pos by considering the movement
                    mouse_pos = self._find_best_template_position(moving_toward,
                                                                  mouse_template,
                                                                  mouse_pos,
                                                                  self.mouse_max_speed)
                
            if self.debug_video:
                # TODO: plot the outline of the mask, not the entire mask
                #self.debug_video.blend_image(moving_toward, mask=(moving_toward == 255))
                self.debug_video.add_contour(moving_toward)
                if mouse_pos is not None:
                    self.debug_video.add_circle(mouse_pos[::-1], 4, 'r')
                    self.debug_video.add_circle(mouse_pos[::-1], self.mouse_size, 'r', thickness=1)

            mouse_trajectory.append(mouse_pos)

        return mouse_trajectory
    