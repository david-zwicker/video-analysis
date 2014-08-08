'''
Created on Aug 5, 2014

@author: zwicker
'''

from __future__ import division

import os
import logging
import numpy as np
import scipy.ndimage as ndimage
import cv2

from video.io import VideoFileStack, VideoFileWriter
from video.filters import FilterMonochrome, FilterFunction, FilterCrop
from video.analysis import measure_mean_std
from video.utils import display_progress
from video.composer import VideoComposer

import debug


class MouseMovie(object):
    """
    analyzes the mouse movie
    """
    
    # determines the rate with which the background is adapted
    background_adaptation_rate = 0.0001
    
    # number of frames considered to estimate the noise 
    noise_estimate_frames= 200
    
    # how much brighter than the background has the mouse to be
    mouse_intensity_threshold = 15
    # radius of the mouse model in pixel
    mouse_size = 25
    # maximal speed in pixel per frame
    mouse_max_speed = 30 

    def __init__(self, folder, frames=None, crop=None, debug_output=None, prefix=''):
        
        self.folder = folder
        
        # initialize video
        self.video = VideoFileStack(os.path.join(folder, 'raw_video/*'))
        
        # setup caches
        self._cache = {} # cache that some functions might want to use
        self.result = {} # dictionary holding result information
        self.debug = {} # dictionary holding debug information
        self.prefix = prefix
        
        # restrict the analysis to an interval of frames
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
            
        # restrict the analysis to a region
        if crop is not None:
            self.video = FilterCrop(self.video, crop)
        
        # check whether a debug video is requested
        debug_output = [] if debug_output is None else debug_output
        if 'video' in debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', prefix + 'video.mov')
            self.debug['video'] = VideoComposer(debug_file, background=self.video)
            
        # only work with the blurred video, to reduce noise effects    
        self.video_blurred = self.get_blurred_video(self.video) 
        
        # register a listener, which takes care of the background model
        self._background = None
        self.video_blurred.register_listener(self.update_background_model)
        
        if 'background' in debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', prefix + 'background.mov')
            self.debug['background'] = VideoFileWriter(debug_file, self.video.size,
                                                       self.video.fps, is_color=False)

        # estimate the noise in the image by looking at the first number of slides
#         std = measure_mean_std(self.video[:self.noise_estimate_frames])[1]
#         self.noise = std.mean()
#         logging.debug('Noise level was determined as %g', self.noise)
            

    def get_blurred_video(self, video, sigma=3):
        """ returns the mouse video with a monochrome and a blur filter """
        
        # add a monochrome filter if necessary
        if video.is_color:
            # take the green channel, which is quicker than considering
            # the mean of all color values
            video = FilterMonochrome(video, 'g')
    
        def blur(frame):
            """ helper function applying the blur """
            return cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=sigma)
        
        # return the video with a blur filter
        return FilterFunction(video, blur)


    def update_background_model(self, frame):
        """ updates the background model using the current frame """
        if self._background is None:
            # initialize background model with first frame
            self._background = np.array(frame, dtype=float)
        
        else:
            # adapt the current background model to current frame
            self._background = np.minimum(
                self._background + self.background_adaptation_rate*frame, frame
            )
            
        if 'background' in self.debug:
            self.debug['background'].write_frame(self._background)


    def get_movie_statistics(self):
        """ returns the mean and std over the entire movie """
        raise NotImplementedError
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
    

    def update_statistics(self, frame):
        """
        updates the estimates of the mean and the standard deviation of each 
        video pixel over time.
        Uses https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm
        """
        raise NotImplementedError
        self._video_stats_count += 1
        
        if self.video_mean is None:
            # initialize the statistics 
            self.video_mean = np.array(frame, dtype=np.int) 
            self.video_var = np.zeros_like(frame, dtype=np.int)
            
        else:
            # update the statistics 
            delta = frame - self.video_mean
            self.video_mean += delta/self._video_stats_count
            
            # use the same time scale as the background_history to estimate
            # the variance
            n = max(self._video_stats_count, self.background_history)
            self.video_var = (n*self.video_var + delta**2)/(n + 1)
            
            
    def get_std(self):
        """ returns the estimated standard deviation of the video """ 
        raise NotImplementedError
        if self._video_stats_count < 2:
            return 0
        else:
            return np.sqrt(self.video_var.mean())
            
        
    #===========================================================================
    # FINDING THE MOUSE
    #===========================================================================
        
                
    def _find_best_template_position(self, image, template, start_pos, max_deviation=None):
        """
        moves a template around until it has the largest overlap with image
        start_pos refers to the the center of the template inside the image
        max_deviation sets a maximal distance the template is moved from start_pos
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
        
        
    def _get_mouse_template(self):
        """ creates a simple template for matching with the mouse.
        This template can be used to update the current mouse position based
        on information about the changes in the video.
        The template consists of a core region of maximal intensity and a ring
        region with gradually decreasing intensity.
        """
        
        # determine the sizes of the different regions
        size_core = self.mouse_size
        size_ring = 3*self.mouse_size
        size_total = size_core + size_ring

        # build a filter for finding the mouse position
        x, y = np.ogrid[-size_total:size_total + 1, -size_total:size_total + 1]
        r = np.sqrt(x**2 + y**2)

        # build the template
#         mouse_template = (
#             # inner circle of ones
#             (r <= size_core).astype(float)
#             # + outer region that falls off
#             + (np.cos((r - size_core)*np.pi/size_ring) + 1)/2  # smooth function from 1 to 0
#               * ((size_core < r) & (r <= size_total))          # mask on ring region
#         )  

        # build the template
        mouse_template = (
            # inner circle of ones
            (r <= size_core).astype(float)
            # + outer region that falls off
            + np.exp(-((r - size_core)/size_core)**2)  # smooth function from 1 to 0
              * (size_core < r)          # mask on ring region
        )  
        
        return mouse_template
        
          
    def _find_mouse_in_binary_image(self, binary_image):
        """ finds the mouse in a binary image """
        
        # build a kernel for morphological closing. We don't cache this kernel,
        # since the current function should only be called once per run
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.mouse_size, self.mouse_size))
        
        # perform morphological closing to combined feature patches that are near 
        moving_toward = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)

        # find all distinct features and label them
        labels, num_features = ndimage.measurements.label(moving_toward)

        # find the largest object (which should be the mouse)
        mouse_label = np.argmax(
            ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
        )
        
        # mouse position is center of mass of largest patch
        mouse_pos = [
            int(p)
            for p in ndimage.measurements.center_of_mass(labels, labels, mouse_label + 1)
        ]
        return mouse_pos

    
    def _find_features_moving_forward(self, frame):
        """ finds moving features in a frame.
        This works by building a model of the current background and subtracting
        this from the current frame. Everything that deviates significantly from
        the background must be moving. Here, we additionally only focus on 
        features that become brighter, i.e. move forward.
        """
        
        # prepare the kernel for morphological opening if necessary
        if 'find_features_moving_forward.kernel_open' not in self._cache:
            self._cache['find_features_moving_forward.kernel_open'] = \
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        
        # calculate the difference to the current background model
        # Note that the first operand determines the dtype of the result.
        diff = -self._background + frame 
        
        # find movement, this should in principle be a factor multiplied by the 
        # noise in the image (estimated by its standard deviation), but a
        # constant factor is good enough right now
        moving_toward = (diff > self.mouse_intensity_threshold)

        # convert the binary image to the normal output
        moving_toward = 255*moving_toward.astype(np.uint8)

        # perform morphological opening to remove noise
        moving_toward = cv2.morphologyEx(moving_toward, cv2.MORPH_OPEN, 
                                         self._cache['find_features_moving_forward.kernel_open'])                        
        
        return moving_toward

    
    def find_mouse(self):
        """
        finds the mouse trajectory in the current video
        """
        
        mouse_template = self._get_mouse_template()
        mouse_pos = None
        mouse_trajectory = []

        # iterate over all frames and find the mouse
        for frame in display_progress(self.video_blurred):

            # find features that indicate that the mouse moved
            moving_toward = self._find_features_moving_forward(frame)

            # check if features have been found            
            if moving_toward.sum() > 0:
                
                if mouse_pos is None:
                    # determine mouse position from largest feature
                    mouse_pos = self._find_mouse_in_binary_image(moving_toward)
                    
                else:
                    # adapt old mouse_pos by considering the movement
                    mouse_pos = self._find_best_template_position(moving_toward,
                                                                  mouse_template,
                                                                  mouse_pos,
                                                                  self.mouse_max_speed)
                
            if 'video' in self.debug:
                debug_video = self.debug['video']
                # plot the contour of the movement
                debug_video.add_contour(moving_toward)
                # indicate the mouse position
                if mouse_pos is not None:
                    debug_video.add_circle(mouse_pos[::-1], 4, 'r')
                    debug_video.add_circle(mouse_pos[::-1], self.mouse_size, 'r', thickness=1)

            mouse_trajectory.append(mouse_pos)

        self.result['mouse_trajectory'] = mouse_trajectory
        return mouse_trajectory
    
    
    #===========================================================================
    # FINDING THE CAGE
    #===========================================================================
    
    
    def find_cage(self):
        
        frame = self.get_blurred_video(self.video, 1)[0]
        
        for r in range(3):
            frame = cv2.pyrDown(frame)
            
        print frame.shape
        edges = cv2.Canny(frame, 30, 120)#, apertureSize=3)
#         edges = cv2.Laplacian(frame, cv2.CV_64F)
#         lines = cv2.HoughLines(edges, 1, .1, 1)#, None, 50, 10)
        lines = cv2.HoughLinesP(edges, 1, .01, 1, None, 50, 10)
        
        print lines
        import matplotlib.pyplot as plt
        
        plt.imshow(edges)
        plt.gray()
        
        for line in lines[0]:
            plt.plot(line[::2], line[1::2], 'r')
        plt.show()