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
from video.filters import FilterBlur, FilterCrop
from video.analysis import get_largest_region, find_bounding_rect
from video.utils import display_progress
from video.composer import VideoComposer

import debug


class MouseMovie(object):
    """
    analyzes the mouse movie
    """
    
    # determines the rate with which the background is adapted
    background_adaptation_rate = 0.01
        
    # how much brighter than the background has the mouse to be
    mouse_intensity_threshold = 15
    # radius of the mouse model in pixel
    mouse_size = 25
    # maximal speed in pixel per frame
    mouse_max_speed = 30 

    def __init__(self, folder, frames=None, crop=None, prefix='', debug_output=None):
        """ initializes the whole mouse tracking and prepares the video filters """
        self.folder = folder
        
        # initialize video
        self.video = VideoFileStack(os.path.join(folder, 'raw_video/*'))
        # restrict the analysis to an interval of frames
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
        
        self.crop = crop
        self.prefix = prefix + '_' if prefix else ''
        self.debug_output = debug_output
        
        # setup internal structures that will be filled by analyzing the video
        self._cache = {} # cache that some functions might want to use
        self.result = {} # dictionary holding result information
        self.debug = {} # dictionary holding debug information
        self.mouse_pos = None # current model of the mouse position
        self._background = None
        self.video_blurred = None
        
        self.tracking_is_prepared = False
  

    def prepare_tracking(self):
        """ prepares tracking by analyzing the first frame. """
        
        self.crop_video_to_cage()
        
        # check whether a debug video is requested
        debug_output = [] if self.debug_output is None else self.debug_output
        if 'video' in debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', self.prefix + 'video.mov')
            self.debug['video'] = VideoComposer(debug_file, background=self.video)
            
        # blur the video to reduce noise effects    
        self.video_blurred = FilterBlur(self.video, 3)

        self.get_color_estimates(self.video_blurred[0])
        
        if 'background' in debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', self.prefix + 'background.mov')
            self.debug['background'] = VideoFileWriter(debug_file, self.video.size,
                                                       self.video.fps, is_color=False)

        self.tracking_is_prepared = True


    def process_video(self):
        """ processes the entire video """
        if not self.tracking_is_prepared:
            self.prepare_tracking()

        for frame in display_progress(self.video_blurred):
            self.update_background_model(frame)
            self.update_mouse_model(frame)


    def update_background_model(self, frame):
        """ updates the background model using the current frame """
        
        if self._background is None:
            # initialize background model with first frame
            self._background = np.array(frame, dtype=float)
            # allocate memory for the background mask, which will be used to
            # adapt the background to a change of the environment
            self._cache['background_mask'] = np.ones_like(frame, dtype=float)
        
        else:
            # adapt the current background model to the current frame
            
            mask = self._cache['background_mask']
            if self.mouse_pos is not None:
                # reset background mask to 1
                mask.fill(1)
                
                # subtract the current mouse model from the mask
                template = self._get_mouse_template()
                pos_x = self.mouse_pos[0] - template.shape[0]//2
                pos_y = self.mouse_pos[1] - template.shape[1]//2
                mask[pos_x:pos_x + template.shape[0], pos_y:pos_y + template.shape[1]] -= template
                
            # use the mask to adapt the background 
            self._background += self.background_adaptation_rate*mask*(frame - self._background)
            
        # write out the background if requested
        if 'background' in self.debug:
            self.debug['background'].write_frame(self._background)

            
    #===========================================================================
    # FINDING THE CAGE
    #===========================================================================
    
    
    def find_cage(self, image):
        """ analyzes a single image and locates the mouse cage in it.
        The rectangle [top, left, height, width] enclosing the cage is returned. """
        
        # do automatic thresholding to find large, bright areas
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find the largest bright area
        cage_mask = get_largest_region(binarized)
        
        # find an enclosing rectangle
        rect_large = find_bounding_rect(cage_mask)
         
        # crop image
        image = image[rect_large[0]:rect_large[2], rect_large[1]:rect_large[3]]
        rect_small = [0, 0, image.shape[0] - 1, image.shape[1] - 1]

        # threshold again, because large distractions outside of cages are now
        # definitely removed. Still, bright objects close to the cage, e.g. the
        # stands or some pipes in the background might distract the estimate.
        # We thus adjust the rectangle in the following  
        _, binarized = cv2.threshold(image, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # move top line down until we hit the cage boundary.
        # We move until more than 10% of the pixel are bright
        width = image.shape[0]
        brightness = binarized[rect_small[0], :].sum()
        while brightness < 0.1*255*width: 
            rect_small[0] += 1
            brightness = binarized[rect_small[0], :].sum()
        
        # move bottom line up until we hit the cage boundary
        # We move until more then 90% of the pixel are bright
        width = image.shape[0]
        brightness = binarized[rect_small[2], :].sum()
        while brightness < 0.9*255*width: 
            rect_small[2] -= 1
            brightness = binarized[rect_small[2], :].sum()

        # return the rectangle as [top, left, height, width]
        top = rect_large[0] + rect_small[0]
        left = rect_large[1] + rect_small[1]
        cage_rect = [top, left,
                     rect_small[2] - rect_small[0], 
                     rect_small[3] - rect_small[1]]

        return cage_rect

  
    def crop_video_to_cage(self):
        """ crops the video to a suitable cropping rectangle given by the cage """
        
        # crop the full video to the region specified by the user
        if self.crop is not None:
            if self.video.is_color:
                # restrict video to green channel
                video_crop = FilterCrop(self.video, self.crop, color_channel=1)
            else:
                video_crop = FilterCrop(self.video, self.crop)
            rect_given = video_crop.rect
            
        else:
            # use the full video
            video_crop = self.video
            rect_given = [0, 0, self.video.size[0] - 1, self.video.size[1] - 1]
        
        # find the cage in the first frame of the movie
        blurred_image = FilterBlur(video_crop, 3)[0]
        rect_cage = self.find_cage(blurred_image)
        
        # TODO: add plausibility test of cage dimensions
        
        # determine the rectangle of the cage in global coordinates
        top = rect_given[0] + rect_cage[0]
        left = rect_given[1] + rect_cage[1]
        height = rect_cage[2] - rect_cage[2] % 2 # make sure its divisible by 2
        width = rect_cage[3] - rect_cage[3] % 2  # make sure its divisible by 2
        cropping_rect = [top, left, height, width]
        
        logging.debug('The cage was determined to lie in the rectangle %s', cropping_rect)

        # crop the video to the cage region
        if self.video.is_color:
            # restrict video to green channel
            self.video = FilterCrop(self.video, cropping_rect, color_channel=1)
        else:
            self.video = FilterCrop(self.video, cropping_rect)
                
                
    def get_color_estimates(self, image):
        """ estimate the colors in the sky region and the sand region """
        
        # add black border around image, which is important for the distance 
        # transform we use later
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        # binarize image
        _, binarized = cv2.threshold(image, 0, 1,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # find sky by locating the largest black areas
        sky_mask = get_largest_region(1 - binarized).astype(np.uint8)*255

        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sky_mask, cv2.cv.CV_DIST_L2, 5)
        _, sky_sure = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)

        # determine the sky color
        sky_img = image[sky_sure.astype(np.bool)]
        self.color_sky = (sky_img.mean(), sky_img.std())
        logging.debug('The sky color was determined to be %s', self.color_sky)

        # find the sand by looking at the largest bright region
        sand_mask = get_largest_region(binarized).astype(np.uint8)*255
        
        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sand_mask, cv2.cv.CV_DIST_L2, 5)
        _, sand_sure = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        # determine the sky color
        sand_img = image[sand_sure.astype(np.bool)]
        self.color_sand = (sand_img.mean(), sand_img.std())
        logging.debug('The sand color was determined to be %s', self.color_sand)

                        
    #===========================================================================
    # FINDING THE MOUSE
    #===========================================================================
        
        
    def _get_mouse_template(self):
        """ creates a simple template for matching with the mouse.
        This template can be used to update the current mouse position based
        on information about the changes in the video.
        The template consists of a core region of maximal intensity and a ring
        region with gradually decreasing intensity.
        """
        
        try:
            return self._cache['mouse_template']
        
        except KeyError:
            
            # determine the sizes of the different regions
            size_core = self.mouse_size
            size_ring = 3*self.mouse_size
            size_total = size_core + size_ring
    
            # build a filter for finding the mouse position
            x, y = np.ogrid[-size_total:size_total + 1, -size_total:size_total + 1]
            r = np.sqrt(x**2 + y**2)
    
            # build the template
            mouse_template = (
                # inner circle of ones
                (r <= size_core).astype(float)
                # + outer region that falls off
                + np.exp(-((r - size_core)/size_core)**2)  # smooth function from 1 to 0
                  * (size_core < r)          # mask on ring region
            )  
            
            self._cache['mouse_template'] = mouse_template
        
            return mouse_template
        
                
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
        ) + 1
        
        # mouse position is center of mass of largest patch
        mouse_pos = [
            int(p)
            for p in ndimage.measurements.center_of_mass(labels, labels, mouse_label)
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
        moving_toward = moving_toward.astype(np.uint8)

        # perform morphological opening to remove noise
        moving_toward = cv2.morphologyEx(moving_toward, cv2.MORPH_OPEN, 
                                         self._cache['find_features_moving_forward.kernel_open'])                        
        
        return moving_toward

    
    def update_mouse_model(self, frame):
        """
        finds the mouse trajectory in the current video
        """
        
        # setup initial data
        if 'mouse_trajectory' not in self.result:
            self.result['mouse_trajectory'] = []

        # find features that indicate that the mouse moved
        moving_toward = self._find_features_moving_forward(frame)

        # check if features have been found
        if moving_toward.sum() > 0:
            
            if self.mouse_pos is not None:
                # adapt old mouse position by considering the movement
                self.mouse_pos = self._find_best_template_position(
                        frame*moving_toward,        # features weighted by intensity
                        self._get_mouse_template(), # search pattern
                        self.mouse_pos, self.mouse_max_speed)
                
            else:
                # determine mouse position from largest feature
                self.mouse_pos = self._find_mouse_in_binary_image(moving_toward)
                    
        if 'video' in self.debug:
            debug_video = self.debug['video']
            # plot the contour of the movement
            debug_video.add_contour(moving_toward)
            # indicate the mouse position
            if self.mouse_pos is not None:
                debug_video.add_circle(self.mouse_pos[::-1], 4, 'r')
                debug_video.add_circle(self.mouse_pos[::-1], self.mouse_size, 'r', thickness=1)

        self.result['mouse_trajectory'].append(self.mouse_pos)

                
    #===========================================================================
    # FINDING THE SAND
    #===========================================================================
    

    def find_sand_profile(self):
        
        image = self.video_blurred[0]
        
        # automatic thresholding to separate bright sand from dark background
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # distance transform to spot large regions of sand
        dist_transform = cv2.distanceTransform(binarized, cv2.cv.CV_DIST_L2, 5)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
        
        # thresholding to get regions, where we are sure that we have sand
        _, sure_sand = cv2.threshold(dist_transform, 0.75*dist_transform.max(), 1, 0)
        
        # determine the sand color
        sand = image[sure_sand.astype(np.bool)]
        self.color_sand = (sand.mean(), sand.std())
        print 'sand color', self.color_sand
        
        # set one sand pixel to mean
        image[max_loc[0], max_loc[1]] = self.color_sand[0]
        
#         cv2.floodFill(image, mask=None, seedPoint=max_loc, newVal=0,
#                       loDiff=self.color_sand[0] - self.color_sand[1],
#                       upDiff=self.color_sand[0] + self.color_sand[1],
#                       flags=cv2.FLOODFILL_FIXED_RANGE)
        
        
        debug.show_image(image)
        