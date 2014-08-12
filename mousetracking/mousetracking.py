'''
Created on Aug 5, 2014

@author: zwicker

Note that the OpenCV convention is to store images in [row, column] format
Thus, a point in an image is referred to as: image[point_y, point_x]
However, a single point is stored as: point = (point_x, point_y)

'''

from __future__ import division

import os
import logging
import itertools

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import minimize_scalar, leastsq
import cv2

from video.io import VideoFileStack, VideoFileWriter
from video.filters import FilterBlur, FilterCrop
from video.analysis.regions import get_largest_region, find_bounding_rect
from video.analysis.curves import curve_length, make_curve_equidistant, simplify_curve
from video.utils import display_progress
from video.composer import VideoComposerListener, VideoComposer

import debug


TRACKING_PARAMETERS_DEFAULT = {
                               
    # radius of the blur filter
    'video.blur': 3,
                               
    # determines the rate with which the background is adapted
    'background.adaptation_rate': 0.01,
    
    # spacing of the points in the sand profile
    'sand_profile.spacing': 20,
    # adapt the sand profile only every number of frames
    'sand_profile.skip_frames': 100,
    # width of the ridge in pixel
    'sand_profile.width': 5,
        
    # `mouse.intensity_threshold` determines how much brighter than the
    # background (usually the sky) has the mouse to be. This value is
    # measured in terms of standard deviations of the sky color
    'mouse.intensity_threshold': 1.5,
    # radius of the mouse model in pixel
    'mouse.size': 25,
    # maximal speed of the mouse in pixel per frame
    'mouse.max_speed': 30, 
    # after how many frames do we think we have lost the mouse entirely
    'mouse.lost_threshold': 100,
}


class MouseMovie(object):
    """
    analyzes mouse movies
    """
    
    def __init__(self, folder, frames=None, crop=None, prefix='', debug_output=None):
        """ initializes the whole mouse tracking and prepares the video filters """
        self.folder = folder
        
        # initialize video
        self.video = VideoFileStack(os.path.join(folder, 'raw_video/*'))
        # restrict the analysis to an interval of frames
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
        
        self.prefix = prefix + '_' if prefix else ''
        self.debug_output = [] if debug_output is None else debug_output
        self.params = TRACKING_PARAMETERS_DEFAULT.copy()
        
        # dictionary holding result information
        self.result = {
            'mouse.has_moved': False,
        }

        # setup internal structures that will be filled by analyzing the video
        # TODO: move all data which might be useful for later analysis to result dictionary
        self._cache = {}           # cache that some functions might want to use
        self.debug = {}            # dictionary holding debug information
        self.sand_profile = None   # current model of the sand profile
        self.mouse_pos = None      # current model of the mouse position
        self._mouse_not_seen = 0   # number of frames in which the mouse has not been seen 
        self._background = None    # current background image
        self._explored_area = None # region the mouse has explored, yet
        
        # restrict the video to the region of interest (the cage)
        self.crop_video_to_cage(crop)

        # blur the video to reduce noise effects    
        self.video_blurred = FilterBlur(self.video, self.params['video.blur'])
        first_frame = self.video_blurred[0]

        # estimate colors of sand and sky
        self.find_color_estimates(first_frame)
        
        # estimate initial sand profile
        self.find_sand_profile(first_frame)


    def process_video(self):
        """ processes the entire video """
        self.debug_setup()
        self.result['sand_profile'] = []

        # iterate over the video and analyze it
        for frame_id, frame in enumerate(display_progress(self.video_blurred)):
            # adapt current background model
            self.update_background_model(frame)
            
            # use the background to find the current sand profile
            if frame_id % self.params['sand_profile.skip_frames'] == 0:
                self.refine_sand_profile(self._background)
            self.result['sand_profile'].append(self.sand_profile)

            # find remaining features                
            self.update_mouse_model(frame)
            self.find_burrows()
            
            # store some information in the debug dictionary
            self.debug_add_frame()
            
        self.debug_finalize()

            
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

  
    def crop_video_to_cage(self, user_crop):
        """ crops the video to a suitable cropping rectangle given by the cage """
        
        # crop the full video to the region specified by the user
        if user_crop is not None:
            if self.video.is_color:
                # restrict video to green channel
                video_crop = FilterCrop(self.video, user_crop, color_channel=1)
            else:
                video_crop = FilterCrop(self.video, user_crop)
            rect_given = video_crop.rect
            
        else:
            # use the full video
            video_crop = self.video
            rect_given = [0, 0, self.video.size[0] - 1, self.video.size[1] - 1]
        
        # find the cage in the first frame of the movie
        blurred_image = FilterBlur(video_crop, self.params['video.blur'])[0]
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
                
            
    #===========================================================================
    # BACKGROUND MODEL AND COLOR ESTIMATES
    #===========================================================================
               
               
    def find_color_estimates(self, image):
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
        self.result['color.sky'] = sky_img.mean()
        self.result['color.sky_std'] = sky_img.std()
        logging.debug('The sky color was determined to be %d +- %d',
                      self.result['color.sky'], self.result['color.sky_std'])

        # find the sand by looking at the largest bright region
        sand_mask = get_largest_region(binarized).astype(np.uint8)*255
        
        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sand_mask, cv2.cv.CV_DIST_L2, 5)
        _, sand_sure = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        # determine the sky color
        sand_img = image[sand_sure.astype(np.bool)]
        self.result['color.sand'] = sand_img.mean()
        self.result['color.sand_std'] = sand_img.std()
        logging.debug('The sand color was determined to be %d +- %d',
                      self.result['color.sand'], self.result['color.sand_std'])
        
        
    def _get_mouse_template_slices(self, pos, i_shape, t_shape):
        """ calculates the slices necessary to compare a template to an image.
        Here, we assume that the image is larger than the template.
        """         

        # get the dimensions of the overlapping region        
        h = min(t_shape[0], i_shape[0] - pos[1])
        w = min(t_shape[1], i_shape[1] - pos[0])
        if h <= 0 or w <= 0:
            raise RuntimeError('Template and image do not overlap')
        
        # get the leftmost point in both images
        if pos[0] >= 0:
            i_x, t_x = pos[0], 0
        elif pos[0] <= -t_shape[1]:
            raise RuntimeError('Template and image do not overlap')
        else: # pos[0] < 0:
            i_x, t_x = 0, -pos[0]
            w += pos[0]
            
        # get the upper point in both images
        if pos[1] >= 0:
            i_y, t_y = pos[1], 0
        elif pos[1] <= -t_shape[0]:
            raise RuntimeError('Template and image do not overlap')
        else: # pos[1] < 0:
            i_y, t_y = 0, -pos[1]
            h += pos[1]
            
        # build the slices used to extract the information
        return ((slice(i_y, i_y + h), slice(i_x, i_x + w)),  # slice for the image
                (slice(t_y, t_y + h), slice(t_x, t_x + w)))  # slice for the template
    
        
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
                pos = (self.mouse_pos[0] - template.shape[0]//2,
                       self.mouse_pos[1] - template.shape[1]//2)
                
                i_s, t_s = self._get_mouse_template_slices(pos, mask.shape, template.shape)
                
                mask[i_s[0], i_s[1]] -= template[t_s[0], t_s[1]]
                
            # use the mask to adapt the background 
            self._background += self.params['background.adaptation_rate'] \
                                *mask*(frame - self._background)

                        
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
            size_core = self.params['mouse.size']
            size_ring = 3*self.params['mouse.size']
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
        
        t_top  = template.shape[0]//2
        t_left = template.shape[1]//2
        
        # calculate the initial top left position of the template
        start_pos = (start_pos[0] - t_left, start_pos[1] - t_top)
        # lists for checking all points surrounding the current one
        points_to_check, seen = [start_pos], set()
        # variables storing the best fit
        best_overlap, best_pos = -np.inf, None
        
        # iterate over all possible points 
        while points_to_check:
            
            # get the next position to check, which refers to the top left image of the template
            pos = points_to_check.pop()
            seen.add(pos)

            # get the slices required for comparing the template to the image
            i_s, t_s = self._get_mouse_template_slices(pos, image.shape, template.shape)
                        
            # calculate the similarity
            overlap = (image[i_s[0], i_s[1]]*template[t_s[0], t_s[1]]).sum()
            
            # compare it to the previously seen one
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = pos
                
                # add points around the current one to the test list
                for p in ((pos[0] - 1, pos[1]), (pos[0], pos[1] - 1),
                          (pos[0] + 1, pos[1]), (pos[0], pos[1] + 1)):
                    
                    # points will only be added if they have not already been checked
                    # and if the associated distance is below the threshold
                    if (not p in seen 
                        and (max_deviation is None or 
                             np.hypot(p[0] - start_pos[0], 
                                      p[1] - start_pos[1]) < max_deviation)
                        ):
                        points_to_check.append(p)
                
        # return the center position of the best template fit
        return (best_pos[0] + t_left, best_pos[1] + t_top)
        
          
    def _find_mouse_in_binary_image(self, binary_image):
        """ finds the mouse in a binary image """
        
        # build a kernel for morphological closing. We don't cache this kernel,
        # since the current function should only be called once per run
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (self.params['mouse.size'],)*2)
        
        # perform morphological closing to combined feature patches that are near 
        moving_toward = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)

        # find all distinct features and label them
        labels, num_features = ndimage.measurements.label(moving_toward)

        # find the largest object (which should be the mouse)
        mouse_label = np.argmax(
            ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
        ) + 1
        
        # mouse position is center of mass of largest patch
        mouse_pos = ndimage.measurements.center_of_mass(labels, labels, mouse_label)
        
        # switch the coordinates, such that they are given as (x, y)
        return np.array(mouse_pos[::-1], np.int)

    
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
        
        # find movement by comparing the difference to a threshold 
        moving_toward = (diff > self.params['mouse.intensity_threshold']*self.result['color.sky_std'])
        moving_toward = moving_toward.astype(np.uint8)

        # perform morphological opening to remove noise
        cv2.morphologyEx(moving_toward, cv2.MORPH_OPEN, 
                         self._cache['find_features_moving_forward.kernel_open'],
                         dst=moving_toward)                        

        # plot the contour of the movement if debug video is enabled
        if 'video' in self.debug:
            self.debug['video'].add_contour(moving_toward, color='g')

        return moving_toward

    
    def update_mouse_model(self, frame):
        """ adapts the current mouse position, if enough information is available """
        
        # setup initial data
        if 'mouse.trajectory' not in self.result:
            self.result['mouse.trajectory'] = []

        if self._explored_area is None:
            self._explored_area = np.zeros(self.video.size, np.uint8)

        # find features that indicate that the mouse moved
        moving_toward = self._find_features_moving_forward(frame)

        if moving_toward.sum() == 0:
            self._mouse_not_seen += 1
            
        else:
            # some moving features have been found in the video 
            
            if self.mouse_pos is None:
                # we have never seen the mouse, yet
                # => determine mouse position from largest feature
                self.mouse_pos = self._find_mouse_in_binary_image(moving_toward)
                self.result['mouse.position_initial'] = self.mouse_pos
                
            elif self._mouse_not_seen > self.params['mouse.lost_threshold']:
                # we have lost the mouse
                # => determine mouse position from largest feature
                self.mouse_pos = self._find_mouse_in_binary_image(moving_toward)
                    
            else:
                # we sort of know where the mouse is                
                # => adapt old mouse position by considering the movement
                self.mouse_pos = self._find_best_template_position(
                                        frame*moving_toward,        # features weighted by intensity
                                        self._get_mouse_template(), # search pattern
                                        self.mouse_pos,             # current known position
                                        self.params['mouse.max_speed'])
                
                # check whether the mouse has moved at all in this video 
                if not self.result['mouse.has_moved']:
                    mouse_dist = np.linalg.norm(self.mouse_pos - self.result['mouse.position_initial'])
                    if mouse_dist > self.params['mouse.size']:
                        self.result['mouse.has_moved'] = True

            # restrict the mask to the mouse region for further analysis
            mask = np.zeros(moving_toward.shape, np.uint8)
            mouse_pos = (int(self.mouse_pos[0]), int(self.mouse_pos[1]))
            cv2.circle(mask, mouse_pos, radius=self.params['mouse.size'], color=255, thickness=-1)
            cv2.bitwise_and(mask, moving_toward, dst=mask)
            
            # check whether the mouse has been seen
            if mask.sum() == 0:
                self._mouse_not_seen += 1
            else:
                self._mouse_not_seen = 0

            # update the area explored by the mouse
            cv2.bitwise_or(self._explored_area, mask, dst=self._explored_area)
                                
        self.result['mouse.trajectory'].append(self.mouse_pos)

                
    #===========================================================================
    # FINDING THE SAND PROFILE
    #===========================================================================
    
    
    def _find_rough_sand_profile(self, image):
        """ determines an estimate of the sand profile from a single image """
        
        # remove 10%/15% of each side of the image
        h = int(0.15*image.shape[0])
        w = int(0.10*image.shape[1])
        image_center = image[h:-h, w:-w]
        
        # binarize image
        cv2.threshold(image_center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=image_center)
        
        # TODO: we might want to replace that with the typical burrow radius
        # do morphological opening and closing to smooth the profile
        s = 4*self.params['sand_profile.spacing']
        ys, xs = np.ogrid[-s:s+1, -s:s+1]
        kernel = (xs**2 + ys**2 <= s**2).astype(np.uint8)

        # widen the mask
        mask = cv2.copyMakeBorder(image_center, s, s, s, s, cv2.BORDER_REPLICATE)
        # make sure their is sky on the top
        mask[:s + h, :] = 0
        # make sure their is and at the bottom
        mask[-s - h:, :] = 255

        # morphological opening to remove noise and clutter
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)

        # morphological closing to smooth the boundary and remove burrows
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE, dst=mask)
        
        # reduce the mask to its original dimension
        mask = mask[s:-s, s:-s]
        
        # get the contour from the mask and store points as (x, y)
        points = [(x + w, np.nonzero(col)[0][0] + h)
                  for x, col in enumerate(mask.T)]            

        # simplify the curve        
        points = np.array(simplify_curve(points, epsilon=2))

        return np.array(points)
   
        
    def refine_sand_profile(self, image):
        """ adapts a sand profile given as points to a given image.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
                
        points = self.sand_profile
        spacing = self.params['sand_profile.spacing']
            
        if not 'sand_profile.model' in self._cache or self._cache['sand_profile.model'].size != spacing:
            self._cache['sand_profile.model'] = \
                    RidgeProfile(spacing, self.params['sand_profile.width'])
                
        # make sure the curve has equidistant points
        sand_profile = make_curve_equidistant(points, spacing)
        sand_profile = np.array(sand_profile)

        # calculate the bounds for the points
        p_min =  spacing 
        y_max, x_max = image.shape[0] - spacing, image.shape[1] - spacing

        # iterate through all points and correct profile
        sand_profile_model = self._cache['sand_profile.model']
        corrected_points = []
        x_previous = spacing
        for k, p in enumerate(sand_profile):
            
            # skip points that are too close to the boundary
            if (p[0] < p_min or p[0] > x_max or 
                p[1] < p_min or p[1] > y_max):
                continue
            
            # determine the local slope of the profile, which fixes the angle 
            if k == 0 or k == len(sand_profile) - 1:
                # we only move these vertically to keep the profile length
                # approximately constant
                angle = np.pi/2
            else:
                dp = sand_profile[k+1] - sand_profile[k-1]
                angle = np.arctan2(dp[0], dp[1]) # y-coord, x-coord
                
            # extract the region image
            region = image[p[1]-spacing : p[1]+spacing+1, p[0]-spacing : p[0]+spacing+1].copy()
            sand_profile_model.set_data(region, angle) 

            # maximize the difference between the colors of the two half planes, which
            # should separate out sky from sand
            x, _, infodict, _, _ = \
                leastsq(sand_profile_model.get_difference, [0], xtol=0.1, full_output=True)
            
            # calculate goodness of fit
            ss_err = (infodict['fvec']**2).sum()
            ss_tot = ((region - region.mean())**2).sum()
            rsquared = 1 - ss_err/ss_tot

            # Note, that we never remove the first and the last point
            if rsquared > 0.1 or k == 0 or k == len(sand_profile) - 1:
                # we are rather confident that this point is better than it was
                # before and thus add it to the result list
                p_x = p[0] + x[0]*np.cos(angle)
                p_y = p[1] - x[0]*np.sin(angle)
                # make sure that we have no overhanging ridges
                if p_x >= x_previous:
                    corrected_points.append((p_x, p_y))
                    x_previous = p_x
            
        #print self.sand_profile[0, 0], corrected_points[0][0]
        self.sand_profile = np.array(corrected_points)
            

    def find_sand_profile(self, image):
        """ finds the sand profile given an image of an antfarm """

        # get an estimate of a profile
        self.sand_profile = self._find_rough_sand_profile(image)
        
        # iterate until the profile does not change significantly anymore
        length_last, length_current = 0, curve_length(self.sand_profile)
        iterations = 0
        while abs(length_current - length_last)/length_current > 0.001 or iterations < 5:
            self.refine_sand_profile(image)
            length_last, length_current = length_current, curve_length(self.sand_profile)
            iterations += 1
            
        logging.info('We found a sand profile of length %g after %d iterations',
                     length_current, iterations)
        
                    
    #===========================================================================
    # FIND BURROWS 
    #===========================================================================


    def find_burrows(self):
        """ locates burrows by combining the information of the sand profile
        and the explored area """
        
        # build a mask with potential burrows
        mask = np.zeros(self.video.size, np.uint8)
        height, width = self.video.size
        
        # create a mask for the region below the current sand profile
        points = np.empty((len(self.sand_profile) + 4, 2), np.int)
        points[:-4, :] = self.sand_profile
        points[-4, :] = (width, points[-5, 1])
        points[-3, :] = (width, height)
        points[-2, :] = (0, height)
        points[-1, :] = (0, points[0, 1])
        cv2.fillPoly(mask, [points], color=128)

        # erode the mask slightly, since the sand profile is not perfect        
        w = 2*self.params['sand_profile.width']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        cv2.erode(mask, kernel, dst=mask)

        # combine with the information of what areas have been explored
        cv2.bitwise_or(mask, self._explored_area, dst=mask)
        
#         debug.show_image(mask)


    #===========================================================================
    # DATA ANALYSIS
    #===========================================================================


    def mouse_underground(self):
        """ checks whether the mouse is under ground """
        pos = self.mouse_pos
        if pos is None:
            return None
        else:
            profile = self.sand_profile
            sand_y = np.interp(pos[0], profile[:, 0], profile[:, 1])
            return pos[1] - self.params['mouse.size']/2 > sand_y


    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        
        # setup the video output, if requested
        if 'video' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', self.prefix + 'video.mov')
            self.debug['video'] = VideoComposerListener(debug_file, background=self.video, is_color=True)
            
        # setup the background output, if requested
        if 'background' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = os.path.join(self.folder, 'debug', self.prefix + 'background.mov')
            self.debug['background.video'] = VideoFileWriter(debug_file, self.video.size,
                                                             self.video.fps, is_color=False)
        if 'explored_area' in self.debug_output:
            # initialize the writer for the explored area video 
            debug_file = os.path.join(self.folder, 'debug', self.prefix + 'explored.mov')
            self.debug['explored_area.video'] = VideoComposer(debug_file, self.video.size,
                                                              self.video.fps, is_color=False)


    def debug_add_frame(self):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the sand profile
            debug_video.add_polygon(self.sand_profile, is_closed=False, color='y')
            debug_video.add_points(self.sand_profile, radius=2, color='y')
        
            # indicate the mouse position
            if self.mouse_pos is not None:
                if not self.result['mouse.has_moved']:
                    color = 'r'
                elif self.mouse_underground():
                    color = 'k'
                else:
                    color = 'w'
                debug_video.add_circle(self.mouse_pos, 4, color)
                debug_video.add_circle(self.mouse_pos, self.params['mouse.size'], color, thickness=1)
                
            debug_video.add_text('not seen: %d' % self._mouse_not_seen, (20, 20), anchor='top')
            

        if 'background.video' in self.debug:
            self.debug['background.video'].write_frame(self._background.copy())

        if 'explored_area.video' in self.debug:
            debug_video = self.debug['explored_area.video']
             
            # set the background
            debug_video.set_frame(128*self._explored_area)
            
            # plot the sand profile
            debug_video.add_polygon(self.sand_profile, is_closed=False, color='y')
            debug_video.add_points(self.sand_profile, radius=2, color='y')


    def debug_finalize(self):
        """ close the video streams when done iterating """
        for identifier in ('video', 'background.video', 'explored_area.video'):
            if identifier in self.debug:
                del self.debug[identifier]
            
    


class RidgeProfile(object):
    """ represents a ridge profile to compare it against an image in fitting """
    
    def __init__(self, size, profile_width=1):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.size = size
        self.ys, self.xs = np.ogrid[-size:size+1, -size:size+1]
        self.width = profile_width
        self.image = None
        
        
    def set_data(self, image, angle):
        """ sets initial data used for fitting
        image denotes the data we compare the model to
        angle defines the direction perpendicular to the profile 
        """
        
        self.image = image
        self.image_mean = image.mean()
        self.image_std = image.std()
        self._sina = np.sin(angle)
        self._cosa = np.cos(angle)
        
        
    def get_difference(self, distance):
        """ calculates the difference between image and model, when the 
        model is moved by a certain distance in its normal direction """ 
        # determine center point
        px =  distance*self._cosa
        py = -distance*self._sina
        
        # determine the distance from the ridge line
        dist = (self.ys - py)*self._sina - (self.xs - px)*self._cosa + 0.5 # TODO: check the 0.5
        
        # apply sigmoidal function
        model = np.tanh(dist/self.width)
     
        return np.ravel(self.image_mean + 1.5*self.image_std*model - self.image)
