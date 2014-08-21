'''
Created on Aug 5, 2014

@author: zwicker

Note that the OpenCV convention is to store images in [row, column] format
Thus, a point in an image is referred to as image[coord_y, coord_x]
However, a single point is stored as point = (coord_x, coord_y)
Similarly, we store rectangles as (coord_x, coord_y, width, height)

Furthermore, the color space in OpenCV is typically BGR instead of RGB

Generally, x-values increase from left to right, while y-values increase from
top to bottom. The origin is thus in the upper left corner.
'''

from __future__ import division

import logging

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import leastsq
from scipy.spatial import distance
import shapely.geometry as geometry
import cv2

from video.io import ImageShow
from video.filters import FilterBlur, FilterCrop
from video.analysis.regions import (get_largest_region, find_bounding_box,
                                    rect_to_slices, corners_to_rect,
                                    get_overlapping_slices)
from video.analysis.curves import curve_length, make_curve_equidistant, simplify_curve, point_distance
from video.utils import display_progress
from video.composer import VideoComposerListener, VideoComposer


from .data_handler import DataHandler
from .mouse_objects import (Object, ObjectTrack, Burrow, BurrowTrack,
                            GroundProfile, RidgeProfile)

import debug



class FirstPass(DataHandler):
    """
    analyzes mouse movies
    """
    
    def __init__(self, folder, prefix='', parameters=None, debug_output=None):
        """ initializes the whole mouse tracking and prepares the video filters """
        
        # initialize the data handler
        super(FirstPass, self).__init__(folder, prefix, parameters)
        self.params = self.data['parameters']
        self.data.create_child('pass1')
        self.result = self.data['pass1'] 
        
        self.log_event('Started initializing the video analysis.')
        
        # setup internal structures that will be filled by analyzing the video
        self._cache = {}               # cache that some functions might want to use
        self.debug = {}                # dictionary holding debug information
        self.ground = None             # current model of the ground profile
        self.tracks = []               # list of plausible mouse models in current frame
        self._mouse_pos_estimate = []  # list of estimated mouse positions
        self.explored_area = None      # region the mouse has explored yet
        self.frame_id = None           # id of the current frame
        self.result['mouse/has_moved'] = False
        self.debug_output = [] if debug_output is None else debug_output
        
        # load the video ... 
        self.video = self.load_video()
        # ... and restrict to the region of interest (the cage)
        self.video, cropping_rect = self.crop_video_to_cage()
        self.data.create_child('video/analyzed', {'frame_count': self.video.frame_count,
                                                  'region_cage': cropping_rect,
                                                  'size': '%d x %d' % self.video.size,
                                                  'fps': self.video.fps})

        # blur the video to reduce noise effects    
        self.video_blurred = FilterBlur(self.video, self.params['video/blur_radius'])
        first_frame = self.video_blurred[0]

        # initialize the background model
        self.background = np.array(first_frame, dtype=float)

        # estimate colors of sand and sky
        self.find_color_estimates(first_frame)
        
        # estimate initial ground profile
        logging.debug('Find the initial ground profile.')
        self.find_initial_ground(first_frame)

        self.data['analysis-status'] = 'Initialized first pass'            
        self.log_event('Finished initializing the video analysis.')

    
    def process_video(self):
        """ processes the entire video """

        self.log_event('Setting up the cache and debug objects.')
        
        self.debug_setup()
        self.setup_processing()

        self.log_event('Started iterating through the video.')
        
        try:
            # iterate over the video and analyze it
            for self.frame_id, frame in enumerate(display_progress(self.video_blurred)):
                
                if self.frame_id % self.params['colors/adaptation_interval'] == 0:
                    self.find_color_estimates(frame)
                
                if self.frame_id >= self.params['video/ignore_initial_frames']:
                    # find the mouse
                    self.find_objects(frame)
                    
                    # use the background to find the current ground profile and burrows
                    if self.frame_id % self.params['ground/adaptation_interval'] == 0:
                        self.refine_ground(self.background)
                        ground = GroundProfile(self.frame_id, self.ground)
                        self.result['ground/profile'].append(ground)
            
                if self.frame_id % self.params['burrows/adaptation_interval'] == 0:
                    self.find_burrows(frame)
            
                # update the background model
                self.update_background_model(frame)
                    
                # store some information in the debug dictionary
                self.debug_add_frame(frame)
                
        except KeyboardInterrupt:
            logging.info('Tracking has been interrupted by user.')
            self.log_event('Analysis run has been interrupted.')
            
        else:
            self.log_event('Finished iterating through the frames.')
        
        frames_analyzed = self.frame_id + 1
        if frames_analyzed == self.video.frame_count:
            self.data['analysis-status'] = 'Finished first pass'
        else:
            self.data['analysis-status'] = 'Partly finished first pass'
        self.data['video/analyzed/frames_analyzed'] = frames_analyzed
                    
        # cleanup and write out of data
        self.debug_finalize()
        self.write_data()


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        
        self.result['objects/tracks'] = []
        self.result['ground/profile'] = []
        self.result['burrows/data'] = []

        # creates a simple template for matching with the mouse.
        # This template can be used to update the current mouse position based
        # on information about the changes in the video.
        # The template consists of a core region of maximal intensity and a ring
        # region with gradually decreasing intensity.
        
        # determine the sizes of the different regions
        size_core = self.params['mouse/model_radius']
        size_ring = 3*self.params['mouse/model_radius']
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
        
        self._cache['mouse.template'] = mouse_template
        
        # prepare kernels for morphological operations
        self._cache['find_moving_features.kernel_open'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._cache['find_moving_features.kernel_close'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # setup more cache variables
        video_shape = (self.video.size[1], self.video.size[0]) 
        self.explored_area = np.zeros(video_shape, np.double)
        self._cache['background.mask'] = np.ones(video_shape, np.double)
        
            
    #===========================================================================
    # FINDING THE CAGE
    #===========================================================================
    
    
    def find_cage(self, image):
        """ analyzes a single image and locates the mouse cage in it.
        Try to find a bounding box for the cage.
        The rectangle [top, left, height, width] enclosing the cage is returned. """
        
        # do automatic thresholding to find large, bright areas
        _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find the largest bright are, which should contain the cage
        cage_mask = get_largest_region(binarized)
        
        # find an enclosing rectangle, which usually overestimates the cage bounding box
        rect_large = find_bounding_box(cage_mask)
         
        # crop image to this rectangle, which should surely contain the cage 
        image = image[rect_to_slices(rect_large)]

        # initialize the rect coordinates
        top = 0 # start on first row
        bottom = image.shape[0] - 1 # start on last row
        width = image.shape[1]

        # threshold again, because large distractions outside of cages are now
        # definitely removed. Still, bright objects close to the cage, e.g. the
        # stands or some pipes in the background might distract the estimate.
        # We thus adjust the rectangle in the following  
        _, binarized = cv2.threshold(image, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # move top line down until we hit the cage boundary.
        # We move until more than 10% of the pixel on a horizontal line are bright
        brightness = binarized[top, :].sum()
        while brightness < 0.1*255*width: 
            top += 1
            brightness = binarized[top, :].sum()
        
        # move bottom line up until we hit the cage boundary
        # We move until more then 90% of the pixel of a horizontal line are bright
        brightness = binarized[bottom, :].sum()
        while brightness < 0.9*255*width: 
            bottom -= 1
            brightness = binarized[bottom, :].sum()

        # return the rectangle defined by two corner points
        p1 = (rect_large[0], rect_large[1] + top)
        p2 = (rect_large[0] + width - 1, rect_large[1] + bottom)
        return corners_to_rect(p1, p2)

  
    def crop_video_to_cage(self):
        """ crops the video to a suitable cropping rectangle given by the cage """
        
        # find the cage in the first frame of the movie
        blurred_image = FilterBlur(self.video, self.params['video/blur_radius'])[0]
        rect_cage = self.find_cage(blurred_image)
        
        # determine the rectangle of the cage in global coordinates
        width = rect_cage[2] - rect_cage[2] % 2   # make sure its divisible by 2
        height = rect_cage[3] - rect_cage[3] % 2  # make sure its divisible by 2
        # Video dimensions should be divisible by two for some codecs

        if not (self.params['cage/width_min'] < width < self.params['cage/width_max'] and
                self.params['cage/height_min'] < height < self.params['cage/height_max']):
            raise RuntimeError('The cage bounding box (%dx%d) is out of the limits.' % (width, height)) 
        
        rect_cage = (rect_cage[0], rect_cage[1], width, height)
        
        logging.debug('The cage was determined to lie in the rectangle %s', rect_cage)

        # crop the video to the cage region
        return FilterCrop(self.video, rect_cage), rect_cage
            
            
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
        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sky_sure = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)

        # determine the sky color
        sky_img = image[sky_sure.astype(np.bool)]
        self.result['colors/sky'] = sky_img.mean()
        self.result['colors/sky_std'] = sky_img.std()
        logging.debug('The sky color was determined to be %d +- %d',
                      self.result['colors/sky'], self.result['colors/sky_std'])

        # find the sand by looking at the largest bright region
        sand_mask = get_largest_region(binarized).astype(np.uint8)*255
        
        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sand_mask, cv2.cv.CV_DIST_L2, 5)
        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sand_sure = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        # determine the sky color
        sand_img = image[sand_sure.astype(np.bool)]
        self.result['colors/sand'] = sand_img.mean()
        self.result['colors/sand_std'] = sand_img.std()
        logging.debug('The sand color was determined to be %d +- %d',
                      self.result['colors/sand'], self.result['colors/sand_std'])
        
                
    def update_background_model(self, frame):
        """ updates the background model using the current frame """
        
        if self._mouse_pos_estimate:
            # load some values from the cache
            mask = self._cache['background.mask']
            template = 1 - self._cache['mouse.template']
            mask.fill(1)
            
            # cut out holes from the mask for each mouse estimate
            for mouse_pos in self._mouse_pos_estimate:
                # get the slices required for comparing the template to the image
                t_s, i_s = get_overlapping_slices(mouse_pos, template.shape, frame.shape)
                mask[i_s[0], i_s[1]] *= template[t_s[0], t_s[1]]
                
        else:
            # disable the mask if no mouse is known
            mask = 1

        # adapt the background to current frame, but only inside the mask 
        self.background += (self.params['background/adaptation_rate']  # adaptation rate 
                             *mask                                     # mask 
                             *(frame - self.background))               # difference to current frame

                        
    #===========================================================================
    # FINDING THE MOUSE
    #===========================================================================
      
    
    def find_moving_features(self, frame):
        """ finds moving features in a frame.
        This works by building a model of the current background and subtracting
        this from the current frame. Everything that deviates significantly from
        the background must be moving. Here, we additionally only focus on 
        features that become brighter, i.e. move forward.
        """
                        
        # calculate the difference to the current background model
        # Note that the first operand determines the dtype of the result.
        diff = -self.background + frame 
        
        # find movement by comparing the difference to a threshold 
        mask_moving = (diff > self.params['mouse/intensity_threshold']*self.result['colors/sky_std'])
        mask_moving = 255*mask_moving.astype(np.uint8)

        # perform morphological opening to remove noise
        cv2.morphologyEx(mask_moving, cv2.MORPH_OPEN, 
                         self._cache['find_moving_features.kernel_open'],
                         dst=mask_moving)    
         
        # perform morphological closing to join distinct features
        cv2.morphologyEx(mask_moving, cv2.MORPH_CLOSE, 
                         self._cache['find_moving_features.kernel_close'],
                         dst=mask_moving)

        # plot the contour of the movement if debug video is enabled
        if 'video' in self.debug:
            self.debug['video'].add_contour(mask_moving, color='g', copy=True)

        return mask_moving


    def find_objects_in_binary_image(self, labels, num_features):
        """ finds objects in a binary image.
        Returns a list with characteristic properties
        """
        
        # find large objects (which could be the mouse)
        objects = []
        largest_obj = Object((0, 0), 0)
        for label in xrange(1, num_features + 1):
            # calculate the image moments
            moments = cv2.moments((labels == label).astype(np.uint8))
            area = moments['m00']
            # get the coordinates of the center of mass
            pos = (moments['m10']/area, moments['m01']/area)
            
            # check whether this object could be a mouse
            if area > self.params['mouse/min_area']:
                objects.append(Object(pos, size=area, label=label))
                
            elif area > largest_obj.size:
                # determine the maximal area during the loop
                largest_obj = Object(pos, size=area, label=label)
        
        if len(objects) == 0:
            # if we haven't found anything yet, we just take the best guess,
            # which is the object with the largest area
            objects.append(largest_obj)
        
        return objects


    def _handle_object_tracks(self, frame, labels, num_features):
        """ analyzes objects in a single frame and tries to add them to
        previous tracks """
        # get potential objects
        objects_found = self.find_objects_in_binary_image(labels, num_features)

        # check if there are previous tracks        
        if len(self.tracks) == 0:
            moving_window = self.params['objects/matching_moving_window']
            moving_threshold = self.params['objects/matching_moving_threshold']
            self.tracks = [ObjectTrack(self.frame_id, obj, moving_window, moving_threshold)
                           for obj in objects_found]
            
            return # there is nothing to do anymore
            
        # calculate the distance between new and old objects
        dist = distance.cdist([obj.pos for obj in objects_found],
                              [obj.predict_pos() for obj in self.tracks],
                              metric='euclidean')
        # normalize distance to the maximum speed
        dist = dist/self.params['mouse/max_speed']
        
        # calculate the difference of areas between new and old objects
        def area_score(area1, area2):
            """ helper function scoring area differences """
            return abs(area1 - area2)/(area1 + area2)
        
        areas = np.array([[area_score(obj_f.size, obj_e.last_size)
                           for obj_e in self.tracks]
                          for obj_f in objects_found])
        # normalize area change such that 1 corresponds to the maximal allowed one
        areas = areas/self.params['mouse/max_rel_area_change']

        # build a combined score from this
        alpha = self.params['objects/matching_weigth']
        score = alpha*dist + (1 - alpha)*areas

        # match previous estimates to this one
        idx_f = range(len(objects_found)) # indices of new objects
        idx_e = range(len(self.tracks))   # indices of old objects
        while True:
            # get the smallest score, which corresponds to best match            
            score_min = score.min()
            
            if score_min > 1:
                # there are no good matches left
                break
            
            else:
                # find the indices of the match
                i_f, i_e = np.argwhere(score == score_min)[0]
                
                # append new object to the track of the old object
                self.tracks[i_e].append(self.frame_id, objects_found[i_f])
                
                # eliminate both objects from further considerations
                score[i_f, :] = np.inf
                score[:, i_e] = np.inf
                idx_f.remove(i_f)
                idx_e.remove(i_e)
                
        # end tracks that had no match in current frame 
        for i_e in reversed(idx_e): # have to go backwards, since we delete items
            logging.debug('%d: Copy mouse track of length %d to results',
                          self.frame_id, len(self.tracks[i_e]))
            # copy track to result dictionary
            self.result['objects/tracks'].append(self.tracks[i_e])
            del self.tracks[i_e]
        
        # start new tracks for objects that had no previous match
        for i_f in idx_f:
            logging.debug('%d: Start new mouse track at %s', self.frame_id, objects_found[i_f].pos)
            # start new track
            moving_window = self.params['objects/matching_moving_window']
            track = ObjectTrack(self.frame_id, objects_found[i_f], moving_window=moving_window)
            self.tracks.append(track)
        
        assert len(self.tracks) == len(objects_found)
        
    
    def update_explored_area(self, tracks, labels, num_features):
        """ update the explored area using the found objects """
        
        # degrade old information
        self.explored_area -= self.params['explored_area/adaptation_rate']
        self.explored_area[self.explored_area < 0] = 0
        
        # add new information
        for track in self.tracks:
            self.explored_area[labels == track.objects[-1].label] = 1

    
    def find_objects(self, frame):
        """ adapts the current mouse position, if enough information is available """

        # find features that indicate that the mouse moved
        mask_moving = self.find_moving_features(frame)
        
        # find all distinct features and label them
        labels, num_features = ndimage.measurements.label(mask_moving)
        self.debug['object_count'] = num_features
        
        if num_features == 0:
            # end all current tracks if there are any
            if len(self.tracks) > 0:
                logging.debug('%d: Copy %d tracks to results', 
                              self.frame_id, len(self.tracks))
                self.result['objects/tracks'].extend(self.tracks)
                self.tracks = []

        else:
            # some moving features have been found in the video 
            self._handle_object_tracks(frame, labels, num_features)
            
            # check whether objects moved and call them a mouse
            obj_moving = [obj.is_moving() for obj in self.tracks]
            if any(obj_moving):
                self.result['mouse/has_moved'] = True
                # remove the tracks that didn't move
                # this essentially assumes that there is only one mouse
                self.tracks = [obj
                               for k, obj in enumerate(self.tracks)
                               if obj_moving[k]]

            self._mouse_pos_estimate = [obj.last_pos for obj in self.tracks]
        
            # keep track of the regions that the mouse explored
            self.update_explored_area(self.tracks, labels, num_features)
                
                
    #===========================================================================
    # FINDING THE GROUND PROFILE
    #===========================================================================
    
    
    def find_rough_ground(self, image):
        """ determines an estimate of the ground profile from a single image """
        
        # remove 10%/15% of each side of the image
        h = int(0.15*image.shape[0])
        w = int(0.10*image.shape[1])
        image_center = image[h:-h, w:-w]
        
        # binarize image
        cv2.threshold(image_center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=image_center)
        
        # TODO: we might want to replace that with the typical burrow radius
        # do morphological opening and closing to smooth the profile
        s = 4*self.params['ground/point_spacing']
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
        points = simplify_curve(points, epsilon=2)

        return np.array(points, np.int32)
   
        
    def refine_ground(self, image):
        """ adapts a ground profile given as points to a given image.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
                
        points = self.ground
        spacing = self.params['ground/point_spacing']
            
        if not 'ground.model' in self._cache or \
            self._cache['ground.model'].size != spacing:
            
            self._cache['ground.model'] = \
                    RidgeProfile(spacing, self.params['ground/width'])
                
        # make sure the curve has equidistant points
        ground = make_curve_equidistant(points, spacing)
        ground = np.array(np.round(ground),  np.int32)
        
        # calculate the bounds for the points
        p_min = spacing 
        y_max, x_max = image.shape[0] - spacing, image.shape[1] - spacing

        # iterate through all points and correct profile
        ground_model = self._cache['ground.model']
        corrected_points = []
        x_previous = spacing
        deviation = 0
        for k, p in enumerate(ground):
            
            # skip points that are too close to the boundary
            if (p[0] < p_min or p[0] > x_max or 
                p[1] < p_min or p[1] > y_max):
                continue
            
            # determine the local slope of the profile, which fixes the angle 
            if k == 0 or k == len(ground) - 1:
                # we only move these vertically to prevent the ground profile
                # from shortening
                angle = np.pi/2
            else:
                dp = ground[k+1] - ground[k-1]
                angle = np.arctan2(dp[0], dp[1]) # y-coord, x-coord
                
            # extract the region around the point used for fitting
            region = image[p[1]-spacing : p[1]+spacing+1, p[0]-spacing : p[0]+spacing+1].copy()
            ground_model.set_data(region, angle) 

            # move the ground profile model perpendicular until it fits best
            x, _, infodict, _, _ = leastsq(ground_model.get_difference, [0],
                                           xtol=0.5, epsfcn=0.5, full_output=True)
            
            # calculate goodness of fit
            ss_err = (infodict['fvec']**2).sum()
            ss_tot = ((region - region.mean())**2).sum()
            if ss_tot == 0:
                rsquared = 0
            else:
                rsquared = 1 - ss_err/ss_tot

            # Note, that we never remove the first and the last point
            if rsquared > 0 or k == 0 or k == len(ground) - 1:
                # we are rather confident that this point is better than it was
                # before and thus add it to the result list
                p_x = p[0] + x[0]*np.cos(angle)
                p_y = p[1] - x[0]*np.sin(angle)
                # make sure that we have no overhanging ridges
                if p_x >= x_previous:
                    corrected_points.append((int(p_x), int(p_y)))
                    x_previous = p_x
            
            # add up the total deviations from the previous profile
            deviation += abs(x[0])
            
        self.ground = np.array(corrected_points)
        return deviation 
            

    def find_initial_ground(self, image):
        """ finds the ground profile given an image of an antfarm """

        # get an estimate of a profile
        self.ground = self.find_rough_ground(image)
        
        # iterate until the profile does not change significantly anymore
        deviation, iterations = np.inf, 0
        while deviation > len(self.ground) and iterations < 50:
            deviation = self.refine_ground(image)
            iterations += 1
            
        logging.info('We found a ground profile of length %g after %d iterations',
                     curve_length(self.ground), iterations)
        
        
    #===========================================================================
    # FINDING BURROWS
    #===========================================================================
   
   
    def estimate_burrow_from_contour(self, contour):
        """ finds the entry of the burrow, which is defined by the points
        closest to the ground line.
        """
        # simplify the contour
        tolerance = self.params['burrows/outline_simplification_threshold']*curve_length(contour)
        contour = simplify_curve(contour, tolerance)
        
        # calculate the distance of each point to the ground
        ground_line = geometry.LineString(np.array(self.ground, np.double))
        dist = np.array([ground_line.distance(geometry.Point(np.array(p, np.double)))
                         for p in contour])
        
        # move points that are close to the boundary onto the boundary
        dist[dist < 2*self.params['burrows/radius']] = 0
        
        # remove points that are surrounded by points on the ground line
        k = 1
        while k < len(dist):
            dr = dist[k+1] if k < len(dist) - 1 else dist[0]
            if dist[k-1] == dr == 0:
                del dist[k]
                del contour[k]
            k += 1
                
        # find the indices of the two closest points
        i1 = np.argmin(dist)
        dist[i1] = np.inf
        i2 = np.argmin(dist)
        i1, i2 = min(i1, i2), max(i1, i2)
        
        # figure out in what direction we have to go around the polygon
        if i2 - i1 > len(contour)//2:
            # the right outline goes from i1 .. i2
            outline = contour[i1:i2+1]
        else:
            # the right outline goes from i2 .. -1 and start 0 .. i1
            outline = np.vstack((contour[i2:], contour[:i1+1]))
        
#         debug.show_shape(geometry.Polygon(np.array(contour, np.double)),
#                          geometry.LineString(np.array(outline, np.double)))
        
        return Burrow(outline)
    
    
    def refine_burrow_outline(self, burrow, ground_mask):
        """ takes a single burrow and refines its outline """
        outline = burrow.outline.tolist()
        
        ground = np.asarray(self.ground, np.double)
        ground_line = geometry.LineString(ground)
        ground_points = geometry.MultiPoint(ground)
        
        # make sure that the anchor points of the burrow lie on the ground outline
        def move_to_ground_profile(point):
            point = geometry.Point(np.array(point, np.double))
            point = ground_line.interpolate(ground_line.project(point))
            return (point.x, point.y)

        outline[ 0][:] = move_to_ground_profile(outline[ 0])
        outline[-1][:] = move_to_ground_profile(outline[-1])
        
        # simplify the burrow outline
        burrow.simplify_outline(self.params['burrows/outline_simplification_threshold'])
        
        # introduce extra support points into long segments
        threshold = 2*self.params['burrows/radius']
        k = 0
        while k < len(outline) - 1:
            if point_distance(outline[k], outline[k+1]) > threshold:
                p = ((outline[k][0] + outline[k+1][0])/2, (outline[k][1] + outline[k+1][1])/2)
                outline.insert(k + 1, p)
            k += 1
        
        def get_closest_point(point):
            """ determines the point on the ground profile closest to `point` """
            point = geometry.Point(point)
            distances = [point.distance(p) for p in ground_points]
            cp = ground_points[np.argmin(distances)]
            return (cp.x, cp.y)

        # determine the angles that restrict the movement of the points        
        angles = []
        for k, p in enumerate(outline):
            if k == 0:
                p0, p2 = get_closest_point(p), p
                self.debug['points'] = [p0]
            elif k == len(outline) - 1:
                p0, p2 = p, get_closest_point(p)
                self.debug['points'].append(p2)
            else:
                p0, p2 = outline[k-1], outline[k+1]
            angles.append(np.arctan2(p2[1] - p0[1], p2[0] - p0[0])) # y-coord, x-coord
            
        # for k=0,-1, we want to move along the angles instead of perpendicular to them 
        angles[ 0] += np.pi/2
        angles[-1] += np.pi/2
        angles = np.array(angles)
                      
        # extract the region of interest and copy it to he burrow object
        rect = burrow.get_bounding_rect(self.params['burrows/fitting_margin'])
        (_, slices_img), rect = get_overlapping_slices(rect[:2], (rect[3], rect[2]), ground_mask.shape,
                                                       anchor='upper left', ret_rect=True)
        
        burrow_image = self.background[slices_img].astype(np.uint8)
        ground_mask = np.asarray(ground_mask[slices_img], np.bool)
        outline = [(p[0] - rect[0], p[1] - rect[1]) for p in outline]
        outline = np.array(outline, np.double)
                        
        # prepare a mask to measure the colors
        model = np.zeros_like(burrow_image, np.uint8)
        cv2.fillPoly(model, np.array([outline], np.int32), color=1)
        burrow_mask = model.astype(np.bool)
        color_burrow = np.median(burrow_image[ground_mask & burrow_mask])
        color_sand = np.median(burrow_image[ground_mask - burrow_mask])

        def get_residual(displacements, show_image=False):
            """ calculates the difference between the model burrow and the image """
            line = outline.copy()
            line[:, 0] += np.cos(angles)*displacements
            line[:, 1] -= np.sin(angles)*displacements
            
            # use the outline to create the burrow image
            model.fill(color_sand)
            cv2.fillPoly(model, np.array([line], np.int32), color=color_burrow)
    
            cv2.GaussianBlur(model, (0, 0), 3, dst=model)
    
            if show_image:
                debug.show_image(model, burrow_image, equalize_colors=True)
    
            return np.ravel(burrow_image[ground_mask] - model[ground_mask])

        # move the ground profile model perpendicular until it fits best
        displacements, _ = leastsq(get_residual, np.zeros(len(outline)),
                                     xtol=0.5, epsfcn=np.sqrt(2))
        
#         get_residual(np.zeros(len(outline)), True)
#         get_residual(displacements, True)
        
        # limit the maximal displacement
        max_dis = self.params['burrows/radius']/2
        displacements = np.clip(displacements, -max_dis, max_dis)
        
#         print 'Corrected burrow by', sum(abs(p) for p in displacements)
        
        # adjust the burrow outline using the given deviations
        outline[:, 0] += np.cos(angles)*np.array(displacements)
        outline[:, 1] -= np.sin(angles)*np.array(displacements)
        
        burrow.outline = np.array([(p[0] + rect[0], p[1] + rect[1]) for p in outline], np.double)
        

    def find_burrows(self, frame):
        """ locates burrows by combining the information of the ground_mask profile
        and the explored area """
        
        # build a mask with potential burrows
        height, width = frame.shape
        ground_mask = np.zeros_like(frame, np.uint8)
        
        # create a mask for the region below the current ground_mask profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(ground_mask, np.array([ground_points], np.int32), color=128)

        # erode the mask slightly, since the ground_mask profile is not perfect        
        w = 2*self.params['mouse/model_radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        ground_mask_small = cv2.erode(ground_mask, kernel)#, dst=ground_mask_small)
        
        # get potential burrows by looking at explored area
        w = self.params['burrows/radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
        explored_area = 255*(self.explored_area >= self.params['explored_area/adaptation_rate'])
        potential_burrows = cv2.morphologyEx(explored_area.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # remove accidental burrows at borders
        potential_burrows[: 30, :] = 0
        potential_burrows[-30:, :] = 0
        potential_burrows[:, : 30] = 0
        potential_burrows[:, -30:] = 0

        # combine with the information of what areas have been explored
        burrows_mask = cv2.bitwise_and(ground_mask_small, potential_burrows)
        
        # find the contours if there are any
        if burrows_mask.sum() == 0:
            contours = []
        else:
            contours, _ = cv2.findContours(burrows_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            
        for contour in contours:
            #convert contour into polygon
            contour_points = np.squeeze(np.asarray(contour, np.double))
            if contour_points.ndim < 2 or len(contour_points) < 3:
                continue
            contour_poly = geometry.Polygon(contour_points)
            
            # check whether the contour overlaps with any of the found burrows
            for burrow_track in self.result['burrows/data']:
                continue
                if burrow_track.last.intersects(contour_poly):
                    # take the data from the previous burrow
                    burrow = burrow_track.last.copy()
                    # extend the burrow with the found contour
                    burrow.extend_outline(contour_poly)
                     
                    self.refine_burrow_outline(burrow, ground_mask)

                    burrow_track.append(self.frame_id, burrow)
                    break
                
            else:
                # this burrow does not seem to correspond to any of the older burrows
                # we thus create a new burrow track
             
                burrow = self.estimate_burrow_from_contour(np.squeeze(contour))
                
                self.refine_burrow_outline(burrow, ground_mask)
                       
                if len(burrow) > 2:
                    # start a new burrow track
                    burrow_track = BurrowTrack(self.frame_id, burrow)
                    self.result['burrows/data'].append(burrow_track)
                    logging.debug('Found new burrow at %s', burrow.polygon.centroid)
            

    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        self.debug['object_count'] = 0
        self.debug['text1'] = ''
        self.debug['text2'] = ''

        # load parameters for video output        
        video_extension = self.params['video/output/extension']
        video_codec = self.params['video/output/codec']
        video_bitrate = self.params['video/output/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('video' + video_extension, 'debug')
            self.debug['video'] = VideoComposerListener(debug_file, background=self.video,
                                                        is_color=True, codec=video_codec,
                                                        bitrate=video_bitrate)
            if 'video.show' in self.debug_output:
                self.debug['video.show'] = ImageShow(self.debug['video'].shape, 'Debug video')

        # set up additional video writers
        for identifier in ('difference', 'background', 'explored_area'):
            if identifier in self.debug_output:
                # determine the filename to be used
                debug_file = self.get_filename(identifier + video_extension, 'debug')
                # set up the video file writer
                video_writer = VideoComposer(debug_file, self.video.size, self.video.fps,
                                               is_color=False, codec=video_codec,
                                               bitrate=video_bitrate)
                self.debug[identifier + '.video'] = video_writer
        

    def debug_add_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            debug_video.add_polygon(self.ground, is_closed=False, color='y')
            debug_video.add_points(self.ground, radius=2, color='y')
        
            # indicate the currently active burrow shapes
            for burrow_track in self.result['burrows/data']:
                if burrow_track.last_seen > self.frame_id - self.params['burrows/adaptation_interval']:
                    burrow = burrow_track.last
                    debug_video.add_polygon(burrow.outline, 'orange', is_closed=False)
                    for p in burrow.outline:
                        debug_video.add_circle(p, 3, 'orange', thickness=-1)
        
            # indicate the mouse position
            if len(self.tracks) > 0:
                for obj in self.tracks:
                    if not self.result['mouse/has_moved']:
                        color = 'r'
                    elif obj.is_moving():
                        color = 'w'
                    else:
                        color = 'b'
                        
                    debug_video.add_polygon(obj.get_track(), '0.5', is_closed=False)
                    debug_video.add_circle(obj.last_pos, self.params['mouse/model_radius'], color, thickness=1)
                
            else: # there are no current tracks
                for mouse_pos in self._mouse_pos_estimate:
                    debug_video.add_circle(mouse_pos, self.params['mouse/model_radius'], 'k', thickness=1)
            
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            debug_video.add_text('#objects:%d' % self.debug['object_count'], (120, 20), anchor='top')
            debug_video.add_text(self.debug['text1'], (300, 20), anchor='top')
            debug_video.add_text(self.debug['text2'], (300, 50), anchor='top')
            if self.debug.get('rect1'):
                debug_video.add_rectangle(self.debug['rect1'])
            if self.debug.get('points'):
                for p in self.debug['points']:
                    debug_video.add_circle(p, radius=4)
            
            if 'video.show' in self.debug:
                self.debug['video.show'].show(debug_video.frame)
                
        if 'difference.video' in self.debug:
            diff = np.clip(frame.astype(int) - self.background + 128, 0, 255)
            self.debug['difference.video'].write_frame(diff.astype(np.uint8))
                
        if 'background.video' in self.debug:
            self.debug['background.video'].write_frame(self.background)

        if 'explored_area.video' in self.debug:
            debug_video = self.debug['explored_area.video']
             
            # set the background
            debug_video.set_frame(128*self.explored_area)
            
            # plot the ground profile
            debug_video.add_polygon(self.ground, is_closed=False, color='y')
            debug_video.add_points(self.ground, radius=2, color='y')


    def debug_finalize(self):
        """ close the video streams when done iterating """
        # close the window displaying the video
        if 'video.show' in self.debug:
            self.debug['video.show'].close()
        
        # close the open video streams
        for i in ('video', 'difference.video', 'background.video', 'explored_area.video'):
            if i in self.debug:
                self.debug[i].close() 

        # remove all windows that may have been opened
        cv2.destroyAllWindows()
            
    
