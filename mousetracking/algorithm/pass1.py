'''
Created on Aug 5, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the second pass of the algorithm


Note that the OpenCV convention is to store images in [row, column] format
Thus, a point in an image is referred to as image[coord_y, coord_x]
However, a single point is stored as point = (coord_x, coord_y)
Similarly, we store rectangles as (coord_x, coord_y, width, height)

Furthermore, the color space in OpenCV is typically BGR instead of RGB

Generally, x-values increase from left to right, while y-values increase from
top to bottom. The origin is thus in the upper left corner.
'''

from __future__ import division

import math
import os
import time
from collections import defaultdict 

import numpy as np
from scipy import ndimage, optimize, signal, spatial
from shapely import affinity, geometry
import cv2
import yaml

from video.io import ImageWindow
from video.filters import FilterCrop
from video.analysis import regions, curves, image
from video.utils import display_progress
from video.composer import VideoComposer

from .data_handler import DataHandler
from .objects.moving import MovingObject, ObjectTrack, ObjectTrackList
from .objects.ground import GroundProfile, GroundProfileList
from .objects.burrow import Burrow, BurrowTrack, BurrowTrackList

import debug  # @UnusedImport



class FirstPass(DataHandler):
    """
    Analyzes mouse movies. Three objects are traced over time:
        1) the mouse position
        2) the ground line 
        3) the burrows
    """
    logging_mode = 'create'    
    
    def __init__(self, name='', parameters=None, **kwargs):
        """ initializes the whole mouse tracking and prepares the video filters """
        
        # initialize the data handler
        super(FirstPass, self).__init__(name, parameters, **kwargs)
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass1')
        self.result['code_status'] = self.get_code_status()
        
        # setup internal structures that will be filled by analyzing the video
        self._cache = {}               # cache that some functions might want to use
        self.debug = {}                # dictionary holding debug information
        self.background = None         # current background model
        self.ground = None             # current model of the ground profile
        self.tracks = []               # list of plausible mouse models in current frame
        self.explored_area = None      # region the mouse has explored yet
        self.frame_id = 0              # id of the current frame
        self.result['objects/moved_first_in_frame'] = None
        self.result['objects/too_many_objects'] = 0
        if self.params['debug/output'] is None:
            self.params['debug/output'] = []
        self.log_event('Pass 1 - Initialized the first pass analysis.')


    def load_video(self, video=None, crop_video=True):
        """ load and prepare the video.
        crop_video indicates whether the cropping to a quadrant (given in the
        parameters dictionary) should be performed. The cropping to the mouse
        cage is performed no matter what. 
        """
        
        # load the video if it is not already loaded 
        video_info = super(FirstPass, self).load_video(video, crop_video=crop_video)
        self.data.create_child('pass1/video', video_info)
        del self.data['pass1/video/filecount']
        
        self.data['analysis-status'] = 'Loaded video'            

    
    def process(self):
        """ processes the entire video """
        self.log_event('Pass 1 - Started initializing the video analysis.')
        
        # restrict the video to the region of interest (the cage)
        if self.params['cage/determine_boundaries']:
            self.video, cropping_rect = self.crop_video_to_cage(self.video)
        else:
            cropping_rect = None
            
        video_info = self.data['pass1/video']
        video_info['cropping_cage'] = cropping_rect
        video_info['frame_count'] = self.video.frame_count
        video_info['size'] = '%d x %d' % tuple(self.video.size),
            
        self.debug_setup()
        self.setup_processing()

        self.log_event('Pass 1 - Started iterating through the video with %d frames.' %
                       self.video.frame_count)
        self.data['analysis-status'] = 'Initialized video analysis'
        start_time = time.time()            
        
        try:
            # skip the first frame, since it has already been analyzed
            self._iterate_over_video(self.video[1:])
                
        except (KeyboardInterrupt, SystemExit):
            # abort the video analysis
            self.video.abort_iteration()
            self.log_event('Pass 1 - Analysis run has been interrupted.')
            self.data['analysis-status'] = 'Partly finished first pass'
            
        else:
            # finished analysis successfully
            self.log_event('Pass 1 - Finished iterating through the frames.')
            self.data['analysis-status'] = 'Finished first pass'
            
        finally:
            # cleanup in all cases 
            
            self._end_current_tracks()
            self.add_processing_statistics(time.time() - start_time)        
                        
            # cleanup and write out of data
            self.video.close()
            self.debug_finalize()
            self.write_data()
        

    def add_processing_statistics(self, time):
        """ add some extra statistics to the results """
        frames_analyzed = self.frame_id + 1
        self.data['pass1/video/frames_analyzed'] = frames_analyzed
        self.result['statistics/processing_time'] = time
        self.result['statistics/processing_fps'] = frames_analyzed/time


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        
        self.result['objects/tracks'] = ObjectTrackList()
        self.result['ground/profile'] = GroundProfileList()
        if self.params['burrows/enabled_pass1']:
            self.result['burrows/tracks'] = BurrowTrackList()

        # create a simple template of the mouse, which will be used to update
        # the background image only away from the mouse.
        # The template consists of a core region of maximal intensity and a ring
        # region with gradually decreasing intensity.
        
        # determine the sizes of the different regions
        size_core = self.params['mouse/model_radius']
        size_ring = 3*self.params['mouse/model_radius']
        size_total = size_core + size_ring

        # build a filter for finding the mouse position
        x, y = np.ogrid[-size_total:size_total + 1, -size_total:size_total + 1]
        r = np.sqrt(x**2 + y**2)

        # build the mouse template
        mouse_template = (
            # inner circle of ones
            (r <= size_core).astype(float)
            # + outer region that falls off
            + np.exp(-((r - size_core)/size_core)**2)  # smooth function from 1 to 0
              * (size_core < r)          # mask on ring region
        )  
        
        self._cache['mouse.template'] = mouse_template
        
        # prepare kernels for morphological operations
        self._cache['find_moving_features.kernel'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
        w = int(self.params['mouse/model_radius'])
        self._cache['get_potential_burrows_mask.kernel_large'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w+1, 2*w+1))
        self._cache['get_potential_burrows_mask.kernel_small'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        w = int(self.params['burrows/width']/2)
        self._cache['update_burrows_mask.kernel'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w+1, 2*w+1))
        
        # setup more cache variables
        video_shape = (self.video.size[1], self.video.size[0]) 
        self.explored_area = np.zeros(video_shape, np.double)
        self._cache['image_uint8'] = np.empty(video_shape, np.uint8)
        self._cache['image_double'] = np.empty(video_shape, np.double)
  

    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """

        blur_sigma = self.params['video/blur_radius']
        blur_sigma_color = self.params['video/blur_sigma_color']

        # iterate over the video and analyze it
        analyzed_never = True
        for self.frame_id, frame in enumerate(display_progress(video)):
            # remove noise using a bilateral filter
            frame_blurred = cv2.bilateralFilter(frame, d=int(blur_sigma),
                                                sigmaColor=blur_sigma_color,
                                                sigmaSpace=blur_sigma)
            
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
            
            # do the main analysis after an initial wait period
            do_analysis = (self.frame_id >= self.params['video/initial_adaptation_frames'])
            do_colors = (self.frame_id % self.params['colors/adaptation_interval'] == 0)
            do_ground = (self.frame_id % self.params['ground/adaptation_interval'] == 0)
            do_burrows = (self.frame_id % self.params['burrows/adaptation_interval'] == 0)
            
            if do_analysis and (analyzed_never or do_colors):
                self.find_color_estimates(frame_blurred)
                analyzed_never = False

            # update the background model
            self.update_background_model(frame)
                
            if do_analysis:
                # do the main analysis after an initial wait period
                # identify moving objects by comparing current frame to background
                self.find_objects(frame_blurred)
                
                # use the background to find the current ground profile and burrows
                if analyzed_never or do_ground:
                    self.ground = self.get_ground_profile(self.ground)
        
                if self.params['burrows/enabled_pass1'] and (analyzed_never or do_burrows):
                    self.find_burrows()
                    
            # store some information in the debug dictionary
            self.debug_process_frame(frame)
                         
                    
    #===========================================================================
    # FINDING THE CAGE
    #===========================================================================
    
    def find_cage_approximately(self, frame, ret_binarized=False):
        """ analyzes a single frame and locates the mouse cage in it.
        Try to find a bounding box for the cage.
        The rectangle [top, left, height, width] enclosing the cage is returned. """
        # do automatic thresholding to find large, bright areas
        _, binarized_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find the largest bright are, which should contain the cage
        cage_mask = regions.get_largest_region(binarized_frame)
        
        # find an enclosing rectangle, which usually overestimates the cage bounding box
        rect_large = regions.find_bounding_box(cage_mask)
        self.debug['cage']['rect_large'] = rect_large
        self.logger.debug('The cage is estimated to be contained in the '
                          'rectangle %s', rect_large)
         
        # crop frame to this rectangle, which should surely contain the cage
        region_slices = regions.rect_to_slices(rect_large) 
        frame = frame[region_slices]

        # initialize the rect coordinates
        left_est, top_est = 0, 0 # start in top right corner
        height_est, width_est = frame.shape
        bottom_est = int(0.95*height_est - 1)
        right_est = width_est - 1
        # the magic 0.95 factor tries to circumvent some problems with the
        # stands underneath each cage

        # threshold again, because large distractions outside of cages are now
        # definitely removed. Still, bright objects close to the cage, e.g. the
        # stands or some pipes in the background might distract the estimate.
        # We thus adjust the rectangle in the following  
        thresh = int(frame.mean() - self.params['cage/threshold_zscore']*frame.std())
        _, binarized = cv2.threshold(frame, thresh=thresh, maxval=255,
                                     type=cv2.THRESH_BINARY)
        
        # move left line to right until we hit the cage boundary
        threshold = self.params['cage/boundary_detection_thresholds'][0]
        while threshold is not None:
            # move bottom line up with current threshold
            for left in xrange(left_est, width_est//2):
                brightness = binarized[:, left].sum()
                if brightness > threshold*255*height_est:
                    # border has been found
                    threshold = None
                    break
            else:
                # border has not been found and we try a lower threshold
                threshold -= 0.05
                    
        # move top line down until we hit the cage boundary
        threshold = self.params['cage/boundary_detection_thresholds'][1]
        while threshold is not None:
            # move bottom line up with current threshold
            for top in xrange(top_est, height_est//2):
                brightness = binarized[top, :].sum()
                if brightness > threshold*255*width_est:
                    # border has been found
                    threshold = None
                    break
            else:
                # border has not been found and we try a lower threshold
                threshold -= 0.05
        
        # move right line to left until we hit the cage boundary
        threshold = self.params['cage/boundary_detection_thresholds'][2]
        while threshold is not None:
            # move bottom line up with current threshold
            for right in xrange(right_est, width_est//2, -1):
                brightness = binarized[:, right].sum()
                if brightness > threshold*255*height_est:
                    # border has been found
                    threshold = None
                    break
            else:
                # border has not been found and we try a lower threshold
                threshold -= 0.05
        
        # move bottom line up until we hit the cage boundary
        threshold = self.params['cage/boundary_detection_thresholds'][3]
        while threshold is not None:
            # move bottom line up with current threshold
            for bottom in xrange(bottom_est, height_est//2, -1):
                brightness = binarized[bottom, :].sum()
                if brightness > threshold*255*width_est:
                    # border has been found
                    threshold = None
                    break
            else:
                # border has not been found and we try a lower threshold
                threshold -= 0.05

        # return the rectangle defined by two corner points
        p1 = (rect_large[0] + left,  rect_large[1] + top)
        p2 = (rect_large[0] + right, rect_large[1] + bottom)
        cage_rect = regions.corners_to_rect(p1, p2)
        self.debug['cage']['rect_cage'] = cage_rect
        
        if ret_binarized:
            binarized[binarized > 0] = 1
            binarized_frame[binarized_frame > 0] = 1
            binarized_frame[region_slices] += 2*binarized
            return cage_rect, binarized_frame
        else:
            return cage_rect


    def produce_cage_debug_image(self, frame, frame_binarized):
        """ saves an image with information on how the frame of the cage was
        obtained """
        
        # create the image from the binarized masks
        r, g, b = [np.zeros_like(frame) for _ in xrange(3)]
        mask_bin1 = (frame_binarized  % 2 == 0)
        mask_bin2 = (frame_binarized // 2 == 0)
        mask_none = ~(mask_bin1 ^ mask_bin2)
        b[mask_bin1] = frame[mask_bin1]
        r[mask_bin2] = frame[mask_bin2]
        g[mask_none] = frame[mask_none]
        frame = cv2.merge((b, g, r))

        # add the cropping rectangle
        rect_large = self.debug['cage']['rect_large']
        p1, p2 = regions.rect_to_corners(rect_large)
        cv2.rectangle(frame, p1, p2, color=(0, 0, 0), thickness=2)
        
        # add the rectangle on top
        rect_cage = self.debug['cage']['rect_cage']
        p1, p2 = regions.rect_to_corners(rect_cage)
        cv2.rectangle(frame, p1, p2, color=(255, 255, 255), thickness=4)

        # save the image
        filename = self.get_filename('cage_estimate.jpg', 'debug')
        cv2.imwrite(filename, frame)

  
    def crop_video_to_cage(self, video):
        """ crops the video to a suitable cropping rectangle given by the cage """
        self.debug['cage'] = defaultdict(dict)
        
        # find the cage in the blurred image
        frame = video[0]
        blurred_frame = cv2.GaussianBlur(frame, ksize=(0, 0),
                                         sigmaX=self.params['video/blur_radius'])
        
        # find the rectangle describing the cage
        rect_cage, frame_binarized = self.find_cage_approximately(blurred_frame,
                                                                  ret_binarized=True)
        
        # determine the rectangle of the cage in global coordinates
        width = rect_cage[2] - rect_cage[2] % 2   # make sure its divisible by 2
        height = rect_cage[3] - rect_cage[3] % 2  # make sure its divisible by 2
        # Video dimensions should be divisible by two for some codecs
        rect_cage = (rect_cage[0], rect_cage[1], width, height)

        if 'cage_estimate' in self.params['debug/output']:
            self.produce_cage_debug_image(frame, frame_binarized)

        if not (self.params['cage/width_min'] < width < self.params['cage/width_max'] and
                self.params['cage/height_min'] < height < self.params['cage/height_max']):
            raise RuntimeError('The cage bounding box (%dx%d) is out of the '
                               'limits.' % (width, height)) 
        
        self.logger.debug('The cage was determined to lie in the rectangle %s', rect_cage)
        
        # crop the video to the cage region
        return FilterCrop(video, rect_cage), rect_cage
            
            
    #===========================================================================
    # BACKGROUND MODEL AND COLOR ESTIMATES
    #===========================================================================
               
    def estimate_sky_and_sand_regions(self, image):
        """ returns estimates for masks that definitely contain sky and sand
        regions, respectively """
                   
        # add black border around image, which is important for the distance 
        # transform we use later
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        # binarize image
        _, binarized = cv2.threshold(image, 0, 1,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # find sky by locating the largest black areas
        sky_mask = regions.get_largest_region(1 - binarized)
        sky_mask = sky_mask.astype(np.uint8, copy=False)*255
        
        # Finding sure sky region using a distance transform
        dist_transform = cv2.distanceTransform(sky_mask, cv2.cv.CV_DIST_L2, 5)

        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sky_sure = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)

        # find the sand by looking at the largest bright region
        sand_mask = regions.get_largest_region(binarized).astype(np.uint8, copy=False)*255
        
        # Finding sure sand region using a distance transform
        dist_transform = cv2.distanceTransform(sand_mask, cv2.cv.CV_DIST_L2, 5)
        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sand_sure = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

        return sky_sure[1:-1, 1:-1], sand_sure[1:-1, 1:-1]
      
               
    def find_color_estimates(self, image):
        """ estimate the colors in the sky region and the sand region """
        
        # get regions of sky and sand in the image
        sky_sure, sand_sure = self.estimate_sky_and_sand_regions(image)
        
        # determine the sky color
        color_std_min = self.params['colors/std_min'] 
        sky_img = image[sky_sure.astype(np.bool, copy=False)]
        self.result['colors/sky'] = sky_img.mean()
        self.result['colors/sky_std'] = max(sky_img.std(), color_std_min)
        
        # determine the sand color
        sand_img = image[sand_sure.astype(np.bool, copy=False)]
        self.result['colors/sand'] = sand_img.mean()
        self.result['colors/sand_std'] = max(sand_img.std(), color_std_min)
        
        # debug output
        self.logger.debug('%d: Colors: Sand %d+-%d, Sky %d+-%d', self.frame_id,
                          self.result['colors/sand'], self.result['colors/sand_std'],
                          self.result['colors/sky'], self.result['colors/sky_std'])
        
                
    def update_background_model(self, frame):
        """ updates the background model using the current frame """
        
        if self.background is None:
            self.background = frame.astype(np.double, copy=True)
        
        # check whether there are currently objects tracked 
        if self.tracks:
            # load some values from the cache
            adaptation_rate = self._cache['image_double']
            template = self._cache['mouse.template']
            adaptation_rate.fill(self.params['background/adaptation_rate'])
            
            # cut out holes from the adaptation_rate for each object estimate
            for obj in self.tracks:
                # get the slices required for comparing the template to the image
                t_s, i_s = regions.get_overlapping_slices(obj.last.pos, template.shape,
                                                          frame.shape)
                adaptation_rate[i_s[0], i_s[1]] *= 1 - template[t_s[0], t_s[1]]
                
        else:
            # disable the adaptation_rate if no mouse is known
            adaptation_rate = self.params['background/adaptation_rate']

        # adapt the background to current frame, but only inside the adaptation_rate 
        self.background += adaptation_rate*(frame - self.background)

                        
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
        # use internal cache to avoid allocating memory
        mask_moving = self._cache['image_uint8']

        # calculate the difference to the current background model
        cv2.subtract(frame, self.background, dtype=cv2.CV_8U, dst=mask_moving)
        # Note that all points where the difference would be negative are set
        # to zero. However, we only need the positive differences.
        
        # find movement by comparing the difference to a threshold 
        moving_threshold = self.params['mouse/intensity_threshold']*self.result['colors/sky_std']
        cv2.threshold(mask_moving, moving_threshold, 255, cv2.THRESH_BINARY,
                      dst=mask_moving)
        
        kernel = self._cache['find_moving_features.kernel']
        # perform morphological opening to remove noise
        cv2.morphologyEx(mask_moving, cv2.MORPH_OPEN, kernel, dst=mask_moving)    
        # perform morphological closing to join distinct features
        cv2.morphologyEx(mask_moving, cv2.MORPH_CLOSE, kernel, dst=mask_moving)

        # plot the contour of the movement if debug video is enabled
        if 'video' in self.debug:
            self.debug['video'].add_contour(mask_moving, color='g', copy=True)

        return mask_moving


    def _find_objects_in_binary_image(self, labels, num_features):
        """ finds objects in a binary image.
        Returns a list with characteristic properties
        """
        
        # find large objects (which could be the mouse)
        objects = []
        largest_obj = MovingObject((0, 0), 0)
        for label in xrange(1, num_features + 1):
            # calculate the image moments
            moments = cv2.moments((labels == label).astype(np.uint8, copy=False))
            area = moments['m00']

            # check whether this object is too large
            if area > self.params['mouse/area_max']:
                continue

            # get the coordinates of the center of mass
            pos = (moments['m10']/area, moments['m01']/area)
            
            # check whether this object could be large enough to be a mouse
            if area > self.params['mouse/area_min']:
                objects.append(MovingObject(pos, size=area, label=label))
                
            elif area > largest_obj.size:
                # determine the maximal area during the loop
                largest_obj = MovingObject(pos, size=area, label=label)
        
        if len(objects) == 0:
            if largest_obj.size == 0:
                # there is not a single object
                return None
            else:
                # if we have found some small objects and we just take the 
                # best guess, which is the object with the largest area
                return [largest_obj]
        else:
            return objects


    def _end_current_tracks(self):
        """ ends all current tracks and copies them to the results """
        self.result['objects/tracks'].extend(self.tracks)
        self.tracks = []


    def _handle_object_tracks(self, frame, labels, num_features):
        """ analyzes objects in a single frame and tries to add them to
        previous tracks """
        # get potential objects
        objects_found = self._find_objects_in_binary_image(labels, num_features)
        
        if objects_found is None:
            # if there are no useful objects, end all current tracks
            self._end_current_tracks()
            return #< there is nothing to do anymore

        # check if there are previous tracks        
        if len(self.tracks) == 0:
            self.tracks = [ObjectTrack([self.frame_id], [obj])
                           for obj in objects_found]
            return #< there is nothing to do anymore
            
        # calculate the distance between new and old objects
        dist = spatial.distance.cdist([obj.pos for obj in objects_found],
                                      [obj.predict_pos() for obj in self.tracks],
                                      metric='euclidean')
        # normalize distance to the maximum speed
        dist /= self.params['mouse/speed_max']
        
        # calculate the difference of areas between new and old objects
        def area_score(area1, area2):
            """ helper function scoring area differences """
            return abs(area1 - area2)/(area1 + area2)
        
        areas = np.array([[area_score(obj_f.size, obj_e.last.size)
                           for obj_e in self.tracks]
                          for obj_f in objects_found])
        # normalize area change such that 1 corresponds to the maximal allowed one
        areas = areas/self.params['mouse/max_rel_area_change']

        # build a combined score from this
        alpha = self.params['tracking/weight']
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
        for i_e in reversed(idx_e): #< have to go backwards, since we delete items
            # copy track to result dictionary
            self.result['objects/tracks'].append(self.tracks[i_e])
            del self.tracks[i_e]
        
        # start new tracks for objects that had no previous match
        for i_f in idx_f:
            # start new track
            track = ObjectTrack([self.frame_id], [objects_found[i_f]])
            self.tracks.append(track)
        
        assert len(self.tracks) == len(objects_found)
        
    
    def find_objects(self, frame):
        """ adapts the current mouse position, if enough information is available """

        # find a binary image that indicates movement in the frame
        moving_objects = self.find_moving_features(frame)
    
        # find all distinct features and label them
        try:
            num_features = ndimage.measurements.label(moving_objects,
                                                      output=moving_objects)
        except RuntimeError:
            # in some rare cases the function wants to store data in a data
            # format with more bits, where it has to create a new array
            moving_objects, num_features = ndimage.measurements.label(moving_objects)
        self.debug['object_count'] = num_features
        
        if num_features == 0:
            # no features found => end all current tracks
            self._end_current_tracks()
            
        elif num_features > self.params['tracking/object_count_max']:
            # too many features found => end all current tracks
            self._end_current_tracks()
            
            self.result['objects/too_many_objects'] += 1
            self.logger.debug('%d: Discarding object information from this frame, '
                              'too many (%d) features were detected.',
                              self.frame_id, num_features)
            
        else:
            # some moving features have been found => handle these 
            self._handle_object_tracks(frame, moving_objects, num_features)

            # check whether objects moved and call them a mouse
            obj_moving = [obj.is_moving() for obj in self.tracks]
            if any(obj_moving):
                # store when the first object moved
                if self.result['objects/moved_first_in_frame'] is None:
                    self.result['objects/moved_first_in_frame'] = self.frame_id
                    
                # remove the tracks that didn't move
                # this essentially assumes that there is only one mouse
                for k, obj in enumerate(self.tracks):
                    if obj_moving[k]:
                        # keep only the moving object in the current list
                        self.tracks = [obj]
                    else:
                        self.result['objects/tracks'].append(obj)

            # add new information to explored area
            for track in self.tracks:
                self.explored_area[moving_objects == track.last.label] = 1

        
    #===========================================================================
    # FINDING THE GROUND PROFILE
    #===========================================================================
        
        
    def _get_ground_template(self, width_estimate, stretch_height=1):
        """ builds the ground template from a stored template.
        width_estimate is the estimated full width of the ground """
        # load the template file once
        if 'ground_template' not in self._cache:
            filename = 'ground_%s.yaml' % self.params['ground/template']
            path = os.path.join(os.path.dirname(__file__), 'assets', filename)
            if not os.path.isfile(path):
                return None
            self._cache['ground_template'] = np.array(yaml.load(open(path)))

        # load the points and scale them to the given width        
        t_points = self._cache['ground_template'].copy()
        t_points *= width_estimate
        t_points[:, 1] *= stretch_height

        # determine template width by subtracting margin
        width_fraction = self.params['ground/template_width_fraction']
        margin_vert = int(self.params['ground/template_margin'])
        t_width = int(np.ceil(width_fraction*width_estimate))
        
        # filter points that are too close to the edge
        width_crop = (width_estimate - t_width)/2
        t_points[:, 0] -= width_crop
        t_points = np.array([p for p in t_points if 0 < p[0] < t_width])
        
        # shift points vertically and determine template height
        t_points[:, 1] += margin_vert - t_points[:, 1].min()
        t_height = int(np.ceil(t_points[:, 1].max())) + margin_vert
        
        # add corner points to point list
        t_points = t_points.tolist()
        t_points.append((t_width, margin_vert))
        t_points.append((t_width, t_height))
        t_points.append((0, t_height))
        t_points.append((0, margin_vert))
        
        # create the mask based on the points 
        template = np.zeros((t_height, t_width), np.uint8)
        cv2.fillPoly(template, [np.array(t_points, np.int32)], 255)
        return template, t_points[:-4]
        
        
    def _get_ground_from_template(self, frame):
        """ determine the ground profile using a template.
        Returns None if ground profile could not be determined. """
        if not self.params['ground/template']:
            return None
        
        correlation_max, points = 0, None
        parameters_max, rect_max = None, None
        # try different aspect ratios
        for aspect_factor in self.params['ground/template_aspect_factors']:
            # try different fractions of the total width                  
            for width_factor in self.params['ground/template_width_factors']:
                width_estimate = width_factor * frame.shape[1] #< width
                template, t_points = self._get_ground_template(width_estimate, aspect_factor)
            
                # convolute template with frame
                conv = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    
                # determine maximum
                _, max_val, _, max_loc = cv2.minMaxLoc(conv)
    
                if max_val > correlation_max:
                    # better match than the previous one
                    correlation_max = max_val
                    parameters_max = (width_factor, aspect_factor)
                    rect_max = max_loc + template.shape[::-1]
                    # shift the points of the template
                    points = curves.translate_points(t_points, max_loc[0], max_loc[1])
        
        self.debug['ground']['template_rect_max'] = rect_max
        self.logger.info('Best ground match for template width %d%% '
                         'and a height factor of %.2g.',
                         parameters_max[0]*100, parameters_max[1])
        
        return points
    
        
    def _get_ground_from_linescans(self, frame):
        """ get a rough series of points on the ground line from vertical
        line scans """
        
        # get region of interest
        frame_margin = int(self.params['ground/frame_margin'])
        rect_frame = [0, 0, frame.shape[1], frame.shape[0]]
        rect_roi = regions.expand_rectangle(rect_frame, amount=-frame_margin)
        slices = regions.rect_to_slices(rect_roi)
        frame = frame[slices]
        
        # build the ground ridge template for matching 
        ridge_width = self.params['ground/ridge_width']
        
        # do vertical line scans and determine ground position
        spacing = int(self.params['ground/point_spacing'])
        points = []
        for k in xrange(frame.shape[1]//spacing):
            pos_x = (k + 0.5)*spacing + frame_margin
            line_scan = frame[:, spacing*k:spacing*(k + 1)].mean(axis=1)

            profile = ndimage.filters.gaussian_filter1d(line_scan, ridge_width)
    
            slopes = np.diff(profile)
            slope_threshold = self.params['ground/slope_detector_max_factor']*slopes.max()
            pos_ys = np.nonzero(slopes > slope_threshold)[0] + frame_margin
    
            if 'video' in self.debug:
                for pos_y in pos_ys:
                    self.debug['video'].add_points([(pos_x, pos_y)], radius=2)

            try:
                pos_y = pos_ys[0]
            except IndexError:
                raise RuntimeError("Ground profile could not be determined "
                                   "since no ridge points could be detected. "
                                   "Try lowering the parameter "
                                   "'ground/slope_detector_max_factor' to "
                                   "increase the number of detected points.")
            
            # add point
            points.append((pos_x, pos_y))

        return points
    
    
    def _refine_ground_points_grabcut(self, frame, points):
        """ refine the ground based on a previous estimate and the current
        image using the GrabCut algorithm """
        # prepare the masks
        mask_ground = self.get_ground_mask(255, GroundProfile(points))

        # restrict to region of interest        
        frame_margin = int(self.params['ground/frame_margin'])
        rect_frame = [0, 0, frame.shape[1], frame.shape[0]]
        rect_roi = regions.expand_rectangle(rect_frame, amount=-frame_margin)
        slices = regions.rect_to_slices(rect_roi)
        frame = frame[slices]
        mask_ground = mask_ground[slices]
        mask = np.zeros_like(frame)
        
        # indicate the estimated border between sand and sky
        color_border = (self.result['colors/sand'] + self.result['colors/sky'])/2
        mask.fill(cv2.GC_PR_BGD) #< probable background
        mask[frame > color_border] = cv2.GC_PR_FGD #< probable foreground

        # set the regions with certain sand and sky
        uncertainty_margin = int(self.params['ground/grabcut_uncertainty_margin'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (uncertainty_margin, uncertainty_margin))
        sure_ground = (cv2.erode(mask_ground, kernel) == 255)
        mask[sure_ground] = cv2.GC_FGD
        sure_sky = (cv2.dilate(mask_ground, kernel) == 0)
        mask[sure_sky] = cv2.GC_BGD
        
#         debug.show_image(debug.get_grabcut_image(mask), frame)

        # run grabCut algorithm
        # have to convert to color image, since cv2.grabCut only supports color, yet
        frame_clr = cv2.cvtColor(frame, cv2.cv.CV_GRAY2RGB)
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv2.grabCut(frame_clr, mask, (0, 0, 1, 1),
                    bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        
        # calculate the mask of the foreground
        mask = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0)
        mask = regions.get_largest_region(mask)

        # iterate through the mask and extract the ground profile
        points = []
        for col_id, column in enumerate(mask.T):
            x = col_id + frame_margin
            try:
                y = np.nonzero(column)[0][0] + frame_margin
            except IndexError:
                pass
            else:
                points.append((x, y))
    
        return points
    
    
    def _revise_ground_points(self, frame, points):
        """ modify ground points to extend to the edge of the cage """
        
        # iterate through points and check slopes
        slope_max = self.params['ground/slope_max']
        k = 1
        while k < len(points):
            p1, p2 = points[k-1], points[k]
            slope = (p2[1] - p1[1])/(p2[0] - p1[0]) # dy/dx
            if slope < -slope_max:
                del points[k-1]
            elif slope > slope_max:
                del points[k]
            else:
                k += 1

        # extend the ground line toward the left edge of the cage
        p_x, p_y = points[0]
        profile = image.line_scan(frame, (0, p_y), (p_x, p_y), 30)
        color_threshold = (self.result['colors/sand'] + frame.max())/2
        try:
            p_x = np.nonzero(profile > color_threshold)[0][-1]
            points.insert(0, (p_x, p_y))
        except IndexError:
            pass
        
        # extend the ground line toward the right edge of the cage
        p_x, p_y = points[-1]
        profile = image.line_scan(frame, (p_x, p_y), (frame.shape[1] - 1, p_y), 30)
        try:
            p_x += np.nonzero(profile > color_threshold)[0][0]
            points.append((p_x, p_y))
        except IndexError:
            pass
        
        # simplify the curve        
        points = curves.simplify_curve(points, epsilon=2)
        # make the curve equidistant
        points = curves.make_curve_equidistant(points, self.params['ground/point_spacing'])
        
        return points
        
                
    def estimate_ground_profile(self):
        """ estimates the ground profile from the current background image """ 
        
        # get the background image from which we extract the ground profile
        frame = self.background.astype(np.uint8)

        # get the ground profile by using a template
        points_est1 = self._get_ground_from_template(frame)
        
        # if this didn't work, fall back to another method
        if points_est1 is None:
            points_est1 = self._get_ground_from_linescans(frame)
            points_est2 = self._refine_ground_points_grabcut(frame, points_est1)
        else:
            points_est2 = points_est1
        
        # refine the ground profile that we found
        points_final = self._revise_ground_points(frame, points_est2[:])

        # plot the ground profiles to the debug video
        if 'video' in self.debug:
            debug_video = self.debug['video']
            debug_video.add_line(points_est1, is_closed=False,
                                 mark_points=True, color='r')
            if points_est1 is not points_est2:
                debug_video.add_line(points_est2, is_closed=False, color='b')
                
        if 'ground_estimate' in self.params['debug/output']:
            self.debug['ground']['estimate1'] = points_est1
            self.debug['ground']['estimate2'] = points_est2
            frame = cv2.cvtColor(frame, cv2.cv.CV_GRAY2BGR)
            cv2.polylines(frame, [np.array(points_est1, np.int32)],
                          isClosed=False, color=(0, 0, 255), thickness=2)
            if points_est1 is not points_est2:
                cv2.polylines(frame, [np.array(points_est2, np.int32)],
                              isClosed=False, color=(255, 0, 0), thickness=2)
            filename = self.get_filename('ground_estimate.jpg', 'debug')
            cv2.imwrite(filename, frame)
        
        return GroundProfile(points_final)

    
    def _get_cage_boundary(self, ground_point, direction='left'):
        """ determines the cage boundary starting from a ground_point
        going in the given direction """

        # check whether we have to calculate anything
        if not self.params['cage/determine_boundaries']:
            if direction == 'left':
                return (0, ground_point[1])
            elif direction == 'right':
                return (self.background.shape[1] - 1, ground_point[1])
            else:
                raise ValueError('Unknown direction `%s`' % direction)
            
        # extend the ground line toward the left edge of the cage
        if direction == 'left':
            border_point = (0, ground_point[1])
        elif direction == 'right':
            image_width = self.background.shape[1] - 1
            border_point = (image_width, ground_point[1])
        else:
            raise ValueError('Unknown direction `%s`' % direction)
        
        # do the line scan
        profile = image.line_scan(self.background, border_point, ground_point,
                                  self.params['cage/linescan_width'])
        
        # smooth the profile slightly
        profile = ndimage.filters.gaussian_filter1d(profile,
                                                    self.params['cage/linescan_smooth'])
        
        # add extra points to make determining the extrema reliable
        profile = np.r_[0, profile, 255]

        # determine first maximum and first minimum after that
        maxima = signal.argrelextrema(profile, comparator=np.greater_equal)
        pos_max = maxima[0][0]
        minima = signal.argrelextrema(profile[pos_max:],
                                      comparator=np.less_equal)
        pos_min = minima[0][0] + pos_max
        # we have to use argrelextrema instead of argrelmax and argrelmin,
        # because the latter don't capture wide maxima like [0, 1, 1, 0]
        
        if pos_min - pos_max >= 2:
            # get steepest point in this interval
            pos_edge = np.argmin(np.diff(profile[pos_max:pos_min + 1])) + pos_max
        else:
            # get steepest point in complete profile
            pos_edge = np.argmin(np.diff(profile))

        if direction == 'right':
            pos_edge = image_width - pos_edge
        
        return (pos_edge, ground_point[1])
    
        
    def refine_ground(self, ground):
        """ adapts a points profile given as points to a given frame.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
        frame = self.background
        spacing = int(self.params['ground/point_spacing'])

        # make sure the curve has equidistant points
        ground.make_equidistant(spacing=spacing)

        # consider all points that are far away from the borders        
        frame_margin = int(self.params['ground/frame_margin'])
        x_max = frame.shape[1] - frame_margin
        points = [point for point in ground.points
                  if frame_margin < point[0] < x_max]
        points = np.array(np.round(points),  np.int32)

        # iterate over all but the boundary points
        curvature_energy_factor = self.params['ground/curvature_energy_factor']
        if 'ground/energy_factor_last' in self.result:
            # load the energy factor for the next iteration
            snake_energy_max = self.params['ground/snake_energy_max']
            energy_factor_last = self.result['ground/energy_factor_last']
            
        else:
            # initialize values such that the energy factor is calculated
            # for the next iteration
            snake_energy_max = np.inf
            energy_factor_last = 1

        # parameters of the line scan            
        ray_len = int(self.params['ground/linescan_length']/2)
        profile_width = self.params['ground/point_spacing']/2
        ridge_width = self.params['ground/ridge_width']
        len_ratio = ray_len/ridge_width
        model_std = math.sqrt(1 - math.tanh(len_ratio)/len_ratio)
        assert 0.5 < model_std < 1
            
        # iterate through all points in random order and correct profile
        energies_image = []
        num_points = len(points) - 2 #< disregard the boundary points
        candidate_points = [None]*num_points
        for k in np.random.permutation(num_points):
            # get local points and slopes
            p_p, p, p_n =  points[k], points[k+1], points[k+2]
            dx, dy = p_n - p_p
            dist = math.hypot(dx, dy)
            if dist == 0: #< something went wrong 
                continue #< skip this point
            dx /= dist; dy /= dist

            # do the line scan perpendicular to the ground line         
            p_a = (p[0] - ray_len*dy, p[1] + ray_len*dx)
            p_b = (p[0] + ray_len*dy, p[1] - ray_len*dx)
            profile = image.line_scan(frame, p_a, p_b, width=profile_width)
            # scale profile to -1, 1
            profile -= profile.mean()
            try:
                profile /= profile.std()
            except RuntimeWarning:
                # this can happen in strange cases where the profile is flat
                continue

            plen = len(profile)
            xs = np.linspace(-plen/2 + 0.5,
                              plen/2 - 0.5,
                              plen)
        
            def energy_image((pos, model_mean, model_std)):
                """ part of the energy related to the line scan """
                # get image part
                model = np.tanh(-(xs - pos)/ridge_width)
                img_diff = profile - model_std*model - model_mean
                squared_diff = np.sum(img_diff**2)
                return energy_factor_last*squared_diff
                # energy_img has units of color^2
                
            def energy_curvature(pos):
                """ part of the energy related to the curvature of the ground line """
                # get curvature part: http://en.wikipedia.org/wiki/Menger_curvature
                p_c = (p[0] + pos*dy, p[1] - pos*dx)
                a = curves.point_distance(p_p, p_c)
                b = curves.point_distance(p_c, p_n)
                c = curves.point_distance(p_n, p_p)
                
                # determine curvature of circle through the three points
                A = regions.triangle_area(a, b, c)
                curvature = 4*A/(a*b*c)*spacing
                # We don't scale by with the arc length a + b, because this 
                # would increase the curvature weight in situations where
                # fitting is complicated (close to burrow entries)
                return curvature_energy_factor*curvature

            def energy_snake(data):
                """ energy function of this part of the ground line """
                return energy_image(data) + energy_curvature(data[0])
            
            # fit the simple model to the line scan profile
            fit_bounds = ((-spacing, spacing), (None, None), (None, None))
            try:      
                #res = optimize.fmin_powell(energy_snake, [0, 0, model_std], disp=False)
                res, snake_energy, _ = \
                    optimize.fmin_l_bfgs_b(energy_snake, approx_grad=True,
                                           x0=np.array([0, 0, model_std]),
                                           bounds=fit_bounds)
            except RuntimeError:
                continue #< skip this point

            # use this point, if it is good enough            
            if snake_energy < snake_energy_max:
                # save final energy for determining the energy scale later
                energies_image.append(energy_image(res))

                pos, _, model_std = res 
                p_x, p_y = p[0] + pos*dy, p[1] - pos*dx
                candidate_points[k] = (int(p_x), int(p_y))

        # filter points, where the fit did not work
        # FIXME: fix overhanging ridge detection (changed sign of comparison recently)
        points = []
        for candidate in candidate_points:
            if candidate is None:
                continue
            
            # check for overhanging ridge
            if len(points) == 0 or candidate[0] > points[-1][0]:
                points.append(candidate)
                
            # there is an overhanging part
            elif candidate[1] < points[-1][0]:
                # current point is above previous point
                # => replace the previous candidate point
                points[-1] = candidate 
                
            else:
                # current point is below previous point
                # => do not add the current candidate point
                pass
                
        # save the energy factor for the next iteration
        energy_factor_last /= np.mean(energies_image)
        self.result['ground/energy_factor_last'] = energy_factor_last

        if len(points) < 2:
            # refinement failed => return original ground
            self.logger.debug('%d: Ground profile shortened too much.',
                              self.frame_id)
            return ground

        # extend the ground line toward the left edge of the cage
        edge_point = self._get_cage_boundary(points[0], 'left')
        if edge_point is not None:
            points.insert(0, edge_point)
            
        # extend the ground line toward the right edge of the cage
        edge_point = self._get_cage_boundary(points[-1], 'right')
        if edge_point is not None:
            points.append(edge_point)
            
        return GroundProfile(points)
            

    def produce_ground_debug_image(self, ground):
        """ saves an image with information on how the ground profile was
        obtained """
        points_est1 = self.debug['ground']['estimate1']
        points_est2 = self.debug['ground']['estimate2']
        frame = cv2.cvtColor(self.background.astype(np.uint8), cv2.cv.CV_GRAY2BGR)
        
        # plot rectangle where the template matched
        if 'template_rect_max' in self.debug['ground']:
            template_rect = self.debug['ground']['template_rect_max']
            p1, p2 = regions.rect_to_corners(template_rect)
            cv2.rectangle(frame, p1, p2, color=(0, 0, 255))
            
        # plot ground lines from several stages
        cv2.polylines(frame, [np.array(points_est1, np.int32)],
                      isClosed=False, color=(0, 0, 255), thickness=2)
        if points_est1 is not points_est2:
            cv2.polylines(frame, [np.array(points_est2, np.int32)],
                          isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.polylines(frame, [np.array(ground.points, np.int32)],
                      isClosed=False, color=(0, 255, 255), thickness=1)
        
        # save the resulting image to a file
        filename = self.get_filename('ground_estimate.jpg', 'debug')
        cv2.imwrite(filename, frame)


    def get_ground_profile(self, ground_estimate=None):
        """ finds the ground profile given an image of an antfarm. """
        
        if ground_estimate is None:
            self.debug['ground'] = defaultdict(dict)
            ground_estimate = self.estimate_ground_profile()
            estimated_ground = True
            
            self.logger.debug('%d: Estimated ground profile of length %g',
                              self.frame_id, ground_estimate.length)
        else:
            estimated_ground = False            
            
        if ground_estimate.length > self.params['ground/length_max']:
            # reject excessively long ground profiles
            self.logger.warn('%d: Ground profile was too long (%g > %g)',
                             self.frame_id, ground_estimate.length,
                             self.params['ground/length_max'])
            ground = None
        
        else:
            # refine estimated ground
            ground = self.refine_ground(ground_estimate)
        
            # add the ground to the result list
            self.result['ground/profile'].append(self.frame_id, ground)
            
        if estimated_ground and 'ground_estimate' in self.params['debug/output']:
            self.produce_ground_debug_image(ground)
        
        return ground
        
    
    def get_ground_mask(self, color=255, ground=None):
        """ returns a binary mask distinguishing the ground from the sky """
        if ground is None:
            ground = self.ground
            if ground is None:
                return None
        
        # build a mask with for the ground
        width, height = self.video.size
        mask_ground = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current mask_ground profile
        ground_points = np.empty((len(ground) + 4, 2), np.int32)
        ground_points[:-4, :] = ground.points
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(mask_ground, np.array([ground_points], np.int32), color=color)

        return mask_ground
    
            
    #===========================================================================
    # FINDING BURROWS
    #===========================================================================
   
        
    def get_potential_burrows_mask(self):
        """ locates potential burrows by searching for underground regions that
        the mouse explored """

        mask_ground = self.get_ground_mask()
        if mask_ground is None:
            return None

        # get potential burrows by looking at explored area
        explored_area = 255*(self.explored_area > 0).astype(np.uint8)
        
        # remove accidental burrows at borders
        margin = int(self.params['burrows/cage_margin'])
        explored_area[: margin, :] = 0
        explored_area[-margin:, :] = 0
        explored_area[:, : margin] = 0
        explored_area[:, -margin:] = 0
        
        # remove all regions that are less than a threshold distance away from
        # the ground line and which are not connected to any other region
        kernel_large = self._cache['get_potential_burrows_mask.kernel_large']
        kernel_small = self._cache['get_potential_burrows_mask.kernel_small']

        # lower the ground to remove artifacts close to the ground line
        mask_ground_low = cv2.erode(mask_ground, kernel_large)
        # find areas which are very likely burrows
        mask_burrows = cv2.bitwise_and(mask_ground_low, explored_area)
        # combine the mask with the sky mask (= ~mask_ground)
        cv2.bitwise_or(255 - mask_ground, mask_burrows, dst=mask_burrows)
        # close this combined mask, which should reconnect the burrows to the ground
        cv2.morphologyEx(mask_burrows, cv2.MORPH_CLOSE,
                         kernel_large, dst=mask_burrows)
        # subtract the ground mask to be left with burrows
        cv2.bitwise_and(mask_burrows, mask_ground, dst=mask_burrows)
        # open the mask slightly to remove sharp edges
        cv2.morphologyEx(mask_burrows, cv2.MORPH_OPEN,
                         kernel_small, dst=mask_burrows)
        
        return mask_burrows
    
        
    def get_burrow_from_mask(self, mask, offset=None):
        """ creates a burrow object given a contour outline.
        If offset=(xoffs, yoffs) is given, all the points are translate.
        May return None if no burrow was found 
        """
        if offset is None:
            offset = (0, 0)

        # find the contour of the mask    
        contours, _ = cv2.findContours(mask.astype(np.uint8, copy=False),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise RuntimeError('Could not find any contour')
        
        # find the contour with the largest area, in case there are multiple
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        contour_id = np.argmax(contour_areas)
        
        if contour_areas[contour_id] < self.params['burrows/area_min']:
            # disregard small burrows
            raise RuntimeError('Burrow is too small')
            
        # simplify the contour
        contour = np.squeeze(np.asarray(contours[contour_id], np.double))
        tolerance = self.params['burrows/outline_simplification_threshold'] \
                        *curves.curve_length(contour)
        contour = curves.simplify_curve(contour, tolerance).tolist()

        # move points close to the ground line onto the ground line
        ground_point_dist = self.params['burrows/ground_point_distance']
        ground_line = affinity.translate(self.ground.linestring,
                                         xoff=-offset[0],
                                         yoff=-offset[1]) 
        for k, p in enumerate(contour):
            point = geometry.Point(p)
            if ground_line.distance(point) < ground_point_dist:
                contour[k] = curves.get_projection_point(ground_line, point)
        
        # simplify contour while keeping the area roughly constant
        threshold = self.params['burrows/simplification_threshold_area']
        contour = regions.simplify_contour(contour, threshold)
        
        # remove potential invalid structures from contour
        if contour:
            contour = regions.regularize_contour_points(contour)
        
#         if offset[0]:
#             debug.show_shape(geometry.LinearRing(contour),
#                              background=mask, wait_for_key=False)
        
        # create the burrow based on the contour
        if contour:
            contour = curves.translate_points(contour,
                                              xoff=offset[0],
                                              yoff=offset[1])
            try:
                return Burrow(contour)
            except ValueError as err:
                raise RuntimeError(err.message)
            
        else:
            raise RuntimeError('Contour is not a simple polygon')
    
    
    def find_burrow_edge(self, profile, direction='up'):
        """ finds a burrow edge in a given profile
        direction denotes whether we are looking for rising or falling edges
        
        returns the position of the edge or None, if the edge was not found
        """
        # load parameters
        edge_width = self.params['burrows/fitting_edge_width']
        template_width = int(2*edge_width)
        
        # create the templates if they are not in the cache
        if 'burrows.template_edge_up' not in self._cache:
            color_sand = self.result['colors/sand']
            color_burrow = self.result['colors/sky']
            
            x = np.linspace(-template_width, template_width, 2*template_width + 1)
            y = (1 + np.tanh(x/edge_width))/2 #< sigmoidal profile
            y = color_burrow + (color_sand - color_burrow)*y
            
            y = np.uint8(y)
            self._cache['burrows.template_edge_up'] = y 
            self._cache['burrows.template_edge_down'] = y[::-1]
            
        # load the template
        if direction == 'up':
            template = self._cache['burrows.template_edge_up']
        elif direction == 'down':
            template = self._cache['burrows.template_edge_down']
        else:
            raise ValueError('Unknown direction `%s`' % direction)
        
        # get the cross-correlation between the profile and the template
        conv = cv2.matchTemplate(profile.astype(np.uint8),
                                 template, cv2.cv.CV_TM_SQDIFF)
        
#         import matplotlib.pyplot as plt
#         plt.plot(profile/profile.max(), 'b', label='profile')
#         plt.plot(conv/conv.max(), 'r', label='conv')
#         plt.axvline(np.argmin(conv), color='r')
#         plt.axvline(np.argmin(conv) + template_width, color='b')
#         plt.legend(loc='best')
#         plt.show()
        
        # find the best match
        pos = np.argmin(conv) + template_width
        
        # calculate goodness of fit
        profile_roi = profile[pos - template_width : pos + template_width + 1]
        ss_tot = ((profile_roi - profile_roi.mean())**2).sum()
        if ss_tot == 0:
            rsquared = 0
        else:
            rsquared = 1 - conv.min()/ss_tot
        
        if rsquared > self.params['burrows/fitting_edge_R2min']:
            return pos
        else:
            return None
    

    def refine_long_burrow(self, burrow):
        """ refines an elongated burrow by doing line scans perpendicular to
        its centerline.
        """
        # keep the points close to the ground line
        ground_line = self.ground.linestring
        distance_threshold = self.params['burrows/ground_point_distance']
        outline_new = sorted([p.coords[0]
                              for p in geometry.asMultiPoint(burrow.outline)
                              if p.distance(ground_line) < distance_threshold])

        # replace the remaining points by fitting perpendicular to the center line
        outline = burrow.outline_ring
        centerline = burrow.get_centerline(self.ground)
        if len(centerline) < 4:
            self.logger.warn('%d: Refining the very short burrows at %s is not '
                             'supported.', self.frame_id, burrow.position)
            return burrow
        
        segment_length = self.params['burrows/centerline_segment_length']
        centerline = curves.make_curve_equidistant(centerline, segment_length)
        
        # HANDLE INNER POINTS OF BURROW
        width_min = self.params['burrows/width_min']
        scan_length = int(2*self.params['burrows/width'])
        centerline_new = [centerline[0]]
        for k, p in enumerate(centerline[1:-1]):
            # get points adjacent to p
            p1, p2 = centerline[k], centerline[k+2]
            
            # find local slope of the centerline
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = np.hypot(dx, dy)
            dx /= dist; dy /= dist

            # find the intersection points with the burrow outline
            d = 1000 #< make sure the ray is outside the polygon
            pa = regions.get_ray_hitpoint(p, (p[0] + d*dy, p[1] - d*dx), outline)
            pb = regions.get_ray_hitpoint(p, (p[0] - d*dy, p[1] + d*dx), outline)
            if pa is not None and pb is not None:
                # put the centerline point into the middle 
                p = ((pa[0] + pb[0])/2, (pa[1] + pb[1])/2)
            
            # do a line scan perpendicular
            p_a = (p[0] + scan_length*dy, p[1] - scan_length*dx)
            p_b = (p[0] - scan_length*dy, p[1] + scan_length*dx)
            background = self.background.astype(np.uint8, copy=False)
            profile = image.line_scan(background, p_a, p_b, 3)
            
            # find the transition points by considering slopes
            k_l = self.find_burrow_edge(profile, direction='down')
            k_r = self.find_burrow_edge(profile, direction='up')

            if k_l is not None and k_r is not None:
                d_l, d_r = scan_length - k_l, scan_length - k_r
                # d_l and d_r are the distance from p
                # Here, d_l > 0 and d_r < 0 accounting for direction

                # ensure a minimal burrow width
                width = d_l - d_r
                if width < width_min:
                    d_r -= (width_min - width)/2
                    d_l += (width_min - width)/2
                
                # save the points
                outline_new.append((p[0] + d_l*dy, p[1] - d_l*dx))
                outline_new.insert(0, (p[0] + d_r*dy, p[1] - d_r*dx))
                
                d_c = (d_l + d_r)/2
                centerline_new.append((p[0] + d_c*dy, p[1] - d_c*dx))
            
            elif pa is not None and pb is not None:
                # add the earlier estimates obtained without fitting 
                outline_new.append(pa)
                outline_new.insert(0, pb)
                centerline_new.append(p)

        # HANDLE BURROW END POINT
        # points at the burrow end
        if len(centerline_new) < 2:
            self.logger.warn('%d: Refining shortened burrow at %s too much.',
                             self.frame_id, burrow.position)
            return None
        
        p1, p2 = centerline_new[-1], centerline_new[-2]
        angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

        # shoot out rays in several angles        
        angles = angle + np.pi/8*np.array((-2, -1, 0, 1, 2))
        points = regions.get_ray_intersections(centerline_new[-1], angles, outline)
        # filter unsuccessful points
        points = (p for p in points if p is not None)
        
        # determine whether the mouse is at the front of the burrow or not
        adaptation_rate = self.params['explored_area/adaptation_rate_burrows']
        if adaptation_rate != 0:
            # check whether the mouse has been away from the burrow long enough
            frames_absent = (
                (1 - self.explored_area[int(p2[1]), int(p2[0])])
                /self.params['explored_area/adaptation_rate_burrows']
            )
            frames_threshold = 3/self.params['background/adaptation_rate']
            mouse_absent = (frames_absent > frames_threshold)
            
        else:
            mouse_absent = True
            distance_threshold = 3*self.params['burrows/width']
            for track in self.tracks:
                pos = geometry.Point(track.last.pos)
                if outline.distance(pos) <  distance_threshold:
                    mouse_absent = False
                    break
        
        point_max, dist_max = None, 0
        point_anchor = centerline_new[-1]
        for point in points:
            if mouse_absent:
                # mouse has been away for a long time
                # => refine point using a line scan along the centerline

                # find local slope of the centerline
                dx, dy = point[0] - point_anchor[0], point[1] - point_anchor[1]
                dist = np.hypot(dx, dy)
                dx /= dist; dy /= dist
                
                # get profile along the centerline
                p1e = (point_anchor[0] + scan_length*dx, point_anchor[1] + scan_length*dy)
                background = self.background.astype(np.uint8, copy=False)
                profile = image.line_scan(background, point_anchor, p1e, 3)

                # determine position of burrow edge
                l = self.find_burrow_edge(profile, direction='up')
                if l is not None:
                    point = (point_anchor[0] + l*dx, point_anchor[1] + l*dy)

            # add the point to the outline                
            outline_new.append(point)
            
            # find the point with a maximal distance from the anchor point
            dist = curves.point_distance(point, point_anchor)
            if dist > dist_max:
                point_max, dist_max = point, dist 
                
        # set the point with a maximal distance as the new centerline end
        if point_max is not None:
            centerline_new.append(point_max)
    
        # HANDLE BURROW EXIT POINT
        # determine the ground exit point by extrapolating from first
        # point until we hit the ground profile
        p1, p2 = centerline[1], centerline[2]
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        point_max, _, _ = regions.get_farthest_ray_intersection(
            centerline_new[1], [angle], ground_line)
        if point_max is not None: 
            centerline_new[0] = point_max

        # make sure that shape is a valid polygon
        outline_new = regions.regularize_contour_points(outline_new)
        if not outline_new:
            self.logger.warn('%d: Refined, long burrow at %s is not a valid '
                             'polygon anymore.', self.frame_id, burrow.position)
            return None

        try:
            return Burrow(outline_new, centerline=centerline_new, refined=True)
        except ValueError:
            # the burrow is not a valid burrow
            return None
    
    
    def refine_bulky_burrow(self, burrow, burrow_prev=None):
        """ refine burrow by thresholding background image using the GrabCut
        algorithm """
        mask_ground = self.get_ground_mask(255)
        frame = self.background
        width_min = self.params['burrows/width_min']
        
        # get region of interest from expanded bounding rectangle
        rect = burrow.get_bounding_rect(5*width_min)
        # get respective slices for the image, respecting image borders 
        (_, slices), rect = regions.get_overlapping_slices(rect[:2],
                                                           (rect[3], rect[2]),
                                                           frame.shape,
                                                           anchor='upper left',
                                                           ret_rect=True)
        
        # extract the region of interest from the frame and the mask
        img = frame[slices].astype(np.uint8)
        mask_ground = mask_ground[slices]
        mask_unexplored = (self.explored_area[slices] <= 0)
        mask = np.zeros_like(mask_ground)        

        # create a mask representing the current estimate of the burrow
        mask_burrow = np.zeros_like(mask)
        outline = curves.translate_points(burrow.outline,
                                          xoff=-rect[0],
                                          yoff=-rect[1])
        cv2.fillPoly(mask_burrow, [np.asarray(outline, np.int)], 255)

        # add to this mask the previous burrow
        if burrow_prev:
            outline = curves.translate_points(burrow_prev.outline,
                                              xoff=-rect[0],
                                              yoff=-rect[1])
            cv2.fillPoly(mask_burrow, [np.asarray(outline, np.int)], 255)

        # prepare the input mask for the GrabCut algorithm by defining 
        # foreground and background regions  
        img[mask_ground == 0] = self.result['colors/sand'] #< turn into background
        mask[:] = cv2.GC_BGD #< surely background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2*width_min), int(2*width_min)))
        mask[cv2.dilate(mask_burrow, kernel) == 255] = cv2.GC_PR_BGD #< probable background
        mask[mask_burrow == 255] = cv2.GC_PR_FGD #< probable foreground
        
        # find sure foreground
        kernel_size = int(2*width_min)
        burrow_core_area_min = self.params['burrows/grabcut_burrow_core_area_min']
        while kernel_size >= 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            burrow_sure = (cv2.erode(mask_burrow, kernel) == 255)
            if burrow_sure.sum() >= burrow_core_area_min:
                # the burrow was large enough that erosion left a good foreground
                mask[burrow_sure] = cv2.GC_FGD #< surely foreground
                break
            else:
                kernel_size //= 2 #< try smaller kernel
        
#         debug.show_image(mask_burrow, mask_ground, img, 
#                          debug.get_grabcut_image(mask),
#                          wait_for_key=False)
        
        # have to convert to color image, since grabCut only supports color
        img = cv2.cvtColor(img, cv2.cv.CV_GRAY2RGB)
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        # run GrabCut algorithm
        try:
            cv2.grabCut(img, mask, (0, 0, 1, 1),
                        bgdmodel, fgdmodel, 2, cv2.GC_INIT_WITH_MASK)
        except:
            # any error in the GrabCut algorithm makes the whole function useless
            self.logger.warn('%d: GrabCut algorithm failed on burrow at %s',
                             self.frame_id, burrow.position)
            return burrow

#         debug.show_image(burrow_mask, ground_mask, img, 
#                          debug.get_grabcut_image(mask),
#                          wait_for_key=False)

        # calculate the mask of the foreground
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
        
        # make sure that the burrow has been explored by the mouse
        mask[mask_unexplored] = 0
        
        # make sure that the burrow is under ground
        mask[mask_ground == 0] = 0
        
        # find the burrow from the mask
        try:
            burrow = self.get_burrow_from_mask(mask.astype(np.uint8), offset=rect[:2])
        except RuntimeError as err:
            burrow = None
            self.logger.debug('%d: Invalid burrow from GrabCut: %s',
                              self.frame_id, err.message)
        
        return burrow
    
    
    def find_burrows(self):
        """ locates burrows by combining the information of the mask_ground
        profile and the explored area """

        # reset the current burrow model
        burrows_mask = self._cache['image_uint8']
        burrows_mask.fill(0)

        # estimate the burrow mask
        potential_burrows = self.get_potential_burrows_mask()
        if potential_burrows is None:
            labels, num_features = potential_burrows, 0
        else:
            labels, num_features = ndimage.measurements.label(potential_burrows)
            
        # iterate through all features that have been found
        for label in xrange(1, num_features + 1):
            # get the burrow object from the contour of region
            try:
                burrow = self.get_burrow_from_mask(labels == label)
            except RuntimeError:
                continue

            # add the unrefined burrow to the debug video
            if 'video' in self.debug:
                self.debug['video'].add_line(burrow.outline, 'w',
                                                is_closed=True, width=2)

            # check whether this burrow belongs to a previously found one                
            adaptation_interval = self.params['burrows/adaptation_interval']
            for track_id, burrow_track in enumerate(self.result['burrows/tracks']):
                if (burrow_track.track_end >= self.frame_id - adaptation_interval
                    and burrow_track.last.intersects(burrow)):
                    
                    # this burrow is active and overlaps with the current one
                    break
            else:
                track_id = None

            # refine the burrow based on its shape
            burrow.get_centerline(self.ground) #< also calculates the length
            burrow_width = burrow.area/burrow.length if burrow.length > 0 else 0
            min_length = self.params['burrows/fitting_length_threshold']
            max_width = self.params['burrows/fitting_width_threshold']
            if (burrow.length > min_length and burrow_width < max_width):
                # it's a long burrow => fit the current burrow
                burrow = self.refine_long_burrow(burrow)
                
            elif track_id is not None:
                # it's a bulky burrow with a known previous burrow
                # => refine burrow taken any previous burrows into account
                burrow_prev = self.result['burrows/tracks'][track_id].last
                burrow = self.refine_bulky_burrow(burrow, burrow_prev)
                
            else:
                # it's a bulky burrow without a known previous burrow
                # => just refine this burrow
                burrow = self.refine_bulky_burrow(burrow)
            
            # add the burrow to our result list if it is valid
            if burrow is not None and burrow.is_valid:
                cv2.fillPoly(burrows_mask, np.array([burrow.outline], np.int32), color=1)
                
                if track_id is not None:
                    # add the burrow to the current mask
                    self.result['burrows/tracks'][track_id].append(self.frame_id, burrow)
                else:
                    # otherwise, start a new burrow track
                    burrow_track = BurrowTrack(self.frame_id, burrow)
                    self.result['burrows/tracks'].append(burrow_track)
                    self.logger.debug('%d: Found new burrow at %s',
                                      self.frame_id, burrow.polygon.centroid)
            
        # degrade information about the mouse position inside burrows
        rate = self.params['explored_area/adaptation_rate_burrows']* \
                self.params['burrows/adaptation_interval']
        if rate != 0:
            cv2.subtract(self.explored_area, rate,
                         dst=self.explored_area,
                         mask=(0 != burrows_mask).astype(np.uint8))
 
        # degrade information about the mouse position outside burrows
        rate = self.params['explored_area/adaptation_rate_outside']* \
                self.params['burrows/adaptation_interval']
        if rate != 0:
            cv2.subtract(self.explored_area, rate,
                         dst=self.explored_area,
                         mask=(0 == burrows_mask).astype(np.uint8))
        

    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        self.debug['object_count'] = 0
        self.debug['video.mark.text1'] = ''
        self.debug['video.mark.text2'] = ''
        debug_output = self.params['debug/output']

        # load parameters for video output        
        video_output_period = int(self.params['output/video/period'])
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in debug_output or 'video.show' in debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('pass1' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                output_period=video_output_period,
                                                codec=video_codec, bitrate=video_bitrate,
                                                debug=self.debug_run)
            
            if 'video.show' in debug_output:
                name = self.name if self.name else ''
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 1 [%s]' % name,
                                           multiprocessing=self.params['debug/use_multiprocessing'],
                                           position=position)
                self.debug['video.show'] = image_window

        # set up additional video writers
        for identifier in ('difference', 'background', 'explored_area'):
            if identifier in debug_output:
                # determine the filename to be used
                debug_file = self.get_filename(identifier + video_extension, 'debug')
                # set up the video file writer
                video_writer = VideoComposer(debug_file, self.video.size, self.video.fps,
                                             is_color=False, output_period=video_output_period,
                                             codec=video_codec, bitrate=video_bitrate,
                                             debug=self.debug_run)
                self.debug[identifier + '.video'] = video_writer
        

    def debug_process_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None: 
                debug_video.add_line(self.ground.points, is_closed=False,
                                     mark_points=True, color='y')
        
            # indicate the currently active burrow shapes
            if self.params['burrows/enabled_pass1']:
                time_interval = self.params['burrows/adaptation_interval']
                for burrow_track in self.result['burrows/tracks']:
                    if burrow_track.track_end > self.frame_id - time_interval:
                        burrow = burrow_track.last
                        burrow_color = 'red' if burrow.refined else 'orange'
                        debug_video.add_line(burrow.outline, burrow_color,
                                             is_closed=True, mark_points=True)
                        debug_video.add_line(burrow.get_centerline(self.ground),
                                             burrow_color, is_closed=False,
                                             width=2, mark_points=True)
        
            # indicate the mouse position
            if len(self.tracks) > 0:
                for obj in self.tracks:
                    if self.result['objects/moved_first_in_frame'] is None:
                        obj_color = 'r'
                    elif obj.is_moving():
                        obj_color = 'w'
                    else:
                        obj_color = 'b'
                    track = obj.get_track()
                    trail_length = self.params['output/video/mouse_trail_length']
                    if len(track) > trail_length:
                        track = track[-trail_length:]
                    debug_video.add_line(track, '0.5', is_closed=False)
                    debug_video.add_circle(obj.last.pos,
                                           self.params['mouse/model_radius'],
                                           obj_color, thickness=1)
                
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            debug_video.add_text('#obj:%d' % self.debug['object_count'],
                                 (120, 20), anchor='top')
            debug_video.add_text(self.debug['video.mark.text1'],
                                 (300, 20), anchor='top')
            debug_video.add_text(self.debug['video.mark.text2'],
                                 (300, 50), anchor='top')
            if self.debug.get('video.mark.rect1'):
                debug_video.add_rectangle(self.debug['rect1'])
            if self.debug.get('video.mark.points'):
                debug_video.add_points(self.debug['video.mark.points'],
                                       radius=4, color='y')
            if self.debug.get('video.mark.highlight', False):
                rect = (0, 0, self.video.size[0], self.video.size[1])
                debug_video.add_rectangle(rect, 'w', 10)
                self.debug['video.mark.highlight'] = False
            
            if 'video.show' in self.debug:
                if debug_video.output_this_frame:
                    self.debug['video.show'].show(debug_video.frame)
                else:
                    self.debug['video.show'].show()
                
        if 'difference.video' in self.debug:
            diff = frame.astype(int, copy=False) - self.background + 128
            diff = np.clip(diff, 0, 255).astype(np.uint8, copy=False)
            self.debug['difference.video'].set_frame(diff)
            self.debug['difference.video'].add_text(str(self.frame_id),
                                                    (20, 20), anchor='top')   
                
        if 'background.video' in self.debug:
            self.debug['background.video'].set_frame(self.background)
            self.debug['background.video'].add_text(str(self.frame_id),
                                                    (20, 20), anchor='top')   

        if 'explored_area.video' in self.debug:
            debug_video = self.debug['explored_area.video']
             
            # set the background
            debug_video.set_frame(128*np.clip(self.explored_area, 0, 1))
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_line(self.ground.points, is_closed=False, color='y')
                debug_video.add_points(self.ground.points, radius=2, color='y')

            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   


    def debug_finalize(self):
        """ close the video streams when done iterating """
        # close the window displaying the video
        if 'video.show' in self.debug:
            self.debug['video.show'].close()
        
        # close the open video streams
        for video in ('video', 'difference.video', 'background.video', 'explored_area.video'):
            if video in self.debug:
                try:
                    self.debug[video].close()
                except IOError:
                    self.logger.exception('Error while writing out the debug '
                                          'video `%s`' % video) 
            
    
