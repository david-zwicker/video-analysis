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

import itertools

import numpy as np
import scipy.ndimage as ndimage
from scipy import optimize, signal, spatial
import shapely.geometry as geometry
import cv2

from video.io import ImageShow
from video.filters import FilterCrop
from video.analysis import regions, curves, image
from video.utils import display_progress
from video.composer import VideoComposer

from .data_handler import DataHandler
from .objects.moving import MovingObject, ObjectTrack, ObjectTrackList
from .objects.ground import GroundProfile, GroundProfileList, RidgeProfile
from .objects.burrow import Burrow, BurrowTrack, BurrowTrackList

import debug  # @UnusedImport
from IPython.qt.console.mainwindow import background



class FirstPass(DataHandler):
    """
    analyzes mouse movies
    """
    logging_mode = 'create'    
    
    def __init__(self, name='', parameters=None, debug_output=None, **kwargs):
        """ initializes the whole mouse tracking and prepares the video filters """
        
        # initialize the data handler
        super(FirstPass, self).__init__(name, parameters, **kwargs)
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass1')
        
        # setup internal structures that will be filled by analyzing the video
        self._cache = {}               # cache that some functions might want to use
        self.debug = {}                # dictionary holding debug information
        self.background = None         # current background model
        self.ground = None             # current model of the ground profile
        self.tracks = []               # list of plausible mouse models in current frame
        self._mouse_pos_estimate = []  # list of estimated mouse positions
        self.explored_area = None      # region the mouse has explored yet
        self.frame_id = None           # id of the current frame
        self.result['mouse/moved_first_in_frame'] = None
        self.debug_output = [] if debug_output is None else debug_output
        self.log_event('Pass 1 - Initialized the first pass analysis.')


    def load_video(self, video=None, crop_video=True):
        """ load and prepare the video.
        crop_video indicates whether the cropping to a quadrant (given in the
        parameters dictionary) should be performed. The cropping to the mouse
        cage is performed no matter what. 
        """
        
        # load the video if it is not already loaded 
        super(FirstPass, self).load_video(video, crop_video=crop_video)
                
        self.data.create_child('video/input', {'frame_count': self.video.frame_count,
                                               'size': '%d x %d' % tuple(self.video.size),
                                               'fps': self.video.fps})
        
        self.data['analysis-status'] = 'Loaded video'            

    
    def process_video(self):
        """ processes the entire video """
        self.log_event('Pass 1 - Started initializing the video analysis.')
        
        # restrict the video to the region of interest (the cage)
        if self.params['cage/determine_boundaries']:
            self.video, cropping_rect = self.crop_video_to_cage(self.video)
        else:
            cropping_rect = None
        self.data.create_child('video/analyzed', {'frame_count': self.video.frame_count,
                                                  'region_cage': cropping_rect,
                                                  'size': '%d x %d' % tuple(self.video.size),
                                                  'fps': self.video.fps})
        
        self.debug_setup()
        self.setup_processing()

        self.log_event('Pass 1 - Started iterating through the video with %d frames.' %
                       self.video.frame_count)
        self.data['analysis-status'] = 'Initialized video analysis'            
        
        try:
            # skip the first frame, since it has already been analyzed
            self._iterate_over_video(self.video[1:])
                
        except (KeyboardInterrupt, SystemExit):
            # abort the video analysis
            self.video.abort_iteration()
            self.logger.info('Pass 1 - Tracking has been interrupted by user.')
            self.log_event('Pass 1 - Analysis run has been interrupted.')
            self.data['analysis-status'] = 'Partly finished first pass'
            
        else:
            # finished analysis successfully
            self.log_event('Pass 1 - Finished iterating through the frames.')
            self.data['analysis-status'] = 'Finished first pass'
            
        finally:
            # add the currently active tracks to the result
            self.result['objects/tracks'].extend(self.tracks)
            # clean up
            self.video.close()
        
        self.data['video/analyzed/frames_analyzed'] = self.frame_id + 1
                    
        # cleanup and write out of data
        self.debug_finalize()
        self.write_data()
        

    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        
        self.result['objects/tracks'] = ObjectTrackList()
        self.result['ground/profile'] = GroundProfileList()
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

        sigma = self.params['video/blur_radius']
        blur_kernel = cv2.getGaussianKernel(int(3*sigma), sigma=sigma)

        # iterate over the video and analyze it
        for self.frame_id, frame in enumerate(display_progress(video)):
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
                
            # blur frame - if the frame is contiguous in memory, we don't need to make a copy
            frame_blurred = np.ascontiguousarray(frame)
            cv2.sepFilter2D(frame_blurred, cv2.CV_8U,
                            blur_kernel, blur_kernel, dst=frame_blurred)
            
            if self.frame_id == self.params['video/initial_adaptation_frames']:
                # prepare the main analysis
                # estimate colors of sand and sky
                self.find_color_estimates(frame_blurred)
                
                # estimate initial ground profile
                self.logger.debug('Find the initial ground profile.')
                self.ground = self.find_initial_ground()
                self.result['ground/profile'].append(self.frame_id, self.ground)
        
            elif self.frame_id > self.params['video/initial_adaptation_frames']:
                # do the main analysis after an initial wait period
                
                # update the color estimates
                if self.frame_id % self.params['colors/adaptation_interval'] == 0:
                    self.find_color_estimates(frame_blurred)

                # identify moving objects by comparing current frame to background
                self.find_objects(frame_blurred)
                
                # use the background to find the current ground profile and burrows
                if self.frame_id % self.params['ground/adaptation_interval'] == 0:
                    self.ground = self.refine_ground(self.ground)
                    self.result['ground/profile'].append(self.frame_id, self.ground)
        
                if self.frame_id % self.params['burrows/adaptation_interval'] == 0:
                    self.find_burrows()
                    
            # update the background model
            self.update_background_model(frame_blurred)
                
            # store some information in the debug dictionary
            self.debug_process_frame(frame_blurred)
                         
                    
    #===========================================================================
    # FINDING THE CAGE
    #===========================================================================
    
    
    def find_cage_approximately(self, frame):
        """ analyzes a single frame and locates the mouse cage in it.
        Try to find a bounding box for the cage.
        The rectangle [top, left, height, width] enclosing the cage is returned. """
        # do automatic thresholding to find large, bright areas
        _, binarized = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find the largest bright are, which should contain the cage
        cage_mask = regions.get_largest_region(binarized)
        
        # find an enclosing rectangle, which usually overestimates the cage bounding box
        rect_large = regions.find_bounding_box(cage_mask)
         
        # crop frame to this rectangle, which should surely contain the cage 
        frame = frame[regions.rect_to_slices(rect_large)]

        # initialize the rect coordinates
        top = 0 # start on first row
        bottom = frame.shape[0] - 1 # start on last row
        width = frame.shape[1]

        # threshold again, because large distractions outside of cages are now
        # definitely removed. Still, bright objects close to the cage, e.g. the
        # stands or some pipes in the background might distract the estimate.
        # We thus adjust the rectangle in the following  
        _, binarized = cv2.threshold(frame, 0, 255,
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
        return regions.corners_to_rect(p1, p2)


    def find_cage_exactly(self, frame, rect_cage):
        """
        """
        return rect_cage

  
    def crop_video_to_cage(self, video):
        """ crops the video to a suitable cropping rectangle given by the cage """
        # find the cage in the blurred image
        blurred_frame = cv2.GaussianBlur(video[0], ksize=(0, 0),
                                         sigmaX=self.params['video/blur_radius'])
        
        # find the rectangle describing the cage
        rect_cage = self.find_cage_approximately(blurred_frame)
        rect_cage = self.find_cage_exactly(blurred_frame, rect_cage)
        
        # determine the rectangle of the cage in global coordinates
        width = rect_cage[2] - rect_cage[2] % 2   # make sure its divisible by 2
        height = rect_cage[3] - rect_cage[3] % 2  # make sure its divisible by 2
        # Video dimensions should be divisible by two for some codecs

        if not (self.params['cage/width_min'] < width < self.params['cage/width_max'] and
                self.params['cage/height_min'] < height < self.params['cage/height_max']):
            raise RuntimeError('The cage bounding box (%dx%d) is out of the '
                               'limits.' % (width, height)) 
        
        rect_cage = (rect_cage[0], rect_cage[1], width, height)
        
        self.logger.debug('The cage was determined to lie in the rectangle %s', rect_cage)

        # crop the video to the cage region
        return FilterCrop(video, rect_cage), rect_cage
            
            
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
        sky_mask = regions.get_largest_region(1 - binarized)
        sky_mask = sky_mask.astype(np.uint8, copy=False)*255

        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sky_mask, cv2.cv.CV_DIST_L2, 5)
        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sky_sure = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)

        # determine the sky color
        sky_img = image[sky_sure.astype(np.bool, copy=False)]
        self.result['colors/sky'] = sky_img.mean()
        self.result['colors/sky_std'] = sky_img.std()
        self.logger.debug('The sky color was determined to be %d +- %d',
                          self.result['colors/sky'], self.result['colors/sky_std'])

        # find the sand by looking at the largest bright region
        sand_mask = regions.get_largest_region(binarized).astype(np.uint8, copy=False)*255
        
        # Finding sure foreground area using a distance transform
        dist_transform = cv2.distanceTransform(sand_mask, cv2.cv.CV_DIST_L2, 5)
        if len(dist_transform) == 2:
            # fallback for old behavior of OpenCV, where an additional parameter
            # would be returned
            dist_transform = dist_transform[0]
        _, sand_sure = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
        
        # determine the sky color
        sand_img = image[sand_sure.astype(np.bool, copy=False)]
        self.result['colors/sand'] = sand_img.mean()
        self.result['colors/sand_std'] = sand_img.std()
        self.logger.debug('The sand color was determined to be %d +- %d',
                          self.result['colors/sand'], self.result['colors/sand_std'])
        
                
    def update_background_model(self, frame):
        """ updates the background model using the current frame """
        
        if self.background is None:
            self.background = frame.astype(np.double, copy=True)
        
        if self._mouse_pos_estimate:
            # load some values from the cache
            adaptation_rate = self._cache['image_double']
            template = self._cache['mouse.template']
            adaptation_rate.fill(self.params['background/adaptation_rate'])
            
            # cut out holes from the adaptation_rate for each mouse estimate
            for mouse_pos in self._mouse_pos_estimate:
                # get the slices required for comparing the template to the image
                t_s, i_s = regions.get_overlapping_slices(mouse_pos, template.shape,
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
            # get the coordinates of the center of mass
            pos = (moments['m10']/area, moments['m01']/area)
            
            # check whether this object could be a mouse
            if area > self.params['mouse/area_min']:
                objects.append(MovingObject(pos, size=area, label=label))
                
            elif area > largest_obj.size:
                # determine the maximal area during the loop
                largest_obj = MovingObject(pos, size=area, label=label)
        
        if len(objects) == 0:
            # if we haven't found anything yet, we just take the best guess,
            # which is the object with the largest area
            objects.append(largest_obj)
        
        return objects


    def _handle_object_tracks(self, frame, labels, num_features):
        """ analyzes objects in a single frame and tries to add them to
        previous tracks """
        # get potential objects
        objects_found = self._find_objects_in_binary_image(labels, num_features)

        # check if there are previous tracks        
        if len(self.tracks) == 0:
            self.tracks = [ObjectTrack([self.frame_id], [obj])
                           for obj in objects_found]
            
            return # there is nothing to do anymore
            
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
            self.logger.debug('%d: Copy mouse track of length %d to results',
                              self.frame_id, len(self.tracks[i_e]))
            # copy track to result dictionary
            self.result['objects/tracks'].append(self.tracks[i_e])
            del self.tracks[i_e]
        
        # start new tracks for objects that had no previous match
        for i_f in idx_f:
            self.logger.debug('%d: New mouse track at %s',
                              self.frame_id, objects_found[i_f].pos)
            # start new track
            track = ObjectTrack([self.frame_id], [objects_found[i_f]])
            self.tracks.append(track)
        
        assert len(self.tracks) == len(objects_found)
        
    
    def find_objects(self, frame):
        """ adapts the current mouse position, if enough information is available """

        # find a binary image that indicates movement in the frame
        moving_objects = self.find_moving_features(frame)
    
        # find all distinct features and label them
        num_features = ndimage.measurements.label(moving_objects,
                                                  output=moving_objects)
        self.debug['object_count'] = num_features
        
        if num_features == 0:
            # end all current tracks if there are any
            if len(self.tracks) > 0:
                self.logger.debug('%d: Copy %d tracks to results', 
                                  self.frame_id, len(self.tracks))
                self.result['objects/tracks'].extend(self.tracks)
                self.tracks = []

        else:
            # some moving features have been found in the video 
            self._handle_object_tracks(frame, moving_objects, num_features)

            # check whether objects moved and call them a mouse
            obj_moving = [obj.is_moving() for obj in self.tracks]
            if any(obj_moving):
                if self.result['mouse/moved_first_in_frame'] is None:
                    self.result['mouse/moved_first_in_frame'] = self.frame_id
                # remove the tracks that didn't move
                # this essentially assumes that there is only one mouse
                for k, obj in enumerate(self.tracks):
                    if obj_moving[k]:
                        # keep only the moving object in the current list
                        self.tracks = [obj]
                    else:
                        self.result['objects/tracks'].append(obj)

            self._mouse_pos_estimate = [obj.last.pos for obj in self.tracks]
        
            # add new information to explored area
            for track in self.tracks:
                self.explored_area[moving_objects == track.last.label] = 1
                
                
    #===========================================================================
    # FINDING THE GROUND PROFILE
    #===========================================================================


    def estimate_ground_new(self):
        """ determines an estimate of the ground profile from a single frame """
        frame = self.background

        # build the ground ridge template for matching 
        width = self.params['ground/width']
        dist_width = int(width + frame.shape[0]/10)
        dist = np.arange(-dist_width, dist_width + 1)
        color_sky = self.result['colors/sky']
        color_sand = self.result['colors/sand']
        model = (1 + np.tanh(dist/width))/2
        model = color_sky + (color_sand - color_sky)*model
        model = model.astype(np.uint8)
        
        # estimate for ground profile very roughly        
        width, height = self.video.size
        dx, dy = width//5, height//3
        x = np.array([dy, 2*dy, dx, 2*dx, 3*dx, 4*dx])
        y1, y2, x1, x2, x3, x4 = x
        points = [[0.1*dx,     y1],
                  [    x1,     y1],
                  [    x2,     y2],
                  [    x3,     y2],
                  [    x4,     y1],
                  [width-0.1*dx,     y1]]
        
        dx = int(self.params['ground/point_spacing']/2)
        dy = 10*dx
        points = curves.make_curve_equidistant(points, 2*dx)
        
        result = []
        for p in points:
            x, y = int(p[0]), int(p[1])
            # get line scan
            profile = frame[y-dy:y+dy+1, x-dx:x+dx+1].mean(axis=1)
            
            conv = cv2.matchTemplate(profile.astype(np.uint8),
                                     model, cv2.cv.CV_TM_CCORR_NORMED)
            # get the minimum, indicating the best match
            pos_y = np.argmax(conv) + dist_width
            # add point
            result.append((x, pos_y))
            
#             import matplotlib.pyplot as plt
#             plt.plot(profile)
#             plt.show()
#             exit()
        
        
        debug.show_shape(geometry.LineString(points),
                         geometry.LineString(result),
                         background=self.background)
        exit()
        
        
        # refine ground profile along the estimated one

        return GroundProfile(points)

        
        
        def make_poly(data):
            y1, y2, x1, x2, x3, x4 = data
            points = [[    0,     y1],
                      [   x1,     y1],
                      [   x2,     y2],
                      [   x3,     y2],
                      [   x4,     y1],
                      [width,     y1],
                      [width, height],
                      [    0, height]]
            return geometry.LinearRing(points)
                
        
        debug.show_shape(make_poly(x), background=self.background)
        exit()

        
        for level in (6, 5, 4, 3, 2, 1):
            
            # create pyramid
            frame = self.background.astype(np.uint8, copy=True)
            for _ in xrange(level):
                frame = cv2.pyrDown(frame)
    
            x /= 2**level
                
            # create model
            model = np.empty_like(frame)
            color_sky = self.result['colors/sky']
            color_sand = self.result['colors/sand']
            
            def get_model(data):
                model.fill(color_sky)
                y1, y2, x1, x2, x3, x4 = data
                points = [[    0,     y1],
                          [   x1,     y1],
                          [   x2,     y2],
                          [   x3,     y2],
                          [   x4,     y1],
                          [width,     y1],
                          [width, height],
                          [    0, height]]
                cv2.fillPoly(model, [np.asarray(points, np.int)], color=color_sand)
                return model
            
            def residual(data):
                res = np.ravel(frame - get_model(data))
                return np.sum(res**2)
                
            x0 = x[:]
            
            min_res = residual(x)
            res_changing = True
            while res_changing:
                res_changing = False 
                for k in xrange(6):
                    for d in (-2, -1, 1, 2):
                        x[k] += d
                        res = residual(x)
                        if res < min_res:
                            print res
                            min_res = res
                            res_changing = True 
                        else:
                            x[k] -= d #revert
                        
            debug.show_image(frame, get_model(x0), get_model(x), equalize_colors=True)   
            
            x *= 2**level     
        
        #debug.show_shape(make_poly(data), background=frame)
        exit()
        

    def estimate_ground(self):
        """ determines an estimate of the ground profile from a single frame """
        frame = self.background.astype(np.uint8)
        
        # build the ground ridge template for matching 
        width = self.params['ground/width']
        dist_width = int(width + frame.shape[0]/10)
        dist = np.arange(-dist_width, dist_width + 1)
        color_sky = self.result['colors/sky']
        color_sand = self.result['colors/sand']
        model = (1 + np.tanh(dist/width))/2
        model = color_sky + (color_sand - color_sky)*model
        model = model.astype(np.uint8)
        
        # do vertical line scans and determine ground position
        spacing = int(self.params['ground/point_spacing'])
        border_width = frame.shape[1] - self.params['cage/width_min'] 
        points = []
        for k in xrange(frame.shape[1]//spacing):
            pos_x = (k + 0.5)*spacing
            if (border_width < pos_x < frame.shape[1] - border_width):
                line_scan = frame[:, spacing*k:spacing*(k + 1)].mean(axis=1)
                # get the cross-correlation between the profile and the template
                conv = cv2.matchTemplate(line_scan.astype(np.uint8),
                                         model, cv2.cv.CV_TM_CCORR_NORMED)
                # get the minimum, indicating the best match
                pos_y = np.argmax(conv)  + dist_width
                
                # add point
                points.append((pos_x, pos_y))

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

        return GroundProfile(points)
   
   
    def refine_ground_snake(self, ground, try_many_distances=False):
        """ adapts a points profile given as points to a given frame.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve.
        
        See http://en.wikipedia.org/wiki/Active_contour_model
        """
                
        spacing = self.params['ground/point_spacing']
        
        # make sure the curve has equidistant points
        points = curves.make_curve_equidistant(ground.line, spacing=spacing)
        
        points = np.array(np.round(points), np.int32)
        
        # make sure that points are not above each other
        for k in xrange(1, len(points)):
            if points[k-1][0] == points[k][0]:
                points[k][0] += 1
        
        snake_stiffness = self.params['ground/snake_bending_energy']
        snake_length = curves.curve_length(points)
        alpha = snake_length*1e3
        beta = snake_length*snake_stiffness
        
        # get the edges of the current background image
        frame = self.background
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=11)
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=11)
        grad = np.hypot(sobel_x, sobel_y)
        
        grad = cv2.GaussianBlur(grad, (0, 0), 5)
        
        frame_energy = cv2.cv.fromarray(-grad) #< convert to cvMat
        
        def get_snake_energy(ps):
            # bending energy from second derivative
            energy_len = 0
            energy_curv = 0#abs(ps[0][1] - ps[1][1]) + abs(ps[-2][1] - ps[-1][1])
            for k in xrange(1, len(ps) - 1):
                curve_len = curves.curve_length(ps[k-1:k+2])
                energy_len += curve_len

                a, b, c = ps[k-1], ps[k], ps[k+1]
                curv = ((b[1] - a[1])/(a[0] - b[0]) + (b[1] - c[1])/(b[0] - c[0]))/(c[0] - a[0])
                #curv = np.sum((ps[k-1] - 2*ps[k] + ps[k+1])**2)
                energy_curv += curv**2 * curve_len
                
            energy_len *= alpha
            energy_curv *= beta
#             print energy
            
            # image energy from integrating along the snake
            energy_ext = 0
            for p1, p2 in itertools.izip(ps[:-1], ps[1:]):
                p1 = (int(p1[0]), int(p1[1]))
                p2 = (int(p2[0]), int(p2[1]))
#                 im = image.line_scan(-grad, p1, p2, 1).sum()
#                 it = sum(cv2.cv.InitLineIterator(frame_energy, p1, p2))
#                 print 'im', im, it
#                 exit()
#                 exit()
                #energy += image.line_scan(-grad, p1, p2, 1).mean()
                energy_ext += sum(cv2.cv.InitLineIterator(frame_energy, p1, p2))
                
            energy = energy_len + energy_curv + energy_ext
                
#             print energy, ps[:, 1].mean(), ps[:, 1].std()
#             print 'second', energy
#             print energy_len/1e7, energy_curv/1e7, energy_ext/1e7
            return energy

        def adapt_snake(ps_y):
            ps = points.copy()
            ps[:, 1] = ps_y
            return get_snake_energy(ps)
        
        import scipy.optimize as so
        
        bounds = np.empty_like(points)
        bounds[:, 0] = 0
        bounds[:, 1] = frame.shape[1]
        
        #points[:, 1] = so.fmin(adapt_snake, points[:, 1], xtol=1)
        #points[:, 1] = so.fmin_bfgs(adapt_snake, points[:, 1], epsilon=1)
        
#         points[:, 1] = so.fmin_tnc(adapt_snake, points[:, 1], epsilon=1,
#                                    approx_grad=True, bounds=bounds)[0]
        points[:, 1] = so.fmin_l_bfgs_b(adapt_snake, points[:, 1],
                                        epsilon=1, approx_grad=True, bounds=bounds)[0]
        return points
                
        if try_many_distances:
            ds = np.array((1, 2, 4, 8, 16, 32))
            distances = np.concatenate((ds, -ds))
        else:
            distances = (-1, 1)
                
        # iterate through all points and correct profile
        snake_energy = get_snake_energy(points)
        energy_reduced = True
        while energy_reduced:
            energy_reduced = False
            for k in xrange(len(points)):
                for d in distances:
                    points[k, 1] += d
                    energy = get_snake_energy(points)
                    if energy < snake_energy:
                        snake_energy = energy
                        energy_reduced = True
                    else:
                        points[k, 1] -= d
        
#         debug.show_shape(geometry.LineString(points.astype(np.double)),
#                          geometry.LineString(points.astype(np.double)),
#                          background=-grad, mark_points=True, wait_for_key=False)
            
        return GroundProfile(points)
    
    
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
        pos_max = signal.argrelmax(profile)[0][0]
        pos_min = signal.argrelmin(profile[pos_max:])[0][0] + pos_max
        
        # get steepest point in this interval
        pos_edge = np.argmin(np.diff(profile[pos_max:pos_min])) + pos_max

        if direction == 'right':
            pos_edge = image_width - pos_edge
        
        return (pos_edge, ground_point[1])
    
        
    def refine_ground(self, ground, **kwargs):
        """ adapts a points profile given as points to a given frame.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
        frame = self.background
        spacing = int(self.params['ground/point_spacing'])
            
        if not 'ground.model' in self._cache or \
            self._cache['ground.model'].size != spacing:
            
            self._cache['ground.model'] = \
                    RidgeProfile(spacing, self.params['ground/width'])
                
        # make sure the curve has equidistant points
        points = curves.make_curve_equidistant(ground.line, spacing)
        points = np.array(np.round(points),  np.int32)
        
        # calculate the bounds for the points
        p_min = spacing 
        y_max, x_max = frame.shape[0] - spacing, frame.shape[1] - spacing

        # iterate through all points and correct profile
        ground_model = self._cache['ground.model']
        corrected_points = []
        x_previous = spacing
        deviation = 0
        
        # iterate over all but the boundary points
        for k, p in enumerate(points[1:-1], 1):
            
            # skip points that are too close to the boundary
            if (p[0] < p_min or p[0] > x_max or 
                p[1] < p_min or p[1] > y_max):
                continue
            
            dp = points[k+1] - points[k-1]
            angle = np.arctan2(dp[0], dp[1])
                
            # extract the region around the point used for fitting
            region = frame[p[1]-spacing : p[1]+spacing+1,
                           p[0]-spacing : p[0]+spacing+1].copy()
            ground_model.set_data(region, angle) 

            # move the points profile model perpendicular until it fits best
            x, _, infodict, _, _ = optimize.leastsq(ground_model.get_difference, [0],
                                                    xtol=0.5, epsfcn=0.5, full_output=True)
            
            # calculate goodness of fit
            ss_err = (infodict['fvec']**2).sum()
            ss_tot = ((region - region.mean())**2).sum()
            if ss_tot == 0:
                rsquared = 0
            else:
                rsquared = 1 - ss_err/ss_tot

            # Note, that we never remove the first and the last point
            if rsquared > 0 or k == 0 or k == len(points) - 1:
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

        # extend the ground line toward the left edge of the cage
        edge_point = self._get_cage_boundary(corrected_points[0], 'left')
        if edge_point is not None:
            corrected_points.insert(0, edge_point)
            
        # extend the ground line toward the right edge of the cage
        edge_point = self._get_cage_boundary(corrected_points[-1], 'right')
        if edge_point is not None:
            corrected_points.append(edge_point)
            
        return GroundProfile(corrected_points)
            

    def find_initial_ground(self):
        """ finds the ground profile given an image of an antfarm """
        ground = self.estimate_ground()
        ground = self.refine_ground(ground, try_many_distances=True)
        
        self.logger.info('Pass 1 - We found a ground profile of length %g',
                         ground.length)
        
        return ground

        frame = self.background.astype(np.uint8)
        
        # remove 10%/15% of each side of the frame
        h = int(0.15*frame.shape[0])
        w = int(0.10*frame.shape[1])
        image_center = frame[h:-h, w:-w]
        
        # turn frame in binary image
        cv2.threshold(image_center, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=image_center)
        
        # do morphological opening and closing to smooth the profile
        s = self.params['burrows/width']
        ys, xs = np.ogrid[-s:s+1, -s:s+1]
        kernel = (xs**2 + ys**2 <= s**2).astype(np.uint8, copy=False)

        # morphological opening to remove noise and clutter
        cv2.morphologyEx(image_center, cv2.MORPH_OPEN, kernel, dst=image_center)

        # get the contour from the mask and store points as (x, y)
        width_video = self.video.size[0]
        top_min = self.params['ground/flat_top_fraction']*width_video
        top_max = width_video - top_min
        points = []
        for x, col in enumerate(image_center.T):
            xp = x + w
            if top_min <= xp <= top_max:
                try:
                    # add topmost white point as estimate 
                    yp = np.nonzero(col)[0][0] + h
                except IndexError:
                    # there are no white points => continue from last time
                    yp = points[-1][1]
                points.append((xp, yp))
        
        # simplify and refine the curve        
        points = curves.simplify_curve(points, epsilon=2)
        points = self.refine_ground(points, try_many_distances=True)
        ground = GroundProfile(points)
        
        self.logger.info('Pass 1 - We found a ground profile of length %g',
                         ground.length)
        
        return ground
        
    
    def get_ground_mask(self, color=255):
        """ returns a binary mask distinguishing the ground from the sky """
        # build a mask with potential burrows
        width, height = self.video.size
        ground_mask = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current ground_mask profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground.line
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(ground_mask, np.array([ground_points], np.int32), color=color)

        return ground_mask
    
            
    #===========================================================================
    # FINDING BURROWS
    #===========================================================================
   
        
    def get_potential_burrows_mask(self):
        """ locates potential burrows by searching for underground regions that
        the mouse explored """

        mask_ground = self.get_ground_mask()

        # get potential burrows by looking at explored area
        explored_area = 255*(self.explored_area > 0).astype(np.uint8, copy=False)
        
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
    
        
    def get_burrow_from_mask(self, mask):
        """ creates a burrow object given a contour outline """
    
        # find the contour of the mask    
        contours, _ = cv2.findContours(mask.astype(np.uint8, copy=False),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = np.squeeze(np.asarray(contours[0], np.double))

        # simplify the contour
        tolerance = self.params['burrows/outline_simplification_threshold'] \
                        *curves.curve_length(contour)
        contour = curves.simplify_curve(contour, tolerance).tolist()

        # remove potential invalid structures from contour
        contour = regions.regularize_contour(contour)
        
        return Burrow(contour)
    
    
    def find_burrow_edge(self, profile, direction='up'):
        """ return the template for finding burrow edges
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
        its centerline """
        # keep the points close to the ground line
        ground_line = self.ground.linestring
        distance_threshold = self.params['burrows/ground_point_distance']
        outline_new = sorted([p.coords[0]
                              for p in geometry.MultiPoint(burrow.outline)
                              if p.distance(ground_line) < distance_threshold])

        # replace the remaining points by fitting perpendicular to the center line
        outline = geometry.LinearRing(burrow.outline)
        centerline = burrow.get_centerline(self.ground)
        if len(centerline) < 3:
            self.logger.warn('Refining of very short burrows is not supported.')
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
                # d_l > 0 and d_r < 0 accounting for direction

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
        if len(centerline_new) >= 2:
            p1, p2 = centerline_new[-1], centerline_new[-2]
            angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    
            # shoot out rays in several angles        
            angles = angle + np.pi/8*np.array((-2, -1, 0, 1, 2))
            points = regions.get_ray_intersections(centerline_new[-1], angles, outline)
            # filter unsuccessful points
            points = (p for p in points if p is not None)
            
            # determine the number of frames the mouse has been absent from the end
            frames_absent = (
                (1 - self.explored_area[int(p2[1]), int(p2[0])])
                /self.params['explored_area/adaptation_rate_burrows']
            )
            
            point_max, dist_max = None, 0
            point_anchor = centerline_new[-1]
            for point in points:
                if frames_absent > 10/self.params['background/adaptation_rate']:
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
        outline_new = regions.regularize_contour(outline_new)

        return Burrow(outline_new, centerline=centerline_new, refined=True)
    
    
    def refine_bulky_burrow(self, burrow):
        """ refines the outline of a bulky burrow """
        # get ground line
        ground_line = self.ground.linestring
        scan_length = int(self.params['burrows/width'])
        
        # wrap around outline points on the edge
        outline = np.vstack((burrow.outline[-1],
                             burrow.outline,
                             burrow.outline[0]))
        
        # iterate through all outline points
        outline_new = []
        for k, p in enumerate(outline[1:-1], 1):
            # refine points away from the ground line
            dist = ground_line.distance(geometry.Point(p))
            if dist > self.params['burrows/ground_point_distance']:
                # find local slope of the outline
                dx = outline[k+1][0] - outline[k-1][0]
                dy = outline[k+1][1] - outline[k-1][1]
                dist = np.hypot(dx, dy)
                dx /= dist; dy /= dist
    
                p_a = (p[0] + scan_length*dy, p[1] - scan_length*dx)
                p_b = (p[0] - scan_length*dy, p[1] + scan_length*dx)
                
#                 import matplotlib.pyplot as plt
#                 plt.imshow(self.background)
#                 plt.plot(outline[:, 0], outline[:, 1], color='w')
#                 plt.plot(p_a[0], p_a[1], 'or')
#                 plt.plot(p_b[0], p_b[1], 'og')
#                 plt.show()
                
                # find the transition points by considering slopes
                background = self.background.astype(np.uint8, copy=False)
                profile = image.line_scan(background, p_a, p_b, 3)
                k = self.find_burrow_edge(profile, direction='up')

                if k is not None:
                    d = scan_length - k
                    p = (p[0] + d*dy, p[1] - d*dx)
            
            outline_new.append(p)

        outline_new = regions.regularize_contour(outline_new)
        burrow.outline = outline_new
        return burrow
    
    
    def find_burrows(self):
        """ locates burrows by combining the information of the ground_mask
        profile and the explored area """

        # reset the current burrow model
        burrows_mask = self._cache['image_uint8']
        burrows_mask.fill(0)

        # estimate the new burrow mask        
        potential_burrows = self.get_potential_burrows_mask()
        labels, num_features = ndimage.measurements.label(potential_burrows)
            
        # iterate through all features that have been found
        for label in xrange(1, num_features + 1):
            # disregard features with a small area
            area = np.sum(labels == label)
            if area < self.params['burrows/area_min']:
                continue
             
            # get the burrow object from the contour of region
            burrow = self.get_burrow_from_mask(labels == label)

            # add the unrefined burrow to the debug video
            if 'video' in self.debug:
                self.debug['video'].add_polygon(burrow.outline, 'w',
                                                is_closed=True, width=2)

            # refine the burrows by fitting
            burrow_length = burrow.get_length(self.ground)
            burrow_width = burrow.area/burrow_length if burrow_length > 0 else 0
            min_length = self.params['burrows/fitting_length_threshold']
            max_width = self.params['burrows/fitting_width_threshold']
            if (burrow_length > min_length and burrow_width < max_width):
                burrow = self.refine_long_burrow(burrow)
            else:
                burrow = self.refine_bulky_burrow(burrow)
            
            # add the burrow to our result list if it is valid
            if burrow.is_valid:
                # add the burrow to the current mask
                cv2.fillPoly(burrows_mask, np.array([burrow.outline], np.int32), color=1)
                
                # see whether this burrow can be appended to an active track
                adaptation_interval = self.params['burrows/adaptation_interval']
                for burrow_track in self.result['burrows/tracks']:
                    if (burrow_track.track_end >= self.frame_id - adaptation_interval
                        and burrow_track.last.intersects(burrow.polygon)):
                        
                        burrow_track.append(self.frame_id, burrow)
                        break
                    
                else:
                    # otherwise, start a new burrow track
                    burrow_track = BurrowTrack(self.frame_id, burrow)
                    self.result['burrows/tracks'].append(burrow_track)
                    self.logger.debug('%d: Found new burrow at %s',
                                      self.frame_id, burrow.polygon.centroid)
            
        # degrade information about the mouse position inside burrows
        rate = self.params['explored_area/adaptation_rate_burrows']* \
                self.params['burrows/adaptation_interval']
        cv2.subtract(self.explored_area, rate,
                     dst=self.explored_area,
                     mask=(0 != burrows_mask).astype(np.uint8))
 
        # degrade information about the mouse position outside burrows
        rate = self.params['explored_area/adaptation_rate_outside']* \
                self.params['burrows/adaptation_interval']
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

        # load parameters for video output        
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('debugvideo' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                codec=video_codec, bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                self.debug['video.show'] = ImageShow(self.debug['video'].shape,
                                                     'Debug video' + ' [%s]' % name)

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
        

    def debug_process_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None: 
                debug_video.add_polygon(self.ground.line, is_closed=False,
                                        mark_points=True, color='y')
        
            # indicate the currently active burrow shapes
            time_interval = self.params['burrows/adaptation_interval']
            for burrow_track in self.result['burrows/tracks']:
                if burrow_track.track_end > self.frame_id - time_interval:
                    burrow = burrow_track.last
                    burrow_color = 'red' if burrow.refined else 'orange'
                    debug_video.add_polygon(burrow.outline, burrow_color,
                                            is_closed=True, mark_points=True)
                    debug_video.add_polygon(burrow.get_centerline(self.ground),
                                            burrow_color, is_closed=False,
                                            width=2, mark_points=True)
        
            # indicate the mouse position
            if len(self.tracks) > 0:
                for obj in self.tracks:
                    if self.result['mouse/moved_first_in_frame'] is None:
                        obj_color = 'r'
                    elif obj.is_moving():
                        obj_color = 'w'
                    else:
                        obj_color = 'b'
                    track = obj.get_track()
                    if len(track) > 1000:
                        track = track[-1000:]
                    debug_video.add_polygon(track, '0.5', is_closed=False)
                    debug_video.add_circle(obj.last.pos,
                                           self.params['mouse/model_radius'],
                                           obj_color, thickness=1)
                
            else: # there are no current tracks
                for mouse_pos in self._mouse_pos_estimate:
                    debug_video.add_circle(mouse_pos,
                                           self.params['mouse/model_radius'],
                                           'k', thickness=1)
            
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            debug_video.add_text('#objects:%d' % self.debug['object_count'],
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
                self.debug['video.show'].show(debug_video.frame)
                
        if 'difference.video' in self.debug:
            diff = frame.astype(int, copy=False) - self.background + 128
            diff = np.clip(diff, 0, 255).astype(np.uint8, copy=False)
            self.debug['difference.video'].write_frame(diff)
            self.debug['difference.video'].add_text(str(self.frame_id),
                                                    (20, 20), anchor='top')   
                
        if 'background.video' in self.debug:
            self.debug['background.video'].write_frame(self.background)
            self.debug['background.video'].add_text(str(self.frame_id),
                                                    (20, 20), anchor='top')   

        if 'explored_area.video' in self.debug:
            debug_video = self.debug['explored_area.video']
             
            # set the background
            debug_video.set_frame(128*np.clip(self.explored_area, 0, 1))
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_polygon(self.ground.line, is_closed=False, color='y')
                debug_video.add_points(self.ground.line, radius=2, color='y')

            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   


    def debug_finalize(self):
        """ close the video streams when done iterating """
        # close the window displaying the video
        if 'video.show' in self.debug:
            self.debug['video.show'].close()
        
        # close the open video streams
        for i in ('video', 'difference.video', 'background.video', 'explored_area.video'):
            if i in self.debug:
                try:
                    self.debug[i].close()
                except IOError:
                    self.logger.exception('Error while writing out the debug video') 

        # remove all windows that may have been opened
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass #< some builds of openCV do not implement destroyAllWindows()
            
    
