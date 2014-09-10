'''
Created on Aug 5, 2014

@author: zwicker

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

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import leastsq
from scipy.spatial import distance
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



class FirstPass(DataHandler):
    """
    analyzes mouse movies
    """
    logging_mode = 'create'    
    
    def __init__(self, name='', parameters=None, debug_output=None):
        """ initializes the whole mouse tracking and prepares the video filters """
        
        # initialize the data handler
        super(FirstPass, self).__init__(name, parameters)
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass1')
        
        # setup internal structures that will be filled by analyzing the video
        self._cache = {}               # cache that some functions might want to use
        self.debug = {}                # dictionary holding debug information
        self.background = None         # current background model
        self.ground = None             # current model of the ground profile
        self.burrows_mask = None       # current mask for found burrows
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
                                               'size': '%d x %d' % self.video.size,
                                               'fps': self.video.fps})
        
        self.data['analysis-status'] = 'Loaded video'            

    
    def process_video(self):
        """ processes the entire video """
        self.log_event('Pass 1 - Started initializing the video analysis.')
        
        # restrict the video to the region of interest (the cage)
        self.video, cropping_rect = self.crop_video_to_cage(self.video)
        self.data.create_child('video/analyzed', {'frame_count': self.video.frame_count,
                                                  'region_cage': cropping_rect,
                                                  'size': '%d x %d' % self.video.size,
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
            self.logger.info('Tracking has been interrupted by user.')
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
        self.result['burrows/data'] = BurrowTrackList()

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
            
        w = self.params['mouse/model_radius']
        self._cache['get_potential_burrows_mask.kernel_large'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w+1, 2*w+1))
        self._cache['get_potential_burrows_mask.kernel_small'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        w = self.params['burrows/width']//2
        self._cache['update_burrows_mask.kernel'] = \
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w+1, 2*w+1))
        
        # setup more cache variables
        video_shape = (self.video.size[1], self.video.size[0]) 
        self.explored_area = np.zeros(video_shape, np.double)
        self._cache['background.mask'] = np.ones(video_shape, np.double)
        self.burrows_mask = np.zeros(video_shape, np.uint8)
  

    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """

        sigma = self.params['video/blur_radius']
        blur_kernel = cv2.getGaussianKernel(3*sigma, sigma=sigma)

        # iterate over the video and analyze it
        for self.frame_id, frame in enumerate(display_progress(video)):
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
                
            # blur frame - if the frame is contiguous in memory, we don't need to make a copy
            frame_blurred = np.ascontiguousarray(frame)
            cv2.sepFilter2D(frame_blurred, cv2.CV_8U, blur_kernel, blur_kernel, dst=frame_blurred)
            
            if self.frame_id == self.params['video/initial_adaptation_frames']:
                # prepare the main analysis
                # estimate colors of sand and sky
                self.find_color_estimates(frame_blurred)
                
                # estimate initial ground profile
                self.logger.debug('Find the initial ground profile.')
                self.find_initial_ground(frame_blurred)
        
            elif self.frame_id > self.params['video/initial_adaptation_frames']:
                # do the main analysis
                if self.frame_id % self.params['colors/adaptation_interval'] == 0:
                    self.find_color_estimates(frame_blurred)

                # find a binary image that indicates movement in the frame
                mask_moving = self.find_moving_features(frame_blurred)
    
                # identify objects from this
                self.find_objects(frame_blurred, mask_moving)
                
                # use the background to find the current ground profile and burrows
                if self.frame_id % self.params['ground/adaptation_interval'] == 0:
                    self.refine_ground(self.background)
                    ground = GroundProfile(self.frame_id, self.ground)
                    self.result['ground/profile'].append(ground)
        
                if self.frame_id % self.params['burrows/adaptation_interval'] == 0:
                    self.find_burrows(mask_moving)
                    
            # update the background model
            self.update_background_model(frame_blurred)
                
            # store some information in the debug dictionary
            self.debug_process_frame(frame_blurred)
                         
                    
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
        cage_mask = regions.get_largest_region(binarized)
        
        # find an enclosing rectangle, which usually overestimates the cage bounding box
        rect_large = regions.find_bounding_box(cage_mask)
         
        # crop image to this rectangle, which should surely contain the cage 
        image = image[regions.rect_to_slices(rect_large)]

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
        return regions.corners_to_rect(p1, p2)

  
    def crop_video_to_cage(self, video):
        """ crops the video to a suitable cropping rectangle given by the cage """
        # find the cage in the blurred image
        blurred_frame = cv2.GaussianBlur(video[0], ksize=(0, 0),
                                         sigmaX=self.params['video/blur_radius'])
        rect_cage = self.find_cage(blurred_frame)
        
        # determine the rectangle of the cage in global coordinates
        width = rect_cage[2] - rect_cage[2] % 2   # make sure its divisible by 2
        height = rect_cage[3] - rect_cage[3] % 2  # make sure its divisible by 2
        # Video dimensions should be divisible by two for some codecs

        if not (self.params['cage/width_min'] < width < self.params['cage/width_max'] and
                self.params['cage/height_min'] < height < self.params['cage/height_max']):
            raise RuntimeError('The cage bounding box (%dx%d) is out of the limits.' % (width, height)) 
        
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
        sky_mask = regions.get_largest_region(1 - binarized).astype(np.uint8, copy=False)*255

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
            mask = self._cache['background.mask']
            template = self._cache['mouse.template']
            mask.fill(1)
            
            # cut out holes from the mask for each mouse estimate
            for mouse_pos in self._mouse_pos_estimate:
                # get the slices required for comparing the template to the image
                t_s, i_s = regions.get_overlapping_slices(mouse_pos, template.shape,
                                                          frame.shape)
                mask[i_s[0], i_s[1]] *= 1 - template[t_s[0], t_s[1]]
                
        else:
            # disable the mask if no mouse is known
            mask = 1

        # adapt the background to current frame, but only inside the mask 
        self.background += (self.params['background/adaptation_rate']  # adaptation rate 
                            *mask                                      # mask 
                            *(frame - self.background))                # difference to current frame

                        
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
        mask_moving = 255*mask_moving.astype(np.uint8, copy=False)

        kernel = self._cache['find_moving_features.kernel']
        # perform morphological opening to remove noise
        cv2.morphologyEx(mask_moving, cv2.MORPH_OPEN, kernel, dst=mask_moving)    
        # perform morphological closing to join distinct features
        cv2.morphologyEx(mask_moving, cv2.MORPH_CLOSE, kernel, dst=mask_moving)

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
        objects_found = self.find_objects_in_binary_image(labels, num_features)

        # check if there are previous tracks        
        if len(self.tracks) == 0:
            self.tracks = [ObjectTrack(self.frame_id, obj) for obj in objects_found]
            
            return # there is nothing to do anymore
            
        # calculate the distance between new and old objects
        dist = distance.cdist([obj.pos for obj in objects_found],
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
            self.logger.debug('%d: New mouse track at %s', self.frame_id, objects_found[i_f].pos)
            # start new track
            track = ObjectTrack(self.frame_id, objects_found[i_f])
            self.tracks.append(track)
        
        assert len(self.tracks) == len(objects_found)
        
    
    def update_explored_area_objects(self, tracks, labels, num_features):
        """ update the explored area using the found objects """
        # add new information
        for track in self.tracks:
            self.explored_area[labels == track.objects[-1].label] = 1

        # the burrow color is similar to the sky color, because both are actually
        # the background behind the cage
        color_sand, color_burrow = self.result['colors/sand'], self.result['colors/sky']
        # normalize frame such that burrows are 0 and sand is 1
        frame_normalized = (self.background - color_burrow)/(color_sand - color_burrow)

        # degrade information about the mouse position inside the burrows
        self.explored_area[0 != self.burrows_mask] -= \
            frame_normalized[0 != self.burrows_mask] \
            *self.params['explored_area/adaptation_rate_burrows']
            
        # degrade information about the mouse position outside the burrows
        self.explored_area[0 == self.burrows_mask] -= \
            frame_normalized[0 == self.burrows_mask] \
            *self.params['explored_area/adaptation_rate_outside']
            
        # restrict the range
        np.clip(self.explored_area, 0, 1, out=self.explored_area)

    
    def find_objects(self, frame, mask_moving):
        """ adapts the current mouse position, if enough information is available """

        # find all distinct features and label them
        labels, num_features = ndimage.measurements.label(mask_moving)
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
            self._handle_object_tracks(frame, labels, num_features)

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
        
            # keep track of the regions that the mouse (or other objects) explored
            self.update_explored_area_objects(self.tracks, labels, num_features)
                
                
    #===========================================================================
    # FINDING THE GROUND PROFILE
    #===========================================================================


    def find_rough_ground(self, frame):
        """ determines an estimate of the ground profile from a single frame """
        
        # remove 10%/15% of each side of the frame
        h = int(0.15*frame.shape[0])
        w = int(0.10*frame.shape[1])
        image_center = frame[h:-h, w:-w]
        
        # binarize frame
        cv2.threshold(image_center, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=image_center)
        
        # do morphological opening and closing to smooth the profile
        s = 5*self.params['burrows/width']
        ys, xs = np.ogrid[-s:s+1, -s:s+1]
        kernel = (xs**2 + ys**2 <= s**2).astype(np.uint8, copy=False)

        # widen the mask
        mask = cv2.copyMakeBorder(image_center, s, s, s, s, cv2.BORDER_REPLICATE)
        # make sure their is sky on the top
        mask[:s + h, :] = 0
        # make sure their is sand at the bottom
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
        ground = curves.make_curve_equidistant(points, spacing)
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
                angle = np.arctan2(dp[0], dp[1])
                
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
            
        self.logger.info('We found a ground profile of length %g after %d iterations',
                         curves.curve_length(self.ground), iterations)
        
    
    def get_ground_mask(self, color=255):
        """ returns a binary mask distinguishing the ground from the sky """
        # TODO: Think about caching this result
        
        # build a mask with potential burrows
        width, height = self.video.size
        ground_mask = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current ground_mask profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground
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
        margin = self.params['burrows/cage_margin']
        explored_area[: margin, :] = 0
        explored_area[-margin:, :] = 0
        explored_area[:, : margin] = 0
        explored_area[:, -margin:] = 0
        
        # remove all regions that are less than a threshold distance away from the ground line
        # and which are not connected to any other region
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
        template_width = 2*edge_width # odd width preferred
        
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
        """ refines an elongated burrow by doing linescans perpendicular to
        its centerline """
        # keep the points close to the ground line
        ground_line = geometry.LineString(np.array(self.ground, np.double))
        outline_new = sorted([p.coords[0]
                              for p in geometry.MultiPoint(burrow.outline)
                              if p.distance(ground_line) < self.params['burrows/ground_point_distance']])

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
            profile = image.line_scan(self.background.astype(np.uint8, copy=False), p_a, p_b, 3)
            
            # find the transition points by considering slopes
            k_l = self.find_burrow_edge(profile, direction='down')
            k_r = self.find_burrow_edge(profile, direction='up')

            if k_l is not None and k_r is not None:
                d_l, d_r = scan_length - k_l, scan_length - k_r
                # d_l and d_r are the distance from p, where d_l > 0 and d_r < 0 accounting for direction

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
                profile = image.line_scan(self.background.astype(np.uint8, copy=False),
                                          point_anchor, p1e, 3)

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
        ground_line = geometry.LineString(np.array(self.ground, np.double))
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
                profile = image.line_scan(self.background.astype(np.uint8, copy=False), p_a, p_b, 3)
                k = self.find_burrow_edge(profile, direction='up')

                if k is not None:
                    d = scan_length - k
                    p = (p[0] + d*dy, p[1] - d*dx)
            
            outline_new.append(p)

        outline_new = regions.regularize_contour(outline_new)
        burrow.outline = outline_new
        return burrow
    
    
    def find_burrows(self, mask_moving):
        """ locates burrows by combining the information of the ground_mask profile
        and the explored area """

        # reset the current burrow model
        self.burrows_mask.fill(0)

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
                self.debug['video'].add_polygon(burrow.outline, 'w', is_closed=True, width=2)

            # refine the burrows by fitting
            if (burrow.eccentricity > self.params['burrows/fitting_eccentricity_threshold'] and 
                    burrow.get_length(self.ground) > self.params['burrows/fitting_length_threshold']):
                burrow = self.refine_long_burrow(burrow)
            else:
                burrow = self.refine_bulky_burrow(burrow)
            
            # add the burrow to our result list if it is valid
            if burrow.is_valid:
                # add the burrow to the current mask
                cv2.fillPoly(self.burrows_mask, np.array([burrow.outline], np.int32), color=255)
                
                # see whether this burrow is already known
                for burrow_track in self.result['burrows/data']:
                    if burrow_track.last.intersects(burrow.polygon):
                        burrow_track.append(self.frame_id, burrow)
                        break
                    
                else:
                    # otherwise, start a new burrow track
                    burrow_track = BurrowTrack(self.frame_id, burrow)
                    self.result['burrows/data'].append(burrow_track)
                    self.logger.debug('%d: Found new burrow at %s',
                                      self.frame_id, burrow.polygon.centroid)
            

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
            debug_file = self.get_filename('video' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                codec=video_codec, bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                self.debug['video.show'] = ImageShow(self.debug['video'].shape,
                                                     'Debug video' + ' [%s]'%self.name if self.name else '')

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
                debug_video.add_polygon(self.ground, is_closed=False, mark_points=True, color='y')
        
            # indicate the currently active burrow shapes
            for burrow_track in self.result['burrows/data']:
                if burrow_track.last_seen > self.frame_id - self.params['burrows/adaptation_interval']:
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
                    debug_video.add_polygon(obj.get_track(), '0.5', is_closed=False)
                    debug_video.add_circle(obj.last.pos, self.params['mouse/model_radius'], obj_color, thickness=1)
                
            else: # there are no current tracks
                for mouse_pos in self._mouse_pos_estimate:
                    debug_video.add_circle(mouse_pos, self.params['mouse/model_radius'], 'k', thickness=1)
            
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            debug_video.add_text('#objects:%d' % self.debug['object_count'], (120, 20), anchor='top')
            debug_video.add_text(self.debug['video.mark.text1'], (300, 20), anchor='top')
            debug_video.add_text(self.debug['video.mark.text2'], (300, 50), anchor='top')
            if self.debug.get('video.mark.rect1'):
                debug_video.add_rectangle(self.debug['rect1'])
            if self.debug.get('video.mark.points'):
                debug_video.add_points(self.debug['video.mark.points'], radius=4, color='y')
            if self.debug.get('video.mark.highlight', False):
                debug_video.add_rectangle((0, 0, self.video.size[0], self.video.size[1]), 'w', 10)
                self.debug['video.mark.highlight'] = False
            
            if 'video.show' in self.debug:
                self.debug['video.show'].show(debug_video.frame)
                
        if 'difference.video' in self.debug:
            diff = np.clip(frame.astype(int, copy=False) - self.background + 128, 0, 255)
            self.debug['difference.video'].write_frame(diff.astype(np.uint8, copy=False))
            self.debug['difference.video'].add_text(str(self.frame_id), (20, 20), anchor='top')   
                
        if 'background.video' in self.debug:
            self.debug['background.video'].write_frame(self.background)
            self.debug['background.video'].add_text(str(self.frame_id), (20, 20), anchor='top')   

        if 'explored_area.video' in self.debug:
            debug_video = self.debug['explored_area.video']
             
            # set the background
            debug_video.set_frame(128*self.explored_area)
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_polygon(self.ground, is_closed=False, color='y')
                debug_video.add_points(self.ground, radius=2, color='y')

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
        cv2.destroyAllWindows()
            
    
