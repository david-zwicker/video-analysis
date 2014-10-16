'''
Created on Oct 2, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the third pass of the algorithm
'''

from __future__ import division

import math
import time

import cv2
import numpy as np
from shapely import affinity, geometry

from .data_handler import DataHandler
from .objects import mouse
from .objects.burrow2 import Burrow, BurrowTrack, BurrowTrackList
from video.analysis import curves, regions
from video.filters import FilterCrop
from video.io import ImageWindow
from video.utils import display_progress
from video.composer import VideoComposer

import debug  # @UnusedImport


class ThirdPass(DataHandler):
    """ class containing methods for the third pass, which locates burrows
    based on the mouse movement """
    
    def __init__(self, name='', parameters=None, **kwargs):
        super(ThirdPass, self).__init__(name, parameters, **kwargs)
        if kwargs.get('initialize_parameters', True):
            self.log_event('Pass 3 - Initialized the third pass analysis.')
        self.initialize_pass()
        

    @classmethod
    def from_second_pass(cls, second_pass):
        """ create the object directly from the second pass """
        # create the data and copy the data from first_pass
        obj = cls(second_pass.name, initialize_parameters=False)
        obj.data = second_pass.data
        obj.params = obj.data['parameters']
        obj.result = obj.data.create_child('pass2')

        # close logging handlers and other files        
        second_pass.close()
        
        # initialize parameters
        obj.initialize_parameters()
        obj.initialize_pass()
        obj.log_event('Pass 3 - Initialized the third pass analysis.')
        return obj
    
    
    def initialize_pass(self):
        """ initialize values necessary for this run """
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass3')
        self.result['code_status'] = self.get_code_status()
        self.debug = {}
        if self.params['debug/output'] is None:
            self.debug_output = []
        else:
            self.debug_output = self.params['debug/output']
            

    def process(self):
        """ processes the entire video """
        self.log_event('Pass 3 - Started initializing the video analysis.')
        
        self.setup_processing()
        self.debug_setup()

        self.log_event('Pass 3 - Started iterating through the video with %d frames.' %
                       self.video.frame_count)
        self.data['analysis-status'] = 'Initialized video analysis'
        start_time = time.time()            
        
        try:
            # skip the first frame, since it has already been analyzed
            self._iterate_over_video(self.video)
                
        except (KeyboardInterrupt, SystemExit):
            # abort the video analysis
            self.video.abort_iteration()
            self.log_event('Pass 3 - Analysis run has been interrupted.')
            self.data['analysis-status'] = 'Partly finished third pass'
            
        else:
            # finished analysis successfully
            self.log_event('Pass 3 - Finished iterating through the frames.')
            self.data['analysis-status'] = 'Finished third pass'
            
        finally:
            # cleanup in all cases 
            self.add_processing_statistics(time.time() - start_time)        
                        
            # cleanup and write out of data
            self.video.close()
            self.debug_finalize()
            self.write_data()

            
    def add_processing_statistics(self, time):
        """ add some extra statistics to the results """
        frames_analyzed = self.frame_id + 1
        self.data['pass3/video/frames_analyzed'] = frames_analyzed
        self.result['statistics/processing_time'] = time
        self.result['statistics/processing_fps'] = frames_analyzed/time


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        # load the video
        cropping_rect = self.data['pass1/video/cropping_rect'] 
        video_info = self.load_video(cropping_rect=cropping_rect)
        
        self.data.create_child('pass3/video', video_info)
        del self.data['pass3/video/filecount']
        
        cropping_cage = self.data['pass1/video/cropping_cage']
        if cropping_cage is not None:
            self.video = FilterCrop(self.video, rect=cropping_cage)
            
        video_info = self.data['pass3/video']
        video_info['cropping_cage'] = cropping_cage
        video_info['frame_count'] = self.video.frame_count
        video_info['size'] = '%d x %d' % tuple(self.video.size),
        
        # get first frame, which will not be used in the iteration
        first_frame = self.video.get_next_frame()
        
        # initialize data structures
        self.frame_id = -1
        self.background = first_frame.astype(np.double)
        self.ground_idx = None  #< index of the ground point where the mouse entered the burrow
        self.mouse_trail = None #< line from this point to the mouse (along the burrow)
        
        if self.params['burrows/enabled_pass3']:
            self.result['burrows/tracks'] = BurrowTrackList()

        
    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """
        
        # load data from previous passes
        mouse_track = self.data['pass2/mouse_trajectory']
        ground_profile = self.data['pass2/ground_profile']

        # iterate over the video and analyze it
        for self.frame_id, frame in enumerate(display_progress(video)):
            
            # adapt the background to current frame 
            adaptation_rate = self.params['background/adaptation_rate']
            self.background += adaptation_rate*(frame - self.background)
            
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
            
            # retrieve data for current frame
            self.mouse_pos = mouse_track.pos[self.frame_id, :]
            self.ground = ground_profile.get_ground_profile(self.frame_id)

            # find out where the mouse currently is        
            self.classify_mouse_state(mouse_track)
            
            if (self.params['burrows/enabled_pass3'] and 
                self.frame_id % self.params['burrows/adaptation_interval'] == 0):
                # find the burrow from the mouse trail
                self.locate_burrows()

            # store some information in the debug dictionary
            self.debug_process_frame(frame, mouse_track)
            
            if self.frame_id % 1000 == 0:
                self.logger.debug('Analyzed frame %d', self.frame_id)

    
    #===========================================================================
    # MOUSE TRACKING
    #===========================================================================


    def extend_mouse_trail(self):
        """ extends the mouse trail using the current mouse position """
        # check points starting from the back of the trail
        last_point = len(self.mouse_trail) - 1
        while last_point >= 1:
            p2 = self.mouse_trail[last_point-1] 
            p1 = self.mouse_trail[last_point]
            angle = curves.angle_between_points(p2, p1, self.mouse_pos)
            if np.abs(angle) < np.pi/2:
                # we found the last point to keep
                dist = curves.point_distance(p2, self.mouse_pos)
                if dist > self.params['burrows/centerline_segment_length']:
                    last_point += 1
                break
            else:
                # remove this point from the mouse trail
                last_point -= 1
    
        del self.mouse_trail[max(1, last_point):]
        self.mouse_trail.append(self.mouse_pos)


    def classify_mouse_state(self, mouse_track):
        """ classifies the mouse in the current frame """
        if (not np.all(np.isfinite(self.mouse_pos)) or
            self.ground is None):
            
            # Not enough information to do anything
            self.mouse_trail = None
            return
        
        # initialize variables
        state = {}
        margin = self.params['mouse/model_radius']/2
                
        # compare y value of mouse and ground (y-axis points down)
        if self.mouse_pos[1] > self.ground.get_y(self.mouse_pos[0]) + margin:
            state['underground'] = True
            
            # handle mouse trail
            if self.mouse_trail is None:
                # start a new mouse trail and initialize it with the                         
                # ground point closest to the mouse       
                ground_point = curves.get_projection_point(self.ground.linestring,
                                                           self.mouse_pos)
                self.mouse_trail = [ground_point, self.mouse_pos]

            else:
                self.extend_mouse_trail()
                
            # get distance the mouse is under ground
            ground_dist = -curves.curve_length(self.mouse_trail)
            
            # score the burrow based on its entry point
            if self.ground_idx is None:
                # only necessary if mouse starts inside burrow
                dist = np.linalg.norm(self.ground.points - self.mouse_pos[None, :], axis=1)
                self.ground_idx = np.argmin(dist)
            entry_point = self.ground.points[self.ground_idx]
            if entry_point[1] > self.ground.midline:
                state['location'] = 'burrow'
            else:
                state['location'] = 'dimple'

        else: 
            state['underground'] = False
            mouse_radius = self.params['mouse/model_radius']
            if self.mouse_pos[1] + 2*mouse_radius < self.ground.get_y(self.mouse_pos[0]):
                state['location'] = 'air'
            elif self.mouse_pos[1] < self.ground.midline:
                state['location'] = 'hill'
            else:
                state['location'] = 'valley'

            self.mouse_trail = None
            # get index of the ground line
            dist = np.linalg.norm(self.ground.points - self.mouse_pos[None, :], axis=1)
            self.ground_idx = np.argmin(dist)
            # get distance from ground line
            mouse_point = geometry.Point(self.mouse_pos)
            ground_dist = self.ground.linestring.distance(mouse_point)
            # report the distance as negative, if the mouse is under the ground line
            if self.mouse_pos[1] > self.ground.get_y(self.mouse_pos[0]):
                ground_dist *= -1
            
        # set the mouse state
        mouse_track.set_state(self.frame_id, state, self.ground_idx, ground_dist)

    
    #===========================================================================
    # BURROW TRACKING
    #===========================================================================


    def get_ground_mask(self):
        """ returns a binary mask distinguishing the ground from the sky """
        # build a mask with potential burrows
        width, height = self.video.size
        mask_ground = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current mask_ground profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground.points
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(mask_ground, np.array([ground_points], np.int32), color=255)

        return mask_ground


    def get_burrow_contour_from_mask(self, mask, offset=None):
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
            contour = regions.regularize_contour(contour)
        
#         if offset[0]:
#             debug.show_shape(geometry.LinearRing(contour),
#                              background=mask, wait_for_key=False)
        
        # create the burrow based on the contour
        if contour:
            contour = curves.translate_points(contour,
                                              xoff=offset[0],
                                              yoff=offset[1])
            try:
                return contour
            except ValueError as err:
                raise RuntimeError(err.message)
            
        else:
            raise RuntimeError('Contour is not a simple polygon')
    
    
    
    def refine_elongated_burrow_centerline(self, burrow):
        spacing = self.params['burrows/centerline_segment_length']
        centerline = curves.make_curve_equidistant(burrow.centerline, spacing)
        outline = burrow.outline_ring
        
        # iterate over all but the boundary points
        ray_len = 10000

        # determine the boundary points for each centerline point
        points, dp, boundary = [centerline[0]], [(0, 0)], []
        for k in xrange(1, len(centerline)):
            # get local points and slopes
            if k == len(centerline) - 1:
                p_p, p_m =  centerline[k-1], centerline[k]
                dx, dy = p_m - p_p
            else:
                p_p, p_m, p_n =  centerline[k-1], centerline[k], centerline[k+1]
                dx, dy = p_n - p_p
            dist = math.hypot(dx, dy)
            if dist == 0: #< something went wrong 
                continue #< skip this point
            dx /= dist; dy /= dist

            # determine the points of intersection with the burrow outline         
            p_a = (p_m[0] - ray_len*dy, p_m[1] + ray_len*dx)
            p_b = (p_m[0] + ray_len*dy, p_m[1] - ray_len*dx)
            line = geometry.LineString((p_a, p_b))
            
            # find the intersections between the ray and the burrow outline
            inter = regions.get_intersections(outline, line)

            if len(inter) < 2:
                # not enough information to proceed
                continue
            
            # find the two closest points
            dist = [curves.point_distance(p, p_m) for p in inter]
            k_a = np.argmin(dist)
            p_a = inter[k_a]
            dist[k_a] = np.inf
            p_b = inter[np.argmin(dist)]
            
            # set boundary point
            points.append(p)
            dp.append((-dy, dx))
            boundary.append((p_a, p_b))


        points = np.array(points)
        dp = np.array(dp)
        boundary = np.array(boundary)
# 
#         def energy_curvature(ps):
#             energy = 0
#             for k in xrange(1, len(ps) - 1):
#                 p_p, p_c, p_n = ps[k-1:k+2]
#                 a = curves.point_distance(p_p, p_c)
#                 b = curves.point_distance(p_c, p_n)
#                 c = curves.point_distance(p_n, p_p)
#  
#                 # determine curvature of circle through the three points
#                 A = regions.triangle_area(a, b, c)
#                 curvature = 4*A/(a*b*c)*spacing#(a + b)
#                 energy += curvature
#             return 50*energy
#          
#         def energy_outline(ps):
#             energy = 0
#             for k, p in enumerate(ps[1:]):
#                 a = curves.point_distance(p, boundary[k][0])
#                 b = curves.point_distance(p, boundary[k][1])
#                 energy += np.hypot(a, b)
#             return energy
#                  
#         ds = np.zeros(len(points))
#         def energy_snake(data):
#             ds[1:] = data
#             ps = points + ds[:, None]*dp
#             print 'curv', energy_curvature(ps) 
#             print 'outl', energy_outline(ps)
#             return energy_curvature(ps) + energy_outline(ps)
#  
#         # fit the simple model to the line scan profile
#         res = optimize.fmin(energy_snake, x0=np.zeros(len(points)-1))
#  
#         ds[1:] = res
#         points_i = (points + ds[:, None]*dp)[1:]

        # get the points, which are neither at the exit nor the front
        points = np.mean(boundary, axis=1).tolist()
        
        # extend the centerline to the burrow front
        angle = np.arctan2(-dp[-1][0], dp[-1][1])
        angles = np.linspace(angle - np.pi/3, angle + np.pi/3, 32)
        p_far, _, _ = regions.get_farthest_ray_intersection(points[-1], angles, outline)

        if p_far is not None:
            points = points + [p_far]
            if curves.point_distance(points[-1], points[-2]) < spacing:
                del points[-2]
            
        # find a better approximation for the burrow exit
        if len(points) >= 2:
            p_a, p_b = points[1], points[0]
            dx, dy = p_b[0] - p_a[0], p_b[1] - p_a[1]
            angle = np.arctan2(dy, dx)
            angles = np.linspace(angle - np.pi/4, angle + np.pi/4, 16)
            p_near, _, _ = regions.get_nearest_ray_intersection(points[0], angles, outline)
            
            if p_near is not None:
                points = [p_near] + points
            
        burrow.centerline = points
    
    
    def refine_burrow_centerline(self, burrow):
        """ refines the centerline of a burrow """
        # check the percentage of outline points close to the ground
        spacing = self.params['burrows/ground_point_distance']
        outline = curves.make_curve_equidistant(burrow.outline, spacing)
        groundline = self.ground.linestring

        num_close = 0
        dist_far, p_far = 0, None
        for p in outline:
            dist = groundline.distance(geometry.Point(p))
            if dist < spacing:
                num_close += 1
            if dist > dist_far:
                dist_far = dist
                p_far = p
                
        shape_threshold = self.params['burrows/shape_threshold_fraction']
        if num_close < shape_threshold*len(outline):
            # burrow has few points close to the ground
            self.refine_elongated_burrow_centerline(burrow)
            burrow.elongated = True
            
        else:
            # burrow is close to the ground
            angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
            p_near, _, _ = regions.get_nearest_ray_intersection(p_far, angles, groundline)
            burrow.elongated = False
            
            self.centerline = [p_near, p_far]
    
    
    def refine_burrow(self, burrow):
        """ refine burrow by thresholding background image using the GrabCut
        algorithm """
        mask_ground = self.get_ground_mask()
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
        mask = np.zeros_like(mask_ground)        
        
        centerline = curves.translate_points(burrow.centerline,
                                             xoff=-rect[0],
                                             yoff=-rect[1])

        if burrow.outline is not None and len(centerline) > 2:
            centerline = geometry.LineString(centerline[:-1])
        else:
            centerline = geometry.LineString(centerline)
        
        def add_to_mask(color, buffer_radius):
            """ adds the region around the centerline to the mask """
            polygon = centerline.buffer(buffer_radius)
            coords = np.asarray(polygon.exterior.xy, np.int).T 
            cv2.fillPoly(mask, [coords], color=int(color))

        # setup the mask for the GrabCut algorithm
        mask.fill(cv2.GC_BGD)
        add_to_mask(cv2.GC_PR_BGD, 2*self.params['burrows/width'])
        add_to_mask(cv2.GC_PR_FGD, self.params['burrows/width'])
        add_to_mask(cv2.GC_FGD, self.params['burrows/width_min']/2)

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
        
        # make sure that the burrow is under ground
        mask[mask_ground == 0] = 0
        
        # find the burrow from the mask
        try:
            contour = self.get_burrow_contour_from_mask(mask.astype(np.uint8),
                                                        offset=rect[:2])
            burrow.outline = contour
            self.refine_burrow_centerline(burrow)
#             burrow.get_centerline(self.ground, p_exit=burrow.centerline[0])
            burrow.refined = True
        except RuntimeError as err:
            self.logger.debug('%d: Invalid burrow from GrabCut: %s',
                              self.frame_id, err.message)
        
        return burrow
    
    
    def active_burrows(self, time_interval=None):
        """ returns a generator to iterate over all active burrows """
        if time_interval is None:
            time_interval = self.params['burrows/adaptation_interval']
        for k, burrow_track in enumerate(self.result['burrows/tracks']):
            if burrow_track.track_end >= self.frame_id - time_interval:
                yield k, burrow_track.last

    
    def locate_burrows(self):
        """ locates burrows based on the mouse's movement """
        burrow_tracks = self.result['burrows/tracks']
        
        # copy all burrows to the this frame
        for burrow_id, burrow in self.active_burrows():
            burrow_tracks[burrow_id].append(self.frame_id, burrow)
        
        # check whether the mouse is in a burrow
        if self.mouse_trail is None:
            # mouse is above ground => all burrows are without mice 
            burrow_with_mouse = -1
        
        else:
            # mouse entered a burrow
            trail_length = curves.curve_length(self.mouse_trail)
            
            # check if we already know this burrow
            for burrow_with_mouse, burrow in self.active_burrows(time_interval=0):
                # determine whether we are inside this burrow
                trail_line = geometry.LineString(self.mouse_trail)
                if burrow.outline is not None:
                    dist = burrow.polygon.distance(trail_line)
                    mouse_close_to_burrow = (dist < self.params['burrows/width']) 
                else:
                    dist = burrow.linestring.distance(trail_line)
                    mouse_close_to_burrow = (dist < 2*self.params['burrows/width']) 
                     
                if mouse_close_to_burrow:
                    burrow.refined = False
                    if trail_length > burrow.length:
                        # update the centerline estimate
                        burrow.centerline = self.mouse_trail[:] #< copy list
                    break #< burrow_with_mouse contains burrow id
            else:
                # create the burrow, since we don't know it yet
                burrow_track = BurrowTrack(self.frame_id, Burrow(self.mouse_trail[:]))
                burrow_tracks.append(burrow_track)
                burrow_with_mouse = len(burrow_tracks) - 1

        # refine burrows
        refined_burrows = []
        for k, burrow in self.active_burrows(time_interval=0):
            # skip burrows with mice in them
            if k == burrow_with_mouse:
                continue
            if not burrow.refined:
                old_shape = burrow.polygon
                while True:
                    burrow = self.refine_burrow(burrow)
                    area = burrow.area
                    # check whether the burrow is small
                    if area < 2*self.params['burrows/area_min']:
                        break
                    # check whether the burrow has changed significantly
                    diff = old_shape.symmetric_difference(burrow.polygon) 
                    if diff.area / area < 0.1:
                        break 
                    old_shape = burrow.polygon
                
                refined_burrows.append(k)
                
#         # check for overlapping burrows
#         for id1 in reversed(refined_burrows):
#             burrow1 = burrows[id1]
#             # check against all the other burrows
#             for id2, burrow2 in enumerate(burrows):
#                 if id1 != id2 and burrow1.intersects(burrow2):
                    
            
            
            

    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        self.debug['video.mark.text1'] = ''
        self.debug['video.mark.text2'] = ''

        # load parameters for video output        
        video_output_period = int(self.params['output/video/period'])
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('pass3' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                output_period=video_output_period,
                                                codec=video_codec, bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 3 [%s]' % name,
                                           multiprocessing=self.params['debug/use_multiprocessing'],
                                           position=position)
                self.debug['video.show'] = image_window


    def debug_process_frame(self, frame, mouse_track):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_line(self.ground.points, is_closed=False,
                                     mark_points=True, color='y')
        
            # indicate the mouse position
            trail_length = self.params['output/video/mouse_trail_length']
            time_start = max(0, self.frame_id - trail_length)
            track = mouse_track.pos[time_start:self.frame_id, :]
            if len(track) > 0:
                debug_video.add_line(track, '0.5', is_closed=False)
                debug_video.add_circle(track[-1], self.params['mouse/model_radius'],
                                       'w', thickness=1)
                
            # indicate the currently active burrow shapes
            if self.params['burrows/enabled_pass3']:
                for _, burrow in self.active_burrows():
                    if hasattr(burrow, 'elongated') and burrow.elongated:
                        burrow_color = 'r'
                    else:
                        burrow_color = '#FF7F00' # orange
                    debug_video.add_line(burrow.centerline, burrow_color, is_closed=False,
                                         mark_points=True, width=2)
                    if burrow.outline is not None:
                        debug_video.add_line(burrow.outline, burrow_color, is_closed=True,
                                             mark_points=False, width=1)
                            
            elif self.mouse_trail:
                debug_video.add_line(self.mouse_trail, '0.5', is_closed=False,
                                     mark_points=True, width=2)
                
            # indicate the mouse state
            mouse_state = mouse_track.states[self.frame_id]
            if mouse_state in mouse.STATES_SHORT:
                debug_video.add_text(mouse.STATES_SHORT[mouse_state],
                                     (120, 20), anchor='top')
                
            # add additional debug information
            debug_video.add_text(str(self.frame_id), (20, 20), anchor='top')   
            if 'video.show' in self.debug:
                if debug_video.output_this_frame:
                    self.debug['video.show'].show(debug_video.frame)
                else:
                    self.debug['video.show'].show()


    def debug_finalize(self):
        """ close the video streams when done iterating """
        # close the window displaying the video
        if 'video.show' in self.debug:
            self.debug['video.show'].close()
        
        # close the open video streams
        if 'video' in self.debug:
            try:
                self.debug['video'].close()
            except IOError:
                    self.logger.exception('Error while writing out the debug '
                                          'video') 
            
    