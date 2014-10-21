'''
Created on Oct 2, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the third pass of the algorithm
'''

from __future__ import division

import time

import cv2
import numpy as np
from shapely import geometry

from .data_handler import DataHandler
from video.analysis import image, curves, regions
from video.io import ImageWindow, VideoFile
from video.filters import FilterMonochrome
from video.utils import display_progress
from video.composer import VideoComposer

import debug  # @UnusedImport


class FourthPass(DataHandler):
    """ class containing methods for the third pass, which locates burrows
    based on the mouse movement """
    
    def __init__(self, name='', parameters=None, **kwargs):
        super(FourthPass, self).__init__(name, parameters, **kwargs)
        if kwargs.get('initialize_parameters', True):
            self.log_event('Pass 3 - Initialized the third pass analysis.')
        self.initialize_pass()
        

    @classmethod
    def from_third_pass(cls, third_pass):
        """ create the object directly from the second pass """
        # create the data and copy the data from first_pass
        obj = cls(third_pass.name, initialize_parameters=False)
        obj.data = third_pass.data
        obj.params = obj.data['parameters']
        obj.result = obj.data.create_child('pass2')

        # close logging handlers and other files        
        third_pass.close()
        
        # initialize parameters
        obj.initialize_parameters()
        obj.initialize_pass()
        obj.log_event('Pass 3 - Initialized the third pass analysis.')
        return obj
    
    
    def initialize_pass(self):
        """ initialize values necessary for this run """
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass4')
        self.result['code_status'] = self.get_code_status()
        self.debug = {}
        if self.params['debug/output'] is None:
            self.debug_output = []
        else:
            self.debug_output = self.params['debug/output']
        self._cache = {}
            

    def process(self):
        """ processes the entire video """
        self.log_event('Pass 4 - Started initializing the video analysis.')
        
        self.setup_processing()
        self.debug_setup()

        self.log_event('Pass 4 - Started iterating through the video with %d frames.' %
                       self.video.frame_count)
        self.data['analysis-status'] = 'Initialized video analysis'
        start_time = time.time()            
        
        try:
            # skip the first frame, since it has already been analyzed
            self._iterate_over_video(self.video)
                
        except (KeyboardInterrupt, SystemExit):
            # abort the video analysis
            self.video.abort_iteration()
            self.log_event('Pass 4 - Analysis run has been interrupted.')
            self.data['analysis-status'] = 'Partly finished third pass'
            
        else:
            # finished analysis successfully
            self.log_event('Pass 4 - Finished iterating through the frames.')
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
        self.data['pass4/video/frames_analyzed'] = frames_analyzed
        self.result['statistics/processing_time'] = time
        self.result['statistics/processing_fps'] = frames_analyzed/time


    def setup_processing(self):
        """ sets up the processing of the video by initializing caches etc """
        # load the video
        #cropping_rect = self.data['pass1/video/cropping_rect'] 
        #video_info = self.load_video(cropping_rect=cropping_rect)
        
        video_extension = self.params['output/video/extension']
        filename = self.get_filename('background' + video_extension, 'debug')
        self.video = FilterMonochrome(VideoFile(filename))
        
        video_info = self.data['pass3/video']
        video_info['frame_count'] = self.video.frame_count
        video_info['size'] = '%d x %d' % tuple(self.video.size),
        
        # initialize data structures
        self.frame_id = -1
        self.background_avg = None

        self.burrows = []
        self.burrow_mask = None
        self._cache['image_uint8'] = np.empty(self.video.shape[1:], np.uint8)

        
    def _iterate_over_video(self, video):
        """ internal function doing the heavy lifting by iterating over the video """
        
        # load data from previous passes
        ground_profile = self.data['pass2/ground_profile']
        adaptation_rate = self.params['background/adaptation_rate']

        # iterate over the video and analyze it
        for background_id, frame in enumerate(display_progress(self.video)):
            self.frame_id = background_id * self.params['output/video/period'] 
            
            # adapt the background to current frame
            if self.background_avg is None:
                self.background_avg = frame.astype(np.double)
            else:
                self.background_avg += adaptation_rate*(frame - self.background_avg)
            
            # copy frame to debug video
            if 'video' in self.debug:
                self.debug['video'].set_frame(frame, copy=False)
            
            # retrieve data for current frame
            self.ground = ground_profile.get_ground_profile(self.frame_id)

            # find the changes in the background
            if background_id >= 0*1/adaptation_rate:
                self.find_burrows(frame)

            # store some debug information
            self.debug_process_frame(frame)
            
            if self.frame_id % 100000 == 0:
                self.logger.debug('Analyzed frame %d', self.frame_id)

    
    #===========================================================================
    # LOCATE CHANGES IN BACKGROUND
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


    def get_initial_burrow_mask(self, frame):
        """ get the burrow mask estimated from the first frame.
        This is mainly the predug, provided in the antfarm experiments """
        ground_mask = self.get_ground_mask()
        w = self.params['burrows/ground_point_distance']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
        ground_mask = cv2.erode(ground_mask, kernel)
        
        color_sand = self.data['pass1/colors/sand']
        color_sky = self.data['pass1/colors/sky']
        color_mean = 0.33*color_sand + 0.67*color_sky
        
        self.burrow_mask = (frame < color_mean).astype(np.uint8)
        self.burrow_mask[ground_mask == 0] = 0
        
        #debug.show_image(frame, burrow_mask, ground_mask)
    

    def get_background_changes(self, frame):
        """ determines a mask of all the burrows """
        mask = self._cache['image_uint8']
        mask.fill(0)
        diff = -self.background_avg + frame

        change_threshold = self.data['pass1/colors/sand_std']
        
#         # shrink burrows
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         mask[:] = (diff > change_threshold)
#         #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
#          
#         self.burrow_mask[mask == 1] = 0

        # enlarge burrows with excavated regions
        mask[:] = (diff < -change_threshold)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        
        self.burrow_mask[mask == 1] = 1
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        self.burrow_mask = cv2.morphologyEx(self.burrow_mask, cv2.MORPH_CLOSE, kernel1)
        
        return self.burrow_mask


    def extend_burrow_to_ground(self, contour):
        """ extends the burrow outline such that it connects to the ground line """

        # check whether the burrow is far away from the ground line
        outline = geometry.LinearRing(contour)
        dist = self.ground.linestring.distance(outline)
        if dist < 1:
            return contour
        
        dist_max = dist + self.params['burrows/width']/2
        ground_line = self.ground.linestring

        # determine burrow points close to the ground and
        # their associated ground points        
        points = []
        for point in contour:
            if ground_line.distance(geometry.Point(point)) < dist_max:
                points.append(point)
                point_ground = curves.get_projection_point(ground_line, point)
                points.append(point_ground)
        
        # get the convex hull of all these points
        hull = geometry.MultiPoint(points).convex_hull
        
        # add this to the burrow outline
        outline = regions.regularize_polygon(geometry.Polygon(contour))
        outline = outline.union(hull.buffer(0.1))
        outline = regions.get_enclosing_outline(outline)
        outline = np.array(outline.coords)
        
        # fill the burrow mask, such that this extension does not have to be done next time again
        cv2.fillPoly(self.burrow_mask, [np.asarray(outline, np.int32)], 1) 
        
        return outline  
        

    def find_burrows(self, frame):
        """ finds burrows from the current frame """
        if self.burrow_mask is None:
            self.get_initial_burrow_mask(frame)
        
        burrow_mask = self.get_background_changes(frame)

        contours, _ = cv2.findContours(burrow_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        self.burrows = []
        for contour in contours:
            props = image.regionprops(contour=contour)
            if props.area < self.params['burrows/area_min']:
                continue
            
            if self.ground.above_ground(props.centroid):
                continue
            
            contour = np.squeeze(np.asarray(contour, np.double))
            
            #contour = self.extend_burrow_to_ground(contour)
            self.burrows.append(contour)


    #===========================================================================
    # DEBUGGING
    #===========================================================================


    def debug_setup(self):
        """ prepares everything for the debug output """
        # load parameters for video output        
        video_output_period = 1#int(self.params['output/video/period'])
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        # set up the general video output, if requested
        if 'video' in self.debug_output or 'video.show' in self.debug_output:
            # initialize the writer for the debug video
            debug_file = self.get_filename('pass4' + video_extension, 'debug')
            self.debug['video'] = VideoComposer(debug_file, size=self.video.size,
                                                fps=self.video.fps, is_color=True,
                                                output_period=video_output_period,
                                                codec=video_codec, bitrate=video_bitrate)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 4 [%s]' % name,
                                           multiprocessing=self.params['debug/use_multiprocessing'],
                                           position=position)
                self.debug['video.show'] = image_window


    def debug_process_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_line(self.ground.points, is_closed=False,
                                     mark_points=True, color='y')
                
#             debug_video.highlight_mask(background_mask == 1, 'g', strength=64)
#             debug_video.highlight_mask(background_mask == 2, 'r', strength=64)
            #debug_video.highlight_mask(self.burrow_mask == 1, 'g', strength=64)
            for burrow in self.burrows:
                debug_video.add_line(burrow, 'r')
                
            # add additional debug information
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
            
    