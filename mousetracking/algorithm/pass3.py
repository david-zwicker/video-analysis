'''
Created on Oct 2, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the third pass of the algorithm
'''

from __future__ import division

import time

import numpy as np
from shapely import geometry

from .data_handler import DataHandler 
from video.analysis import curves
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
            self._iterate_over_video(self.video[1:])
                
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
            
        frames = self.data['pass1/video/frames']
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
        
        video_info = self.data['pass3/video']
        video_info['cropping_cage'] = cropping_cage
        video_info['frame_count'] = self.video.frame_count
        video_info['frames'] = frames
        video_info['size'] = '%d x %d' % tuple(self.video.size),
        
        # get first frame, which will not be used in the iteration
        first_frame = self.video.get_next_frame()
        
        # initialize data structures
        self.frame_id = -1
        self.background = first_frame.astype(np.double)
        self.ground_idx = None  #< index of the ground point where the mouse entered the burrow
        self.mouse_trail = None #< line from this point to the mouse (along the burrow)
        
        
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
            
            # load data for current frame
            self.mouse_pos = mouse_track.pos[self.frame_id, :]
            self.ground = ground_profile.get_ground_profile(self.frame_id)

            # do the actual work            
            self.classify_mouse_state(mouse_track)
            # TODO: use the current mouse tracks to know where the burrows are
            # use self.background to fit the burrows
            #self.locate_burrows()

            # store some information in the debug dictionary
            self.debug_process_frame(frame)

    
    #===========================================================================
    # MOUSE TRACKING
    #===========================================================================


    def classify_mouse_state(self, mouse_track):
        """ classifies the mouse in the current frame """
        if (not np.all(np.isfinite(self.mouse_pos)) or
            self.ground is None):
            
            # Not enough information to do anything
            return
        
        # initialize variables
        state = {}
                
        # compare y value of mouse and ground
        # Note that the y-axis points down
        if self.mouse_pos[1] > self.ground.get_y(self.mouse_pos[0]):
            state['underground'] = True
            
            # handle mouse trail
            if self.mouse_trail is None:
                # start a new mouse trail and initialize it with the                         
                # ground point closest to the mouse       
                ground_point = curves.get_projection_point(self.ground.linestring,
                                                           self.mouse_pos)
                self.mouse_trail = [ground_point, self.mouse_pos]

            else:
                # work with an existing mouse trail
                p_trail = self.mouse_trail[-2]
                
                trail_spacing = self.params['burrows/centerline_segment_length']
                if curves.point_distance(p_trail, self.mouse_pos) < trail_spacing:
                    # old trail should be modified
                    if len(self.mouse_trail) > 2:
                        # check whether the trail has to be shortened
                        p_trail = self.mouse_trail[-3]
                        if curves.point_distance(p_trail, self.mouse_pos) < trail_spacing:
                            del self.mouse_trail[-1] #< shorten trail
                        
                    self.mouse_trail[-1] = self.mouse_pos
                else:
                    # old trail must be extended
                    self.mouse_trail.append(self.mouse_pos)
                
            # get distance the mouse is under ground
            ground_dist = -curves.curve_length(self.mouse_trail)
                
        else: 
            state['underground'] = False
            mouse_radius = self.params['mouse/model_radius']
            if self.mouse_pos[1] + mouse_radius < self.ground.get_y(self.mouse_pos[0]):
                state['location'] = 'air'
            elif self.mouse_pos[1] < self.ground.midline:
                state['location'] = 'hill'
            else:
                state['location'] = 'valley'

            self.mouse_trail = None
            # get index of the ground line
            dist = np.linalg.norm(self.ground.line - self.mouse_pos[None, :], axis=1)
            self.ground_idx = np.argmin(dist)
            # get distance from ground line
            mouse_point = geometry.Point(self.mouse_pos)
            ground_dist = self.ground.linestring.distance(mouse_point)
            
        # set the mouse state
        mouse_track.set_state(self.frame_id, state, self.ground_idx, ground_dist)

    
    #===========================================================================
    # BURROW TRACKING
    #===========================================================================


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
                                                codec=video_codec, bitrate=video_bitrate,
                                                debug=self.debug_run)
            
            if 'video.show' in self.debug_output:
                name = self.name if self.name else ''
                position = self.params['debug/window_position']
                image_window = ImageWindow(self.debug['video'].shape,
                                           title='Debug video pass 3 [%s]' % name,
                                           multiprocessing=self.params['debug/use_multiprocessing'],
                                           position=position)
                self.debug['video.show'] = image_window


    def debug_process_frame(self, frame):
        """ adds information of the current frame to the debug output """
        
        if 'video' in self.debug:
            debug_video = self.debug['video']
            
            # plot the ground profile
            if self.ground is not None:
                debug_video.add_polygon(self.ground.line, is_closed=False,
                                        mark_points=True, color='y')
        
            # indicate the mouse position
            track = self.data['pass2/mouse_trajectory'].pos
            trail_length = self.params['output/video/mouse_trail_length']
            time_start = max(0, self.frame_id - trail_length)
            track = track[time_start:self.frame_id, :]
            if len(track) > 0:
                debug_video.add_polygon(track, '0.5', is_closed=False)
                debug_video.add_circle(track[-1], self.params['mouse/model_radius'],
                                       'w', thickness=1)
                
            if self.mouse_trail is not None:
                debug_video.add_polygon(self.mouse_trail, 'w', is_closed=False,
                                        mark_points=True, width=2)
                
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
                self.logger.exception('Error while writing out the debug video') 