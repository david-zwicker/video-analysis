'''
Created on Aug 19, 2014

@author: zwicker

Module that contains the class responsible for the second pass of the algorithm
'''

from __future__ import division

import itertools

import numpy as np
import networkx as nx
from shapely import geometry

from .data_handler import DataHandler
from .objects import GroundProfileTrack, MouseTrack
from video.analysis import curves
from video.composer import VideoComposer
from video.filters import FilterCrop
from video.utils import display_progress

import debug  # @UnusedImport


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, name='', parameters=None, debug_output=None, **kwargs):
        super(SecondPass, self).__init__(name, parameters, **kwargs)
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass2')

        self.debug = {} #< dictionary holding debug information
        self.debug_output = [] if debug_output is None else debug_output
        self.log_event('Pass 2 - Initialized the second pass analysis.')
        

    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        # create the data and copy the data from first_pass
        obj = cls(first_pass.name)
        obj.data = first_pass.data
        obj.params = first_pass.data['parameters']
        obj.tracks = first_pass.tracks
        # initialize parameters
        obj.initialize_parameters()
        return obj
    

    def process_data(self):
        """ do the second pass of the analysis """
        
        self.find_mouse_track()
        self.smooth_ground_profile()
        #self.smooth_burrows() # this should be 'morphing'
        self.classify_mouse_track()
        
        self.data['analysis-status'] = 'Finished second pass'
        self.log_event('Pass 2 - Finished second pass.')
        
        self.write_data()

    
    def load_video(self, video=None, crop_video=True):
        """ load the video, potentially using a previously analyzed video """
        # load the video with the standard method
        super(SecondPass, self).load_video(video, crop_video=crop_video)
        if self.data['video/analyzed']:
            # apparently a video has already been analyzed
            # ==> we thus use the previously determined cage cropping rectangle
            crop_rect = self.data['video/analyzed/region_cage']
            self.video = FilterCrop(self.video, crop_rect)
        

    #===========================================================================
    # CONNECT TEMPORAL DATA -- TRACKING
    #===========================================================================


    def get_track_graph(self, tracks, threshold):
        """ builds a weighted, directed graph representing the possible trajectories """
        graph = nx.DiGraph()
        
        # find all possible connections
        time_scale = self.params['tracking/time_scale']
        for a_idx, a in enumerate(tracks):
            for b in tracks[a_idx + 1:]:
                gap_length = b.start - a.end #< time gap between the two chunks
                if gap_length > -self.params['tracking/tolerated_overlap']:
                    # calculate the weight of this graph
                    # lower is better; all terms should be normalized to one
                    
                    distance = curves.point_distance(a.last.pos, b.first.pos)
                    
                    weight = (
                        #+ (2 - a.mouse_score - b.mouse_score)       # is it a mouse? 
                        + distance/self.params['mouse/speed_max']    # how close are the mice
                        + abs(gap_length)/time_scale                 # is it a long gap?
                    )
                    
                    # add the edge if the weight is not too high
                    if weight < threshold:
                        graph.add_weighted_edges_from([(a, b, weight)])
        return graph
            
                
    def get_best_track(self, tracks):
        """ finds the best connection of tracks """
        if not tracks:
            return []

        # sort them according to their start time
        tracks = sorted(tracks, key=lambda track: track.start)
        
        # get some statistics about the tracks
        start_time = min(track.start for track in tracks)
        end_time = max(track.end for track in tracks)
        end_node_interval = self.params['tracking/end_node_interval']
        endtoend_nodes = [track for track in tracks
                          if track.start <= start_time + end_node_interval and
                             track.end >= end_time - end_node_interval]
        
        threshold = self.params['tracking/initial_score_threshold']
        
        # try different thresholds until we found a result        
        track_found = False
        while not track_found:
            self.logger.info('Building tracking graph of %d nodes and with threshold %g',
                             len(tracks), threshold) 
            graph = self.get_track_graph(tracks, threshold)
            graph.add_nodes_from(endtoend_nodes) 
            self.logger.info('Built tracking graph with %d nodes and %d edges',
                             graph.number_of_nodes(), graph.number_of_edges()) 

            if graph.number_of_nodes() > 0:
    
                # find start and end nodes
                start_nodes = [node for node in graph
                               if graph.in_degree(node) == 0 and 
                                   node.start <= start_time + end_node_interval]
                end_nodes = [node for node in graph
                             if graph.out_degree(node) == 0 and
                                 node.end >= end_time - end_node_interval]
        
                self.logger.info('Found %d start node(s) and %d end node(s) in tracking graph.',
                                 len(start_nodes), len(end_nodes)) 
                
                # find paths between start and end nodes
                paths = []
                for start_node in start_nodes:
                    for end_node in end_nodes:
                        try:
                            # find best path to reach this out degree
                            path = nx.shortest_path(graph, start_node, end_node, weight='weight')
                            paths.append(path)
                        except nx.exception.NetworkXNoPath:
                            continue # check the next node
                        else:
                            track_found = True

            threshold *= 2
        
        self.logger.info('Found %d good tracking paths', len(paths))
        
        # identify the best path
        path_best, score_best = None, np.inf 
        for path in paths:
            weight = sum(graph.get_edge_data(a, b)['weight']
                         for a, b in itertools.izip(path, path[1:]))
            length = 1 + path[-1].end - path[0].start
            score = (1 + weight)/length # lower is better
            if score < score_best:
                path_best, score_best = path, score
                
#         debug.show_tracking_graph(graph, path_best)
                
        return path_best
            
        
    def find_mouse_track(self):
        """ identifies the mouse trajectory by connecting object tracks.
        
        This function takes the tracks in 'pass1/objects/tracks', connects
        suitable parts, and interpolates gaps.
        """
        self.log_event('Pass 2 - Started identifying mouse trajectory.')
        
        tracks = self.data['pass1/objects/tracks']

        #tracks = [track for track in tracks if track.start < 10000]
        
        # get the best collection of tracks that best fit mouse
        path = self.get_best_track(tracks)
        
        # build a single trajectory out of this
        trajectory = np.empty((self.data['video/input/frame_count'], 2))
        trajectory.fill(np.nan)
        
        time, obj = None, None        
        for track in path:
            # interpolate between the last track and the current one
            if obj is not None:
                time_now, obj_now = track.start, track.first
                frames = np.arange(time + 1, time_now)
                times = np.linspace(0, 1, len(frames) + 2)[1:-1]
                x1, x2 = obj.pos[0], obj_now.pos[0]
                trajectory[frames, 0] = x1 + (x2 - x1)*times
                y1, y2 = obj.pos[1], obj_now.pos[1]
                trajectory[frames, 1] = y1 + (y2 - y1)*times
            
            # add the data of this track directly
            for time, obj in track:
                trajectory[time, :] = obj.pos
        
        self.data['pass2/mouse_trajectory'] = MouseTrack(trajectory)
    
                        
    #===========================================================================
    # SMOOTH GROUND AND BURROWS
    #===========================================================================


    def smooth_ground_profile(self):
        """ smooth the ground profile """
        
        # convert data to different format
        profile_list = self.data['pass1/ground/profile']
        profile = GroundProfileTrack.create_from_ground_profile_list(profile_list)
        
        # standard deviation for smoothing [in number of profiles]
        sigma = self.params['ground/smoothing_sigma']/self.params['ground/adaptation_interval']
        profile.smooth(sigma)
         
        # store the result
        self.data['pass2/ground_profile'] = profile
        
        
    #===========================================================================
    # HANDLE THE MOUSE
    #===========================================================================


    def classify_mouse_track(self):
        """ classifies the mouse at all times """
        self.log_event('Pass 2 - Start classifying the mouse.')
        
        # load the mouse, the ground, and the burrows
        mouse_track = self.data['pass2/mouse_trajectory']
        ground_profile = self.data['pass2/ground_profile']
        burrow_tracks = self.data['pass1/burrows/tracks']
        
        # load some variables
        mouse_radius = self.data['parameters/mouse/model_radius']
        
        for frame_id, mouse_pos in enumerate(mouse_track.pos):
            if not np.all(np.isfinite(mouse_pos)):
                # the mouse position is invalid
                continue
            
            # initialize variables
            state = {}
                    
            # check the mouse position
            ground = ground_profile.get_ground(frame_id)
            if ground is not None:
                # compare y value of mouse and ground
                # Note that the y-axis points down
                if mouse_pos[1] < ground.get_y(mouse_pos[0]):
                    state['underground'] = False
                    if mouse_pos[1] + mouse_radius < ground.get_y(mouse_pos[0]):
                        state['location'] = 'air'
                    elif mouse_pos[1] < ground.midline:
                        state['location'] = 'hill'
                    else:
                        state['location'] = 'valley'
                    
                else: 
                    state['underground'] = True
                            
                    # check whether the mouse is inside a burrow
                    mouse_point = geometry.Point(mouse_pos)
                    for burrow in burrow_tracks.get_burrows(frame_id):
                        if burrow and burrow.polygon.contains(mouse_point):
                            state['location'] = 'burrow'
                            break
            
            # set the mouse state
            mouse_track.set_state(frame_id, state)
        
        self.log_event('Pass 2 - Finished classifying the mouse.')


    #===========================================================================
    # PRODUCE VIDEO
    #===========================================================================


    def produce_video(self):
        """ prepares everything for the debug output """
        # load parameters for video output        
        video_extension = self.params['output/video/extension']
        video_codec = self.params['output/video/codec']
        video_bitrate = self.params['output/video/bitrate']
        
        if self.video is None:
            self.load_video()
        
        filename = self.get_filename('video' + video_extension, 'results')
        video = VideoComposer(filename, size=self.video.size, fps=self.video.fps,
                              is_color=True, codec=video_codec, bitrate=video_bitrate)
        
        mouse_track = self.data['pass2/mouse_trajectory']
        ground_profile = self.data['pass2/ground_profile']
        
        self.logger.info('Start producing final video with %d frames', len(self.video))
        
        # we used the first frame to determine the cage dimensions in the first pass
        source_video = self.video[1:]
        
        track_first = 0
        tracks = sorted(self.data['pass1/objects/tracks'], key=lambda track: track.start)
        burrow_tracks = self.data['pass1/burrows/tracks']
        
        for frame_id, frame in enumerate(display_progress(source_video)):
            # set real video as background
            video.set_frame(frame)
        
            # plot the ground profile
            ground_line = ground_profile.get_profile(frame_id)
            video.add_polygon(ground_line, is_closed=False, mark_points=False, color='y')

            # indicate burrow centerline
            for burrow_track in burrow_tracks:
                try:
                    burrow = burrow_track.get_burrow(frame_id)
                except IndexError:
                    continue
                video.add_polygon(burrow.get_centerline(ground_line),
                                  'k', is_closed=False, width=2)
        
            # find the first track which is still active
            while tracks[track_first].end < frame_id:
                track_first += 1
            # draw all tracks that are still active
            for track in tracks[track_first:]:
                if track.start > frame_id:
                    # this is the last track that can possibly be active
                    break
                elif track.start <= frame_id <= track.end:
                    video.add_circle(track.get_pos(frame_id),
                                     self.params['mouse/model_radius'], '0.5', thickness=1)

            # indicate the mouse position
            if np.all(np.isfinite(mouse_track.pos[frame_id])):
                video.add_circle(mouse_track.pos[frame_id],
                                 self.params['mouse/model_radius'], 'w', thickness=2)
                    
            
#                 # add additional debug information
            
            video.add_text(str(frame_id), (20, 20), anchor='top')   
#                 video.add_text(str(self.frame_id/self.fps), (20, 20), anchor='top')   
                
        video.close()
            
    