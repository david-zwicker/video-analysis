'''
Created on Aug 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that contains the class responsible for the second pass of the algorithm
'''

from __future__ import division

import itertools

import numpy as np
import networkx as nx
from shapely import geometry

from .data_handler import DataHandler
from .objects import GroundProfile, GroundProfileTrack, MouseTrack
from video.analysis import curves
from video.composer import VideoComposer
from video.filters import FilterCrop
from video.utils import display_progress

import debug  # @UnusedImport


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, name='', parameters=None, **kwargs):
        super(SecondPass, self).__init__(name, parameters, **kwargs)
        self.params = self.data['parameters']
        self.result = self.data.create_child('pass2')
        self.result['code_status'] = self.get_code_status()
        if kwargs.get('initialize_parameters', True):
            self.log_event('Pass 2 - Initialized the second pass analysis.')
        

    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        # create the data and copy the data from first_pass
        obj = cls(first_pass.name, initialize_parameters=False)
        obj.data = first_pass.data
        obj.params = obj.data['parameters']
        obj.result = obj.data.create_child('pass2')

        # close logging handlers and other files        
        first_pass.close()
        
        # initialize parameters
        obj.initialize_parameters()
        obj.log_event('Pass 2 - Initialized the second pass analysis.')
        return obj
    

    def process(self):
        """ do the second pass of the analysis """
        
        self.find_mouse_track()
        self.smooth_ground_profile()
        #self.classify_mouse_track()
        
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


    def get_track_graph(self, tracks, threshold, highlight_nodes=None):
        """ builds a weighted, directed graph representing the possible trajectories """
        if highlight_nodes:
            highlight_nodes = set(highlight_nodes)
        else:
            highlight_nodes = set()
        
        graph = nx.DiGraph()
        
        # find all possible connections
        time_scale = self.params['tracking/time_scale']
        tolerated_overlap = self.params['tracking/tolerated_overlap']
        look_back_count = int(tolerated_overlap) + 5
        for a_idx, a in enumerate(tracks):
            # compare to other nodes (look back into past, too) 
            for b in tracks[max(a_idx - look_back_count, 0):]:
                if a is b or graph.has_edge(a, b):
                    continue # don't add self-loops or multiple loops
                
                gap_length = b.start - a.end #< time gap between the two chunks
                if gap_length > -tolerated_overlap:
                    # calculate the cost of this gap
                    # lower is better; all terms should be normalized to one
                    
                    distance = curves.point_distance(a.last.pos, b.first.pos)
                    
                    cost = (
                        #+ (2 - a.mouse_score - b.mouse_score)       # is it a mouse? 
                        + distance/self.params['mouse/speed_max']    # how close are the positions
                        + abs(gap_length)/time_scale                 # is it a long gap?
                    )

                    # add the edge if the cost is not too high
                    if cost < threshold:
                        graph.add_edge(a, b, cost=cost)
                        
            # highlight node if necessary
            if a in graph:
                graph.node[a]['highlight'] = (a_idx in highlight_nodes)
                        
        return graph
            
                
    def find_paths_in_track_graph(self, graph, start_nodes, end_nodes, find_all=True):
        """ finds all shortest paths from start_nodes to end_nodes in the graph """
        # filter start and end nodes to make sure that they actually are in the graph
        start_nodes = (s for s in start_nodes if s in graph)
        end_nodes = (s for s in end_nodes if s in graph)

        # find paths between start and end nodes
        paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                try:
                    # find best path to reach this out degree
                    path = nx.shortest_path(graph, start_node, end_node, weight='cost')
                except nx.exception.NetworkXNoPath:
                    # there are no connections between the start and the end node 
                    continue #< check the next node
                else:
                    paths.append(path)
                    if not find_all:
                        return paths #< return as early as possible
        return paths
           
           
    def find_outer_nodes(self, graph, start_time=None, end_time=None):
        """ locate the start and end nodes in a graph """
        # determine start and end times automatically if necessary
        if start_time is None:
            start_time = min(node.start for node in graph.nodes_iter())
            start_time += self.params['tracking/end_node_interval']
        if end_time is None:
            end_time = max(node.end for node in graph.nodes_iter())
            end_time -= self.params['tracking/end_node_interval']
        
        # find the nodes which are at the edges
        start_nodes, end_nodes = [], []
        for node in graph:
            if node.start <= start_time:
                # potential start node
                if all(neighbor.start >= node.start
                       for neighbor in graph.neighbors(node)):
                    start_nodes.append(node)
            if node.end >= end_time:
                # potential end node
                if all(neighbor.end <= node.end
                       for neighbor in graph.neighbors(node)):
                    end_nodes.append(node)
        return start_nodes, end_nodes
         
         
    def get_best_track_connection(self, tracks, start_nodes=None, end_nodes=None):
        """ determines a good path, possibly choosing from several start and 
        end nodes """   
        if len(tracks) < 2:
            return tracks
        
        threshold = self.params['tracking/initial_score_threshold']
        threshold_max = self.params['tracking/score_threshold_max']

        # set flags if the end nodes have to be determined automatically
        determine_start_nodes = (start_nodes is None)
        determine_end_nodes = (end_nodes is None)

        # try different thresholds until we found a result
        successful_iterations = 0
        while successful_iterations < 2:
            # build the tracking graph
            graph = self.get_track_graph(tracks, threshold)

            if graph.number_of_edges() > 0:
                # get the end nodes if necessary
                if determine_start_nodes:
                    start_nodes, _ = self.find_outer_nodes(graph)
                if determine_end_nodes:
                    _, end_nodes = self.find_outer_nodes(graph)
    
                # find possible paths between the start and end nodes
                find_all = (successful_iterations >= 1)
                paths = self.find_paths_in_track_graph(graph, start_nodes,
                                                       end_nodes, find_all)
            elif threshold > threshold_max:
                # stop the iteration if the threshold becomes too large 
                if determine_start_nodes or determine_end_nodes:
                    paths = []
                else:
                    paths = [[start_nodes[0], end_nodes[0]]]
                break
                
            else:
                paths = False

            if paths:
                # we'll do an additional search with an increased threshold
                successful_iterations += 1
                threshold *= 10
            else:
                threshold *= 2
        
        # filter erroneous paths
        paths = (p for p in paths if p is not None)

        # identify the best path
        path_best, score_best = None, np.inf
        for path in paths:
            cost = sum(graph.get_edge_data(a, b)['cost']
                       for a, b in itertools.izip(path, path[1:]))
            length = 1 + path[-1].end - path[0].start
            score = (1 + cost)/length
            if score < score_best:  #< lower is better
                path_best, score_best = path, score
                
#         debug.show_tracking_graph(graph, path_best)
        
        return path_best
         
                
    def get_best_track(self, tracks, sure_tracks):
        """ finds the best connection of tracks """
        if not tracks:
            return []

        # sort them according to their start time
        tracks.sort(key=lambda track: track.start)

        # break apart long tracks to facilitate graph matching
        track_len_orig = len(tracks)
        tracks.break_long_tracks(self.params['tracking/splitting_duration_min'])
        if len(tracks) != track_len_orig:
            self.logger.info('Pass 2 - Increased the track count from %d to %d '
                             'by splitting long, overlapping tracks.',
                             track_len_orig, len(tracks))
        
        # get some statistics about the tracks
        start_time = min(track.start for track in tracks)
        end_time = max(track.end for track in tracks)
        # find tracks that span the entire time
        end_node_interval = self.params['tracking/end_node_interval']
        endtoend_nodes = [track for track in tracks
                          if track.start <= start_time + end_node_interval and
                             track.end >= end_time - end_node_interval]
        
        threshold = self.params['tracking/initial_score_threshold']
        
        # try different thresholds until we found a result
        successful_iterations = 0
        while successful_iterations < 2:
            self.logger.info('Pass 2 - Building tracking graph of %d nodes '
                             'with threshold %g', len(tracks), threshold) 
            graph = self.get_track_graph(tracks, threshold, sure_tracks)
            graph.add_nodes_from(endtoend_nodes) 
            self.logger.info('Pass 2 - Built tracking graph with %d nodes and '
                             '%d edges',
                             graph.number_of_nodes(), graph.number_of_edges()) 

            # find start and end nodes
            start_nodes, end_nodes = self.find_outer_nodes(graph,
                                                           start_time + end_node_interval,
                                                           end_time - end_node_interval)
    
            self.logger.info('Pass 2 - Found %d start node(s) and %d end '
                             'node(s) in tracking graph.',
                             len(start_nodes), len(end_nodes)) 

            # find possible paths
            find_all = (successful_iterations >= 1)
            paths = self.find_paths_in_track_graph(graph, start_nodes,
                                                   end_nodes, find_all)

            if paths:
                # we'll do an additional search with an increased threshold
                successful_iterations += 1
                threshold *= 10
            else:
                threshold *= 2
        
        self.logger.info('Pass 2 - Found %d good tracking paths', len(paths))
        
        # identify the best path
        path_best, score_best = None, np.inf 
        for path in paths:
            cost = sum(graph.get_edge_data(a, b)['cost']
                       for a, b in itertools.izip(path, path[1:]))
            length = 1 + path[-1].end - path[0].start
            score = (1 + cost)/length
            if score < score_best:  #< lower is better
                path_best, score_best = path, score
                
        debug.show_tracking_graph(graph, path_best)
                
        return path_best
    

    def find_sure_mouse_tracks(self, tracks):
        """ identifies tracks which surely describe mice. These tracks can then
        be used to split up the connection problem into distinct parts """
        smoothing_window = self.params['tracking/position_smoothing_window']
        dist_threshold = self.params['tracking/mouse_distance_threshold']
        min_mean_speed = self.params['tracking/mouse_min_mean_speed']
        max_speed = self.params['mouse/speed_max']
        min_length = int(dist_threshold/(0.25 * max_speed))
         
        sure_tracks = []
        for track_id, track in enumerate(tracks):
            if len(track) < min_length:
                continue
            
            # get smoothed trajectory
            trajectory = track.get_trajectory(smoothing_window)
            dist = curves.curve_length(trajectory)
            
            if dist > dist_threshold and dist/len(track) > min_mean_speed:
                sure_tracks.append(track_id)

        return sure_tracks
        

    def add_tracks_to_trajectory(self, tracks, trajectory):
        """ iterates through all tracks and adds them to the trajectory, using
        linear interpolation where necessary and appropriate """
        
        time_last, obj_last = None, None        
        for track in tracks:
            # check the connection between the previous point and the current one
            if obj_last is not None:
                time_now, obj_now = track.start, track.first
                time_gap = time_now - time_last
                small_gap = (time_gap < self.params['tracking/maximal_gap'])
                obj_dist = curves.point_distance(obj_last.pos, obj_now.pos)
                small_jump = (obj_dist < self.params['tracking/maximal_jump'])
                
                # check whether we should interpolate 
                if small_gap and small_jump:
                    frames = np.arange(time_last + 1, time_now)
                    ratio = np.linspace(0, 1, len(frames) + 2)[1:-1]
                    x1, x2 = obj_last.pos[0], obj_now.pos[0]
                    trajectory[frames, 0] = x1 + (x2 - x1)*ratio
                    y1, y2 = obj_last.pos[1], obj_now.pos[1]
                    trajectory[frames, 1] = y1 + (y2 - y1)*ratio
                    
                # check whether the tracking went wrong
                if time_gap <= 1 and not small_jump:
                    # make sure that there is a gap indicated by nan between
                    # the two chunks 
                    trajectory[time_now - 1:time_last + 1, :] = np.nan
            
            # add the data of this track directly
            for time, obj in track:
                if np.all(np.isfinite(trajectory[time, :])):
                    # average overlapping tracks
                    trajectory[time, :] = (trajectory[time, :] + obj.pos)/2
                else:
                    trajectory[time, :] = obj.pos
                    
            time_last, obj_last = time, obj
        
                
    def find_mouse_track(self):
        """ identifies the mouse trajectory by connecting object tracks.
        
        This function takes the tracks in 'pass1/objects/tracks', connects
        suitable parts, and interpolates gaps.
        """
        self.log_event('Pass 2 - Started identifying mouse trajectory.')
        
        # get tracks and sort them according to their start time
        tracks = self.data['pass1/objects/tracks']
        tracks.sort(key=lambda track: track.start)

        # find tracks where we are sure that they correspond to a mouse
        sure_tracks = self.find_sure_mouse_tracks(tracks)
        if len(sure_tracks) > 1:
            consecutive_tracks = itertools.izip(sure_tracks, sure_tracks[1:])
            max_gap = max(b.start - a.end for a, b in consecutive_tracks)
            self.logger.info('Found %d sure mouse tracks. The longest gap is '
                             '%d frames.', len(sure_tracks), max_gap)
        else:
            self.logger.info('Found %d sure mouse tracks.', len(sure_tracks))

        if sure_tracks:
            # connect all these sure tracks with the best match
            path = []
            # add the first trajectory with automatic start nodes
            tracks_part = tracks[:sure_tracks[0] + 1]
            connection = self.get_best_track_connection(tracks_part,
                                                        end_nodes=[tracks_part[-1]])
            path.extend(connection)
            
            # connect all the sure tracks
            for track_start, track_end in itertools.izip(sure_tracks[:-1],
                                                         sure_tracks[1:]):
                tracks_part = tracks[track_start:track_end + 1]
                connection = self.get_best_track_connection(tracks_part,
                                                            start_nodes=[tracks_part[0]],
                                                            end_nodes=[tracks_part[-1]])
                path.extend(connection[1:])
    
            # add the last trajectory with automatic end nodes
            tracks_part = tracks[sure_tracks[-1]:]
            connection = self.get_best_track_connection(tracks_part,
                                                        start_nodes=[tracks_part[0]])
            if connection:
                path.extend(connection[1:])
            
        else:
            path = self.get_best_track_connection(tracks)
              
        # build a single trajectory out of this
        trajectory = np.empty((self.data['pass1/video/frames_analyzed'], 2))
        trajectory.fill(np.nan)
        self.add_tracks_to_trajectory(path, trajectory)

        # save the result
        self.data['pass2/mouse_trajectory'] = MouseTrack(trajectory)
    
                        
    #===========================================================================
    # SMOOTH GROUND AND BURROWS
    #===========================================================================


    def smooth_ground_profile(self):
        """ smooth the ground profile """
        
        # convert data to different format
        profile_list = self.data['pass1/ground/profile']
        
        # determine the minimal and maximal extend of the ground profile
        x_min, x_max = np.inf, 0
        for ground in profile_list.grounds:
            x_min = min(x_min, ground.points[ 0, 0])
            x_max = max(x_max, ground.points[-1, 0])
            
        # extend all ground profiles that do not reach to these limits
        cage_width = []
        for ground in profile_list.grounds:
            points = ground.points.tolist()
            if points[0][0] > x_min:
                points.insert(0, (x_min, points[0][1]))
            if points[-1][0] < x_max:
                points.append((x_max, points[-1][1]))
            ground.points = points
            cage_width.append(points[-1][0] - points[0][0])
        
        # create a ground profile track from the list
        profile = GroundProfileTrack.create_from_ground_profile_list(profile_list)
        
        # standard deviation for smoothing [in number of profiles]
        sigma = self.params['ground/smoothing_sigma']/self.params['ground/adaptation_interval']
        profile.smooth(sigma)
         
        # store the result
        width_mean, width_std = np.mean(cage_width), np.std(cage_width)
        self.data['pass2/ground_profile'] = profile
        self.data['pass2/cage/width_mean'] = width_mean
        self.data['pass2/cage/width_std'] = width_std
        self.data['pass2/pixel_size_cm'] = self.params['cage/width_cm']/width_mean
        
        
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
        mouse_radius = self.params['mouse/model_radius']
        trail_spacing = self.params['burrows/centerline_segment_length']
        burrow_next_change = 0
        mouse_trail = None
        for frame_id, mouse_pos in enumerate(mouse_track.pos):
            if not np.all(np.isfinite(mouse_pos)):
                # the mouse position is invalid
                continue
            
            # initialize variables
            state = {}
                    
            # check the mouse position
            ground = ground_profile.get_ground_profile(frame_id)
            if ground is not None:
                if ground.above_ground(mouse_pos):
                    state['underground'] = False
                    if mouse_pos[1] + mouse_radius < ground.get_y(mouse_pos[0]):
                        state['location'] = 'air'
                    elif mouse_pos[1] < ground.midline:
                        state['location'] = 'hill'
                    else:
                        state['location'] = 'valley'

                    mouse_trail = None
                    # get index of the ground line
                    dist = np.linalg.norm(ground.points - mouse_pos[None, :], axis=1)
                    ground_idx = np.argmin(dist)
                    # get distance from ground line
                    ground_dist = ground.linestring.distance(geometry.Point(mouse_pos))

                else:
                    state['underground'] = True
                    # check the burrow structure
                    if frame_id >= burrow_next_change:
                        burrows, burrow_next_change = \
                            burrow_tracks.find_burrows(frame_id, ret_next_change=True)
                    
                    # check whether the mouse is inside a burrow
                    mouse_point = geometry.Point(mouse_pos)
                    for burrow in burrows:
                        if burrow.polygon.contains(mouse_point):
                            state['location'] = 'burrow'
                            break

                    # keep the ground index from last time
                    ground_idx = mouse_track.ground_idx[frame_id - 1]

                    # handle mouse trail
                    if mouse_trail is None:
                        # start a new mouse trail and initialize it with the                         
                        # ground point closest to the mouse       
                        mouse_prev = mouse_track.pos[frame_id - 1]              
                        ground_point = curves.get_projection_point(ground.linestring, mouse_prev)
                        mouse_trail = [ground_point, mouse_pos]

                    else:
                        # work with an existing mouse trail
                        p_trail = mouse_trail[-2]
                        
                        if curves.point_distance(p_trail, mouse_pos) < trail_spacing:
                            # old trail should be modified
                            if len(mouse_trail) > 2:
                                # check whether the trail has to be shortened
                                p_trail = mouse_trail[-3]
                                if curves.point_distance(p_trail, mouse_pos) < trail_spacing:
                                    del mouse_trail[-1] #< shorten trail
                                
                            mouse_trail[-1] = mouse_pos
                        else:
                            # old trail must be extended
                            mouse_trail.append(mouse_pos)
                        
                    # get distance the mouse is under ground
                    ground_dist = -curves.curve_length(mouse_trail)
                    
            # set the mouse state
            mouse_track.set_state(frame_id, state, ground_idx, ground_dist)
        
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
        
        filename = self.get_filename('video' + video_extension, 'video')
        video = VideoComposer(filename, size=self.video.size,
                              fps=self.video.fps, is_color=True,
                              output_period=self.params['output/output_period'],
                              codec=video_codec, bitrate=video_bitrate)
        
        mouse_track = self.data['pass2/mouse_trajectory']
        ground_profile = self.data['pass2/ground_profile']
        
        self.logger.info('Pass 2 - Start producing final video with %d frames', len(self.video))
        
        # we used the first frame to determine the cage dimensions in the first pass
        source_video = self.video[1:]
        
        track_first = 0
        tracks = self.data['pass1/objects/tracks']
        tracks.sort(key=lambda track: track.start)
        burrow_tracks = self.data['pass1/burrows/tracks']
        
        for frame_id, frame in enumerate(display_progress(source_video)):
            # set real video as background
            video.set_frame(frame)
        
            # plot the ground profile
            ground_line = ground_profile.get_groundline(frame_id)
            video.add_line(ground_line, is_closed=False,
                              mark_points=False, color='y')

            # indicate burrow centerline
            for burrow in burrow_tracks.find_burrows(frame_id):
                ground = GroundProfile(ground_line)
                video.add_line(burrow.get_centerline(ground),
                                  'k', is_closed=False, width=2)
        
            # indicate all moving objects
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
                                     self.params['mouse/model_radius'],
                                     '0.5', thickness=1)

            # indicate the mouse position
            if np.all(np.isfinite(mouse_track.pos[frame_id])):
                video.add_circle(mouse_track.pos[frame_id],
                                 self.params['mouse/model_radius'],
                                 'w', thickness=2)
                    
            
#                 # add additional debug information
            
            video.add_text(str(frame_id), (20, 20), anchor='top')   
#                 video.add_text(str(self.frame_id/self.fps), (20, 20), anchor='top')   
                
        video.close()
            
    