'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Contains a class that can be used to analyze results from the tracking
'''

from __future__ import division

import collections
import copy
import itertools

import numpy as np
import networkx as nx
from scipy import ndimage
from shapely import geometry

from ..algorithm import FirstPass, SecondPass, ThirdPass, FourthPass
from .data_handler import DataHandler
from .objects import mouse
from utils.data_structures import OmniContainer
from utils.math import contiguous_int_regions_iter, is_equidistant
from external.kids_cache import cache
from video.analysis import curves

try:
    import pint
    UNITS_AVAILABLE = True
except (ImportError, ImportWarning):
    UNITS_AVAILABLE = False



class Analyzer(DataHandler):
    """ class contains methods to analyze the results of a video """
    # flag indicating whether results are reported with pint units
    use_units = True
    # list of classes that manage analysis passes
    pass_classes = [FirstPass, SecondPass, ThirdPass, FourthPass]
    # list of mouse states that are returned by default
    mouse_states_default = ['.A.', '.H.', '.V.', '.D.', '.B ', '.[B|D]E', '...']
    
    
    def __init__(self, *args, **kwargs):
        """ initialize the analyzer """
        super(Analyzer, self).__init__(*args, **kwargs)
        self.params = self.data['parameters'] #< TODO use this throughout class
        
        if self.use_units and not UNITS_AVAILABLE:
            raise ValueError('Outputting results with units is not available. '
                             'Please install the `pint` python package.')

        # set the dimensions        
        self.time_scale_mag = 1/self.data['video/fps']
        self.length_scale_mag = self.data['pass2/pixel_size_cm']
        
        if self.use_units:
            # use a unit registry to keep track of all units
            self.units = pint.UnitRegistry()
            # define custom units
            self.units.define('frames = %g * second' % self.time_scale_mag)
            self.units.define('pixel = %g * centimeter' % self.length_scale_mag)
            # augment the dimension with the appropriate units
            self.time_scale = self.time_scale_mag * self.units.second
            self.length_scale = self.length_scale_mag * self.units.centimeter
            
        else:
            self.time_scale = self.time_scale_mag
            self.length_scale = self.length_scale_mag

        self.speed_scale = self.length_scale / self.time_scale
        self.burrow_pass = self.data['parameters/analysis/burrow_pass']
        self._cache = {}
        
        
    @cache
    def get_frame_range(self):
        """ returns the range of frames that is going to be analyzed """
        # get the frames that were actually analyzed in the video
        frames_video = copy.copy(self.data['pass1/video/frames'])
        if not frames_video[0]:
            frames_video[0] = 0
        
        adaptation_frames = self.data['parameters/video/initial_adaptation_frames'] 
        if adaptation_frames:
            frames_video[0] += adaptation_frames
            
        # get the frames that are chosen for analysis
        frames_analysis = self.data['parameters/analysis/frames']
        if frames_analysis is None:
            frames_analysis = frames_video
        else:
            if frames_analysis[0] is None:
                frames_analysis[0] = frames_video[0]
            else:
                frames_analysis[0] = max(frames_analysis[0], frames_video[0])
            if frames_analysis[1] is None:
                frames_analysis[1] = frames_video[1]
            else:
                frames_analysis[1] = min(frames_analysis[1], frames_video[1])

        return frames_analysis
    
    
    def get_frame_roi(self, frame_ids=None):
        """ returns a slice object indicating where the given frame_ids lie in
        the range that was chosen for analysis """ 
        fmin, fmax = self.get_frame_range()
        if frame_ids is None:
            return slice(fmin, fmax + 1)
        else:
            return slice(np.searchsorted(frame_ids, fmin, side='left'),
                         np.searchsorted(frame_ids, fmax, side='right'))
        
        
    #===========================================================================
    # GROUND STATISTICS
    #===========================================================================

    
    def get_ground_changes(self, frames=None):
        """ returns shape polygons or polygon collections of ground area that
        was removed and added, respectively, over the chosen frames """
        if frames is None:
            frames = self.get_frame_range()

        # load the ground profiles
        ground_profile = self.data['pass2/ground_profile']
        ground0 = ground_profile.get_ground_profile(frames[0])
        ground1 = ground_profile.get_ground_profile(frames[1])
        
        # get the ground polygons
        left = min(ground0.points[0, 0], ground1.points[0, 0])
        right = max(ground0.points[-1, 0], ground1.points[-1, 0])
        depth = max(ground0.points[:, 1].max(), ground1.points[:, 1].max()) + 1
        poly0 = ground0.get_polygon(depth, left, right)
        poly1 = ground1.get_polygon(depth, left, right)
        
        # get the differences in the areas
        area_removed = poly0.difference(poly1)
        area_accrued = poly1.difference(poly0)
        
        return area_removed, area_accrued
        
        
    def get_ground_change_areas(self, frames=None):
        """ returns the area of the ground that was removed and added """
        area_removed, area_accrued = self.get_ground_changes(frames)
        return (area_removed.area * self.length_scale**2,
                area_accrued.area * self.length_scale**2)
        
        
    #===========================================================================
    # BURROW STATISTICS
    #===========================================================================
        
        
    def _get_burrow_tracks(self):
        """ return the burrow tracks or throw an error if not avaiable """
        pass_id = self.burrow_pass 
        try:
            return self.data['pass%d/burrows/tracks' % pass_id]
        except KeyError:
            raise ValueError('Data from pass %d is not available.' % pass_id)

   
    def find_burrow_predug(self, ret_track_id=False):
        """ identifies the predug from the burrow traces """

        burrow_tracks = self._get_burrow_tracks()
        predug_analyze_time = self.params['burrows/predug_analyze_time']
        area_threshold = self.params['burrows/predug_area_threshold']
        
        # iterate over all burrow tracks and find the burrow that was largest
        # after a short initial time, which indicates that it was the predug
        predug, predug_track_id, predug_area = None, None, 0 
        for track_id, burrow_track in enumerate(burrow_tracks):
            # get the burrow shortly after it has been detected
            t_analyze = burrow_track.track_start + predug_analyze_time
            try:
                burrow = burrow_track.get_burrow(t_analyze)
            except IndexError:
                # burrow is exists for shorter than predug_analyze_time
                pass
            else:
                if burrow.area > area_threshold and burrow.area > predug_area:
                    predug = burrow
                    predug_track_id = track_id
                    predug_area = burrow.area

        if ret_track_id:
            return predug, predug_track_id
        else:                    
            return predug
        
    
    @cache
    def get_burrow_predug(self, ret_track_id=False):
        """ loads the predug from the data.
        If `ret_track_id` is given, the burrow track that overlaps most with
        the predug is also returned.
        """
        try:
            predug_data = self.data['pass1/burrows/predug']
        except KeyError:
            self.logger.warn('Predug is not available.')
            predug_data = None
        
        predug = geometry.Polygon(predug_data)

        if ret_track_id:
            # find the burrow track that overlaps most with the predug
            predug_track_id = None
            
            if predug_data:
                # look at all the burrow tracks
                burrow_tracks = self._get_burrow_tracks()
                area_max = 0
                for track_id, burrow_track in enumerate(burrow_tracks):
                    burrow = burrow_track.last
                    area = burrow.polygon.intersection(predug).area
                    if area > area_max:
                        area_max = area
                        predug_track_id = track_id
                
            return predug, predug_track_id
        else:                    
            return predug
        
        
    def burrow_has_predug(self, burrow):
        """ check whether the burrow had a predug """ 
        predug = self.get_burrow_predug()
        return burrow.intersects(predug)

        
    def get_burrow_lengths(self):
        """ returns a list of burrows containing their length over time """
        burrow_tracks = self._get_burrow_tracks()
        
        results = []
        for burrow_track in burrow_tracks:
            # read raw data
            times = np.asarray(burrow_track.times)
            lenghts = [burrow.length for burrow in burrow_track.burrows]
            
            # get the indices in the analysis range
            idx = self.get_frame_roi(times)
            
            # append the data to the result
            data = np.c_[times[idx]*self.time_scale, lenghts[idx]]
            results.append(data)
                  
        return results
    
    
    def get_main_burrow(self):
        """ returns the track of the main burrow, which is defined to be the
        longest burrow """
        burrow_tracks = self._get_burrow_tracks()
        
        # find the longest burrow 
        main_track, main_length = None, 0
        for burrow_track in burrow_tracks:
            max_length = burrow_track.get_max_length()
            if max_length > main_length:
                main_track, main_length = burrow_track, max_length
                  
        return main_track
    
    
    def get_burrow_initiated(self, burrow):
        """ determines when the `burrow` was initiated """

        # determine the area threshold used for initiation detection
        area_threshold = self.params['burrows/initiation_threshold']
        if self.burrow_has_predug(burrow.last):
            area_threshold += self.get_burrow_predug().area
        
        # initiation is defined as the time point when the
        # burrow grew larger than the predug
        for time, burrow in itertools.izip(burrow.times, burrow.burrows):
            if burrow.area > area_threshold:
                time_start = time * self.time_scale
                break
        else:
            # burrow never got larger then the predug
            time_start = None
        return time_start

                    
    def get_burrow_peak_activity(self, burrow_track):
        """ determines the time point of the main burrowing activity for the
        given burrow """
        if burrow_track is None:
            return None
        times = burrow_track.times
        assert is_equidistant(times)
        time_delta =  (times[-1] - times[0])/(len(times) - 1)

        # ignore the initial frames
        ignore_interval = self.params['burrows/activity_ignore_interval']
        start = int(ignore_interval / time_delta)
        times = times[start:]
        
        if len(times) == 0:
            return None 
        
        # collect all the burrow areas
        areas = np.array([burrow.area
                          for burrow in burrow_track.burrows[start:]])
        
        # correct for predug if present
        predug = self.get_burrow_predug()
        if predug.area > 0:
            if burrow_track.last.intersects(predug):
                areas = np.clip(areas - predug.area, 0, np.inf)
        
        # do some Gaussian smoothing to get rid of fluctuations
        sigma = self.params['burrows/activity_smoothing_interval'] / time_delta
        ndimage.filters.gaussian_filter1d(areas, sigma, mode='nearest',
                                          output=areas)
        
        # calculate the rate of area increase
        area_rate = np.gradient(areas)
         
        # determine the time point of the maximal rate
        return times[np.argmax(area_rate)] * self.time_scale
    
    
    def get_burrow_growth_statistics(self, frames=None):
        """ calculates the area that was excavated during the frames given """
        if frames is None:
            frames = self.get_frame_range()

        burrow_tracks = self._get_burrow_tracks()

        # load the predug, if available
        predug, predug_track_id = self.get_burrow_predug(ret_track_id=True)
        # if it is not availble, both values will be None

        # find the burrow areas 
        area_excavated = 0
        burrows_grew_times = {}
        for track_id, burrow_track in enumerate(burrow_tracks):
            # check whether the track overlaps with the chosen frames
            if (burrow_track.track_start >= frames[1]
                  or burrow_track.track_end <= frames[0]):
                continue
            
            # get the burrow area at the beginning of the time interval
            try:
                b1_idx = burrow_track.get_burrow_index(frames[0])
            except IndexError:
                b1_idx = 0
                b1_area = 0 #< burrow did not exist in the beginning
            else:
                b1_area = burrow_track.burrows[b1_idx].area

            # get the burrow area at the end of the time interval
            try:
                b2_idx = burrow_track.get_burrow_index(frames[1])
            except IndexError:
                b2_idx = len(burrow_track)
                b2_area = burrow_track.last.area #< take the last known burrow
            else:
                b2_area = burrow_track.burrows[b2_idx].area
                
            # correct for the predug
            if predug_track_id == track_id:
                b1_area = max(0, b1_area - predug.area)
                b2_area = max(0, b2_area - predug.area)
                
            # get the excavated area
            area_excavated += b2_area - b1_area
            
            # check at what times the burrow grew
            burrow = burrow_track.burrows[b1_idx]
            for b_idx in xrange(b1_idx + 1, b2_idx):
                burrow_next = burrow_track.burrows[b_idx]
                
                # check whether the burrow grew and is larger than the predug
                if (burrow_next.area > burrow.area and #< burrow grew
                    (predug_track_id != track_id or    #< burrow has no predug
                     burrow.area > predug.area)        #< burrow is large enough
                    ):
                    burrows_grew_times[burrow_track.times[b_idx]] = True
                    
                burrow = burrow_next
                
        # calculate the total time during which burrows were extended
        time_burrow_grew = (len(burrows_grew_times) 
                            * self.params['burrows/adaptation_interval'])
                
        return area_excavated, time_burrow_grew
 
            
    #===========================================================================
    # MOUSE STATISTICS
    #===========================================================================


    def get_mouse_track_data(self, attribute='pos', night_only=True):
        """ returns information about the tracked mouse position or velocity """
        try:
            # read raw data for the frames that we are interested in 
            mouse_track = self.data['pass2/mouse_trajectory']
            
        except KeyError:
            raise RuntimeError('The mouse trajectory has to be determined '
                               'before the track data can be extracted.')
        
        else:            
            # extract the right attribute from the mouse track
            if attribute == 'trajectory_smoothed':
                sigma = self.data['parameters/tracking/position_smoothing_window']
                data = mouse_track.trajectory_smoothed(sigma)
                
            elif attribute == 'velocity':
                sigma = self.data['parameters/tracking/position_smoothing_window']
                mouse_track.calculate_velocities(sigma=sigma)
                data = mouse_track.velocity
                
            else:
                data = getattr(mouse_track, attribute)
                
            # restrict the data to the night period
            if night_only:
                data = data[self.get_frame_roi()]
                
        return data


    def get_mouse_trajectory(self, smoothed=True):
        """ returns the mouse positions, smoothed if requested """
        if smoothed:
            return self.get_mouse_track_data('trajectory_smoothed')
        else:
            return self.get_mouse_track_data('pos')


    def get_mouse_distance(self):
        """ returns the total distance traveled by the mouse """
        trajectory = self.get_mouse_track_data('trajectory_smoothed')

        # calculate distance
        valid = np.isfinite(trajectory[:, 0])
        distance = curves.curve_length(trajectory[valid, :])
        return distance

    
    def get_mouse_velocities(self, invalid=0):
        """ returns an array with mouse velocities as a function of time.
        Velocities at positions where we have no information about the mouse
        are set to `invalid`, if `invalid` is not None """
        velocity = self.get_mouse_track_data('velocity')
        if invalid is not None:
            velocity[np.isnan(velocity)] = invalid
        return velocity * self.length_scale / self.time_scale


    def get_mouse_running_peak(self):
        """ determines the time point of the main running activity of the mouse
        """
        # get the velocity of the mouse and replace nan with zeros
        velocity = self.get_mouse_track_data('velocity')
        velocity = np.nan_to_num(velocity)

        # calculate scalar speed
        speed = np.hypot(velocity[:, 0], velocity[:, 1])

        # get smoothing range
        sigma = self.params['mouse/activity_smoothing_interval']

        # compress the data by averaging over consecutive windows
        window = max(1, int(sigma/100))
        if window > 1:
            end = int(len(velocity) / window) * window
            speed = speed[:end].reshape((-1, window)).mean(axis=1)

        # do some Gaussian smoothing to get rid of fluctuations
        ndimage.filters.gaussian_filter1d(speed, sigma/window, mode='constant',
                                          cval=0, output=speed)
        
        # determine the time point of the maximal rate
        return np.argmax(speed) * window * self.time_scale


    def get_mouse_state_vector(self, states=None, ret_states=False):
        """ returns the a vector of integers indicating in what state the mouse
        was at each point in time 
        
        If a list of `states` is given, only these states are included in
            the result.
        If `ret_states` is True, the states that are considered are returned
            as a list
        """
        mouse_state = self.get_mouse_track_data('states')

        # set the default list of states if it not already set
        if states is None:
            states = self.mouse_states_default
            
        # convert the mouse states into integers according to the defined states
        lut = mouse.state_converter.get_state_lookup_table(states)
        state_cat = np.array([-1 if lut[state] is None else lut[state]
                              for state in mouse_state])
            
        if ret_states:
            return states, state_cat
        else:
            return state_cat


    def get_mouse_state_durations(self, states=None, ret_states=False):
        """ returns the durations the mouse spends in each state 
        
        If a list of `states` is given, only these states are included in
            the result.
        If `ret_states` is True, the states that are considered are returned
            as a list
        """
        states, state_cat = self.get_mouse_state_vector(states, ret_states=True)
            
        # get durations
        durations = collections.defaultdict(list)
        for state, start, end in contiguous_int_regions_iter(state_cat):
            durations[state].append(end - start)
            
        # convert to numpy arrays and add units
        durations = {states[key]: np.array(value) * self.time_scale
                     for key, value in durations.iteritems()}
            
        if ret_states:
            return durations, states
        else:
            return durations
    
    
    def get_mouse_state_transitions(self, states=None, duration_threshold=0,
                                    ret_states=False):
        """ returns the durations the mouse spends in each state before 
        transitioning to another state
        
        If a list of `states` is given, only these states are included in
        the result.
        Transitions with a duration [in seconds] below duration_threshold will
        not be included in the results.
        """
        mouse_state = self.get_mouse_track_data('state')

        # set the default list of states if it not already set
        if states is None:
            states = self.mouse_states_default
            
        # add a unit to the duration threshold if it does not have any
        if self.use_units and isinstance(duration_threshold, self.units.Quantity):
            duration_threshold /= self.units.second

        # cluster mouse states according to the defined states
        lut = mouse.state_converter.get_state_lookup_table(states)
        state_cat = [-1 if lut[state] is None else lut[state]
                     for state in mouse_state]
            
        # get transitions
        transitions = collections.defaultdict(list)
        last_trans = 0
        for k in np.nonzero(np.diff(state_cat) != 0)[0]:
            if state_cat[k] < 0 or state_cat[k + 1] < 0:
                # this transition involves uncategorized states
                continue
            duration = (k - last_trans)
            if duration > duration_threshold:
                trans = (states[state_cat[k]], states[state_cat[k + 1]])
                transitions[trans].append(duration)
            last_trans = k
            
        # convert to numpy arrays and add units
        transitions = {key: np.array(value) * self.time_scale
                       for key, value in transitions.iteritems()}
            
        if ret_states:
            return transitions, states
        else:
            return transitions
            
    
    def get_mouse_transition_matrices(self, ret_states=False, **kwargs):
        """ returns the matrix of transition rates between different states.
        ret_states indicates whether a list indicating which row/column corresponds
            to which state should also be returned. """
            
        transitions, states = self.get_mouse_state_transitions(ret_states=True,
                                                               **kwargs)
            
        # build the matrix
        rates = np.empty((len(states), len(states)))
        rates.fill(np.nan)
        counts = np.zeros_like(rates)
        states = sorted(states)
        lut = {s: k for k, s in enumerate(states)}
        for trans, lengths in transitions.iteritems():
            # calculate the transition rate            
            rate = 1/np.mean(lengths)
            rates[lut[trans[0]], lut[trans[1]]] = rate
            counts[lut[trans[0]], lut[trans[1]]] = len(lengths)
            
        if ret_states:
            return rates, counts, states
        else:
            return rates, counts
            
    
    def get_mouse_transition_graph(self, **kwargs):
        """ calculate the graph representing the transitions between
        different states of the mouse """ 
        transitions = self.get_mouse_state_transitions(**kwargs)

        graph = nx.MultiDiGraph()
        nodes = collections.defaultdict(int)
        for trans, lengths in transitions.iteritems():
            # get node names 
            u = mouse.state_converter.symbols_repr(trans[0])
            v = mouse.state_converter.symbols_repr(trans[1])

            # get statistics            
            rate = 1/np.mean(lengths)
            nodes[u] += sum(lengths)

            # add the edge
            graph.add_edge(u, v, rate=rate, count=len(lengths))
        
        # add the nodes with additional data
        for node, duration in nodes.iteritems():
            graph.add_node(node, duration=duration)
        
        return graph
    
    
    def plot_mouse_transition_graph(self, ax=None, **kwargs):
        """ show the graph representing the transitions between
        different states of the mouse.
        
        Node size relates to the average duration the mouse spend there
        Line widths are related to the number of times the link was used
        Line colors relate to the transition rate between different states
        """
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib import patches, cm, colors, colorbar
        
        # get the transition graph
        graph = self.get_mouse_transition_graph(**kwargs)
        
        def log_scale(values, range_from, range_to):
            """ scale values logarithmically, where the interval range_from is
            mapped onto the interval range_to """
            values = np.asarray(values)
            log_from = np.log(range_from)
            scaled = (np.log(values) - log_from[0])/(log_from[1] - log_from[0])
            return scaled*(range_to[1] - range_to[0]) + range_to[0]
        
        # hard-coded node positions
        pos = {'unknown': (2, 2),
               'dimple': (0, 1),
               'air': (1.5, 3),
               'hill': (0, 2),
               'valley': (1.2, 1.5),
               'sand': (1.5, 0),
               'burrow': (0, 0)}

        # prepare the plot
        if ax is None:
            ax = plt.gca()
        ax.axis('off')

        # plot nodes
        nodes = graph.nodes(data=True)
        durations = [node[1].get('duration', np.nan) for node in nodes]
        max_duration = max(durations)
        node_sizes = log_scale(durations,
                               range_from=(1, max_duration),
                               range_to=(10, 5000))
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, ax=ax)
        
        # plot the edges manually because of directed graph
        edges = graph.edges(data=True)
        max_rate = max(edge[2]['rate'] for edge in edges)
        max_count = max(edge[2]['count'] for edge in edges)
        curve_bend = 0.08 #< determines the distance of the two edges between nodes
        colormap = cm.autumn
        for u, v, data in edges:
            # calculate edge properties
            width = log_scale(data['count'],
                              range_from=[10, max_count],
                              range_to=[1, 5])
            width = np.clip(width, 0, 10)
            color = colormap(data['rate']/max_rate)
            
            # get points
            p1, p2 = np.array(pos[u]), np.array(pos[v])
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            pm = np.array((p1[0] + dx/2 - curve_bend*dy,
                           p1[1] + dy/2 + curve_bend*dx))
        
            # plot Bezier curve
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path((p1, pm, p2), codes)
            patch = patches.PathPatch(path, facecolor='none',
                                      edgecolor=color, lw=width)
            ax.add_patch(patch)
            
            # add arrow head
            if width > 1:
                pm = np.array((p1[0] + dx/2 - 0.75*curve_bend*dy,
                               p1[1] + dy/2 + 0.75*curve_bend*dx))
                dp = p2 - pm
                dp /= np.linalg.norm(dp)
                pc_diff = 0.1*dp
                pc2 = p2 - 0.6*dp
                ax.arrow(pc2[0], pc2[1], pc_diff[0], pc_diff[1],
                         head_width=0.1,
                         edgecolor='none', facecolor=color)
                
        # add a colorbar explaining the color scheme
        cax = ax.figure.add_axes([0.87, 0.1, 0.03, 0.8])
        norm = colors.Normalize(vmin=0, vmax=1)
        cb = colorbar.ColorbarBase(cax, cmap=colormap, norm=norm,
                                   orientation='vertical', ticks=[0, 1])
        cb.set_label('Transition rate')
        cax.set_yticklabels(('lo', 'hi'))

        # plot the labels manually, since nx.draw_networkx_labels seems to be broken on mac
        for node in graph.nodes():
            x, y = pos[node]
            ax.text(x, y, node,
                    horizontalalignment='center',
                    verticalalignment='center')
        
        plt.sca(ax)
        return ax

    
    #===========================================================================
    # GENERAL ROUTINES
    #===========================================================================


    def get_mouse_ground_distances(self, nigth_only=False):
        """ return the distance of the mouse to the ground for all time points.
        Negative distances indicate that the mouse is below the ground.
        """
        # try loading the distances from the tracked data
        try:
            mouse_ground_dists = self.get_mouse_track_data(
                                           'ground_dist', night_only=nigth_only)
        except RuntimeError:
            # data from second pass is not available
            mouse_ground_dists = None

        # if this it not possible, try to estimate it from the first pass
        if mouse_ground_dists is None:
            raise NotImplementedError
            #< TODO fill in the code

        # make sure that the data is not completely useless
        if np.all(np.isnan(mouse_ground_dists)):
            raise RuntimeError('The distance of the mouse to the ground is not '
                               'available. Either the second pass has not '
                               'finished yet or there was a general problem '
                               'with the video analysis.')
            
        return mouse_ground_dists


    def get_mouse_ground_distance_max(self, frame_ivals):
        """ determines the maximal distance of the mouse to the ground line
        during the given frame_slices """
        
        # load data
        trajectory = self.get_mouse_track_data()
        trail_lengths = self.data['pass2/mouse_trajectory'].ground_dist
        ground_profile = self.data['pass2/ground_profile']
        
        # iterate over all frame intervals
        res_diagonal, res_vertical = [], []
        for a, b in frame_ivals:
            max_diagonal, max_vertical = -np.inf, -np.inf
            # iterate over all frames in this interval
            for frame_id in xrange(a, b):
                # retrieve mouse position
                try:
                    pos = trajectory[frame_id]
                except IndexError:
                    break
                if np.isnan(pos[0]):
                    continue
                trail_length = trail_lengths[frame_id]
                
                if max_diagonal < 0 or trail_length > 0:
                    # mouse was either never under ground or it is currently
                    # under ground
                     
                    ground = ground_profile.get_ground_profile(frame_id)

                    # get vertical distance
                    dist_vert = pos[1] - ground.get_y(pos[0])
                    max_vertical = max(max_vertical, dist_vert)
                
                    # Here, we use that max_vertical >= max_diagonal
                    if dist_vert > max_diagonal:
                        dist_diagonal = ground.get_distance(pos, signed=True)
                        max_diagonal = max(max_diagonal, dist_diagonal)
                
            res_diagonal.append(max_diagonal)
            res_vertical.append(max_vertical)
            
        return np.array(res_diagonal), np.array(res_vertical)              

    
    def get_statistics_periods(self, keys=None, slice_length=None):
        """ calculate statistics given in `keys` for consecutive time slices.
        If `keys` is None, all statistics are calculated.
        If `slice_length` is None, the statistics are calculated for the entire
            video. """
        if keys is None:
            keys = OmniContainer()
            
        # determine the frame slices
        frame_range = self.get_frame_range()
        if slice_length:
            frame_range = range(frame_range[0], frame_range[1],
                                int(slice_length))
        frame_ivals = [(a, b + 1) for a, b in itertools.izip(frame_range,
                                                             frame_range[1:])]
        frame_slices = [slice(a, b + 1) for a, b in frame_ivals]

        # length of the periods
        period_durations = (np.array([(b - a + 1) for a, b in frame_ivals]) 
                            * self.time_scale) 

        # save the time slices used for analysis
        result = {'frame_interval': frame_ivals,
                  'period_start': [a*self.time_scale for a, _ in frame_ivals],
                  'period_end': [b*self.time_scale for _, b in frame_ivals],
                  'period_duration': [(b - a)*self.time_scale 
                                      for a, b in frame_ivals]}

        # get the area changes of the ground line
        if 'ground_removed' in keys or 'ground_accrued' in keys:
            area_removed, area_accrued = [], []
            for f in frame_slices:
                poly_rem, poly_acc = self.get_ground_changes((f.start, f.stop))
                area_removed.append(poly_rem.area)
                area_accrued.append(poly_acc.area)

            if 'ground_removed' in keys:
                result['ground_removed'] = area_removed * self.length_scale**2
            if 'ground_accrued' in keys:
                result['ground_accrued'] = area_accrued * self.length_scale**2

        # get durations of the mouse being in different states        
        for key, pattern in (('time_spent_moving', '...M'), 
                             ('time_at_burrow_end', '.(B|D)E.')):
            # special case in which the calculation has to be done
            c = (key == 'time_at_burrow_end' and 'mouse_digging_rate' in keys)
            # alternatively, the computation might be requested directly
            if c or key in keys:
                states = self.get_mouse_state_vector([pattern])
                duration = [np.count_nonzero(states[t_slice] == 0)
                            for t_slice in frame_slices]
                result[key] = duration * self.time_scale
                key_fraction = key.replace('time_', 'fraction_')
                result[key_fraction] = np.array(result[key]) / period_durations

        # get velocity statistics
        speed_statistics = {
            'mouse_speed_mean': lambda x: np.nan_to_num(np.array(x)).mean(),
            'mouse_speed_mean_valid': lambda x: np.nanmean(x),
            'mouse_speed_max': lambda x: np.nanmax(x)
        }
        if any(key in keys for key in speed_statistics.keys()):
            velocities = self.get_mouse_velocities()
            speed = np.hypot(velocities[:, 0], velocities[:, 1])
            for key, stat_func in speed_statistics.iteritems():
                res = [stat_func(speed[t_slice]) for t_slice in frame_slices]
                result[key] = np.array(res) * self.speed_scale
        
        # get distance statistics
        if 'mouse_distance_covered' in keys:
            trajectory = self.get_mouse_trajectory()
            dist = []
            for t_slice in frame_slices:
                trajectory_part = trajectory[t_slice]
                valid = np.isfinite(trajectory_part[:, 0])
                dist.append(curves.curve_length(trajectory_part[valid]))
            result['mouse_distance_covered'] = dist * self.length_scale

        if 'mouse_trail_longest' in keys:
            ground_dist = self.get_mouse_track_data('ground_dist')
            dist = [-np.nanmin(ground_dist[t_slice])
                    for t_slice in frame_slices]
            result['mouse_trail_longest'] = dist * self.length_scale
            
        if 'mouse_deepest_diagonal' in keys or 'mouse_deepest_vertical' in keys:
            dist_diag, dist_vert = self.get_mouse_ground_distance_max(frame_ivals)
            if 'mouse_deepest_diagonal' in keys:
                result['mouse_deepest_diagonal'] = dist_diag * self.length_scale
            if 'mouse_deepest_vertical' in keys:
                result['mouse_deepest_vertical'] = dist_vert * self.length_scale

        # get statistics about the burrow evolution
        if any(key in keys for key in ('burrow_area_excavated',
                                       'mouse_digging_rate',
                                       'time_burrow_grew')):
            
            stats = [self.get_burrow_growth_statistics((f.start, f.stop))
                     for f in frame_slices]
            stats = np.array(stats)
            
            if 'burrow_area_excavated' in keys or 'mouse_digging_rate' in keys:
                try:
                    area_excavated = stats[:, 0] * self.length_scale**2
                except IndexError:
                    area_excavated = []
                result['burrow_area_excavated'] = area_excavated
                 
            if 'time_burrow_grew' in keys:
                try:
                    burrow_time = stats[:, 1] * self.time_scale
                except IndexError:
                    burrow_time = []
                result['time_burrow_grew'] = burrow_time 
                result['fraction_burrow_grew'] = burrow_time / period_durations 
                                        
        # calculate the digging rate by considering burrows and the mouse
        if 'mouse_digging_rate' in keys:
            time_min = self.params['mouse/digging_rate_time_min']
            if self.use_units:
                time_min *= self.time_scale
                unit_rate = self.length_scale**2 / self.time_scale
                area_min = 0 * self.length_scale**2
            else:
                unit_rate = 1
                area_min = 0
            # calculate the digging rate
            digging_rate = []
            for area, time in itertools.izip(result['burrow_area_excavated'],
                                             result['time_at_burrow_end']):
                if area > area_min and time > time_min:
                    digging_rate.append(area / time)
                else:
                    digging_rate.append(np.nan * unit_rate)
            result['mouse_digging_rate'] = digging_rate

        # determine the remaining keys
        if not isinstance(keys, OmniContainer):
            keys = set(keys) - set(result.keys())  

        if not isinstance(keys, OmniContainer):
            keys = set(keys) - set(result.keys())
            if keys:
                # report statistics that could not be calculated
                self.logger.warn('The following statistics are not defined in '
                                 'the algorithm and could therefore not be '
                                 'calculated: %s', ', '.join(keys))
        
        return result
        
    
    def get_statistics(self, keys=None):
        """ calculate statistics given in `keys` for the entire experiment.
        If `keys` is None, all statistics are calculated. """
        if keys is None:
            keys = OmniContainer()
        
        result = {}
        
        # predug statistics
        if 'predug_area' in keys:
            predug = self.get_burrow_predug()
            result['predug_area'] = predug.area * self.length_scale**2
        
        # check if the burrows need to be analyzed
        if any(key in keys for key in ('burrow_area_total',
                                       'burrow_length_total',
                                       'burrow_length_max')):
            # load the burrow tracks and get the last time frame
            length_max, length_total, area_total = 0, 0, 0
            burrow_tracks = self._get_burrow_tracks()
            if burrow_tracks:
                last_time = max(bt.track_end for bt in burrow_tracks)
                # gather burrow statistics
                for bt in burrow_tracks:
                    if bt.track_end == last_time:
                        length_total += bt.last.length
                        if bt.last.length > length_max:
                            length_max = bt.last.length
                        area_total += bt.last.area
                        
            # correct for predug
            predug = self.get_burrow_predug()
            area_total -= predug.area

            # save the data
            if 'burrow_length_max' in keys:
                result['burrow_length_max'] = length_max * self.length_scale
            if 'burrow_length_total' in keys:
                result['burrow_length_total'] = length_total * self.length_scale
            if 'burrow_area_total' in keys:
                result['burrow_area_total'] = area_total * self.length_scale**2

        # check if the main burrow needs to be analyzed
        if any(key in keys for key in ('burrow_main_initiated',
                                       'burrow_main_peak_activity')):
            # determine the main burrow
            burrow_main = self.get_main_burrow()
            
            # check when it was initiated
            if 'burrow_main_initiated' in keys:
                if burrow_main:
                    initated = self.get_burrow_initiated(burrow_main)
                else:
                    initated = None
                result['burrow_main_initiated'] = initated
                
            # check for the peak activity
            if 'burrow_main_peak_activity' in keys:
                time_peak = self.get_burrow_peak_activity(burrow_main)
                result['burrow_main_peak_activity'] = time_peak

        # calculate statistics of the mouse trajectory
        if 'mouse_running_peak' in keys:
            result['mouse_running_peak'] = self.get_mouse_running_peak()
        
        # determine the remaining keys
        if not isinstance(keys, OmniContainer):
            keys = set(keys) - set(result.keys())  
        
        if keys:
            # fill in the remaining keys by using the statistics function that
            # usually is used for calculating statistics on regular periods
            result_periods = self.get_statistics_periods(keys)
            for k, v in result_periods.iteritems():
                result[k] = v[0]
        
        return result
            
    
    def find_problems(self):
        """ checks for certain common problems in the results and returns a
        dictionary with identified problems """
        problems = {}
        
        for pass_cls in self.pass_classes:
            try:
                state = pass_cls.get_pass_state(self.data)
            except AttributeError:
                continue
            if state['state'] == 'error':
                problems.update(problems)
                
        return problems

