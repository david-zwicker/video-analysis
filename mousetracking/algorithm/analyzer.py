'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Contains a class that can be used to analyze results from the tracking
'''

from __future__ import division

import collections

import numpy as np
import networkx as nx

from .data_handler import DataHandler
from .objects import mouse
from video.utils import contiguous_int_regions_iter

try:
    import pint
    UNITS_AVAILABLE = True
except (ImportError, ImportWarning):
    UNITS_AVAILABLE = False



class Analyzer(DataHandler):
    """ class contains methods to analyze the results of a video """
    
    use_units = True
    
    def __init__(self, *args, **kwargs):
        super(Analyzer, self).__init__(*args, **kwargs)
        
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
        
        
    def get_frame_range(self):
        """ returns the range of frames that is going to be analyzed """
        frames = self.data['parameters/analysis/frames']
        frames_video = self.data['pass1/video/frames']
        if frames is None:
            frames = frames_video
        else:
            frames = (max(frames[0], frames_video[0]),
                      min(frames[1], frames_video[1]))
        return frames
    
    
    def get_frame_roi(self, frame_ids=None):
        """ returns a slice object indicating where the given frame_ids lie in
        the range that was choosen for analysis """ 
        fmin, fmax = self.get_frame_range()
        if frame_ids is None:
            return slice(fmin, fmax + 1)
        else:
            return slice(np.searchsorted(frame_ids, fmin, side='left'),
                         np.searchsorted(frame_ids, fmax, side='right'))
        
        
    #===========================================================================
    # BURROW STATISTICS
    #===========================================================================

        
    def get_burrow_lengths(self):
        """ returns a list of burrows containing their length over time """
        burrow_tracks = self.data['pass1/burrows/tracks']
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
    
    
    #===========================================================================
    # MOUSE STATISTICS
    #===========================================================================


    def get_mouse_velocities(self):
        """ returns an array with mouse velocities as a function of time """
        # read data
        sigma = self.data['parameters/mouse/speed_smoothing_window']
        mouse_trajectory = self.data['pass2/mouse_trajectory']
        
        # calculate and return velocity
        mouse_trajectory.calculate_velocities(sigma=sigma)
        velocity = mouse_trajectory.velocity[self.get_frame_roi()]
        return velocity * self.length_scale / self.time_scale

    
    def get_mouse_state_durations(self, states=None, ret_states=False):
        """ returns the durations the mouse spends in each state 
        
        If a list of `states` is given, only these states are included in
        the result.
        """
        try:
            # read raw data for the frames that we are interested in 
            mouse_state = self.data['pass2/mouse_trajectory'].states
            mouse_state = mouse_state[self.get_frame_roi()]
        except KeyError:
            raise RuntimeError('The mouse trajectory has to be determined '
                               'before the transitions can be analyzed.')

        # set the default list of states if it not already set
        if states is None:
            states = ('.A.', '.H.', '.V.', '.D.', '.B ', '.BE', '...')
            
        # cluster mouse states according to the defined states
        lut = mouse.state_converter.get_state_lookup_table(states)
        state_cat = [-1 if lut[state] is None else lut[state]
                     for state in mouse_state]
            
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
        try:
            # read raw data for the frames that we are interested in 
            mouse_state = self.data['pass2/mouse_trajectory'].states
            mouse_state = mouse_state[self.get_frame_roi()]
        except KeyError:
            raise RuntimeError('The mouse trajectory has to be determined '
                               'before the transitions can be analyzed.')

        # set the default list of states if it not already set
        if states is None:
            states = ('.A.', '.H.', '.V.', '.D.', '.B ', '.BE', '...')
            
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

    
    def find_problems(self):
        """ checks for certain common problems in the results and returns a
        dictionary with identified problems """
        problems = {}
        
        # check for ground line that went up to the roof
        ground_profile = self.data['pass2/ground_profile']
        if np.max(ground_profile.profiles[-1, :, 1]) < 2:
            problems['ground_through_roof'] = True
            
        # check the number of frames that were analyzed
        frame_count_1 = self.data['pass1/video/frames_analyzed']
        frame_count_3 = self.data['pass3/video/frames_analyzed']
        if frame_count_3 < 0.99*frame_count_1:
            problems['pass_3_stopped_early'] = True
        
        return problems
    
        


                    
                
                    