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


class Analyzer(DataHandler):
    """ class contains methods to analyze the results of a video """
    
    def __init__(self, *args, **kwargs):
        super(Analyzer, self).__init__(*args, **kwargs)
        
        self.time_scale = self.data['video/analyzed/fps']
        
    
    def get_burrow_lengths(self):
        """ returns a list of burrows containing their length over time """
        burrow_tracks = self.data['pass1/burrows/tracks']
        results = []
        for burrow_track in burrow_tracks:
            times = np.asarray(burrow_track.times)/self.time_scale
            lenghts = [burrow.length for burrow in burrow_track.burrows]
            data = np.c_[times, lenghts]
            results.append(data)
                  
        return results
    
    
    def get_mouse_state_transitions(self):
        """ returns the durations the mouse spends in each state before 
        transitioning to another state """
        try:
            mouse_state = self.data['pass2/mouse_trajectory'].states
        except KeyError:
            self.logger('The mouse trajectory has to be determined before '
                        'the transitions can be analyzed.')
            
        # get transitions
        transitions = collections.defaultdict(list)
        last_trans = 0
        for k in np.nonzero(np.diff(mouse_state) != 0)[0]:
            trans = (mouse_state[k], mouse_state[k + 1])
            transitions[trans].append(k - last_trans)
            last_trans = k
            
        return transitions
            
    
    def get_mouse_transition_graph(self):
        """ calculate the graph representing the transitions between
        different states of the mouse """ 
        transitions = self.get_mouse_state_transitions()

        graph = nx.MultiDiGraph()
        nodes = collections.defaultdict(int)
        for trans, lengths in transitions.iteritems():
            # get node names 
            u = mouse.STATES[trans[0]]
            v = mouse.STATES[trans[1]]

            # get statistics            
            rate = 1/np.mean(lengths)
            nodes[u] += sum(lengths)
            
            # add the edge
            graph.add_edge(u, v, rate=rate, count=len(lengths))
        
        # add the nodes with additional data
        for node, duration in nodes.iteritems():
            graph.add_node(node, duration=duration)
        
        return graph
    
    
    def show_mouse_transition_graph(self):
        """ show the graph representing the transitions between
        different states of the mouse """
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib import patches
        
        # get the transition graph
        graph = self.get_mouse_transition_graph()
        
        def log_scale(values, range_from, range_to):
            """ scale values logarithmically, where the interval range_from is
            mapped onto the interval range_to """
            values = np.asarray(values)
            log_from = np.log(range_from)
            scaled = (np.log(values) - log_from[0])/(log_from[1] - log_from[0])
            return scaled*(range_to[1] - range_to[0]) + range_to[0]
        
        # hard-coded node positions
        pos = {'unknown': (2, 2),
               'air': (1.5, 3),
               'hill': (0, 2),
               'valley': (1, 1.5),
               'sand': (1.5, 0),
               'burrow': (0, 0)}

        # plot nodes
        nodes = graph.nodes(data=True)
        max_duration = max(node[1]['duration'] for node in nodes)
        node_sizes = log_scale([node[1]['duration'] for node in nodes],
                               range_from=(1, max_duration),
                               range_to=(10, 5000))
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes)
        
        # plot the edges
        edges = graph.edges(data=True)
        max_rate = max(edge[2]['rate'] for edge in edges)
        max_count = max(edge[2]['count'] for edge in edges)
        curve_bend = 0.05
        for u, v, data in edges:
            # calculate edge properties
            width = log_scale(data['count'],
                              range_from=[10, max_count],
                              range_to=[1, 5])
            width = np.clip(width, 0, 5)
            
            # get points
            p1, p2 = pos[u], pos[v]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            pm = (p1[0] + dx/2 - curve_bend*dy,
                  p1[1] + dy/2 + curve_bend*dx)
        
            # plot Bezier curve
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path((p1, pm, p2), codes)
            patch = patches.PathPatch(path, facecolor='none',
                                      edgecolor=str(0.3 + 0.7*data['rate']/max_rate),
                                      lw=width)
            plt.gca().add_patch(patch)

        # plot the labels manually, since nx.draw_networkx_labels seems to be broken on mac
        for label, (x, y) in pos.iteritems(): 
            plt.text(x, y, label,
                     horizontalalignment='center',
                     verticalalignment='center'
                     )
        
        # tweak display
        plt.axis('off')
        plt.show()
                    
                
                    