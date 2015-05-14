'''
Created on Apr 28, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>


contains functions that are useful for image analysis
'''

from __future__ import division

import cv2
import numpy as np
import networkx as nx
from shapely import geometry

import curves

        
        
class MorphologicalGraph(nx.MultiGraph):
    """ class that represents a morphological graph.
    Note that a morphological graph generally might have parallel edges.
    """
    
    def __init__(self, *args, **kwargs):
        super(MorphologicalGraph, self).__init__(*args, **kwargs)
        self._unique_node_id = 0 #< used to ensure unique nodes
        
    
    def get_node_points(self):
        """ returns the coordinates of all nodes """
        return nx.get_node_attributes(self, 'coords').values()

    
    def get_edge_curves(self):
        """ returns a list of all edge curves """
        return nx.get_edge_attributes(self, 'curve').values()

    
    def add_node_point(self, coords):
        """ adds a node to the graph """
        # check whether this node already exists
        for key, value in nx.get_node_attributes(self, 'coords').iteritems():
            if np.allclose(coords, value):
                # it does already exist => return its node_id
                node_id = key
                break
        else:
            # it does not exist => create a new node
            self._unique_node_id += 1
            node_id = self._unique_node_id 
            self.add_node(node_id, {'coords': coords})
        return node_id
    
    
    def add_edge_line(self, n1, n2, curve):
        """ adds an edge to the graph """
        curve = np.asarray(curve)
        p1, p2 = curve[0], curve[-1]
        
        coords1, coords2 = self.node[n1]['coords'], self.node[n2]['coords']
        
        if (np.allclose(p1, coords1) and np.allclose(p2, coords2)):
            data = {'curve': curve,
                    'length': curves.curve_length(curve)}
            
        elif (np.allclose(p1, coords2) and  np.allclose(p2, coords1)):
            data = {'curve': curve[::-1],
                    'length': curves.curve_length(curve)}
            
        else:
            raise ValueError('The curve given by `curve` does not connect the '
                             'specified nodes.')

        if data['length'] > 1e-6:
            self.add_edge(n1, n2, attr_dict=data)
    
    
    def insert_node_into_edge(self, edge, point_id):
        """ inserts a node into the given edge at the point given by point id """
        if len(edge) == 2:
            edge = next(self.edges_iter(nbunch=edge, keys=True))
        
        # get the curve of the edge to intersect with
        curve = self.edge[edge[0]][edge[1]][edge[2]]['curve']
        
        # insert a new node    
        node_id = self.add_node_point(curve[point_id])

        # check which side is connected to which node
        p1, p2 = curve[0], curve[-1]
        coords1 = self.node[edge[0]]['coords']
        coords2 = self.node[edge[1]]['coords']
        
        if (np.allclose(p1, coords1) and np.allclose(p2, coords2)):
            self.add_edge_line(edge[0], node_id, curve[:point_id + 1])
            self.add_edge_line(node_id, edge[1], curve[point_id:])
            
        elif (np.allclose(p1, coords2) and  np.allclose(p2, coords1)):
            self.add_edge_line(edge[1], node_id, curve[:point_id + 1])
            self.add_edge_line(node_id, edge[0], curve[point_id:])
            
        else:
            raise ValueError('The edge (%d, %d) is inconsistent with its curve'
                             % edge[:2])
        
        # break the old edge at the intersection point
        self.remove_edge(*edge)
        
        return node_id

    def connect_point_to_edge(self, point, edge, point_id):
        """ adds a node representing the point and connect this point to
        the given edge by intersecting the edge at the given support point.
        An edge is specified by the ids of its two end nodes and an optional 
        integer if there are multiple edges between the nodes """
        # add new nodes and the edge between them
        node_point = self.add_node_point(point)
        node_int = self.insert_node_into_edge(edge, point_id) 
        
        self.add_edge_line(node_point, node_int,
                           curve=[point, self.node[node_int]['coords']])
    
    
    def translate(self, x, y):
        """ translate the whole graph in space by x and y """
        # change all nodes
        for _, data in self.nodes_iter(data=True):
            c = data['coords']
            data['coords'] = (c[0] + x, c[1] + y)
        # change all edges
        offset = np.array([x, y])
        for _, _, data in self.edges_iter(data=True):
            data['curve'] += offset
            
            
    def remove_short_edges(self, length_min=1):
        """ removes very short edges """
        for n1, n2, key, data in self.edges_iter(data=True, keys=True):
            if data['length'] < length_min:
                degrees = self.degree((n1, n2)) 
                if (1 in degrees.values()):
                    # edge connected to at least one end point
                    self.remove_edge(n1, n2)
                    for n, d in degrees.iteritems():
                        if d == 1:
                            self.remove_node(n)
                elif n1 == n2:
                    # loop
                    self.remove_edge(n1, n2, key)
    
    
    def simplify(self, epsilon=0):
        """ remove nodes with degree=2 and simplifies the curves describing the
        edges if epsilon is larger than 0 """
        # remove nodes with degree=2
        while True:
            for n, d in self.degree_iter():
                if d == 2:
                    try:
                        n1, n2 = self.neighbors(n)
                    except ValueError:
                        # this can happen for a self-loop, where n1 == n2
                        continue
                    # get the points
                    points1 = self.get_edge_data(n, n1)['curve']
                    points2 = self.get_edge_data(n, n2)['curve']

                    # remove the node and the edges
                    self.remove_edges_from([(n, n1), (n, n2)])
                    self.remove_node(n)
                    
                    # add the new edge
                    points = curves.merge_curves(points1, points2)
                    self.add_edge_line(n1, n2, points)
                    break
            else:
                break #< no changes => we're done!
    
        # simplify the curves describing the edges
        if epsilon > 0:
            for _, _, data in self.edges_iter(data=True):
                data['curve'] = curves.simplify_curve(data['curve'], epsilon)
                data['length'] = curves.curve_length(data['curve'])
    
    
    def get_point_on_edge(self, n1, n2, point_id):
        """ returns the point on the edge (n1, n2) that has the point_id """
        return self.get_edge_data(n1, n2)['curve'][point_id, :] 
    
    
    def get_closest_node(self, point):
        """ get the node that is closest to a given point.
        This function returns three values:
            * the id of the closest node
            * its coordinates
            * the distance of this node to the given point
        """
        node_min, coord_min, dist_min = None, None, np.inf
        for node, data in self.nodes_iter(data=True):
            dist = curves.point_distance(point, data['coords'])
            if dist < dist_min:
                node_min, coord_min, dist_min = node, data['coords'], dist
        
        return node_min, coord_min, dist_min
    
    
    def get_closest_edge(self, point):
        """ get the edge that is closest to a given point. This function returns
        three values:
            * the edge (indicated by the two nodes that it connects)
            * the id of the point on the connection curve
            * the distance between this point and the given point
        """
        point = geometry.Point(point)
        edge_min, dist_min, data_min = None, np.inf, None
        for n1, n2, key, data in self.edges_iter(data=True, keys=True):
            if 'linestring' in data:
                line = data['linestring']
            else:
                line = geometry.LineString(data['curve'])
                data['linestring'] = line
            dist = line.distance(point)
            if dist < dist_min:
                edge_min, dist_min, data_min = (n1, n2, key), dist, data

        # calculate the projection point
        if edge_min:
            # find the index of the projection point on the line
            coords = data_min['curve']
            #coords = self.get_edge_data(*edge_min)['curve']
            dists = np.linalg.norm(coords - np.array(point.coords), axis=1)
            projection_id = np.argmin(dists)
            dist = dists[projection_id]
        else:                
            projection_id = None
            
        return edge_min, projection_id, dist
                
            
    @classmethod
    def from_skeleton(cls, skeleton, copy=True, post_process=True):
        """ determines the morphological graph from the image `skeleton`
        `copy` determines whether the skeleton is copied before its modified
        `post_process` determines whether some post processing is performed
            that removes spurious edges and nodes
        """
        if copy:
            skeleton = skeleton.copy()
        graph = cls()
        
        # count how many neighbors each point has
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbors = cv2.filter2D(skeleton, -1, kernel) * skeleton

        # find an point with minimal neighbors to start iterating from
        neighbors_min = neighbors[neighbors > 0].min()
        ps = np.nonzero(neighbors == neighbors_min)        
        start_point = (ps[1][0], ps[0][0])
        
        # initialize graph by adding first node
        start_node  = graph.add_node_point(start_point)
        edge_seeds = {start_point: start_node}
        
        # iterate over all edges
        while edge_seeds:
            # pick new point from edge_seeds and initialize the point list
            p, start_node = edge_seeds.popitem()
            points = [graph.node[start_node]['coords']]
            
            # iterate along the edge
            while True:
                # handle the current point
                points.append(p)
                skeleton[p[1], p[0]] = 0
                
                # look at the neighborhood of the current point
                ps_n = skeleton[p[1]-1 : p[1]+2, p[0]-1 : p[0]+2]
                neighbor_count = ps_n.sum()
                #print neighbor_count
                
                if neighbor_count == 1:
                    # find the next point along this edge
                    dy, dx = np.nonzero(ps_n)
                    p = (p[0] + dx[0] - 1, p[1] + dy[0] - 1)
                else:
                    # the current point ends the edge
                    break
                
            # check whether we are close to another edge seed
            for p_seed in edge_seeds:
                dist = curves.point_distance(p, p_seed)
                if dist < 1.5:
                    # distance should be either 1 or sqrt(2)
                    if dist > 0:
                        points.append(p_seed)
                    node = edge_seeds.pop(p_seed)
                    points.append(graph.node[node]['coords'])
                    break
            else:
                # could not find a close edge seed => this is an end point
                node = graph.add_node_point(p)
                    
                # check whether we have to branch off other edges
                if neighbor_count > 0:
                    assert neighbor_count > 1
                    # current points is a crossing => branch off new edge seeds
                    # find all points from which we have to branch off
                    dps = np.transpose(np.nonzero(ps_n))
                    # initialize all edge seeds
                    seeds = set([(p[0] + dx - 1, p[1] + dy - 1)
                                 for dy, dx in dps])
                    while seeds:
                        # check the neighbor hood of the seed point
                        p_seed = seeds.pop()
                        skeleton[p_seed[1], p_seed[0]] = 0
                        ps_n = skeleton[p_seed[1]-1 : p_seed[1]+2,
                                        p_seed[0]-1 : p_seed[0]+2]
                        neighbor_count = ps_n.sum()
                        #print 'test_seed', p_seed, neighbor_count
                        if neighbor_count == 1:
                            edge_seeds[p_seed] = node
                        else:
                            # add more seeds
                            dps = np.transpose(np.nonzero(ps_n))
                            for dy, dx in dps:
                                p = (p_seed[0] + dx - 1, p_seed[1] + dy - 1)
                                if p not in seeds and p not in edge_seeds:
                                    seeds.add(p)
                
            # add the edge to the graph
            graph.add_edge_line(start_node, node, points)
              
        if post_process:
            # remove small edges and self-loops
            graph.remove_short_edges(4)
            # remove nodes of degree 2
            graph.simplify()
              
        return graph
    
    
        return 

    
    def debug_visualization(self, **kwargs):
        """ visualizes the morphological graph.
        Keyword arguments are passed on to the `debug.show_shape` function
        """
        from .. import debug
     
        debug.show_shape(geometry.MultiLineString(self.get_edge_curves()),
                         geometry.MultiPoint(self.get_node_points()),
                         **kwargs)
        
            
        