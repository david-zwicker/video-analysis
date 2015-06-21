#!/usr/bin/env python2
'''
Created on Jan 30, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import functools
import logging
import multiprocessing as mp
import os.path
import operator
import sys
import traceback

import cv2
import numpy as np
from scipy import spatial
from shapely import geometry
import pint

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)

from projects.mouse_burrows.algorithm.objects import Burrow, GroundProfile
from video.analysis import curves, image, regions, shapes
from utils import data_structures, math

from video import debug  # @UnusedImport


default_parameters = {
    'burrow_parameters': {'ground_point_distance': 2},
    'burrow/area_min': 10000,
    'burrow/branch_length_min': 50,
    'cage/width_norm': 85.5,
    'cage/width_min': 80,
    'cage/width_max': 95,
    'colors/burrow': (1, 1, 0),      #< burrow color in RGB
    'colors/ground_line': (0, 1, 0), #< ground line color in RGB
    'colors/scale_bar': (1, 1, 1),   #< scale bar color in RGB
    'colors/isolation_closing_radius': 10, #< radius of mask for closing op.
    'scale_bar/area_max': 1000,
    'scale_bar/length_min': 100,
    'scale_bar/dist_bottom': 0.1,
    'scale_bar/dist_left': 0.1,
    'scale_bar/length_cm': 10,
}


ScaleBar = collections.namedtuple('ScaleBar', ['size', 'angle'])



class AntfarmShapes(object):
    """ class that manages shapes in an antfarm """
    
    def __init__(self, parameters=None, name='', output_folder=None):
        """ initializes the polygon collection
        `polygons` is a list of polygons
        `parameters` are parameters for the algorithms of this class
        `name` is the name of the collection
        `debug_output` can be a folder to which debug output will be written 
        """
        self.name = name
        self.output_folder = output_folder

        self.burrows = []
        self.ground_line = None
        self.scale_bar = None
        
        self.params = default_parameters.copy()
        if parameters is not None:
            self.params.update(parameters)
    
    
    @classmethod
    def load_from_file(cls, path, **kwargs):
        """ load polygons from a file """
        # handle the file 
        _, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        obj = cls(None, name=name, **kwargs)
        
        # determine which loader to use for the individual files
        if ext == '.jpg' or ext == '.png':
            logging.debug('Use OpenCV image loader')
            if obj.output_folder:
                output_file = os.path.join(obj.output_folder, filename)
            else:
                output_file = None
                
            image = cv2.imread(path)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image) #< convert to RGB
            obj.load_from_image(image, output_file)
            
        else:
            raise ValueError("Don't know how to read `*%s` files" % ext)
            
        return obj
    

    def load_from_image(self, image, output_file=None):
        """ load the data from an image """
        # find the scale bar
        scale_mask = self.isolate_color(image, self.params['colors/scale_bar'])
        self.scale_bar = self.get_scalebar_from_image(scale_mask)

        # find the ground line
        ground_mask = self.isolate_color(image, self.params['colors/ground_line'],
                                         dilate=1)
        self.ground_line = self.get_groundline_from_image(ground_mask)

        # find all the burrows
        burrow_mask = self.isolate_color(image, self.params['colors/burrow'])
        self.burrows = self.get_burrows_from_image(burrow_mask, self.ground_line)
        
        # determine additional burrow properties
        for burrow in self.burrows:
            self.calculate_burrow_properties(burrow, self.ground_line)
            
        if output_file:
            self.make_debug_output(image, output_file)
         
         
    def _add_burrow_angle_statistics(self, burrow, ground_line):
        """ adds statistics about the burrow slopes to the burrow object """
        # determine the fraction of the burrow that goes upwards
        cline = burrow.centerline
        clen = curves.curve_segment_lengths(cline)
        length_upwards = clen[np.diff(cline[:, 1]) < 0].sum()
        
        # distinguish left from right burrows
        center = (ground_line.points[0, 0] + ground_line.points[-1, 0]) / 2
        burrow_on_left = (burrow.centroid[0] < center)
        cline_left2right = (cline[0, 0] < cline[-1, 0])
        if burrow_on_left ^ cline_left2right:
            # burrow is on the left and centerline goes from right to left
            # or burrow is on the right and centerline goes left to right
            # => We measured the correct portion of the burrow
            burrow.length_upwards = min(burrow.length, length_upwards)
            
        else:
            # burrow is on the left and centerline goes from left to right
            # or burrow is on the right and centerline goes right to left
            # => We measured the compliment of what we actually want
            burrow.length_upwards = max(0, burrow.length - length_upwards)
            
        
    def calculate_burrow_properties(self, burrow, ground_line=None):
        """ calculates additional properties of the burrow """
        
        # determine the burrow end points
        burrow.get_endpoints(ground_line)
        
        # determine the morphological graph
        graph = burrow.get_morphological_graph()
        length_min = self.params['burrow/branch_length_min']
        graph.remove_short_edges(length_min=length_min)
        graph.simplify()
        graph.debug_visualization()
        burrow.morphological_graph = graph
        
        if ground_line:
            self._add_burrow_angle_statistics(burrow, ground_line)
                
                
    def make_debug_output(self, image, filename):
        """ make debug output image """
        logging.info('Creating debug output')
        
        for burrow in self.burrows:
            # draw the morphological graph
            for points in burrow.morphological_graph.get_edge_curves():
                cv2.polylines(image, [np.array(points, np.int)],
                              isClosed=False, color=(255, 255, 255),
                              thickness=2)
            
            # draw the smooth centerline
            cline = burrow.centerline
            cv2.polylines(image, [np.array(cline, np.int)],
                          isClosed=False, color=(255, 0, 0), thickness=3)

            # mark the end points
            for e_p in burrow.endpoints:
                if e_p.is_exit:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                coords = tuple([int(c) for c in e_p.coords])
                cv2.circle(image, coords, 10, color, thickness=-1)

        cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image) #< convert to BGR
        cv2.imwrite(filename, image)
        
        logging.info('Wrote output file `%s`' % filename)


    def _get_line_from_contour(self, contour):
        """ determines a line described by a contour """
        # calculate the distance between all points on this contour
        dist = spatial.distance.pdist(contour, 'euclidean')
        dist = spatial.distance.squareform(dist)
        
        # start from the left most point and find all points
        p_cur = np.argmin(contour[:, 0])
        p_avail = np.ones(len(contour), np.bool)
        p_avail[p_cur] = False
        
        points = []
        while True:
            # add the current point to our list
            points.append(contour[p_cur, :])

            # find the closest points
            p_close = np.where((dist[p_cur, :] < 4) & p_avail)[0]
            if len(p_close) == 0:
                break
            
            # find the next point
            k = np.argmax(dist[p_cur, p_close])
            p_cur = p_close[k]
            
            # remove all old points that are in the same surrounding
            p_avail[p_close] = False
        return points
    
    
    def get_scalebar_from_image(self, mask):
        """ finds the scale bar in the image """
        # determine contours in the mask
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)[1]
        
        # pick the largest contour
        contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, closed=True))

        # determine the rectangle that describes the contour best
        _, (w, h), rot = cv2.minAreaRect(contour)
        
        if max(w, h) > self.params['scale_bar/length_min']:
            # we found the scale bar
            if w > h:
                scale_bar = ScaleBar(size=w, angle=rot)
            else:
                scale_bar = ScaleBar(size=h, angle=(rot + 90) % 180)
                
        else:
            logging.debug('Did not find any scale bar.')
            scale_bar = None
            
        return scale_bar

    
    def get_groundline_from_image(self, mask):
        """ load burrow polygons from an image """
        # get the skeleton of the image
        mask = image.mask_thinning(mask)
        
        # get the path between the two points in the mask that are farthest
        points = regions.get_farthest_points(mask, ret_path=True)
        
        # build the ground line from this
        ground_line = GroundProfile(points)
        
#         debug.show_shape(ground_line.linestring, background=mask)

        return ground_line
        
            
    def get_burrows_from_image(self, mask, ground_line):
        """ load burrow polygons from an image """
        # turn image into gray scale
        height, width = mask.shape
        
        # get a polygon for cutting away the sky
        above_ground = ground_line.get_polygon(0, left=0, right=width)         

        # determine contours in the mask
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]

        # iterate through the contours
        burrows = []
        for contour in contours:
            points = contour[:, 0, :]
            if len(points) <= 2:
                continue

            # get the burrow area
            area = cv2.contourArea(contour)

            if area < self.params['scale_bar/area_max']:
                # object could be a scale bar
                rect = shapes.Rectangle(*cv2.boundingRect(contour))

                at_left = (rect.left < self.params['scale_bar/dist_left']*width)
                max_dist_bottom = self.params['scale_bar/dist_bottom']
                at_bottom = (rect.bottom > (1 - max_dist_bottom) * height)
                hull = cv2.convexHull(contour) 
                hull_area = cv2.contourArea(hull)
                is_simple = (hull_area < 2*area)
                
                if at_left and at_bottom and is_simple:
                    # the current polygon is the scale bar
                    _, (w, h), _ = cv2.minAreaRect(contour)
                    
                    if max(w, h) > self.params['scale_bar/length_min']:
                        raise RuntimeError('Found something that looks like a '
                                           'scale bar')

            if area > self.params['burrow/area_min']:
                # build polygon out of the contour points
                burrow_poly = geometry.Polygon(points)

                # regularize the points to remove potential problems                
                burrow_poly = regions.regularize_polygon(burrow_poly)
                
#                 debug.show_shape(geometry.Polygon(points), above_ground,
#                                  background=mask)
                
                # build the burrow polygon by removing the sky
                burrow_poly = burrow_poly.difference(above_ground)
                
                # create a burrow from the outline
                boundary = regions.get_enclosing_outline(burrow_poly)
                burrow = Burrow(boundary.coords,
                                parameters=self.params['burrow_parameters'])
                burrows.append(burrow)
            
        logging.info('Found %d polygons' % len(burrows))
        return burrows
             
        
    def isolate_color(self, image, color, white_background=None, dilate=0):
        """ isolates a certain color channel from the image. Color should be a
        binary vector only containing 0 and 1 """
        # determine whether the background is white or black if not given
        if white_background is None:
            white_background = (np.mean(image) > 128)
            if white_background:
                logging.debug('Image appears to have a white background.')
            else:
                logging.debug('Image appears to have a black background.')
        
        # determine the limits of the color function
        if white_background:
            limits = ((0, 230), (128, 255))
        else:
            limits = ((0, 30), (30, 255))
        bounds = np.array([limits[int(c)] for c in color], np.uint8)

        # find the mask highlighting the respective colors
        mask = cv2.inRange(image, bounds[:, 0], bounds[:, 1])

        # dilate the mask to close gaps in the outline
        w = self.params['colors/isolation_closing_radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
        mask_dilated = cv2.dilate(mask, kernel)

        # fill the objects
        contours = cv2.findContours(mask_dilated.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]
        
        for contour in contours:
            cv2.fillPoly(mask_dilated, [contour[:, 0, :]], color=(255, 255, 255))
            
        # erode the mask and return it
        if dilate != 0:
            w = self.params['colors/isolation_closing_radius'] - dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
        mask = cv2.erode(mask_dilated, kernel)
        
        return mask
    
    
    def _get_burrow_exit_length(self, burrow):        
        """ calculates the length of all exists of the given burrow """
        # identify all points that are close to the ground line
        dist_max = burrow.parameters['ground_point_distance']
        g_line = self.ground_line.linestring
        points = burrow.contour
        exitpoints = [g_line.distance(geometry.Point(point)) < dist_max
                      for point in points]
        
        # find the indices of contiguous true regions
        indices = math.contiguous_true_regions(exitpoints)

        # find the total length of all exits        
        exit_length = 0
        for a, b in indices:
            exit_length += curves.curve_length(points[a : b+1])
        
        # handle the first and last point if they both belong to an exit 
        if exitpoints[0] and exitpoints[-1]:
            exit_length += curves.point_distance(points[0], points[-1])
            
        return exit_length
        
        
    def get_statistics(self):
        """ returns statistics for all the polygons """
        result = {'name': self.name}
            
        # save results about ground line
        points = self.ground_line.points
        ground_width_px = abs(points[0, 0] - points[-1, 0])
        ground_cm_per_pixel = self.params['cage/width_norm'] / ground_width_px
        result['ground'] = {'ground_length': self.ground_line.length,
                            'ground_width_pixel': ground_width_px,
                            'cm_per_pixel': ground_cm_per_pixel}  
        
        # check the scale bar
        if self.scale_bar:
            logging.info('Found %d pixel long scale bar' % self.scale_bar.size)
            cm_per_pixel = self.params['scale_bar/length_cm']/self.scale_bar.size
            units = pint.UnitRegistry()
            scale_factor = cm_per_pixel * units.cm

            # check the ground line
            points = self.ground_line.points
            len_x_cm = abs(points[0, 0] - points[-1, 0]) * scale_factor
            w_min = self.params['cage/width_min'] * units.cm
            w_max = self.params['cage/width_max'] * units.cm
            if not w_min < len_x_cm < w_max:
                raise RuntimeError('The length (%s cm) of the ground line is '
                                   'off.' % len_x_cm)
                
            result['scale_bar'] = {'length_pixel': self.scale_bar.size,
                                   'cm_per_pixel': cm_per_pixel}
        else:
            scale_factor = 1
            result['scale_bar'] = None
        
        # collect result of all burrows
        result['burrows'] = []
        for burrow in self.burrows:
            perimeter_exit = self._get_burrow_exit_length(burrow)
            #graph = burrow.morphological_graph 
            data = {'pos_x': burrow.centroid[0] * scale_factor,
                    'pos_y': burrow.centroid[1] * scale_factor,
                    'area': burrow.area * scale_factor**2,
                    'length': burrow.length * scale_factor,
                    'length_upwards': burrow.length_upwards * scale_factor,
                    'fraction_upwards': burrow.length_upwards / burrow.length,
                    #'total_length': graph.get_total_length() * scale_factor,
#                     'branch_count': graph.number_of_edges(),
                    'perimeter': burrow.perimeter * scale_factor,
                    'perimeter_exit': perimeter_exit * scale_factor,
                    'openness': perimeter_exit / burrow.perimeter}
            result['burrows'].append(data)
            
        return result
        
        

def process_polygon_file(path, output_folder=None, suppress_exceptions=False):
    """ process a single shape file given by path """
    if suppress_exceptions:
        # run the function within a catch-all try-except-block
        try:
            result = process_polygon_file(path, output_folder, False)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            traceback.print_exc()
            result = []
            
    else:
        # do the actual computation
        logging.info('Analyzing file `%s`' % path)
        pc = AntfarmShapes.load_from_file(path, output_folder=output_folder)
        result = pc.get_statistics()

    return result



def main():
    """ main routine of the program """
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Analyze antfarm polygons')
    parser.add_argument('-c', '--result_csv', dest='result_csv', type=str,
                        metavar='FILE.csv',
                        help='csv file to which statistics about the burrows '
                             'are written')
    parser.add_argument('-p', '--result_pkl', dest='result_pkl', type=str,
                        metavar='FILE.pkl',
                        help='python pickle file to which all results from the '
                             'algorithm are written')
    parser.add_argument('-l', '--load_pkl', dest='load_pkl', type=str,
                        metavar='FILE.pkl',
                        help='python pickle file from which data is loaded')
    parser.add_argument('-f', '--folder', dest='folder', type=str,
                        help='folder where output images will be written to')
    parser.add_argument('-m', '--multi-processing', dest='multiprocessing',
                        action='store_true', help='turns on multiprocessing')
    parser.add_argument('files', metavar='FILE', type=str, nargs='*',
                        help='files to analyze')

    args = parser.parse_args()

    if args.load_pkl:
        # load file from pickled data
        logging.info('Loading data from file `%s`.' % args.load_pkl)    
        with open(args.load_pkl, "rb") as fp:
            results = pickle.load(fp)
        
    else:
        # get files to analyze
        files = args.files
        logging.info('Analyzing %d files.' % len(files))
        
        # collect burrows from all files
        if args.multiprocessing:
            # use multiple processes to analyze data
            job_func = functools.partial(process_polygon_file,
                                         output_folder=args.folder,
                                         suppress_exceptions=True)
            pool = mp.Pool()
            results = pool.map(job_func, files)
            
        else:
            # analyze data in the current process
            job_func = functools.partial(process_polygon_file,
                                         output_folder=args.folder,
                                         suppress_exceptions=False)
            results = map(job_func, files)
            
        # write complete results as pickle file if requested
        if args.result_pkl:
            with open(args.result_pkl, "wb") as fp:
                pickle.dump(results, fp)
        
    # write burrow results as csv file if requested
    if args.result_csv:
        # create a dictionary of lists
        table = collections.defaultdict(list) 
        # iterate through all experiments and save information about the burrows
        for data in results:
            if data:
                # sort the burrows from left to right
                burrows = sorted(data['burrows'], key=operator.itemgetter('pos_x'))
                # create a single row per burrow
                for burrow_id, properties in enumerate(burrows, 1):
                    properties['burrow_id'] = burrow_id
                    properties['experiment'] = data['name']
                    # iterate over all burrow properties
                    for k, v in properties.iteritems():
                        table[k].append(v)
                       
        # write the data to a csv file     
        first_columns = ['experiment', 'burrow_id']
        data_structures.save_dict_to_csv(table, args.result_csv,
                                         first_columns=first_columns)



if __name__ == '__main__':
    main()
    
    