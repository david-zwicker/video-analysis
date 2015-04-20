#!/usr/bin/env python2
'''
Created on Jan 30, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import collections
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
import pint

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)

from projects.mouse_burrows.algorithm.objects import Burrow, GroundProfile
from video.analysis import shapes, image
from utils.data_structures import save_dict_to_csv

from video import debug  # @UnusedImport


default_parameters = {
    'burrow/area_min': 10000,
    'colors/burrow': (0, 0, 1),      #< burrow color in RGB
    'colors/ground_line': (0, 1, 0), #< ground line color in RGB
    'colors/scale_bar': (0, 0, 0),   #< scale bar color in RGB
    'scale_bar/area_max': 1000,
    'scale_bar/length_min': 100,
    'scale_bar/dist_bottom': 0.1,
    'scale_bar/dist_left': 0.1,
    'scale_bar/length_cm': 10,
}


ScaleBar = collections.namedtuple('ScaleBar', ['size', 'angle'])



#FIXME: use burrow and ground line objects from tracking

# class BurrowPolygon(shapes.Polygon):
#     """ class representing a single burrow """ 
#     @cached_property
#     def centerline(self):
#         return self.get_centerline_smoothed()
#     
#     @cached_property
#     def length(self):
#         return curves.curve_length(self.centerline)
    
#     
#     
# class GroundLine(geometry.LineString):
#     """ class that represents the ground line """
#     pass
#     


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
        self.scale_factor = 1

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
        
        # determine which loaded to use for the individual files
        if ext == '.jpg' or ext == '.png':
            logging.debug('Use OpenCV image loader')
            if obj.output_folder:
                output_file = os.path.join(obj.output_folder, filename)
            else:
                output_file = None
                
            image = cv2.imread(path)
            cv2.cvtColor(image, cv2.cv.CV_BGR2RGB, image) #< convert to RGB
            obj.load_from_image(image, output_file)
            
        else:
            raise ValueError("Don't know how to read `*%s` files" % ext)
            
        return obj
    
    
    def load_from_image(self, image, output_file=None):
        """ load the data from an image """
        # find the ground line
        ground_mask = self.isolate_color(image, self.params['colors/ground_line'])
        self.ground_line = self.get_groundline_from_image(ground_mask)

        # find all the burrows
        burrow_mask = self.isolate_color(image, self.params['colors/burrow'])
        self.burrows = self.get_burrows_from_image(burrow_mask)
        
        # determine the end points
        for burrow in self.burrows:
            burrow.get_endpoints(self.ground_line)
                
        if output_file:
            
            logging.info('Creating debug output')
            for burrow in self.burrows:
                cline = burrow.centerline
                cv2.polylines(image, [np.array(cline, np.int)],
                              isClosed=False, color=(255, 0, 0), thickness=3)

                for e_p in burrow.endpoints:
                    if e_p.is_exit:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    coords = tuple([int(c) for c in e_p.coords])
                    cv2.circle(image, coords, 10, color, thickness=-1)

            cv2.cvtColor(image, cv2.cv.CV_RGB2BGR, image) #< convert to BGR
            cv2.imwrite(output_file, image)
            
            logging.info('Wrote output file `%s`' % output_file)

    
    def get_groundline_from_image(self, mask, output_file=None):
        """ load burrow polygons from an image """
        # find the longest contour, which should be the ground line
#         mask_thin = image.mask_thinning(mask)
                
        # determine contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        
        # pick the largest contour
        contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, closed=True))
        contour = contour[:, 0, :]
        
#         debug.show_shape(geometry.MultiPoint(contour), background=mask)
        
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

        ground_line = GroundProfile(points)

        logging.info('Found a ground line of length %g ' % ground_line.length)
        return ground_line
        
            
    def get_burrows_from_image(self, mask, output_file=None):
        """ load burrow polygons from an image """
        # turn image into gray scale
        height, width = mask.shape

        # determine contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # iterate through the contours
        burrows = []
        scale_bar = None
        for contour in contours:
            points = contour[:, 0, :]
            if len(points) <= 2:
                continue

            area = cv2.contourArea(contour) #< get area of burrow

            if area < self.params['scale_bar/area_max']:
                # object could be a scale bar
                rect = shapes.Rectangle(*cv2.boundingRect(contour))

                at_left = (rect.left < self.params['scale_bar/dist_left']*width)
                max_dist_bottom = self.params['scale_bar/dist_bottom']
                at_bottom = (rect.bottom > (1 - max_dist_bottom)*height)
                hull = cv2.convexHull(contour) 
                hull_area = cv2.contourArea(hull)
                is_simple = (hull_area < 2*area)
                
                if at_left and at_bottom and is_simple:
                    # the current polygon is the scale bar
                    _, (w, h), rot = cv2.minAreaRect(contour)
                    
                    if max(w, h) > self.params['scale_bar/length_min']:
                        # we found the scale bar
                        if w > h:
                            scale_bar = ScaleBar(size=w, angle=rot)
                        else:
                            scale_bar = ScaleBar(size=h, angle=(rot + 90) % 180)
                        continue #< object has been processed

            if area > self.params['burrow/area_min']:
                burrows.append(Burrow(points))
            
        if scale_bar:
            raise RuntimeError('Found something that looks like a scale bar')
            logging.info('Found scale bar of length %d' % scale_bar.size)
            self.scale_factor = self.params['scale_bar/length_cm']/scale_bar.size
            units = pint.UnitRegistry()
            self.scale_factor *= units.cm

        logging.info('Found %d polygons' % len(burrows))
        return burrows
             
        
    def isolate_color(self, image, color, white_background=True):
        """ isolates a certain color channel from the image. Color should be a
        binary vector only containing 0 and 1 """
        if white_background:
            limits = ((0, 230), (128, 255))
        else:
            limits = ((0, 30), (128, 255))
        bounds = np.array([limits[int(c)] for c in color], np.uint8)

        # find the mask highlighting the respective colors
        mask = cv2.inRange(image, bounds[:, 0], bounds[:, 1])

        # dilate the mask to close gaps in the outline
        w = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
        mask_dilated = cv2.dilate(mask, kernel)

        # fill the objects
        contours, _ = cv2.findContours(mask_dilated.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            cv2.fillPoly(mask_dilated, [contour[:, 0, :]], color=(255, 255, 255))
            
        # erode the mask and return it
        mask = cv2.erode(mask_dilated, kernel)
        
        return mask
    
        
    def get_statistics(self):
        """ returns statistics for all the polygons """
        result = []
        # iterate through all polygons
        for burrow in self.burrows:
            data = {'area': burrow.area * self.scale_factor**2,
                    'length': burrow.length * self.scale_factor,
                    'pos_x': burrow.centroid[0] * self.scale_factor,
                    'pos_y': burrow.centroid[1] * self.scale_factor}
            if self.name:
                data['name'] = self.name
            result.append(data)
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
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='file to which the output statistics are written')
    parser.add_argument('-f', '--folder', dest='folder', type=str,
                        help='folder where output will be written to')
    parser.add_argument('-m', '--multi-processing', dest='multiprocessing',
                        action='store_true', help='turns on multiprocessing')
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='files to analyze')

    args = parser.parse_args()
    
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
        result = pool.map(job_func, files)
        
    else:
        # analyze data in the current process
        job_func = functools.partial(process_polygon_file,
                                     output_folder=args.folder,
                                     suppress_exceptions=False)
        result = map(job_func, files)
        
    # create a dictionary of lists
    table = collections.defaultdict(list) 
    for burrows in result: #< iterate through all experiments
        # sort the burrows from left to right
        burrows = sorted(burrows, key=operator.itemgetter('pos_x'))
        for burrow_id, properties in enumerate(burrows, 1): #< iter. all burrows
            properties['burrow_id'] = burrow_id
            for k, v in properties.iteritems(): #< iter. all burrow properties
                table[k].append(v)
                   
    # write the data to a csv file     
    first_columns = ['name', 'burrow_id']
    save_dict_to_csv(table, args.output, first_columns=first_columns)



if __name__ == '__main__':
    main()
    
    