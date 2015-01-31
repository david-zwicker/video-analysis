#!/usr/bin/env python2
'''
Created on Jan 30, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import collections
import os.path
import operator
import sys
import logging

import cv2
import numpy as np

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)

from data_structures.cache import cached_property
from video.analysis import regions, curves
from mousetracking.algorithm.utils import save_dict_to_csv

from video import debug  # @UnusedImport


default_parameters = {
    'burrow/area_min': 500,
    'scale_bar/area_max': 1000,
    'scale_bar/dist_bottom': 0.1,
    'scale_bar/dist_left': 0.1,
}


ScaleBar = collections.namedtuple('ScaleBar', ['size', 'angle'])



class BurrowPolygon(regions.Polygon):
    """ class representing a single burrow """ 
    @cached_property
    def centerline(self):
        return self.get_centerline_smoothed()
    
    @cached_property
    def length(self):
        return curves.curve_length(self.centerline)
            


class PolygonCollection(object):
    """ class that manages collections of polygons """
    
    def __init__(self, polygons, parameters=None, name='', output_folder=None):
        """ initializes the polygon collection
        `polygons` is a list of polygons
        `parameters` are parameters for the algorithms of this class
        `name` is the name of the collection
        `debug_output` can be a folder to which debug output will be written 
        """
        self.polygons = polygons
        self.name = name
        self.output_folder = output_folder
        
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
        if ext == '.jpg':
            logging.debug('Use jpeg loader')
            if obj.output_folder:
                output_file = os.path.join(obj.output_folder, filename)
            else:
                output_file = None
                
            image = cv2.imread(path)
            obj.load_from_image(image, output_file)
            
        else:
            raise ValueError("Don't know how to read `*%s` files" % ext)
            
        return obj 

    
    def load_from_image(self, image, output_file=None):
        """ load polygons from an image """
        # turn image into gray scale
        image_gray = np.mean(image, axis=2).astype(np.uint8)
        height, width = image_gray.shape
                
        # threshold image to get a mask
        mask = cv2.threshold(image_gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # determine contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # iterate through the contours
        self.polygons = []
        scale_bar = None
        for contour in contours:
            points = contour[:, 0, :]
            if len(points) <= 2:
                continue

            area = cv2.contourArea(contour)

            if area < self.params['scale_bar/area_max']:
                # object could be a scale bar
                rect = regions.Rectangle(*cv2.boundingRect(contour))

                at_left = (rect.left < self.params['scale_bar/dist_left']*width)
                at_bottom = (rect.bottom > (1 - self.params['scale_bar/dist_bottom'])*height)
                hull = cv2.convexHull(contour) 
                hull_area = cv2.contourArea(hull)
                is_simple = (hull_area < 2*area)
                
                if at_left and at_bottom and is_simple:
                    # the current polygon is the scale bar
                    _, (w, h), rot = cv2.minAreaRect(contour)
                    if w > h:
                        scale_bar = ScaleBar(size=w, angle=rot)
                    else:
                        scale_bar = ScaleBar(size=h, angle=(rot + 90) % 180)
                    continue #< object has been processed

            if area > self.params['burrow/area_min']:
                self.polygons.append(BurrowPolygon(points))
            
        if scale_bar:
            logging.info('Found scale bar of length %d' % scale_bar.size)
        else:
            logging.info('Did not find any scale bar')

        logging.info('Found %d polygons' % len(self.polygons))
        
        if output_file:
            logging.info('Creating debug output')
            for poly in self.polygons:
                cline = poly.centerline
                cv2.polylines(image, [np.array(cline, np.int)],
                              isClosed=False, color=(255, 0, 0), thickness=3)
            cv2.imwrite(output_file, image)
            
            logging.info('Wrote output file')
        
        
    def get_statistics(self):
        """ returns statistics for all the polygons """
        result = []
        # iterate through all polygons
        for polygon in self.polygons:
            data = {'area': polygon.area,
                    'length': polygon.length,
                    'pos_x': polygon.centroid[0],
                    'pos_y': polygon.centroid[1]}
            if self.name:
                data['name'] = self.name
            result.append(data)
        return result
        
        

def process_file(path, output_folder=None):
    """ process a single shape file given by path """
    pc = PolygonCollection.load_from_file(path, output_folder=output_folder)
    return pc.get_statistics()



def main():
    """ main routine of the program """
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Analyze antfarm polygons')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='file to which the output statistics are written')
    parser.add_argument('-f', '--folder', dest='folder', type=str,
                        help='folder where output will be written to')
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='files to analyze')

    args = parser.parse_args()
    
    # collect all burrows
    result = []
    for path in args.files[:2]:
        # get all burrows from this file 
        burrows = process_file(path, args.folder)
        # sort the burrows from left to right
        burrows = sorted(burrows, key=operator.itemgetter('pos_x'))
        for burrow_id, burrow in enumerate(burrows, 1):
            burrow['burrow_id'] = burrow_id
            result.append(burrow)
            
    # create a dictionary of lists
    result_dict = collections.defaultdict(list)
    for item in result:
        print item
        for k, v in item.iteritems():
            result_dict[k].append(v)
                   
    # write the data to a csv file     
    first_columns = ['name', 'burrow_id']
    save_dict_to_csv(result_dict, args.output, first_columns=first_columns)



if __name__ == '__main__':
    main()
    
    