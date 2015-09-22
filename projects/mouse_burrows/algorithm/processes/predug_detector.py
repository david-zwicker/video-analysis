'''
Created on Aug 25, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os

import numpy as np
import cv2
from shapely import geometry
import yaml

from video.analysis import regions, shapes, curves
from ..objects.burrow import Burrow 

# from video import debug



class PredugDetector(object):
    """ class capsulating the code necessary for detecing the predug """
    
    def __init__(self, background_image, ground, parameters):
        self.image = background_image
        self.ground = ground
        self.params = parameters


    def get_ground_mask(self):
        """ returns a binary mask distinguishing the ground from the sky """
        # build a mask with for the ground
        height, width = self.image.shape
        mask_ground = np.zeros((height, width), np.uint8)
        
        # create a mask for the region below the current mask_ground profile
        ground_points = np.empty((len(self.ground) + 4, 2), np.int32)
        ground_points[:-4, :] = self.ground.points
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(mask_ground, np.array([ground_points], np.int32), color=1)

        return mask_ground
    

    def _search_predug_in_region_old(self, region):
        """ searches for the predug in the rectangular `region` """
        slice_x, slice_y = region.slices

        # determine the image in the region, only considering under ground
        img = self.image[slice_y, slice_x]
        mask = self.get_ground_mask()[slice_y, slice_x]
        mask = ~mask.astype(np.bool)
        img[mask] = np.nan

        # calculate the statistics of this image 
        img_mean = np.nanmean(img)
        img_std = np.nanstd(img)
        
        # convert image to uint8 to use with opencv functions
        img[mask] = 255
        img = img.astype(np.uint8)
        
        # threshold image
        z_score = self.params['predug/search_zscore_threshold']
        img_thresh = int(img_mean - z_score * img_std)
        _, predug_mask = cv2.threshold(img, img_thresh, 255,
                                       cv2.THRESH_BINARY_INV)
        
        # slight smoothing using morphological transformation
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.morphologyEx(predug_mask, cv2.MORPH_CLOSE, kernel)
        
        # determine the predug polygon
        try:
            contour = regions.get_contour_from_largest_region(predug_mask)
        except RuntimeError:
            # could not find any contour => return null polygon
            return shapes.Polygon(np.zeros((3, 2)))
        
        # simplify the contour to make it more useable
        threshold = self.params['predug/simplify_threshold']
        contour = regions.simplify_contour(contour, threshold)
        
        # shift the contour back to the original position
        contour += np.array((region.x, region.y))
        return shapes.Polygon(contour)
   

    def _search_predug_in_region(self, region, reflect=False):
        """ searches the predug using a template """
        slice_x, slice_y = region.slices

        # determine the image in the region, only considering under ground
        img = self.image[slice_y, slice_x].astype(np.uint8)

        # load the file that describes the template data 
        filename = self.params['predug/template_file']
        path = os.path.join(os.path.dirname(__file__), '..', 'assets', filename)
        if not os.path.isfile(path):
            logging.warn('Predug template file `%s` was not found', path)
            return None

        with open(path, 'r') as infile:
            yaml_content = yaml.load(infile)
            
        if not yaml_content: 
            logging.warn('Predug template file `%s` is empty', path)
            return None
        
        # load the template image
        img_file = yaml_content['image']
        coords = np.array(yaml_content['coordinates'])
        path = os.path.join(os.path.dirname(__file__), '..', 'assets', img_file)
        
        # read the image from the file as grey scale
        template = cv2.imread(path)[:, :, 0]
        
        # get the scaled size of the template image
        target_size = (self.params['predug/template_width'],
                       self.params['predug/template_height'])
        
        # scale the predug coordinates to match the template size
        coords[:, 0] *= target_size[0] / template.shape[1] #< x-coordinates
        coords[:, 1] *= target_size[1] / template.shape[0] #< y-coordinates

        # scale the template image itself
        template = cv2.resize(template, target_size)
        
        if reflect:
            # reflect the image on the vertical center line
            template = cv2.flip(template, 1)
            template_width = template.shape[1]
            coords[:, 0] = template_width - coords[:, 0] - 1
            
        # do the template matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        # create a Burrow from this estimate
        coords = curves.translate_points(coords, *max_loc)
        coords = curves.translate_points(coords, region.x, region.y)
                
        return max_val, shapes.Polygon(coords)  


    def detect(self):
        """ tries to locate the predug using the ground line and the background
        image """

        # determine the height of the search region
        ground_points = self.ground.points
        y_deep = ground_points[:, 1].max()
        factor = self.params['predug/search_height_factor']
        height = factor * (y_deep - self.ground.points[:, 1].min())
        
        # determine the width of the search region
        y_mid = ground_points[:, 1].mean()
        mid_line = geometry.LineString(((0, y_mid),
                                        (ground_points[:, 0].max(), y_mid)))
        points = mid_line.intersection(self.ground.linestring)

        if len(points) != 2:
            logging.warn('The ground line crossed its midline not exactly '
                         'twice => The ground line is likely messed up.')
            return
        
        x1, x2 = sorted((points[0].x, points[1].x))
        factor = self.params['predug/search_width_factor']
        width = factor * (x2 - x1)
        
        # determine the two serach regions
        region_l = shapes.Rectangle.from_centerpoint((x1, y_deep), width, height)
        region_r = shapes.Rectangle.from_centerpoint((x2, y_deep), width, height)
        
        # determine the possible contours in both regions
        score_l, candidate_l = self._search_predug_in_region(region_l, False)
        score_r, candidate_r = self._search_predug_in_region(region_r, True)

        if score_r > score_l:
            logging.info('Located predug on the right side.')
            candidate = candidate_r
        else:
            logging.info('Located predug on the left side.')
            candidate = candidate_l        
            
        area_min = self.params['burrows/area_min']
        if candidate.area > area_min:
            return Burrow(candidate.contour)
        else:
            return None    
        

    