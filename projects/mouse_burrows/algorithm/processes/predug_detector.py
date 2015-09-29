'''
Created on Aug 25, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os

import numpy as np
import cv2
from shapely import affinity, geometry
import yaml

from video.analysis import regions, shapes, curves

# from video import debug



class PredugDetector(object):
    """ class capsulating the code necessary for detecing the predug """
    
    def __init__(self, background_image, ground, parameters):
        self.image = background_image
        self.ground = ground
        self.params = parameters
        
        self.predug = None
        self.predug_rect = None
        self.predug_location = None
        self.search_rectangles = []

    
    def _refine_predug(self, candidate):
        """ uses the color information directly to specify the predug """
        # determine a bounding rectangle
        region = candidate.bounds
        size = max(region.width, region.height)
        region.buffer(0.5*size) #< increase by 50% in each direction
        
        # extract the region from the image
        slice_x, slice_y = region.slices
        img = self.image[slice_y, slice_x].astype(np.uint8, copy=True)

        # build the estimate polygon
        poly_p = affinity.translate(candidate.polygon, -region.x, -region.y)        
        poly_s = affinity.translate(self.ground.get_sky_polygon(),
                                    -region.x, -region.y)        
        
        def fill_mask(color, margin=0):
            """ fills the mask with the buffered regions """
            for poly in (poly_p, poly_s):
                pts = np.array(poly.buffer(margin).boundary.coords, np.int32)
                cv2.fillPoly(mask, [pts], color)
                
        # prepare the mask for the grabCut algorithm
        burrow_width = self.params['burrows/width'] 
        mask = np.full_like(img, cv2.GC_BGD, dtype=np.uint8) #< sure background
        fill_mask(cv2.GC_PR_BGD, 0.25*burrow_width) #< possible background
        fill_mask(cv2.GC_PR_FGD, 0) #< possible foreground
        fill_mask(cv2.GC_FGD, -0.25*burrow_width) #< sure foreground

        # run GrabCut algorithm
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, mask, (0, 0, 1, 1),
                        bgdmodel, fgdmodel, 2, cv2.GC_INIT_WITH_MASK)
        except:
            # any error in the GrabCut algorithm makes the whole function useless
            logging.warn('GrabCut algorithm failed for predug')
            return candidate
        
        # turn the sky into background
        pts = np.array(poly_s.boundary.coords, np.int32)
        cv2.fillPoly(mask, [pts], cv2.GC_BGD)

        # extract a binary mask determining the predug 
        predug_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        predug_mask = predug_mask.astype(np.uint8)
        
        # simplify the mask using binary operations
        w = int(0.5*burrow_width)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
        predug_mask = cv2.morphologyEx(predug_mask, cv2.MORPH_OPEN, kernel)
         
        # extract the outline of the predug
        contour = regions.get_contour_from_largest_region(predug_mask)
        
        # translate curves back into global coordinate system
        contour = curves.translate_points(contour, region.x, region.y)
        
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
        
        if self.params['predug/scale_predug']:
            # get the scaled size of the template image
            target_size = (int(self.params['predug/template_width']),
                           int(self.params['predug/template_height']))
            
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
#         res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
#         val_best, _, loc_best, _ = cv2.minMaxLoc(res)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, val_best, _, loc_best = cv2.minMaxLoc(res)
        
        # determine the rough outline of the predug in the region 
        coords = curves.translate_points(coords, *loc_best)
        # determine the outline of the predug in the video 
        coords = curves.translate_points(coords, region.x, region.y)
                
        return val_best, shapes.Polygon(coords)  


    def detect(self):
        """ tries to locate the predug using the ground line and the background
        image """

        # determine the height of the search region
        ground_points = self.ground.points
        y_min = ground_points[:, 1].min()
        y_max = ground_points[:, 1].max()
        factor = self.params['predug/search_height_factor']
        height = factor * (y_max - y_min)
        
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
        y_top = y_max - 0.15*height
        region_l = shapes.Rectangle.from_centerpoint((x1, y_top), width, height)
        region_r = shapes.Rectangle.from_centerpoint((x2, y_top), width, height)

        self.search_rectangles = [region_l, region_r] 
        
        # determine the possible contours in both regions
        score_l, candidate_l = self._search_predug_in_region(region_l, False)
        score_r, candidate_r = self._search_predug_in_region(region_r, True)
        logging.debug('Predug template matching scores: left=%g, right=%g',
                      score_l, score_r)

        predug_location = self.params['predug/location']
        if predug_location == 'left':
            logging.info('Predug was specified to be on the left side.')
            self.predug_rect = candidate_l
            self.predug_location = 'left'

        if predug_location == 'right':
            logging.info('Predug was specified to be on the right side.')
            self.predug_rect = candidate_r
            self.predug_location = 'right'

        elif predug_location == 'auto':
            if score_r > score_l:
                logging.info('Located predug on the right side.')
                self.predug_rect = candidate_r
                self.predug_location = 'right'
            else:
                logging.info('Located predug on the left side.')
                self.predug_rect = candidate_l    
                self.predug_location = 'left'
                
        else:
            raise ValueError('Unknown predug location `%s`' % predug_location) 
            
        # refine the predug
        self.predug = self._refine_predug(self.predug_rect)   
        
#         debug.show_shape(geometry.Polygon(self.predug.contour),
#                          background=self.image)
            
        return self.predug
        

    