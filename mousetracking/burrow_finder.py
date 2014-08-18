'''
Created on Aug 16, 2014

@author: zwicker
'''

import numpy as np
import cv2

from video.analysis.curves import make_curve_equidistant, simplify_curve
from video.analysis.regions import expand_rectangle, rect_to_slices

import debug


class BurrowFinder(object):
    """ class devoted to finding burrows in a image.
    This is a separate class because it might run in a separate process and
    the logic should separated from the main run
    """
    
    def __init__(self, tracking_parameters):
        self.params = tracking_parameters
        
                    
    #===========================================================================
    # FIND BURROWS 
    #===========================================================================

    def find_burrows(self, frame, explored_area, sand_profile):
        """ locates burrows by combining the information of the sand profile
        and the explored area """
        
        # build a mask with potential burrows
        height, width = frame.shape
        sand_mask = np.zeros_like(frame, np.uint8)
        
        # create a mask for the region below the current sand profile
        sand_points = np.empty((len(sand_profile) + 4, 2), np.int)
        sand_points[:-4, :] = sand_profile
        sand_points[-4, :] = (width, sand_points[-5, 1])
        sand_points[-3, :] = (width, height)
        sand_points[-2, :] = (0, height)
        sand_points[-1, :] = (0, sand_points[0, 1])
        sand_contour = np.array([sand_points], np.int)
        cv2.fillPoly(sand_mask, np.array([sand_points], np.int), color=128)

        # erode the mask slightly, since the sand profile is not perfect        
        w = self.params['sand_profile.width'] + self.params['mouse.model_radius']
        # TODO: cache this kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        cv2.erode(sand_mask, kernel, dst=sand_mask)
        
        w = self.params['burrows.radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        cv2.morphologyEx(explored_area, cv2.MORPH_CLOSE, kernel, dst=explored_area)

        # combine with the information of what areas have been explored
        burrows_mask = cv2.bitwise_and(sand_mask, explored_area)
        
        # find the contours of the features
        contours, hierarchy = cv2.findContours(burrows_mask.copy(), # we want to use the mask later again
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        
        print '-'
        
        for contour in np.array(contours, np.int):
            # get enclosing rectangle 
            rect = cv2.boundingRect(contour)
            rect = expand_rectangle(rect, 30)
            
            # focus on this part of the problem
            slices = rect_to_slices(rect)
            sand_mask_roi = sand_mask[slices]
            burrow_mask = burrows_mask[slices]
            frame_roi = frame[slices]
            contour = np.squeeze(contour) - np.array([[rect[0], rect[1]]], np.int)

            # find the combined contour of burrow and sand profile
            combined_mask = cv2.bitwise_xor(burrow_mask, sand_mask_roi)
            points, hierarchy = cv2.findContours(combined_mask,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
            
            assert len(points) == 1
            points = points[0]
            
            # simplify the curve
            epsilon = 0.02*cv2.arcLength(points, True)
            points = cv2.approxPolyDP(points, epsilon, True)
            points = points[:, 0, :]

            # identify points that are free to be modified in the fitting
            free_points = np.ones(len(points), np.bool)
            roi_h, roi_w = burrow_mask.shape
            for k, p in enumerate(points):
                if p[0] == 1 or p[1] == 1 or p[0] == roi_w - 2 or p[1] == roi_h - 2:
                    free_points[k] = False
                         
            # draw burrow
            cv2.drawContours(frame_roi, np.array([points], np.int), -1, 255, 1)
            for k, p in enumerate(points):
                color = 255 if free_points[k] else 128
                cv2.circle(frame_roi, (int(p[0]), int(p[1])), 3, color, thickness=-1)
            debug.show_image(frame_roi)



class Burrow(object):
    """ represents a single burrow to compare it against an image in fitting """
    
    def __init__(self, outline, image):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.outline = outline
        self.image = None
        
        
    def get_centerline(self):
        raise NotImplementedError
    
        
    def adjust_outline(self, deviations):
        """ adjust the current outline by moving points perpendicular by
        a distance given by `deviations` """
        pass
    
        
    def get_difference(self, deviations):
        """ calculates the difference between image and model, when the 
        model is moved by a certain distance in its normal direction """
        raise NotImplementedError 
        
        dist = 1
           
        # apply sigmoidal function
        model = np.tanh(dist/self.width)
     
        return np.ravel(self.image_mean + 1.5*self.image_std*model - self.image)
    