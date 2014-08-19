'''
Created on Aug 16, 2014

@author: zwicker
'''

import logging

import numpy as np
import cv2
import shapely.geometry as geometry

from video.analysis.curves import make_curve_equidistant, simplify_curve
from video.analysis.regions import expand_rectangle, rect_to_slices

import debug


class BurrowFinder(object):
    """ class devoted to finding burrows in a image.
    This is a separate class because it might run in a separate process and
    the logic should separated from the main run
    """
    
    def __init__(self, tracking_parameters, debug=None):
        self.params = tracking_parameters
        self.debug = {} if debug is None else debug
    
    
    #===========================================================================
    # FIND BURROWS 
    #===========================================================================


    def refine_burrow_outline(self, burrow, ground_profile, offset):
        """ takes a single burrow and refines its outline """

        # identify points that are free to be modified in the fitting
        ground = geometry.LineString(np.array(ground_profile, np.double))
        roi_h, roi_w = burrow.image.shape

        # move points close to the ground profile on top of it
        in_burrow = False
        outline = []
        last_point = None
        for p in burrow.outline:
            # get the point in global coordinates
            point = geometry.Point((p[0] + offset[0], p[1] + offset[1]))

            if p[0] == 1 or p[1] == 1 or p[0] == roi_w - 2 or p[1] == roi_h - 2:
                # points at the boundary are definitely outside the burrow
                point_outside = True
                
            elif point.distance(ground) < self.params['burrows/radius']:
                # points close to the ground profile are outside
                point_outside = True
                # we also move these points onto the ground profile
                # see http://stackoverflow.com/a/24440122/932593
                point_new = ground.interpolate(ground.project(point))
                p = (point_new.x - offset[0], point_new.y - offset[1])
                
            else:
                point_outside = False
            
            if point_outside:
                # current point is a point outside the burrow
                if in_burrow:
                    # if we currently reaping, add this point and exit
                    outline.append(p)
                    break
                # otherwise save the point, since we might need it in the next iteration
                last_point = p
                    
            else:
                # current point is inside the burrow
                if not in_burrow:
                    # start reaping points
                    in_burrow = True
                    if last_point is not None:
                        outline = [last_point]
                # definitely add this point
                outline.append(p)
                
        # simplify the burrow outline
        outline = geometry.LineString(np.array(outline, np.double))
        tolerance = self.params['burrows/outline_simplification_threshold'] * outline.length
        outline = outline.simplify(tolerance, preserve_topology=False)
        burrow.outline = list(outline.coords)                
        
        # adjust the outline until it explains the burrow best 
        
        
        # remove the fixed points from the final burrow object
        points = [(p[0] + offset[0], p[1] + offset[1]) for p in burrow.outline]
        
        return points


    def find_burrows(self, frame_id, frame, explored_area, ground_profile):
        """ locates burrows by combining the information of the ground profile
        and the explored area """
        
        # build a mask with potential burrows
        height, width = frame.shape
        ground = np.zeros_like(frame, np.uint8)
        
        # create a mask for the region below the current ground profile
        ground_points = np.empty((len(ground_profile) + 4, 2), np.int32)
        ground_points[:-4, :] = ground_profile
        ground_points[-4, :] = (width, ground_points[-5, 1])
        ground_points[-3, :] = (width, height)
        ground_points[-2, :] = (0, height)
        ground_points[-1, :] = (0, ground_points[0, 1])
        cv2.fillPoly(ground, np.array([ground_points], np.int32), color=128)

        # erode the mask slightly, since the ground profile is not perfect        
        w = 2*self.params['mouse/model_radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        ground_mask = cv2.erode(ground, kernel)#, dst=ground_mask)
        
        w = self.params['burrows/radius']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w)) 
        potential_burrows = cv2.morphologyEx(explored_area, cv2.MORPH_CLOSE, kernel)
        
        # remove accidental burrows at borders
        potential_burrows[: 30, :] = 0
        potential_burrows[-30:, :] = 0
        potential_burrows[:, : 30] = 0
        potential_burrows[:, -30:] = 0

        # combine with the information of what areas have been explored
        burrows_mask = cv2.bitwise_and(ground_mask, potential_burrows)
        
        if burrows_mask.sum() == 0:
            contours = []
        else:
            # find the contours of the features
            contours, hierarchy = cv2.findContours(burrows_mask.copy(), # we want to use the mask later again
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        
        burrows = []
        for contour in contours:
            contour = np.array(contour, np.int32) 

            # get enclosing rectangle
            rect = cv2.boundingRect(contour)
            burrow_fitting_margin = self.params['burrows/fitting_margin']
            rect = expand_rectangle(rect, burrow_fitting_margin)

            # focus on this part of the problem            
            slices = rect_to_slices(rect)
            ground_roi = ground[slices]
            burrow_mask = cv2.bitwise_and(ground[slices], potential_burrows[slices])
            #burrow_mask = cv2.morphologyEx(explored_area[], cv2.MORPH_CLOSE, kernel)

            #burrow_mask = burrows_mask[slices]
            frame_roi = frame[slices].astype(np.uint8)
            contour = np.squeeze(contour) - np.array([[rect[0], rect[1]]], np.int32)

            # find the combined contour of burrow and ground profile
            mask = cv2.bitwise_xor(burrow_mask, ground_roi)
            
            points, hierarchy = cv2.findContours(mask,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
            
            if len(points) == 1:
                # define the burrow and refine its outline
                burrow = Burrow(np.squeeze(points), image=frame_roi)
                outline = self.refine_burrow_outline(burrow, ground_profile, (rect[0], rect[1]))
                
                if len(outline) > 0:
                    burrows.append(Burrow(outline, time=frame_id))
                    
            else:
                logging.warn('We found multiple potential burrows in a small region. '
                             'This part will not be analyzed.')
            
        return burrows
                         


class Burrow(object):
    """ represents a single burrow to compare it against an image in fitting """
    
    array_columns = ['Time', 'Position X', 'Position Y']
    index_columns = 0 #< there could be multiple burrows at each time point
    # Hence, time can not be used as an index
    
    def __init__(self, outline, time=None, image=None):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.outline = outline
        self.image = image
        self.time = time

        
    def __len__(self):
        return len(self.outline)
        
        
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
    
    
    def show_image(self, mark_points):
        # draw burrow
        image = self.image.copy()
        cv2.drawContours(image, np.array([self.outline], np.int32), -1, 255, 1)
        for k, p in enumerate(self.outline):
            color = 255 if mark_points[k] else 128
            cv2.circle(image, (int(p[0]), int(p[1])), 3, color, thickness=-1)
        debug.show_image(image)
    
    
    def to_array(self):
        """ converts the internal representation to a single array """
        self.outline = np.asarray(self.outline)
        time_array = np.zeros((len(self.outline), 1), np.int32) + self.time
        return np.hstack((time_array, self.outline))


    @classmethod
    def from_array(cls, data):
        data = np.asarray(data)
        return cls(outline=data[1:, :], time=data[0, 0])
        
    
