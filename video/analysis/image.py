'''
Created on Aug 22, 2014

@author: zwicker


contains functions that are useful for image analysis
'''

from __future__ import division

import numpy as np
import scipy.ndimage as ndimage

import cv2



def line_scan(image, p1, p2, width=5):
    """ returns the average intensity of an image along a strip of a given
    width, ranging from point p1 to p2.
    """ 
    
    # get corresponding points between the two images
    length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    p0 = (p1[0] + width*np.sin(angle), p1[1] - width*np.cos(angle))
    pts1 = np.array((p0, p1, p2), np.float32)
    pts2 = np.array(((0, 0), (0, width), (length, width)), np.float32)

    # determine and apply the affine transformation
    matrix = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(image, matrix, (int(length), 2*width))

    # return the profile
    return res.mean(axis=0)



def get_steepest_point(profile, direction=1, smoothing=0):
    """ returns the index where the profile is steepest.
    
    profile is a 1D array of intensities
    direction determines whether ascending (direction=1) or
        descending (direction=-1) slopes are search for
    smoothing determines the standard deviation of a Gaussian smoothing
        filter that is applied before looking for the slope
    """ 
    if len(profile) < 2:
        return np.nan

    if smoothing > 0:
        profile = ndimage.filters.gaussian_filter1d(profile, smoothing)
        
    i_max = np.argmax(direction*np.diff(profile))
    
    return i_max + 0.5



class regionprops(object):
    """ calculates useful properties of regions in binary images """
    def __init__(self, mask):
        self.moments = cv2.moments(mask.astype(np.uint8))
        
    @property
    def area(self):
        return self.moments['m00']
    
    @property
    def centroid(self):
        m = self.moments
        return (m['m10']/m['m00'], m['m01']/m['m00'])
    
    @property
    def eccentricity(self):
        m = self.moments
        a, b, c = m['mu20'], -m['mu11'], m['mu02']
        e1 = (a + c) + np.sqrt(4*b**2 + (a - c)**2)
        e2 = (a + c) - np.sqrt(4*b**2 + (a - c)**2)
        if e1 == 0:
            return 0
        else:
            return np.sqrt(1 - e2/e1)

    @property
    def orientation(self):
        m = self.moments
        a, b, c = m['mu20'], m['mu11'], m['mu02']
        if a - c == 0:
            if b > 0:
                return -np.pi/4
            else:
                return np.pi/4
        else:
            return -np.arctan2(2*b, (a - c))/2
