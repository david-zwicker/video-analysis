'''
Created on Aug 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>


contains functions that are useful for image analysis
'''

from __future__ import division

import numpy as np
from scipy import ndimage

import cv2

from utils import cached_property



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
    res = cv2.warpAffine(image, matrix, (int(length), int(2*width)))

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
    """ calculates useful properties of regions in binary images.
    Much of this code was inspired by the excellent skimage package, which is
    available at http://scikit-image.org
    The original source code can be found on github at
        https://github.com/scikit-image/scikit-image
        
    The license coming with scikit reads:
    
    Copyright (C) 2011, the scikit-image team
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    
    def __init__(self, mask):
        self.moments = cv2.moments(mask.astype(np.uint8))
        
    @property
    def area(self):
        return self.moments['m00']
    
    @cached_property
    def centroid(self):
        m = self.moments
        return (m['m10']/m['m00'], m['m01']/m['m00'])

    @cached_property
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
    
    @cached_property
    def inertia_tensor_eigvals(self):
        m = self.moments
        a, b, c = m['mu20']/m['m00'], -m['mu11']/m['m00'], m['mu02']/m['m00']
        # eigenvalues of inertia tensor
        e1 = (a + c) + np.sqrt(4*b**2 + (a - c)**2)
        e2 = (a + c) - np.sqrt(4*b**2 + (a - c)**2)
        return e1, e2
            
    @cached_property
    def eccentricity(self):
        e1, e2 = self.inertia_tensor_eigvals()
        if e1 == 0:
            return 0
        else:
            return np.sqrt(1 - e2/e1)
        
    @cached_property
    def major_axis_length(self):
        e1, _ = self.inertia_tensor_eigvals
        return 4*np.sqrt(e1)

    @cached_property
    def minor_axis_length(self):
        _, e2 = self.inertia_tensor_eigvals
        return 4*np.sqrt(e2)