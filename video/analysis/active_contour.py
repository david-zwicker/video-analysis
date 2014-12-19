'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections

import numpy as np
import cv2

import curves
import image



class ActiveContour(object):
    """ class that manages an algorithm for using active contours for edge
    detection [http://en.wikipedia.org/wiki/Active_contour_model]
    
    This implementation is inspired by the following articles:
        http://www.pagines.ma1.upc.edu/~toni/files/SnakesAivru86c.pdf
        http://www.cb.uu.se/~cris/blog/index.php/archives/217
    """
    
    max_iterations = 50  #< maximal number of iterations
    max_cache_count = 20 #< maximal number of cache entries
    residual_tolerance = 1 #< stop iteration when reaching this residual value
    
    
    def __init__(self, blur_radius=10, alpha=0, beta=1e2, gamma=0.001,
                 point_spacing=None, closed_loop=False, keep_end_x=False):
        """ initializes the active contour model
        blur_radius sets the length scale of the attraction to features.
            As a drawback, this is also the largest feature size that can be
            resolved by the contour.
        alpha is the line tension of the contour (high alpha leads to shorter
            contours)
        beta is the stiffness of the contour (high beta leads to straighter
            contours)
        gamma is the time scale of the convergence (high gamma might lead to 
            overshoot)
        point_spacing should only be set in production runs. It can help
            speeding up the process since matrices can be reused
        closed_loop indicates whether the contour is a closed loop
        keep_end_x indicates whether the x coordinates of the end points of the
            contour are kept fixed
        """
        
        self.blur_radius = blur_radius
        self.alpha = alpha  #< line tension
        self.beta = beta    #< stiffness 
        self.gamma = gamma  #< convergence rate
        self.point_spacing = point_spacing
        self.closed_loop = closed_loop
        self.keep_end_x = keep_end_x
        
        self._Pinv_cache = collections.OrderedDict()


    def clear_cache(self):
        """ clears the cache. This method should be called if any of the
        parameters of the model are changed """
        self._Pinv_cache = collections.OrderedDict()

        
    def get_evolution_matrix(self, N, ds):
        """ calculates the evolution matrix """
        # scale parameters
        alpha = self.alpha/ds**2 # tension ~1/ds^2
        beta = self.beta/ds**4   # stiffness ~ 1/ds^4
        
        # calculate matrix entries
        a = self.gamma*(2*alpha + 6*beta) + 1
        b = self.gamma*(-alpha - 4*beta)
        c = self.gamma*beta
        
        if self.closed_loop:
            # matrix for closed loop
            P = (
                np.diag(np.zeros(N) + a) +
                np.diag(np.zeros(N-1) + b, 1) + np.diag(   [b], -N+1) +
                np.diag(np.zeros(N-1) + b,-1) + np.diag(   [b],  N-1) +
                np.diag(np.zeros(N-2) + c, 2) + np.diag([c, c], -N+2) +
                np.diag(np.zeros(N-2) + c,-2) + np.diag([c, c],  N-2)
            )
            
        else:
            # matrix for open end with vanishing derivatives
            P = (
                np.diag(np.zeros(N) + a) +
                np.diag(np.zeros(N-1) + b, 1) +
                np.diag(np.zeros(N-1) + b,-1) +
                np.diag(np.zeros(N-2) + c, 2) +
                np.diag(np.zeros(N-2) + c,-2)
            )
            P[0, 1] = P[-1, -2] = 2*b
            P[0, 2] = P[-1, -3] = 2*c
            P[0, 2] = P[-1, -3] = 2*c
            P[1, 1] = P[-2, -2] = a + c

        # create inverse matrix for iteration                
        return np.linalg.inv(P)
        
        
    def find_contour(self, potential, points):
        """ adapts the contour given by points to the potential image """
        points = np.asarray(points, np.double)
        
        # determine point spacing if it is not given
        if self.point_spacing is None:
            ds = curves.curve_length(points)/(len(points) - 1)
        else:
            ds = self.point_spacing

        # try loading the evolution matrix from the cache            
        cache_key = (len(points), ds)
        Pinv = self._Pinv_cache.get(cache_key, None)
        if not Pinv:
            # make sure that the cache does not grow indefinitely
            while len(self._Pinv_cache) >= self.max_cache_count - 1:
                self._Pinv_cache.popitem(last=False)
            # add new item to cache
            Pinv = self.get_evolution_matrix(len(points), ds)
            self._Pinv_cache[cache_key] = Pinv
    
        # get image gradient
        if self.blur_radius > 0:
            potential = cv2.GaussianBlur(potential, (0, 0), self.blur_radius)
        fx = cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=5)
        fy = cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=5)
    
        # create intermediate array
        ps = points.copy()
    
        for _ in xrange(self.max_iterations):
            # calculate external force
            fex = image.subpixels(fx, points)
            fey = image.subpixels(fy, points)
            
            # Move control points
            if self.keep_end_x:
                # move all but end points in x direction
                ps[1:-1, 0] = np.dot(Pinv[1:-1, :],
                                     points[:, 0] + self.gamma*fex)
            else:
                # move all points in x-direction
                ps[:, 0] = np.dot(Pinv, points[:, 0] + self.gamma*fex)
            # move all points in y-direction
            ps[:, 1] = np.dot(Pinv, points[:, 1] + self.gamma*fey)
            
            # check the distance that we evolved
            residual = np.abs(ps - points).sum()

            # Restrict control points to potential
            points[:, 0] = np.clip(ps[:, 0], 0, potential.shape[1] - 2)
            points[:, 1] = np.clip(ps[:, 1], 0, potential.shape[0] - 2)

            if residual < self.residual_tolerance * self.gamma:
                break
            
    
        return points
    