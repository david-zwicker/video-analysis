'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import cv2
import numpy as np
from scipy import spatial

from utils.data_structures.cache import DictFiniteCapacity
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
                 closed_loop=False):
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
        closed_loop indicates whether the contour is a closed loop
        """
        
        self.blur_radius = blur_radius
        self.alpha = float(alpha)  #< line tension
        self.beta = float(beta)    #< stiffness 
        self.gamma = float(gamma)  #< convergence rate
        self.closed_loop = closed_loop
        
        self.clear_cache()  #< also initializes the cache
        self.fx = self.fy = None
        self.info = {}


    def clear_cache(self):
        """ clears the cache. This method should be called if any of the
        parameters of the model are changed """
        self._Pinv_cache = DictFiniteCapacity(capacity=self.max_cache_count)

        
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
        
    
    def set_potential(self, potential):
        """ sets the potential and calculates the associated derivatives """
        # get image gradient
        if self.blur_radius > 0:
            potential = cv2.GaussianBlur(potential, (0, 0), self.blur_radius)
        self.fx = cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=5)
        self.fy = cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=5)
    
        
    def find_contour(self, curve, anchor_x=None, anchor_y=None):
        """ adapts the contour given by points to the potential image
        anchor_x can be a list of indices for those points whose x-coordinate
            should be kept fixed.
        anchor_y is the respective argument for the y-coordinate
        """
        if self.fx is None:
            raise RuntimeError('Potential must be set before the contour can '
                               'be adapted.')

        # curve must be equidistant for this implementation to work
        curve = np.asarray(curve)    
        points = curves.make_curve_equidistant(curve)
        
        # check for marginal small cases
        if len(points) <= 2:
            return points
        
        def _get_anchors(indices, coord):
            """ helper function for determining the anchor points """
            if indices is None or len(indices) == 0:
                return tuple(), tuple()
            # get points where the coordinate `coord` has to be kept fixed
            ps = curve[indices, :] 
            # find the points closest to the anchor points
            dist = spatial.distance.cdist(points, ps)
            return np.argmin(dist, axis=0), ps[:, coord] 
        
        # determine anchor_points if requested
        if anchor_x is not None or anchor_y is not None:
            has_anchors = True
            x_idx, x_vals = _get_anchors(anchor_x, 0)
            y_idx, y_vals = _get_anchors(anchor_y, 1)
        else:
            has_anchors = False

        # determine point spacing if it is not given
        ds = curves.curve_length(points)/(len(points) - 1)
            
        # try loading the evolution matrix from the cache            
        cache_key = (len(points), ds)
        Pinv = self._Pinv_cache.get(cache_key, None)
        if Pinv is None:
            # add new item to cache
            Pinv = self.get_evolution_matrix(len(points), ds)
            self._Pinv_cache[cache_key] = Pinv
    
        # restrict control points to shape of the potential
        points[:, 0] = np.clip(points[:, 0], 0, self.fx.shape[1] - 2)
        points[:, 1] = np.clip(points[:, 1], 0, self.fx.shape[0] - 2)

        # create intermediate array
        points_initial = points.copy()
        ps = points.copy()
    
        for k in xrange(self.max_iterations):
            # calculate external force
            fex = image.subpixels(self.fx, points)
            fey = image.subpixels(self.fy, points)
            
            # move control points
            ps[:, 0] = np.dot(Pinv, points[:, 0] + self.gamma*fex)
            ps[:, 1] = np.dot(Pinv, points[:, 1] + self.gamma*fey)
            
            # enforce the position of the anchor points
            if has_anchors:
                ps[x_idx, 0] = x_vals
                ps[y_idx, 1] = y_vals
            
            # check the distance that we evolved
            residual = np.abs(ps - points).sum()

            # restrict control points to shape of the potential
            points[:, 0] = np.clip(ps[:, 0], 0, self.fx.shape[1] - 2)
            points[:, 1] = np.clip(ps[:, 1], 0, self.fx.shape[0] - 2)

            if residual < self.residual_tolerance * self.gamma:
                break
            
        # collect additional information
        self.info['iteration_count'] = k + 1
        self.info['total_variation'] = np.abs(points_initial - points).sum()
    
        return points

    