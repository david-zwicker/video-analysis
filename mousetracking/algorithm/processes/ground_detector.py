'''
Created on Dec 3, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import math

import numpy as np
from scipy import ndimage, optimize, signal
import cv2

from ..objects import GroundProfile
from video.analysis import image, curves, regions

from ..debug import *  # @UnusedWildImport


class GroundDetector(object):
    """ class that handles the detection and adaptation of the ground line """
    
    def __init__(self, ground, parameters):
        self.ground = ground
        self.params = parameters
        self._cache = {}
        
    
    def _get_cage_boundary(self, ground_point, frame, direction='left'):
        """ determines the cage boundary starting from a ground_point
        going in the given direction """

        # check whether we have to calculate anything
        if not self.params['cage/determine_boundaries']:
            if direction == 'left':
                return (0, ground_point[1])
            elif direction == 'right':
                return (frame.shape[1] - 1, ground_point[1])
            else:
                raise ValueError('Unknown direction `%s`' % direction)
            
        # extend the ground line toward the left edge of the cage
        if direction == 'left':
            border_point = (0, ground_point[1])
        elif direction == 'right':
            image_width = frame.shape[1] - 1
            border_point = (image_width, ground_point[1])
        else:
            raise ValueError('Unknown direction `%s`' % direction)
        
        # do the line scan
        profile = image.line_scan(frame, border_point, ground_point,
                                  self.params['cage/linescan_width'])
        
        # smooth the profile slightly
        profile = ndimage.filters.gaussian_filter1d(profile,
                                                    self.params['cage/linescan_smooth'])
        
        # add extra points to make determining the extrema reliable
        profile = np.r_[0, profile, 255]

        # determine first maximum and first minimum after that
        maxima = signal.argrelextrema(profile, comparator=np.greater_equal)
        pos_max = maxima[0][0]
        minima = signal.argrelextrema(profile[pos_max:],
                                      comparator=np.less_equal)
        pos_min = minima[0][0] + pos_max
        # we have to use argrelextrema instead of argrelmax and argrelmin,
        # because the latter don't capture wide maxima like [0, 1, 1, 0]
        
        if pos_min - pos_max >= 2:
            # get steepest point in this interval
            pos_edge = np.argmin(np.diff(profile[pos_max:pos_min + 1])) + pos_max
        else:
            # get steepest point in complete profile
            pos_edge = np.argmin(np.diff(profile))

        if direction == 'right':
            pos_edge = image_width - pos_edge
        
        return (pos_edge, ground_point[1])
        
    
    def refine_ground(self, frame):
        """ adapts a points profile given as points to a given frame.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
        # make sure the curve has equidistant points
        spacing = int(self.params['ground/point_spacing'])
        self.ground.make_equidistant(spacing=spacing)

        # consider all points that are far away from the borders        
        frame_margin = int(self.params['ground/frame_margin'])
        x_max = frame.shape[1] - frame_margin
        points = [point for point in self.ground.points
                  if frame_margin < point[0] < x_max]
        points = np.array(np.round(points),  np.int32)

        # iterate over all but the boundary points
        curvature_energy_factor = self.params['ground/curvature_energy_factor']
        if 'ground/energy_factor_last' in self._cache:
            # load the energy factor for the next iteration
            snake_energy_max = self.params['ground/snake_energy_max']
            energy_factor_last = self._cache['energy_factor_last']
            
        else:
            # initialize values such that the energy factor is calculated
            # for the next iteration
            snake_energy_max = np.inf
            energy_factor_last = 1

        # parameters of the line scan            
        ray_len = int(self.params['ground/linescan_length']/2)
        profile_width = self.params['ground/point_spacing']/2
        ridge_width = self.params['ground/ridge_width']
        len_ratio = ray_len/ridge_width
        model_std = math.sqrt(1 - math.tanh(len_ratio)/len_ratio)
        assert 0.5 < model_std < 1
            
        # iterate through all points in random order and correct profile
        energies_image = []
        num_points = len(points) - 2 #< disregard the boundary points
        candidate_points = [None]*num_points
        for k in np.random.permutation(num_points):
            # get local points and slopes
            p_p, p, p_n =  points[k], points[k+1], points[k+2]
            dx, dy = p_n - p_p
            dist = math.hypot(dx, dy)
            if dist == 0: #< something went wrong 
                continue #< skip this point
            dx /= dist; dy /= dist

            # do the line scan perpendicular to the ground line         
            p_a = (p[0] - ray_len*dy, p[1] + ray_len*dx)
            p_b = (p[0] + ray_len*dy, p[1] - ray_len*dx)
            profile = image.line_scan(frame, p_a, p_b, width=profile_width)
            # scale profile to -1, 1
            profile -= profile.mean()
            try:
                profile /= profile.std()
            except RuntimeWarning:
                # this can happen in strange cases where the profile is flat
                continue

            plen = len(profile)
            xs = np.linspace(-plen/2 + 0.5, plen/2 - 0.5, plen)
        
            def energy_image((pos, model_mean, model_std)):
                """ part of the energy related to the line scan """
                # get image part
                model = np.tanh(-(xs - pos)/ridge_width)
                img_diff = profile - model_std*model - model_mean
                squared_diff = np.sum(img_diff**2)
                return energy_factor_last*squared_diff
                # energy_image has units of color^2
                
            def energy_curvature(pos):
                """ part of the energy related to the curvature of the ground line """
                # get curvature part: http://en.wikipedia.org/wiki/Menger_curvature
                p_c = (p[0] + pos*dy, p[1] - pos*dx)
                a = curves.point_distance(p_p, p_c)
                b = curves.point_distance(p_c, p_n)
                c = curves.point_distance(p_n, p_p)
                
                # determine curvature of circle through the three points
                A = regions.triangle_area(a, b, c)
                curvature = 4*A/(a*b*c)*spacing
                # We don't scale by with the arc length a + b, because this 
                # would increase the curvature weight in situations where
                # fitting is complicated (close to burrow entries)
                return curvature_energy_factor*curvature

            def energy_snake(data):
                """ energy function of this part of the ground line """
                return energy_image(data) + energy_curvature(data[0])
            
            # fit the simple model to the line scan profile
            fit_bounds = ((-spacing, spacing), (None, None), (None, None))
            try:      
                #res = optimize.fmin_powell(energy_snake, [0, 0, model_std], disp=False)
                res, snake_energy, _ = \
                    optimize.fmin_l_bfgs_b(energy_snake, approx_grad=True,
                                           x0=np.array([0, 0, model_std]),
                                           bounds=fit_bounds)
            except RuntimeError:
                continue #< skip this point

            # use this point, if it is good enough            
            if snake_energy < snake_energy_max:
                # save final energy for determining the energy scale later
                energies_image.append(energy_image(res))

                pos, _, model_std = res
                p_x, p_y = p[0] + pos*dy, p[1] - pos*dx
                candidate_points[k] = (int(p_x), int(p_y))

        # filter points, where the fit did not work
        # FIXME: fix overhanging ridge detection (changed sign of comparison Oct. 27)
        points = []
        for candidate in candidate_points:
            if candidate is None:
                continue
            
            # check for overhanging ridge
            if len(points) == 0 or candidate[0] > points[-1][0]:
                points.append(candidate)
                
            # there is an overhanging part, since candidate[0] <= points[-1][0]:
            elif candidate[1] < points[-1][1]:
                # current point is above previous point
                # => replace the previous candidate point
                points[-1] = candidate
                
            else:
                # current point is below previous point
                # => do not add the current candidate point
                pass
                
        # save the energy factor for the next iteration
        energy_factor_last /= np.mean(energies_image)
        self._cache['ground/energy_factor_last'] = energy_factor_last

        if len(points) < 0.5*num_points:
            # refinement failed for too many points => return original ground
            self.logger.debug('%d: Ground profile shortened too much.',
                              self.frame_id)
            return self.ground

        # extend the ground line toward the left edge of the cage
        edge_point = self._get_cage_boundary(points[0], frame, 'left')
        if edge_point is not None:
            points.insert(0, edge_point)
            
        # extend the ground line toward the right edge of the cage
        edge_point = self._get_cage_boundary(points[-1], frame, 'right')
        if edge_point is not None:
            points.append(edge_point)
            
        self.ground = GroundProfile(points)
        return self.ground

    
    
class ActiveContour(object):
    
    max_iterations = 50
    
    def __init__(self, blur_radius=10, alpha=0, beta=1e2, gamma=0.001,
                 point_spacing=None, closed_loop=False, keep_end_x=False):
        self.blur_radius = blur_radius
        self.alpha = alpha  #< line tension
        self.beta = beta    #< stiffness 
        self.gamma = gamma  #< convergence rate
        self.point_spacing = point_spacing
        self.closed_loop = closed_loop
        self.keep_end_x = keep_end_x
        
        self._Pinv_cache = {}


    def clear_cache(self):
        self._Pinv_cache = {}

        
    def get_evolution_matrix(self, N, ds):
        # calculate evolution matrix

        # scale parameters
        alpha = self.alpha/ds**2 # tension ~1/ds^2
        beta = self.beta/ds**4 # stiffness ~ 1/ds^4
        
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
        """ adapts the contour given by points to the image
        ds is the line spacing, which
        """
        points = np.asarray(points, np.double)
        
        if self.point_spacing is None:
            ds = curves.curve_length(points)/(len(points) - 1)
        else:
            ds = self.point_spacing

        # try loading the inverse matrix from the cache            
        N = len(points)
        cache_key = (N, ds)
        Pinv = self._Pinv_cache.get(cache_key, None)
        if not Pinv:
            Pinv = self.get_evolution_matrix(N, ds)
            self._Pinv_cache[cache_key] = Pinv
    
        # get image gradient
        if self.blur_radius > 0:
            potential = cv2.GaussianBlur(potential, (0, 0), self.blur_radius)
        fx = cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=5)
        fy = cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=5)
    
#         show_image(potential, (fx, fy))
    
        for _ in xrange(self.max_iterations):
            # TODO: find better stopping criterium
            # calculate external force
            fex = image.subpixels(fx, points)
            fey = image.subpixels(fy, points)
            
            # Move control points
            if self.keep_end_x:
                # move all but end points in x direction
                points[1:-1, 0] = np.dot(Pinv[1:-1, :],
                                         points[:, 0] + self.gamma*fex)
            else:
                # move all points in x-direction
                points[:, 0] = np.dot(Pinv, points[:, 0] + self.gamma*fex)
            # move all points in y-direction
            points[:, 1] = np.dot(Pinv, points[:, 1] + self.gamma*fey)
            
            # Restrict control points to potential
            np.clip(points[:, 0], 0, potential.shape[1] - 2, out=points[:, 0])
            np.clip(points[:, 1], 0, potential.shape[0] - 2, out=points[:, 1])
    
        return points
    
    
    
class GroundDetectorGlobal(GroundDetector):
    """ class that handles the detection and adaptation of the ground line """
    

    def __init__(self, *args, **kwargs):
        super(GroundDetectorGlobal, self).__init__(*args, **kwargs)
        self.img_shape = None
        self.blur_radius = 20
        self.contour_finder = None

    
    def get_buffers(self, indices, shape):
        # prepare buffers
        if ('img_buffer' not in self._cache or
            len(self._cache['img_buffer']) <= max(indices)):
            self.img_shape = shape
            self._cache['img_buffer'] = [np.zeros(self.img_shape, np.double)
                                         for _ in xrange(max(indices) + 1)]
            
        return [self._cache['img_buffer'][i] for i in indices]

    
    def get_gradient_strenght(self, frame):
        """ calculates the gradient strength of the image in frame
        This function returns its result in buffer1
        """
        buffer1, buffer2, buffer3 = self.get_buffers((0, 1, 2), frame.shape)
        
        # smooth the frame_blurred to be able to find smoothed edges
        blur_radius = self.params['ground/ridge_width']
        frame_blurred = buffer1
        cv2.GaussianBlur(frame, (0, 0), blur_radius, dst=frame_blurred)
        
        # do Sobel filtering to find the frame_blurred edges
        sobel_x, sobel_y = buffer2, buffer3
        cv2.divide(frame_blurred, 256, dst=frame_blurred) #< scale frame_blurred to [0, 1]
        cv2.Sobel(frame_blurred, cv2.CV_64F, 1, 0, ksize=5, dst=sobel_x)
        cv2.Sobel(frame_blurred, cv2.CV_64F, 0, 1, ksize=5, dst=sobel_y)

        # restrict to edges that go from dark to white (from top to bottom)        
        sobel_y[sobel_y < 0] = 0
        
        # calculate the gradient strength
        gradient_mag = frame_blurred #< reuse memory
        np.hypot(sobel_x, sobel_y, out=gradient_mag)
        
        return gradient_mag
        
        
    def get_gradient_vector_flow(self, potential):
        # use Eq. 12 from Paper `Gradient Vector Flow: A New External Force for Snakes`
        fx, fy, fxy, u, v = self.get_buffers(range(1, 6), potential.shape)
        
        cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=5, dst=fx)
        cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=5, dst=fy)
        np.add(fx**2, fy**2, out=fxy)
        
#         print fx.max(), fy.max(), fxy.max()
#         show_image(potential, fx, fy, fxy, u, v)
        
        mu = 10
        def dudt(u):
            return mu*cv2.Laplacian(u, cv2.CV_64F) - (u - fx)*fxy
        def dvdt(v):
            return mu*cv2.Laplacian(v, cv2.CV_64F) - (v - fy)*fxy
        
         
        N = 10000 #< maximum number of steps that the integrator is allowed
        dt = 1e-4 #< time step

        
        for n in xrange(N):
            rhs = dudt(u)
            residual = np.abs(rhs).sum()
            if n % 1000 == 0:
                print n*dt, '%e' % residual 
            u += dt*rhs
        
        for n in xrange(N):
            rhs = dvdt(v)
#             residual = np.abs(rhs).sum()
#             if n % 100 == 0:
#                 print n*dt, '%e' % residual 
            v += dt*rhs
        
        
        show_image(potential, (u, v))
        return (u, v)
    
    
    def refine_ground(self, frame):
        """ adapts a points profile given as points to a given frame.
        Here, we fit a ridge profile in the vicinity of every point of the curve.
        The only fitting parameter is the distance by which a single points moves
        in the direction perpendicular to the curve. """
        
        # prepare image
        potential = self.get_gradient_strenght(frame)
        
        # TODO: Try using gradient vector flow from this paper:
        # Gradient Vector Flow: A New External Force for Snakes
        
        # make sure the curve has equidistant points
        spacing = int(self.params['ground/point_spacing'])
        self.ground.make_equidistant(spacing=spacing)

        frame_margin = int(self.params['ground/frame_margin'])
        x_max = frame.shape[1] - frame_margin
        points = [point for point in self.ground.points
                  if frame_margin < point[0] < x_max]
        
        if self.contour_finder is None:
            # first contour fitting
            while self.blur_radius > 0:
                self.contour_finder = ActiveContour(blur_radius=self.blur_radius,
                                                    keep_end_x=True,
                                                    beta=0*1e1, alpha=0)
                points = self.contour_finder.find_contour(potential, points)
                if self.blur_radius < 2:
                    self.blur_radius = 0
                else:
                    self.blur_radius /= 2
        else:
            # reuse the previous contour finder
            points = self.contour_finder.find_contour(potential, points)

        points = points.tolist()

#         show_shape(geometry.LineString(points),
#                    background=potential, mark_points=True)
                
        # extend the ground line toward the left edge of the cage
        edge_point = self._get_cage_boundary(points[0], frame, 'left')
        if edge_point is not None:
            points.insert(0, edge_point)
             
        # extend the ground line toward the right edge of the cage
        edge_point = self._get_cage_boundary(points[-1], frame, 'right')
        if edge_point is not None:
            points.append(edge_point)

        self.ground = GroundProfile(points)
        return self.ground    
