'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import measurements
from scipy.spatial import distance
from shapely import geometry

from video import debug
from video.io import VideoFile, ImageWindow
from video.composer import VideoComposer
from video.utils import display_progress
from video.filters import FilterMonochrome
from video.analysis import curves
from video.analysis.active_contour import ActiveContour
from data_structures.cache import cached_property


class Tail(object):
    
    def __init__(self, contour):
        self.contour = contour
        
        
    @property
    def contour(self):
        return self._contour
    
    @contour.setter
    def contour(self, points):
        points = curves.make_curve_equidistant(points, spacing=20)
        if geometry.LinearRing(points).is_ccw:
            self._contour = np.array(points[::-1])
        else:
            self._contour = np.array(points)
        self._cache = {}
        

    @cached_property
    def outline(self):
        return geometry.LinearRing(self.contour)
    
    @cached_property
    def polygon(self):
        return geometry.Polygon(self.contour)


    @cached_property
    def bounds(self):
        return np.array(self.outline.bounds, np.int)
    
    
    @cached_property
    def mask(self):
        x_min, y_min, x_max, y_max = self.bounds
        shape = (y_max - y_min) + 3, (x_max - x_min) + 3
        offset =  (-x_min + 1, -y_min + 1)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [self.contour.astype(np.int)], 255,
                     offset=offset)
        return mask, offset
    
    
    @cached_property
    def endpoint_indices(self):
        """ locate the end points as contour points with maximal distance """
        # get the points which are farthest away from each other
        dist = distance.squareform(distance.pdist(self.contour))
        indices = np.unravel_index(np.argmax(dist), dist.shape)
        
        # determine the surrounding mass of tissue to determine posterior end
        # TODO: We might have to determine the posterior end from previous
        # tails, too
        mass = []
        for k in indices:
            p = geometry.Point(self.contour[k]).buffer(500)
            mass.append(self.polygon.intersection(p).area)
            
        # determine posterior end point by measuring the surrounding
        if mass[1] > mass[0]:
            return indices
        else:
            return indices[::-1]
        
        
    @cached_property
    def endpoints(self):
        j, k = self.endpoint_indices
        return self.contour[j], self.contour[k]
    
    
    def _get_both_contour_sides(self):
        k1, k2 = self.endpoint_indices
        if k2 > k1:
            ps = [self.contour[k1:k2 + 1],
                  np.r_[self.contour[k2:], self.contour[:k1 + 1]]]
        else:
            ps = [self.contour[k2:k1 + 1][::-1],
                  np.r_[self.contour[k1:], self.contour[:k2 + 1]][::-1, :]]
        return ps
            
        
    def determine_ventral_side(self):
        """ determines the ventral side from the curvature of the tail """
        # define a line connecting both end points
        k1, k2 = self.endpoint_indices
        line = geometry.LineString([self.contour[k1], self.contour[k2]])
        
        # cut the shape using this line and return the largest part
        parts = self.polygon.difference(line.buffer(0.1))
        areas = [part.area for part in parts]
        polygon = parts[np.argmax(areas)].buffer(0.1)
        
        # get the two contour lines connecting the end points
        cs = self._get_both_contour_sides()
        
        # measure the fraction of points that lie in the polygon
        fs = []
        for c in cs:
            mp = geometry.MultiPoint(c)
            frac = len(mp.intersection(polygon))/len(mp)
            fs.append(frac)

        return cs[np.argmax(fs)]
    
    
    def update_ventral_side(self, tail_prev):
        """ determines the ventral side by comparing to an earlier shape """
        # get the two contour lines connecting the end points
        cs = self._get_both_contour_sides()
        
        # get average distance of these two lines to the previous dorsal line
        line_prev = geometry.LineString(tail_prev.ventral_side)
        dists = [np.mean([line_prev.distance(geometry.Point(p))
                          for p in c])
                 for c in cs]
        
        return cs[np.argmin(dists)]

        
    @property
    def ventral_side(self):
        """ returns the points along the ventral side """
        if 'ventral_side' not in self._cache:
            self._cache['ventral_side'] = self.determine_ventral_side()
        return self._cache['ventral_side']
    
    
    @cached_property
    def centerline(self):
        """ determine the center line of the tail """
        mask, offset = self.mask
        dist_map = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 5)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=5,
                           closed_loop=False,
                           alpha=0, #< line length is constraint by beta
                           beta=1e6,
                           gamma=1e-2)
        ac.max_iterations = 500
        ac.set_potential(dist_map)
        
        # find centerline starting from the ventral_side
        points = curves.translate_points(self.ventral_side, *offset)
        points = curves.make_curve_equidistant(points, spacing=50)
        # use the active contour algorithm
        points = ac.find_contour(points)
        
        points = curves.translate_points(points, -offset[0], -offset[1])
        
        
        return points
    
    
    @cached_property
    def measurement_line(self):
        """ determines the measurement line, we should use the central line
        and the ventral line to determine where to measure """
        # measurement line is defined by an offset to the ventral side
        ventral_line = self.ventral_side
        centerline = self.centerline
        
        # find the line between the centerline and the ventral line
        points = []
        for p in centerline:
            pp = curves.get_projection_point(ventral_line, p)
            points.append((0.5*(p[0] + pp[0]), 0.5*(p[1] + pp[1])))
            
        # do spline fitting
        tck, _ = interpolate.splprep(np.transpose(points),
                                     k=2, s=10*len(points))
        
        points = interpolate.splev(np.linspace(-0.5, 1.5, 100), tck)
        points = zip(*points) #< transpose list

        # restrict centerline to object
        cline = geometry.LineString(points).intersection(self.polygon)
            
        return np.array(cline.coords)
        
#         lines = [ventral_line.parallel_offset(100, side)
#                  for side in ('left', 'right')]
#         
#         # pick the side that lies within the object
#         overlap = [self.polygon.intersection(l).length
#                    for l in lines]
#         line = lines[np.argmax(overlap)]
# 
#         # pick longest line if there are many due to strange geometries
#         if isinstance(line, geometry.MultiLineString):
#             line = line[np.argmax([l.length for l in line])]
#         
#         return line.coords
    



class TailSegmentationTracking(object):
    """ class managing the tracking of mouse tails in videos """
    
    def __init__(self, video_file, output_file, show_video=False):
        """
        `video_file` is the input video
        `output_file` is the video file where the output is written to
        `show_video` indicates whether the video should be shown while processing
        """
        self.video = self.load_video(video_file)
        # setup debug output 
        self.output = VideoComposer(output_file, size=self.video.size,
                                    fps=self.video.fps, is_color=True,
                                    zoom_factor=1)
        if show_video:
            self.debug_window = ImageWindow(self.output.shape, title=video_file,
                                            multiprocessing=False)
        else:
            self.debug_window = None
        
        self.tails = None


    def load_video(self, filename):
        """ loads and returns the video """
        video = VideoFile(filename) 
        video = FilterMonochrome(video)
        return video

    
    def process(self):
        """ processes the video """
        self.process_first_frame(self.video[0])
    
        for self.frame_id, frame in enumerate(display_progress(self.video)):
            self.adapt_tail_contours(frame, self.tails)
            
#             for tail in self.tails:
#                 tail.measurement_line
            
            self.update_video_output(frame)
            
        self.close()

    
    def process_first_frame(self, frame):
        """ process the first frame to localize the tails """
        # locate tails roughly
        self.tails = self.locate_tails_roughly(frame)
        # refine tails
        self.adapt_tail_contours_initially(frame, self.tails)
        # refine tails a couple of times because convergence is sometimes bad
        for _ in xrange(2):#0):
            self.adapt_tail_contours(frame, self.tails)
        
                    
    #===========================================================================
    # CONTOUR FINDING
    #===========================================================================
    
    
    def locate_tails_roughly(self, img):
        """ locate tail objects in thresholded image """
        for _ in xrange(3):
            img = cv2.pyrDown(img)
    
        # blur away noise
        img = cv2.GaussianBlur(img, (0, 0), 2)
    
        # scale frame_blurred to [0, 1]
        img = cv2.divide(img.astype(np.double), 256)
    
        # do Sobel filtering to find the frame_blurred edges
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
        # calculate the gradient strength
        gradient_mag = np.hypot(grad_x, grad_y)
        # scale gradient magnitude back to [0, 255]
        gradient_mag = (gradient_mag*256/gradient_mag.max()).astype(np.uint8)
    
        # threshold gradient boundary to find boundaries
        thresh = gradient_mag.mean() #+ gradient_mag.std()
        _, bw = cv2.threshold(gradient_mag, thresh, 255, cv2.THRESH_BINARY)
    
        # do morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
        # do morphological closing to locate objects
        w = 5
        bw = cv2.copyMakeBorder(bw, w, w, w, w, cv2.BORDER_CONSTANT, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw = bw[w:-w, w:-w].copy()
    
        labels, num_features = measurements.label(bw)
    
        tails = []
        for label in xrange(1, num_features + 1):
            mask = (labels == label)
            if mask.sum() > 5000:
                w = 30
                mask = cv2.copyMakeBorder(mask.astype(np.uint8), w, w, w, w, cv2.BORDER_CONSTANT, 0)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask[w:-w, w:-w].copy()
    
                # find objects in the image
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                contour = np.squeeze(contours) * 8
                tails.append(Tail(contour))
    
        return tails
    
    
    def get_gradient_strenght(self, frame):
        """ calculates the gradient strength of the image in frame """
        # smooth the frame_blurred to be able to find smoothed edges
        frame_blurred = cv2.GaussianBlur(frame.astype(np.double), (0, 0), 10)
        
        # scale frame_blurred to [0, 1]
        cv2.divide(frame_blurred, 256, dst=frame_blurred)

        # do Sobel filtering to find the frame_blurred edges
        grad_x = cv2.Sobel(frame_blurred, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(frame_blurred, cv2.CV_64F, 0, 1, ksize=5)

        # calculate the gradient strength
        gradient_mag = frame_blurred #< reuse memory
        np.hypot(grad_x, grad_y, out=gradient_mag)
        
        return gradient_mag
    
        
    def adapt_tail_contours_initially(self, frame, tails):
        """ adapt tail contour to frame, assuming that they could be quite far
        away """
        # get potential
        potential = self.get_gradient_strenght(frame)

        # setup active contour algorithm
        ac = ActiveContour(blur_radius=30,
                           closed_loop=True,
                           alpha=0, #< line length is constraint by beta
                           beta=1e1,
                           gamma=1e0)
        ac.max_iterations = 300
        ac.set_potential(potential)
        
        # iterate through the contours
        for tail in tails:
            for _ in xrange(10):
                tail.contour = ac.find_contour(tail.contour)
#         self.debug_tails(tails, potential)

    
    def adapt_tail_contours(self, frame, tails):
        """ adapt tail contour to frame, assuming that they are already close """

        # get potential
        potential = self.get_gradient_strenght(frame)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=5,
                           closed_loop=True,
                           alpha=0, #< line length is constraint by beta
                           beta=1e6,
                           gamma=1e0)
        ac.max_iterations = 300
        ac.set_potential(potential)
        
        # iterate through the contours
        for tail in tails:
            tail.contour = ac.find_contour(tail.contour)
#         self.debug_tails(tails, frame)

    
    #===========================================================================
    # SEGMENT FINDING
    #===========================================================================
    
    #===========================================================================
    # OUTPUT
    #===========================================================================
    
    def debug_tails(self, tails, image=None):
        gtail = [tail.outline for tail in tails]
        debug.show_shape(*gtail, background=image)
    
    
    def update_video_output(self, frame):
        """ updates the video output to both the screen and the file """
        self.output.set_frame(frame, copy=False)
        for tail in self.tails:
            # mark all the lines that are important in the video
            mark_points = (self.output.zoom_factor < 3)
            self.output.add_line(tail.contour, color='b', is_closed=True,
                                 width=3, mark_points=mark_points)
            self.output.add_line(tail.ventral_side, color='g', is_closed=False,
                                 width=4, mark_points=mark_points)
            self.output.add_line(tail.centerline, color='g',
                                 is_closed=False, width=5, mark_points=mark_points)
            self.output.add_line(tail.measurement_line, color='r',
                                 is_closed=False, width=5, mark_points=mark_points)
            
            # mark the points that we identified
            self.output.add_circle(tail.endpoints[0], 10, 'g')
            self.output.add_circle(tail.endpoints[1], 10, 'b')
            
        if self.debug_window:
            self.debug_window.show(self.output.frame)
    
    
    def close(self):
        self.output.close()
        if self.debug_window:
            self.debug_window.close()
        