'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from collections import defaultdict
import cPickle as pickle 
import itertools
import logging
import math
import os

import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import measurements
from shapely import geometry

from data_structures.cache import cached_property
from video.io import VideoFile, ImageWindow
from video.composer import VideoComposer
from video.utils import display_progress
from video.filters import FilterMonochrome, FilterResize
from video.analysis import curves, image, regions
from video.analysis.active_contour import ActiveContour

from video import debug  # @UnusedImport

from .tail import Tail
from .parameters import parameters_tracking, parameters_tracking_special
from segmentation_tracking.annotation import TackingAnnotations, SegmentPicker



class TailSegmentationTracking(object):
    """ class managing the tracking of mouse tails in videos """
    
    def __init__(self, video_file, output_file, parameter_set='default',
                 parameters=None, show_video=False):
        """
        `video_file` is the input video
        `output_file` is the video file where the output is written to
        `show_video` indicates whether the video should be shown while processing
        """
        self.video_file = video_file
        self.name = os.path.splitext(video_file)[0]
        self.annotations = TackingAnnotations(self.name)
        
        # load the parameters for the tracking
        self.params = parameters_tracking[parameter_set].copy()
        if self.name in parameters_tracking_special:
            logging.info('There are special parameters for this video.')
            logging.debug('The parameters are: %s',
                          parameters_tracking_special[self.name])
            self.params.update(parameters_tracking_special[self.name])
        if parameters is not None:
            self.params.update(parameters)

        # initialize the video
        self.video = self.load_video(video_file)
        
        # setup debug output 
        zoom_factor = self.params['output/zoom_factor']
        self.output = VideoComposer(output_file, size=self.video.size,
                                    fps=self.video.fps, is_color=True,
                                    zoom_factor=zoom_factor)
        if show_video:
            self.debug_window = ImageWindow(self.output.shape, title=video_file,
                                            multiprocessing=False)
        else:
            self.debug_window = None
        
        # setup structure for saving data
        self.kymographs = defaultdict(lambda: [[], []])
        self.tails = None
        self.frame_id = None
        self.frame = None
        self._frame_cache = {}


    def load_video(self, filename):
        """ loads and returns the video """
        # load video and make it grey scale
        video = VideoFile(filename)
        video = FilterMonochrome(video)
        
        if self.params['input/zoom_factor'] != 1:
            video = FilterResize(video, self.params['input/zoom_factor'],
                                 even_dimensions=True)
        
        # restrict video to requested frames
        if self.params['input/frames']:
            frames = self.params['input/frames']
            video = video[frames[0]: frames[1]]
            
        return video


    def load_first_frame(self):
        """ loads the first frame into self.frame """
        # TODO: correct the frame id for skipping
        if self.frame_id != 0:
            self._frame_cache = {} #< delete cache per frame
            self.frame_id = 0
            self.frame = self.video[0]
        return self.frame

    
    def process(self):
        """ processes the video """
        # process first frame to find objects
        self.process_first_frame()
    
        # iterate through all frames
        # TODO: correct the frame id for skipping
        for self.frame_id, self.frame in enumerate(display_progress(self.video)):
            self._frame_cache = {} #< delete cache per frame
            self.set_video_background()

            # adapt the object outlines
            #self.adapt_tail_contours(self.tails, blur_radius=30)
            self.adapt_tail_contours(self.tails)#, blur_radius=5)
            
            # do the line scans in each object
            for tail_id, tail in enumerate(self.tails):
                linescans = self.tail_linescans(self.frame, tail)
                self.kymographs[tail_id][0].append(linescans[0]) 
                self.kymographs[tail_id][1].append(linescans[1])
            
            # update the debug output
            self.update_video_output(self.frame)
            
        # save the data and close the videos
        self.save_kymographs()
        self.close()
        
    
    def set_video_background(self):
        """ sets the background of the video """
        if self.params['output/background'] == 'original':
            self.output.set_frame(self.frame, copy=True)
            
        elif self.params['output/background'] == 'potential':
            image = self.contour_potential
            lo, hi = image.min(), image.max()
            image = 255*(image - lo)/(hi - lo)
            self.output.set_frame(image)
            
        elif self.params['output/background'] == 'gradient':
            image = self.get_gradient_strenght(self.frame)
            lo, hi = image.min(), image.max()
            image = 255*(image - lo)/(hi - lo)
            self.output.set_frame(image)
            
        elif self.params['output/background'] == 'gradient_thresholded':
            image = self.get_gradient_strenght(self.frame)
            image = self.threshold_gradient_strength(image)
            lo, hi = image.min(), image.max()
            image = 255*(image - lo)/(hi - lo)
            self.output.set_frame(image)
            
        elif self.params['output/background'] == 'features':
            image, num_features = self.get_features(use_annotations=False)
            if num_features > 0:
                self.output.set_frame(image*(128//num_features))
            else:
                self.output.set_frame(np.zeros_like(self.frame))
            
        else:
            self.output.set_frame(np.zeros_like(self.frame))

    
    def process_first_frame(self):
        """ process the first frame to localize the tails """
        self.load_first_frame() #< stores frame in self.frame
        # locate tails roughly
        self.tails = self.locate_tails_roughly()
        # refine tails
        self.adapt_tail_contours_initially(self.tails)
        # refine tails a couple of times because convergence is sometimes bad
        #for _ in xrange(2):#0):
        #self.adapt_tail_contours(self.tails)
        
                    
    #===========================================================================
    # CONTOUR FINDING
    #===========================================================================
    
    
    def locate_tails_roughly_old(self, img):
        """ locate tail objects using thresholding """
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
    
    
    @cached_property(cache='_frame_cache')
    def frame_blurred(self):
        """ blurs the current frame """
        return cv2.GaussianBlur(self.frame.astype(np.double), (0, 0),
                                self.params['gradient/blur_radius'])
        
    
    def get_gradient_strenght(self, frame):
        """ calculates the gradient strength of the image in frame """
        # smooth the image to be able to find smoothed edges
        if frame is self.frame:
            # take advantage of possible caching
            frame_blurred = self.frame_blurred
        else:
            frame_blurred = cv2.GaussianBlur(frame.astype(np.double), (0, 0),
                                             self.params['gradient/blur_radius'])
        
        # scale frame_blurred to [0, 1]
        cv2.divide(frame_blurred, 256, dst=frame_blurred)

        # do Sobel filtering to find the frame_blurred edges
        grad_x = cv2.Sobel(frame_blurred, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(frame_blurred, cv2.CV_64F, 0, 1, ksize=5)

        # calculate the gradient strength
        gradient_mag = frame_blurred #< reuse memory
        np.hypot(grad_x, grad_y, out=gradient_mag)
        
        return gradient_mag
    
        
    def threshold_gradient_strength(self, gradient_mag):
        """ thresholds the gradient strength such that features are emphasized
        """
        lo, hi = gradient_mag.min(), gradient_mag.max()
        threshold = lo + self.params['gradient/threshold']*(hi - lo)
        bw = (gradient_mag > threshold).astype(np.uint8)
        
        for _ in xrange(2):
            bw = cv2.pyrDown(bw)

        # do morphological opening to remove noise
        w = 2#0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
        # do morphological closing to locate objects
        w = 2#0
        bw = cv2.copyMakeBorder(bw, w, w, w, w, cv2.BORDER_CONSTANT, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*w + 1, 2*w + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw = bw[w:-w, w:-w].copy()

        for _ in xrange(2):
            bw = cv2.pyrUp(bw)
        
        return bw
        

    def get_features(self, use_annotations=False, ret_raw=False):
        """ calculates a feature mask based on the image statistics """
        # calculate image statistics
        ksize = self.params['detection/statistics_window']
        _, var = image.get_image_statistics(self.frame, kernel='circle',
                                            ksize=ksize)

        # threshold the variance to locate features
        threshold = self.params['detection/statistics_threshold']*np.median(var)
        bw = (var > threshold).astype(np.uint8)
        
        if ret_raw:
            return bw
        
        # add features from the previous frames if present
        if self.tails:
            for tail in self.tails:
                # fill the features with the interior of the former tail
                polys = tail.polygon.buffer(-self.params['detection/shape_max_speed'])
                if not isinstance(polys, geometry.MultiPolygon):
                    polys = [polys]
                for poly in polys: 
                    cv2.fillPoly(bw, [np.array(poly.exterior.coords, np.int)],
                                 color=1)
                
                # calculate the distance to other tails to bound the current one
                buffer_dist = self.params['detection/shape_max_speed']
                for tail2 in self.tails:
                    if tail is not tail2:
                        dist = tail.polygon.distance(tail2.polygon)
                        buffer_dist = min(dist/2, buffer_dist)
                    
                # make sure that this tail is separated from all the others
                bound = tail.polygon.buffer(buffer_dist).exterior
                cv2.polylines(bw, [np.array(bound.coords, np.int)],
                              isClosed=True, color=0, thickness=2)
        
        if use_annotations:
            lines = self.annotations['segmentation_dividers']
            if lines:
                for line in lines:
                    cv2.line(bw, tuple(line[0]), tuple(line[1]), 0, thickness=3)

        # remove features at the edge of the image
        border = self.params['detection/border_distance']
        image.set_image_border(bw, size=border, color=0)
        
        # remove very thin features
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3)), dst=bw)

        # find features in the binary image
        contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # determine the rectangle where objects can lie in
        h, w = self.frame.shape
        rect = regions.Rectangle(x=0, y=0, width=w, height=h)
        rect.buffer(-2*border)
        rect = geometry.Polygon(rect.outline)

        bw[:] = 0
        num_features = 0  
        for contour in contours:
            if cv2.contourArea(contour) > self.params['detection/area_min']:
                # check whether the object touches the border
                feature = geometry.Polygon(np.squeeze(contour))
                if rect.exterior.intersects(feature):
                    # fill the hole in the feature
                    difference = rect.difference(feature)
                    
                    if isinstance(difference, geometry.Polygon):
                        difference = [difference] #< make sure we handle a list
                        
                    for diff in difference:
                        if diff.area < self.params['detection/area_max']:
                            feature = feature.union(diff)
                
                # reduce feature, since detection typically overshoots
                features = feature.buffer(-0.5*self.params['detection/statistics_window'])
                
                if not isinstance(features, geometry.MultiPolygon):
                    features = [features]
                
                for feature in features:
                    if feature.area > self.params['detection/area_min']:
                        #debug.show_shape(feature, background=self.frame)
                        
                        # extract the contour of the feature 
                        contour = regions.get_enclosing_outline(feature)
                        contour = np.array(contour.coords, np.int)
                        
                        # fill holes inside the objects
                        num_features += 1
                        cv2.fillPoly(bw, [contour], num_features)

#         debug.show_image(self.frame, var, bw, wait_for_key=False)

        return bw, num_features
        
        
    def locate_tails_roughly(self):
        """ locate tail objects using thresholding """
        # find features, using annotations in the first frame        
        use_annotations = (self.frame_id == 0)
        labels, _ = self.get_features(use_annotations)

        # find the contours of these features
        contours, _ = cv2.findContours(labels, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # locate the tails using these contours
        tails = []
        for contour in contours:
            print 'Area', cv2.contourArea(contour)
            if cv2.contourArea(contour) > self.params['detection/area_min']:
                tails.append(Tail(contour[:, 0, :]))
                
#         debug.show_shape(*[t.outline for t in tails], background=self.frame,
#                          wait_for_key=False)
    
        logging.info('Found %d tail(s) in frame %d', len(tails), self.frame_id)
        return tails
        
        
    @cached_property(cache='_frame_cache')
    def contour_potential(self):
        """ calculates the contour potential """ 
        #features = self.get_features[0]
        #gradient_mag = self.get_gradient_strenght(features > 0)
        
        gradient_mag = self.get_gradient_strenght(self.frame)
        return gradient_mag
        
        
    def adapt_tail_contours_initially(self, tails):
        """ adapt tail contour to frame, assuming that they could be quite far
        away """
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=self.params['outline/blur_radius_initial'],
                           closed_loop=True,
                           alpha=self.params['outline/line_tension'], 
                           beta=self.params['outline/bending_stiffness'],
                           gamma=self.params['outline/adaptation_rate'])
        ac.max_iterations = self.params['outline/max_iterations']
        ac.set_potential(self.contour_potential)
        
        # iterate through the contours
        for tail in tails:
            tail.contour = ac.find_contour(tail.contour)

#         debug.show_shape(*[t.outline for t in tails],
#                          background=self.contour_potential)


    def adapt_tail_contours(self, tails, blur_radius=10):
        """ adapt tail contour to frame, assuming that they are already close """
        # locate tails in this frame
        tails_new = self.locate_tails_roughly()
        #self.adapt_tail_contours_initially(tails)

        assert len(tails_new) == len(tails)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=20,#blur_radius,
                           closed_loop=True,
                           alpha=self.params['outline/line_tension'],
                           beta=self.params['outline/bending_stiffness'],
                           gamma=self.params['outline/adaptation_rate'])
        ac.max_iterations = self.params['outline/max_iterations']
        #potential_approx = self.threshold_gradient_strength(gradient_mag)
        ac.set_potential(self.contour_potential)

#         debug.show_shape(*[t.outline for t in tails],
#                          background=self.contour_potential)        
        
        for k, tail in enumerate(tails):
            # find the tail that is closest
            idx = np.argmin([curves.point_distance(tail.center, t.center)
                             for t in tails_new])
            
            # adapt this contour to the potential
            tail = tails_new.pop(idx)
            contour = ac.find_contour(tail.contour)
            
            # update the old tail to keep the identity of sides
            tails[k].update_contour(contour)
            
#         debug.show_shape(*[t.outline for t in tails],
#                          background=self.contour_potential)        
    
    
    def adapt_tail_contours_old(self, frame, tails, blur_radius=10):
        """ adapt tail contour to frame, assuming that they are already close """

        # threshold the gradient strength
        #potential_approx = self.threshold_gradient_strength(gradient_mag)
        
        # setup active contour algorithm
        ac = ActiveContour(blur_radius=20,#blur_radius,
                           closed_loop=True,
                           alpha=self.params['outline/line_tension'],
                           beta=self.params['outline/bending_stiffness'],
                           gamma=self.params['outline/adaptation_rate'])
        ac.max_iterations = self.params['outline/max_iterations']
        ac.set_potential(self.contour_potential)
        
        # iterate through the contours
        for tail in tails:
            contour = ac.find_contour(tail.contour)
            tail.update_contour(contour)
    
    
    #===========================================================================
    # TRACKING ANNOTATIONS
    #===========================================================================
    
    
    def annotate(self):
        """ add annotations to the video to help the segmentation """
        # determine the features of the first frame
        self.load_first_frame()
        features = self.get_features(ret_raw=True)

        # load previous annotations
        lines = self.annotations['segmentation_dividers']
        
        # use the segmentation picker to alter these segments
        picker = SegmentPicker(self.frame, features, lines)
        result = picker.run()
        if result == 'ok':
            # save the result if desired
            self.annotations['segmentation_dividers'] = picker.segments
            
    
    #===========================================================================
    # SEGMENT FINDING
    #===========================================================================
    
    
    def get_measurement_lines(self, tail):
        """
        determines the measurement segments that are used for the line sca
        """
        f_c = self.params['measurement/line_offset']
        f_o = 1 - f_c

        centerline = tail.centerline
        result = []
        for side in tail.sides:
            # find the line between the centerline and the ventral line
            points = []
            for p_c in centerline:
                p_o = curves.get_projection_point(side, p_c) #< outer line
                points.append((f_c*p_c[0] + f_o*p_o[0],
                               f_c*p_c[1] + f_o*p_o[1]))
                
            # do spline fitting to smooth the line
            smoothing = self.params['measurement/spline_smoothing']*len(points)
            tck, _ = interpolate.splprep(np.transpose(points),
                                         k=2, s=smoothing)
            
            points = interpolate.splev(np.linspace(-0.5, .8, 100), tck)
            points = zip(*points) #< transpose list
    
            # restrict centerline to object
            mline = geometry.LineString(points).intersection(tail.polygon)
            
            # pick longest line if there are many due to strange geometries
            if isinstance(mline, geometry.MultiLineString):
                mline = mline[np.argmax([l.length for l in mline])]
                
            result.append(np.array(mline.coords))
            
        return result   
    
    
    def tail_linescans(self, frame, tail):
        """ do line scans along the measurement segments of the tails """
        l = self.params['measurement/line_scan_width']
        w = 2 #< width of each individual line scan
        result = []
        for line in self.get_measurement_lines(tail):
            ps = curves.make_curve_equidistant(line, spacing=2*w)
            profile = []
            for pp, p, pn in itertools.izip(ps[:-2], ps[1:-1], ps[2:]):
                # slope
                dx, dy = pn - pp
                dlen = math.hypot(dx, dy)
                dx /= dlen; dy /= dlen
                
                # get end points of line scan
                p1 = (p[0] + l*dy, p[1] - l*dx)
                p2 = (p[0] - l*dy, p[1] + l*dx)
                
                lscan = image.line_scan(frame, p1, p2, width=w)
                profile.append(lscan.mean())
                
                self.output.add_points([p1, p2], 1, 'w')
                
            result.append(profile)
            
        return result
    
    
    def save_kymographs(self):
        """ saves all kymographs as pickle files """
        for key, tail_data in self.kymographs.iteritems():
            outfile = self.video_file.replace('.avi', '_kymo_%s.pkl' % key)
            with open(outfile, 'w') as fp:
                pickle.dump(tail_data, fp)
                
    
    #===========================================================================
    # OUTPUT
    #===========================================================================
    

    def update_video_output(self, frame):
        """ updates the video output to both the screen and the file """
        video = self.output
        # add information on all tails
        for tail_id, tail in enumerate(self.tails):
            # mark all the segments that are important in the video
            mark_points = (video.zoom_factor < 1)
            video.add_line(tail.contour, color='b', is_closed=True,
                           width=3, mark_points=mark_points)
            video.add_line(tail.ventral_side, color='g', is_closed=False,
                           width=4, mark_points=mark_points)
            video.add_line(tail.centerline, color='g',
                           is_closed=False, width=5, mark_points=mark_points)
            for k, line in enumerate(self.get_measurement_lines(tail)):
                video.add_line(line[:-3], color='r', is_closed=False, width=5,
                               mark_points=mark_points)
                video.add_text(tail.line_names[k], line[-1], color='r',
                               anchor='center middle')
            
            # mark the points that we identified
            video.add_circle(tail.endpoints[0], 10, 'g')
            video.add_circle(tail.endpoints[1], 10, 'b')
            video.add_text(str('tail %d' % tail_id), tail.center,
                           color='w', anchor='center middle')
        
        # add general information
        video.add_text(str(self.frame_id), (20, 20), size=2, anchor='top')
        
        if self.debug_window:
            self.debug_window.show(video.frame)
    
    
    def close(self):
        self.output.close()
        if self.debug_window:
            self.debug_window.close()
        