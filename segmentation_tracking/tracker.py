'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from collections import defaultdict
import cPickle as pickle 
import itertools
import math

import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage import measurements
from shapely import geometry

from video.io import VideoFile, ImageWindow
from video.composer import VideoComposer
from video.utils import display_progress
from video.filters import FilterMonochrome
from video.analysis import curves, image
from video.analysis.active_contour import ActiveContour

from video import debug  # @UnusedImport

from tail import Tail



class TailSegmentationTracking(object):
    """ class managing the tracking of mouse tails in videos """
    
    measurement_line_offset = 0.5 #< determines position of the measurement line
    spline_smoothing = 20 #< smoothing factor for measurement lines
    line_scan_width = 40 #< width of the line scan along the measurement lines

    
    def __init__(self, video_file, output_file, show_video=False):
        """
        `video_file` is the input video
        `output_file` is the video file where the output is written to
        `show_video` indicates whether the video should be shown while processing
        """
        self.video_file = video_file
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
        
        # setup structure for saving data
        self.kymographs = defaultdict(lambda: [[], []])
        self.tails = None


    def load_video(self, filename):
        """ loads and returns the video """
        video = VideoFile(filename) 
        video = FilterMonochrome(video)
        return video

    
    def process(self):
        """ processes the video """
        # process first frame to find objects
        self.process_first_frame(self.video[0])
    
        # iterate through all frames
        for self.frame_id, frame in enumerate(display_progress(self.video)):
            self.output.set_frame(frame, copy=True)
            
            # adapt the object outlines
            self.adapt_tail_contours(frame, self.tails)
            
            # do the line scans in each object
            for tail_id, tail in enumerate(self.tails):
                linescans = self.tail_linescans(frame, tail)
                self.kymographs[tail_id][0].append(linescans[0]) 
                self.kymographs[tail_id][1].append(linescans[1])
            
            # update the debug output
            self.update_video_output(frame)
            
        # save the data and close the videos
        self.save_kymographs()
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
            tail.update_contour(ac.find_contour(tail.contour))

    
    #===========================================================================
    # SEGMENT FINDING
    #===========================================================================
    
    
    def get_measurement_lines(self, tail):
        """
        determines the measurement lines that are used for the line sca
        """
        f_c = self.measurement_line_offset
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
            smoothing = self.spline_smoothing*len(points)
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
        """ do line scans along the measurement lines of the tails """
        l = self.line_scan_width
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
        for tail_id, tail in enumerate(self.tails):
            # mark all the lines that are important in the video
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
            
        if self.debug_window:
            self.debug_window.show(video.frame)
    
    
    def close(self):
        self.output.close()
        if self.debug_window:
            self.debug_window.close()
        