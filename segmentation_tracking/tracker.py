'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import cv2
from scipy.ndimage import measurements
from shapely import geometry


from video import debug
from video.io import VideoFile
from video.composer import VideoComposer
from video.utils import display_progress
from video.filters import FilterMonochrome
from video.analysis import curves
from video.analysis.active_contour import ActiveContour


class Tail(object):
    def __init__(self, contour):
        self.contour = contour
        
    @property
    def contour(self):
        return self._contour
    
    @contour.setter
    def contour(self, points):
        self._contour = curves.make_curve_equidistant(points, spacing=20)        



class TailSegmentationTracking(object):
    
    def __init__(self, video_file, output_file):
        self.video = self.load_video(video_file) 
        self.output = VideoComposer(output_file, size=self.video.size,
                                    fps=self.video.fps, is_color=True)
        
        self.tails = None


    def load_video(self, filename):
        video = VideoFile(filename) 
        video = FilterMonochrome(video)
        return video

    
    def process(self):
        self.process_first_frame(self.video[0])
    
        for self.frame_id, frame in enumerate(display_progress(self.video)):
            self.adapt_tail_contours(frame, self.tails)
            
            self.output.set_frame(frame, copy=False)
            for tail in self.tails:
                self.output.add_line(tail.contour, color='b', is_closed=True,
                                     width=3)
    
    
    def debug_tails(self, tails, image=None):
        gtail = [geometry.LinearRing(t.contour) for t in tails]
        debug.show_shape(*gtail, background=image)
    
        
    def process_first_frame(self, frame):
        """
        """
        # locate tails roughly
        self.tails = self.locate_tails_roughly(frame)
        # refine tails
        self.adapt_tail_contours_initially(frame, self.tails)
        # refine tails a couple of times because convergence is sometimes bad
        for _ in xrange(20):
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
        """ calculates the gradient strength of the image in frame
        """
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
    