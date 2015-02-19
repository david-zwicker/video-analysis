'''
Created on Feb 14, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from utils.concurrency import WorkerThread
from video.analysis import regions



class BackgroundExtractor(object):
    """ a class that averages multiple frames of a movie to do background
    extraction """
    
    def __init__(self, parameters, blur_function=None, object_radius=0,
                 use_threads=True):
        """ initialize the background extractor with
        `parameters` is a dictionary of parameters influencing the algorithm
        `blur_function` is an optional function that, if given, supplies a
            blurred image of the background via the `blurred` property
        `object_radius` is an additional parameter that influences how the
            background extraction is done.
        """
        self.image = None
        self.image_uint8 = None
        self._adaptation_rate = None
        self.params = parameters
        
        self._blurred = None
        if blur_function:
            self._blur_worker = WorkerThread(blur_function,
                                             use_threads=use_threads)
        else:
            self._blur_worker = None
        
        if object_radius > 0:
            # create a simple template of the mouse, which will be used to update
            # the background image only away from the mouse.
            # The template consists of a core region of maximal intensity and a ring
            # region with gradually decreasing intensity.
            
            # determine the sizes of the different regions
            size_core = object_radius
            size_ring = 3*object_radius
            size_total = size_core + size_ring
    
            # build a filter for finding the mouse position
            x, y = np.ogrid[-size_total:size_total + 1, -size_total:size_total + 1]
            r = np.sqrt(x**2 + y**2)
    
            # build the mouse template
            object_mask = (
                # inner circle of ones
                (r <= size_core).astype(float)
                # + outer region that falls off
                + np.exp(-((r - size_core)/size_core)**2)  # smooth function from 1 to 0
                  * (size_core < r)          # mask on ring region
            )  
            
            self._object_mask = 1 - object_mask
        
    
    def update(self, frame, tracks=None):
        """ update the background with the current frame """
        if self.image is None:
            self.image = frame.astype(np.double, copy=True)
            self._blur_worker.put(self.image) #< initialize background worker
            self.image_uint8 = frame.astype(np.uint8, copy=True)
            self._adaptation_rate = np.empty_like(frame, np.double)
        
        # check whether there are currently objects tracked 
        if tracks:
            # load some values from the cache
            adaptation_rate = self._adaptation_rate
            adaptation_rate.fill(self.params['adaptation_rate'])
            
            # cut out holes from the adaptation_rate for each object estimate
            for obj in tracks:
                # get the slices required for comparing the template to the image
                t_s, i_s = regions.get_overlapping_slices(obj.last.pos,
                                                          self._object_mask.shape,
                                                          frame.shape)
                adaptation_rate[i_s[0], i_s[1]] *= self._object_mask[t_s[0], t_s[1]]
                
        else:
            # use the default adaptation rate everywhere when mouse is unknown
            adaptation_rate = self.params['adaptation_rate']

        # adapt the background to current frame, but only inside the mask 
        self.image += adaptation_rate*(frame - self.image)
        
        # initialize the blurring of the image if requested
        if self._blur_worker:
            self._blurred = self._blur_worker.get()
            self._blur_worker.put(self.image)
        

    @property
    def blurred(self):
        """ returns a blurred version of the image if the `blur_function` was
        defined. This blurred image might be from the last background image and
        not the current one, which shouldn't make any difference since the
        background typically evolves slowly """
        if self._blurred is None:
            self._blurred = self._blur_worker.get()
        return self._blurred
            