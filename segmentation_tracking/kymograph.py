'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
import cv2

from video import debug  # @UnusedImport




def _get_alignment_score(data, feature_scale=31, blur_radius=1):
    """ calculates a score for a particular alignment """
    # blur the buffer to filter some noise
    idx = np.isnan(data)
    data = cv2.GaussianBlur(data, (0, 0), blur_radius)
    data[idx] = np.nan

    # perform time average to find stationary features
    time_average = np.nanmean(data, axis=0)

    # filter spatially to locate features on specific scale
    kernel = np.ones(feature_scale, np.double)/feature_scale
    space_average = np.convolve(time_average, kernel, mode='same') 

    # judge the strength of the features by standard deviation
    score = np.nanstd(time_average - space_average)
    return score


class Kymograph(object):
    """ class representing a single kymograph """
    
    align_window = 100   
    
    def __init__(self, linescans, normalize=True):
        """ create a kymograph from line scans """
        if normalize:
            self.data = [(line - np.mean(line))/np.std(line)
                         for line in linescans]
        else:
            self.data = [line for line in linescans]
        self.offsets = np.zeros(len(linescans), np.int)
        
    
    def _make_image(self, offsets):
        """ produces an image where all the line scans are left aligned """
        offsets = (offsets - self.offsets.min()).astype(np.int)
        
        # determine maximal length of a single line scan
        size = max(len(d) for d in self.data)
        size += offsets.max()
        
        # collect the full data
        img = np.zeros((len(self.data), size)) + np.nan
        
        # iterate over all time points
        for l in xrange(len(self.data)):
            start = offsets[l]
            img[l, start:start + len(self.data[l])] = self.data[l]
            
        return img 
    
           
    def get_image(self):
        """ produces an image with the current alignment """
        return self._make_image(self.offsets)


    def align_left(self):
        """ aligns the data at the left side """
        self.offsets[:] = 0

    
    def align_features_linearly(self):
        """ align features roughly by trying linear transformations """
        # prepare image to work with
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = self.align_window
        
        # initialize buffer that will be used
        data = np.empty((height, width + 2*window), np.double)

        score_best, offset_best = 0, None
        for offset in xrange(-window, window + 1):
            offsets = offset * np.linspace(0, 1, height)
        
            # prepare an array to hold the data
            height, width = image.shape
            data[:] = np.nan
            for y, offset in enumerate(offsets):
                s = int(window + offset)
                data[y, s:s + width] = image[y, :]
                            
            # judge this offset
            score = _get_alignment_score(data)
            
            if score > score_best:
                score_best = score
                offset_best = offset
            
        offsets = offset_best * np.linspace(0, 1, height)
        self.offsets += offsets.astype(np.int)
        
        
    def align_features_individually(self, max_dist=1):
        """ run through all lines individually and try to move them """
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = self.align_window

        # prepare an array to hold the data
        data = np.empty((height, width + 2*window), np.double) + np.nan
        data[:, window:-window] = image

        # set distances that we test to [-max_dist, ..., -1, 1, ..., max_dist]
        offset_values = np.arange(-max_dist, max_dist)
        offset_values[offset_values >= 0] += 1

        score = _get_alignment_score(data)
        
        # initialize buffer that will be used
        data_best = data.copy()
        improved, score_best = 1, 0 
        while improved > 0:
            improved = 0
            for y in xrange(height):
                for offset in offset_values: #< try left and right shift
                    d = abs(offset)
                    if offset< 0:
                        data[y, :-d] = data_best[y, d:] #< shift row to left
                    else:
                        data[y, d:] = data_best[y, :-d] #< shift row to right
                    score = _get_alignment_score(data)
                    if score > score_best:
                        improved += 1
                        score_best = score
                        self.offsets[y] += offset
                        data_best = data.copy()
                        break
                else:
                    # reset row if no better state was found
                    data[y, :] = data_best[y, :]
                    
            logging.debug('Improved %d lines.' % improved)
    
            
    def align_features(self, mode='individually'):
        """ align features in the kymograph """
        self.align_features_linearly()
        if mode == 'individually':
            self.align_features_individually(1)


