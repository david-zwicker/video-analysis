'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os
import cPickle as pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt

from video import debug  # @UnusedImport



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


    def _get_alignment_score(self, data, feature_scale=31, blur_radius=1):
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
            score = self._get_alignment_score(data)
            
            if score > score_best:
                score_best = score
                offset_best = offset
            
        offsets = offset_best * np.linspace(0, 1, height)
        self.offsets += offsets.astype(np.int)
        
        
    def align_features_individually(self):
        """ run through all lines individually and try to move them """
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = self.align_window

        # prepare an array to hold the data
        data = np.empty((height, width + 2*window), np.double) + np.nan
        data[:, window:-window] = image

        score = self._get_alignment_score(data) 
        
        # initialize buffer that will be used
        data_best = data.copy()
        improved, score_best = 1, 0 
        while improved > 0:
            improved = 0
            for y in xrange(height):
                for d in (-1, 1): #< try left and right shift
                    if d == -1:
                        data[y, :-1] = data_best[y, 1:] #< shift row to left
                    else:
                        data[y, 1:] = data_best[y, :-1] #< shift row to right
                    score = self._get_alignment_score(data)
                    if score > score_best:
                        improved += 1
                        score_best = score
                        self.offsets[y] += d
                        data_best = data.copy()
                        break
                else:
                    # reset row if no better state was found
                    data[y, :] = data_best[y, :]
                    
            logging.debug('Improved %d lines.' % improved)
    
            
    def align_features(self):
        """ align features in the kymograph """
        self.align_features_linearly()
        self.align_features_individually()
            


class TrackingResult(object):
    """ class managing the result of a a tracking """ 
    
    def __init__(self, result_file):
        """ load the kymograph from a file """
        self.name = os.path.splitext(result_file)[0]
        self.load_from_file(result_file)
        
        
    def load_from_file(self, result_file):
        """ load the kymograph data from a file """
        with open(result_file) as fp:
            self.data = pickle.load(fp)


    def calculate_kymographs(self, align=False):
        """ calculates the kymographs from the tracking result data """
        # iterate over all tails found in the movie
        for tail_data in self.data['tails']:
            tail_data['kymograph'] = []
            # iterate over all line scans (ventral and dorsal)
            for data in tail_data['line_scans']:
                kymograph = Kymograph(data)
                if align:
                    kymograph.align_features()
                tail_data['kymograph'].append(kymograph)


    def plot_kymographs(self, outfile=None):
        """ plots a kymograph of the line scan data """
        plt.figure(figsize=(10, 4))
        
        for tail_id, tail_data in enumerate(self.data['tails']):
            for side, kymograph in enumerate(tail_data['kymograph']):
                plt.subplot(1, 2, side + 1)
                # create image
                plt.imshow(kymograph.get_image(), aspect='auto',
                           interpolation='none')
                plt.gray()
                plt.xlabel('Distance from posterior [4 pixels]')
                plt.ylabel('Time [frames]')
                plt.title(['ventral', 'dorsal'][side])
        
            plt.suptitle(self.name + ' Tail %d' % tail_id)
            if outfile is None:
                plt.show()
            else:
                plt.savefig(outfile % tail_id)


