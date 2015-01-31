'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import random
import os
import cPickle as pickle

import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from simanneal import Annealer

from video import debug  # @UnusedImport


def trim_nan(data):
    """ removes nan values from the ends of the array """
    for s in xrange(len(data)):
        if not np.isnan(data[s]):
            break
    for e in xrange(len(data) - 1, s, -1):
        if not np.isnan(data[e]):
            break
    else:
        return []
    return data[s:e + 1]
    


def moving_average(data, window=1):
    """ calculates a moving average with a given window along the first axis
    of the given data.
    """
    height = len(data)
    result = np.zeros_like(data) + np.nan
    size = 2*window + 1
    assert height >= size
    for pos in xrange(height):
        # determine the window
        if pos < window:
            rows = slice(0, size)
        elif pos > height - window:
            rows = slice(height - size, height)
        else:
            rows = slice(pos - window, pos + window + 1)
            
        # find indices where all values are valid
        cols = np.all(np.isfinite(data[rows, :]), axis=0)
        result[pos, cols] = data[rows, cols].mean(axis=0)
    return result
            
            
class AlignmentOptimizer(Annealer):
    copy_strategy = 'method'
    align_window = 20   
    
    Tmax = 1e1
    Tmin = 1e-5
    steps = 100000
    

    def __init__(self, image):
        fill = np.empty((len(image), self.align_window), np.double) + np.nan
        state = np.c_[fill, image, fill]
        print state.shape
        super(AlignmentOptimizer, self).__init__(state)
        
    def move(self):
        y = random.randrange(0, len(self.state))
        if random.randrange(2):
            # move row to the right
            self.state[y, 1:] = self.state[y, :-1]
        else:
            # move row to the left
            self.state[y, :-1] = self.state[y, 1:]
            
    def energy(self):
        return -np.nanmean(self.state, axis=0).std()
            


class Kymograph(object):
    """ class representing a single kymograph """
    
    align_window = 20   
    
    def __init__(self, linescans):
        """ create a kymograph from line scans """
        self.data = [(line - np.mean(line))/np.std(line)
                     for line in linescans]
#         self.data = [line for line in linescans]
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


    def _get_signal_offset(self, line1, line2):
        """ compares two signals with each other
        Returns the offset between the two signals and the two signals 
        themselves, clean from any non-finite values """
        # determine the offset of the second line with respect to the first
        line1 = np.asarray(trim_nan(line1), np.float32)
        line2 = np.asarray(trim_nan(line2), np.float32)
 
        # determine smaller template from second line
        w_len = self.align_window
        left = w_len
        right = min(len(line1), len(line2)) - w_len
        template = line2[left:right]
         
#                 conv = signal.convolve(ref, data, mode='full')
        res = cv2.matchTemplate(np.atleast_2d(line1),
                                np.atleast_2d(template),
                                method=cv2.TM_SQDIFF_NORMED)[0, :]
        offset = np.argmin(res) - w_len
        return offset, line1, line2
     
         
    def _get_line_offsets(self, image):
        """ determine the offset for each line recursively
        returns the offsets for each line in the image and a overall pattern
        describing the current image after alignment
        """
        if len(image) == 1:
            # trivial case of single line
            offsets = np.array([0], np.int)
            line = np.asarray(trim_nan(image[0, :]), np.float32)
            return offsets, line
        
        if len(image) == 2:
            # smallest case where we align two lines
            diff, l1, l2 = self._get_signal_offset(image[0], image[1])
            offsets = np.array([0, diff], np.int)

        else:
            # more complex case where we use recursion for alignment
            offsets = np.empty(len(image), np.int)
            mid = len(image)//2
            offsets1, l1 = self._get_line_offsets(image[:mid])
            offsets2, l2 = self._get_line_offsets(image[mid:])
            
            diff = self._get_signal_offset(l1, l2)[0]
            offsets = np.r_[offsets1, offsets2 + diff]

        # determine the pattern for the current image using the difference
        if diff > 0:
            # second line is to the right of the first line
            length = min(len(l1), len(l2) - diff)
            line = 0.5*(l1[:length] + l2[diff:diff + length])
        else:
            # second line is to the left of the first line
            diff *= -1
            length = min(len(l1) - diff, len(l2))
            line = 0.5*(l1[diff:diff + length] + l2[:length])
                 
        return offsets, line

    
    def align_features_linearly(self):
        # prepare image to work with
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = 100#self.align_window
        
        # initialize buffer that will be used
        data = np.empty((height, width + 2*window), np.double)

        offset_best, result_best, strength_max = None, None, 0
        for offset in xrange(-window, window + 1):
            # write the skewed image into the buffer
            data[:] = np.nan
            d = offset/height
            for y in xrange(height):
                s = int(d*y) + window
                data[y, s:s + width] = image[y, :]
                
            # blur the buffer to filter some noise 
            data = cv2.GaussianBlur(data, (0, 0), 2)

            # perform time average to find stationary features
            time_average = np.nanmean(data, axis=0)
            # filter spatially to locate features on specific scale
            w = 31
            kernel = np.ones(w, np.double)/w
            space_average = np.convolve(time_average, kernel, mode='same') 

            # judge the strength of the features by standard deviation
            strength = np.nanstd(time_average - space_average)
            
            if strength > strength_max:
                strength_max = strength
                offset_best = offset
                result_best = data.copy()
            
        offsets = np.arange(height)*offset_best/height
        #debug.show_image(result_best, aspect='auto')
        return offsets
        
            
    def align_features(self):
        self.offsets = self.align_features_linearly()
#         plt.plot(result)
#         plt.show()
            


    def align_features_anneal(self):
        # try global shift 
        image = self._make_image(self.offsets)
        optimizer = AlignmentOptimizer(image)
#         schedule = optimizer.auto(1)
#         optimizer.set_schedule(schedule)
        state_best, energy_best = optimizer.anneal()
        debug.show_image(state_best, self._make_image(self.offsets), aspect='auto')
        print state_best
        

    def align_features_rec(self):
        image = self._make_image(self.offsets)
        self.offsets = self._get_line_offsets(image)[0]
        image = self._make_image(self.offsets)


    def align_features_old(self):
        """ aligns the data such that features are preserved
        Use algorithm presented in Noble2013
        """
        offsets = np.zeros(len(self.data), np.int)
        
        for window_size in (1, 5, 11, 15, 21, 25):
            # get reference image
            img = self._make_image(self.offsets)
            reference = moving_average(img, window_size)
            
            debug.show_image(img, reference, wait_for_key=False, aspect='auto')
            
            # iterate through all lines
            for l in xrange(len(self.data)):
                # prepare the data that we want to compare
                ref = np.asarray(trim_nan(reference[l]), np.float32)
                data = np.asarray(trim_nan(img[l]), np.float32)
                
                w_len = self.align_window
                left = w_len
                right = min(len(data), len(ref)) - 2 * w_len
                data = data[left:right]
                
#                 conv = signal.convolve(ref, data, mode='full')
                res = cv2.matchTemplate(np.atleast_2d(ref),
                                        np.atleast_2d(data),
                                        method=cv2.TM_SQDIFF_NORMED)
                
                res = res[0, :]
                
                offset = np.argmin(res) - w_len
#                 pos = len(ref) #+ offsets[l]
#                 window = conv[pos - w_len:pos + w_len + 1]
# 
#                 offset = np.argmax(window) - w_len + 1

#                 plt.plot(np.arange(len(ref)), ref/max(ref), 'b', label='ref')
#                 plt.plot(np.arange(len(data)) + np.argmin(res), data/max(data), 'g', label='data')
#                 plt.plot(res/max(res), 'r', label='conv')
#                 plt.title(offset)
#                 plt.legend(loc='best')
#                 plt.show()
#                 if l > 2:
#                     exit()
                
                offsets[l] += offset
            print offsets
#             offsets = moving_average(offsets.reshape(-1, 1), window_size)
#             print offsets
            
        exit()
        self.offsets = offsets



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


