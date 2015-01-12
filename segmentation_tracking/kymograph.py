'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt



def plot_kymograph(infile, outfile):
    """ plots a kymograph of the line scan data """
    with open(infile) as fp:
        tail_data = pickle.load(fp)

    plt.figure(figsize=(10, 4))
    
    
    for side, data in enumerate(tail_data):
        # get kymograph data
        maxlen = max(len(d) for d in data)
        img = np.zeros((len(data), maxlen)) + np.nan
        for l, d in enumerate(data):
            img[l, :len(d)] = d#[::-1]


        plt.subplot(1, 2, side + 1)
        # create image
        plt.imshow(img, aspect='auto', interpolation='none')
        plt.gray()
        plt.xlabel('Distance from posterior [4 pixels]')
        plt.ylabel('Time [frames]')
        plt.title(['ventral', 'dorsal'][side])

    plt.suptitle(infile)
    plt.savefig(outfile)
