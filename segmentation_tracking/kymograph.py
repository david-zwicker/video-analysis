'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt



def plot_kymograph(infile, outfile):
    with open(infile) as fp:
        tail_data = pickle.load(fp)

    plt.figure(figsize=(10, 4))
    
    
    for side, data in enumerate(tail_data):
        # get kymograph data
        maxlen = max(len(d) for d in data)
        img = np.zeros((len(data), maxlen)) + np.nan
        for l, d in enumerate(data):
            img[l, :len(d)] = d#[::-1]


        plt.subplot(1, 2, side)
        # create image
        plt.imshow(img, aspect='auto')
        plt.gray()
        plt.xlabel('Distance from posterior [pixels]')
        plt.ylabel('Time [frames]')
        #cb = plt.colorbar(label='Mean')

    plt.suptitle(infile)
    plt.savefig(outfile)
