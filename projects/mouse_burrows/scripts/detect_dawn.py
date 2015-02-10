#!/usr/bin/env python2
'''
Created on Nov 26, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from contextlib import closing
import sys
import os

import h5py
import numpy as np
from scipy import ndimage

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)
from utils.misc import display_progress
from video.io import load_any_video



def determine_average_frame_brightness(video_file_or_pattern,
                                       output_hdf5_file=None):
    """ iterates a video and determines its intensity, which will be stored
    in a hdf5 file, if the respective file name is given"""
    # read video data
    with closing(load_any_video(video_file_or_pattern)) as video:
        brightness = np.empty(video.frame_count, np.double)
        for k, frame in enumerate(display_progress(video)):
            brightness[k] = frame.mean()
        # restrict the result to the number of actual frames read
        if k < video.frame_count:
            brightness = brightness[:k + 1]

    # write brightness data
    if output_hdf5_file:
        with h5py.File(output_hdf5_file, "w") as fd:
            fd.create_dataset("brightness", data=brightness)
            
    return brightness



def get_dawn_from_brightness(brightness, output_file=None,
                             averaging_window=100, smoothing_sigma=25,
                             margin=500, debug_image=None):
    """ determines the frame where dawn sets.
    output_file: is a file into which the frame number of dawn is written
    averaging_window: denotes the number of frames which are averaged together
        in order to reduce the total amount of data
    smoothing_sigma: standard deviation used for Gaussian smoothing of the 
        resulting data
    margin: regions of data at the beginning and the end, which will not be used
        in the further analysis
    debug_image: filename to which an image is written, which can then be used
        to check the result of the algorithm
    """
    # average over window to reduce amount of data
    data_len = len(brightness) // averaging_window
    data_raw = np.empty(data_len, np.double)
    for i in xrange(data_len):
        ia = i*averaging_window
        data_raw[i] = np.mean(brightness[ia: ia + averaging_window])
    
    # filter the data
    data = ndimage.filters.gaussian_filter1d(data_raw, smoothing_sigma,
                                             mode='nearest')
    
    # determine the maximal change in brightness
    data_roi = data[margin : -margin]
    pos_max = np.argmax(np.gradient(data_roi))
    frame_dawn = (pos_max + margin) * averaging_window
    
    if output_file:
        with open(output_file, 'w') as fp:
            fp.write(str(frame_dawn))

    if debug_image:
        import matplotlib.pyplot as plt
        
        num_points = min(data_len, 512)
        ks = np.linspace(0, data_len, num_points, endpoint=False).astype(int)
        ts = ks*averaging_window
        
        plt.figure()
        plt.axvspan(0, margin*averaging_window, color='0.5')
        tmax = data_len*averaging_window
        plt.axvspan(tmax - margin*averaging_window, tmax, color='0.5')
        
        plt.plot(ts, data_raw[ks], label='Raw intensity')
        plt.plot(ts, data[ks], label='Smoothed intensity', lw=2)
        plt.axvline(frame_dawn, color='r', label='Detected dawn')
        
        plt.xlabel('Frame number')
        plt.ylabel('Average brightness')
        plt.xlim(0, tmax)
        plt.legend(loc='best')
        plt.savefig(debug_image)
        plt.close()
    
    return frame_dawn



def main():
    """ main routine of the program """
    # determine the video file
    try:
        video_file = sys.argv[1]
    except IndexError:
        raise ValueError('The input video has to be specified')
    print('Analyze video file `%s`' % video_file)
    
    # determine the output file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(video_file)[0] + '_dawn.txt'
    print('Write the result to `%s`' % output_file)

    # determine the brightness file
    if len(sys.argv) > 3:
        output_hdf5_file = sys.argv[3]
    else:
        output_hdf5_file = os.path.splitext(video_file)[0] + '_brightness.hdf5'
    print('Write the brightness data to `%s`' % output_hdf5_file)
    
    # calculate the brightness
    brightness = determine_average_frame_brightness(video_file, output_hdf5_file)
    
    # determine the frame where the light is switched on
    frame_dawn = get_dawn_from_brightness(brightness, output_file)
    
    print('Lights are switch on in frame %d' % frame_dawn)
    


if __name__ == '__main__':
    main()
