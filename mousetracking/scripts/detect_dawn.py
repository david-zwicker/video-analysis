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

# add the root of the video-analysis project to the path
this_path = os.path.dirname(__file__)
video_analysis_path = os.path.join(this_path, '..', '..')
sys.path.append(video_analysis_path)
from video.io import load_any_video



def determine_average_frame_brightness(video_file_or_pattern, output_hdf5_file=None):
    """ iterates a video and determines its intensity, which will be stored
    in a hdf5 file, if the respective file name is given"""
    # read video data
    with closing(load_any_video(video_file_or_pattern)) as video:
        brightness = np.empty(video.frame_count, np.double)
        for k, frame in enumerate(video):
            brightness[k] = frame.mean()
        # restrict the result to the number of actual frames read
        if k < video.frame_count:
            brightness = brightness[:k + 1]

    # write brightness data
    if output_hdf5_file:
        with h5py.File(output_hdf5_file, "w") as fd:
            fd.create_dataset("brightness", data=brightness)
            
    return brightness



def main():
    # determine the video file
    try:
        video_file = sys.argv[1]
    except IndexError:
        raise ValueError('The input video has to be specified')
    print('Analyze video file `%s`' % video_file)
    
    # determine the output file
    if len(sys.argv) > 2:
        output_hdf5_file = sys.argv[2]
    else:
        output_hdf5_file = os.path.splitext(video_file)[0] + '.hdf5'
    print('Write the result to `%s`' % output_hdf5_file)
    
    determine_average_frame_brightness(video_file, output_hdf5_file)
    


if __name__ == '__main__':
    main()
