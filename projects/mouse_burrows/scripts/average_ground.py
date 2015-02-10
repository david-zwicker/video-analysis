#!/usr/bin/env python2
'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import logging
import os.path
import cPickle as pickle

import yaml
import numpy as np
import matplotlib.pyplot as plt

from ..simple import load_result_file
from video.analysis import curves



def average_ground_shape(result_files, time_point=60*60, parameters=None,
                         ret_traces=False):
    """ determines an average ground shape from a list of results """
    
    # get all the profiles
    profiles = []
    for filename in result_files:
        # load the results
        try:
            result = load_result_file(filename, parameters)
            ground_profile = result.data['pass2/ground_profile']
        except KeyError:
            logging.warn('Data of `%s` could not be read', filename)
        
        # retrieve profile at the right time point
        if result.use_units:
            frame_id = time_point/(result.time_scale/result.units.second)
        else:
            frame_id = time_point/result.time_scale
        profile = ground_profile.get_ground_profile(frame_id)
        
        # scale profile such that width=1
        points = profile.line
        scale = points[-1][0] - points[0][0]
        points[:, 0] = (points[:, 0] - points[0, 0])/scale
        points[:, 1] = (points[:, 1] - points[:, 1].mean())/scale
        profiles.append(points)
        
    # average the profiles
    profile_avg = curves.average_normalized_functions(profiles) 
    
    if ret_traces:
        return profile_avg, profiles
    else:
        return profile_avg
    
    
    
def main(folder):
    """ main routine of the program """
    # parse the command line arguments
    description = 'Average ground profiles from several analyzed videos'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='file to which the output statistics are written')
    parser.add_argument('-p', '--plot', dest='plot',
                        action='store_true', help='plots the result')
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='files to analyze')

    args = parser.parse_args()
    
    # get files to analyze
    filenames = args.files
    
    # analyze all the files
    parameters = {'logging/enabled': False}
    ground, traces = average_ground_shape(filenames, ret_traces=True,
                                          parameters=parameters)
    
    if args.plot:
        # show the result if requested
        plt.plot(ground[:, 0], ground[:, 1], 'b-', lw=3)
        
        # determine traces that are far from the average
        dist = []
        for k, trace in enumerate(traces):
            ys = np.interp(ground[:, 0], trace[:, 0], trace[:, 1])
            dist.append(np.linalg.norm(ground[:, 1] - ys))
            if dist[-1] < 0.3:
                plt.plot(trace[:, 0], trace[:, 1], 'k-', alpha=0.2)
                if dist[-1] > 0.2:
                    print 'Possibly bad ground profile', filenames[k]
            else:
                plt.plot(trace[:, 0], trace[:, 1], 'r-', alpha=0.2)
                print 'Bad ground profile', filenames[k]
        
        traces = [t for k, t in enumerate(traces)
                  if dist[k] < 0.3]
        ground = curves.average_normalized_functions(traces)
        
        plt.plot(ground[:, 0], ground[:, 1], 'g-', lw=3)
        plt.gca().invert_yaxis()
        plt.show()
        
    if args.output:
        # write to file
        ext = os.path.splitext(args.output)[1]
        if ext == '.pkl':
            pickle.dump(ground, open(args.output, 'w'))
        elif ext == '.yaml':
            yaml.dump(ground.tolist(), open(args.output, 'w'))
        else:
            raise ValueError('Unknown output format `%s`' % ext[1:])

    else:
        # write to stdout
        print ground.tolist()
        
        
    
if __name__ == '__main__':
    main()

