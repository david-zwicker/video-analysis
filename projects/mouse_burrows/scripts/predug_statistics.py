#!/usr/bin/env python2
'''
Created on Apr 6, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import cPickle as pickle
import logging
import multiprocessing as mp
import os.path
import sys

import yaml
import numpy as np 

# add the root of the video-analysis project to the path
# append base path to sys.path
script_path = os.path.split(os.path.realpath(__file__))[0]
video_analysis_path = os.path.join(script_path, '..', '..', '..')
sys.path.append(video_analysis_path)

from projects.mouse_burrows.simple import load_result_file
from video import debug



def _get_predug_from_file(args):
    """ analyze a single experiment and return the predug """
    # unpack the arguments
    filename, parameters, verbose, pass_id = args
    
    if verbose:
        print('Analyze `%s`' % filename)
    
    # load the results and find the predug
    try:
        result = load_result_file(filename, parameters)
        predug = result.get_burrow_predug(pass_id)
    except KeyError:
        logging.warn('Data of `%s` could not be read', filename)
        predug = None
    except Exception as e:
        logging.warn('Exception occurred: %s' % e)
        predug = 'error'
    
    return predug



def get_predug_statistics(result_files, ret_shapes=False, parameters=None,
                          pass_id=3, verbose=False, multiprocessing=False):
    """ gather statistics about the predug in all result files.
    `ret_shapes` determines whether the list of shapes is returned
    `parameters` are a dictionary of parameters for reading the results
    `pass_id` determines the id of the processing pass for the burrowing data 
    """
    # analyze all the files
    jobs = [(filename, parameters, verbose, pass_id)
            for filename in result_files[:32]]
    
    if multiprocessing:
        results = mp.Pool().map(_get_predug_from_file, jobs)
    else:
        results = map(_get_predug_from_file, jobs) 
        
    # sort the resulting data into two categories
    predugs = []
    no_predug_count, error_count = 0, 0
    for res in results:
        if res is None:
            no_predug_count += 1
        elif res == 'error':
            error_count += 1
        else:
            predugs.append(res)
    
    # gather statistics about predugs
    areas = [predug.area for predug in predugs]
    statistics = {'predug_count': len(predugs),
                  'no_predug_count': no_predug_count,
                  'error_count': error_count,
                  'area_mean': np.mean(areas),
                  'area_std': np.std(areas)}
    
    if ret_shapes:
        return statistics, predugs
    else:
        return statistics
    


def main():
    """ main routine of the program """
    # parse the command line arguments
    description = 'Gather statistics about the predugs in the experiments.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='file to which the output statistics are written')
    parser.add_argument('-p', '--plot', dest='plot',
                        action='store_true', help='plots the result')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', help='outputs more information')
    parser.add_argument('-m', '--multiprocessing', dest='multiprocessing',
                        action='store_true', help='uses multiple cores')
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='files to analyze')

    args = parser.parse_args()
    
    # get files to analyze
    filenames = args.files
    
    # analyze all the files
    parameters = {'logging/enabled': False}
    statistics, predugs = get_predug_statistics(filenames, ret_shapes=True,
                                                parameters=parameters,
                                                verbose=args.verbose,
                                                multiprocessing=args.multiprocessing)

    if args.plot:
        shapes = [poly.contour_ring for poly in predugs]
        debug.show_shape(*shapes)
        
    if args.output:
        # write to file
        ext = os.path.splitext(args.output)[1]
        if ext == '.pkl':
            pickle.dump(statistics, open(args.output, 'w'))
        elif ext == '.yaml':
            yaml.dump(statistics, open(args.output, 'w'))
        else:
            raise ValueError('Unknown output format `%s`' % ext[1:])

    else:
        # write to stdout
        print statistics
        
        
    
if __name__ == '__main__':
    main()
