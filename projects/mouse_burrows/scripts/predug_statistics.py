'''
Created on Apr 6, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import argparse
import cPickle as pickle
import logging
import os.path

import yaml
import numpy as np 

from ..simple import load_result_file



def get_predug_statistics(result_files, ret_shapes=False, parameters=None,
                          pass_id=3):
    """ gather statistics about the predug in all result files.
    `ret_shapes` determines whether the list of shapes is returned
    `parameters` are a dictionary of parameters for reading the results
    `pass_id` determines the id of the processing pass for the burrowing data 
    """
    # get all the profiles
    predugs = []
    no_predug_count = 0
    for filename in result_files:
        # load the results and find the predug
        try:
            result = load_result_file(filename, parameters)
        except KeyError:
            logging.warn('Data of `%s` could not be read', filename)
        else:
            predug = result.get_burrow_predug(pass_id)
        
        if predug:
            predugs.append(predug)
        else:
            no_predug_count += 1
        
    # gather statistics about predugs
    statistics = {'predug_count': len(predugs),
                  'no_predug_count': no_predug_count,
                  'area_mean': np.mean(predug.area for predug in predugs),
                  'area_std': np.std(predug.area for predug in predugs)}
    
    if ret_shapes:
        return statistics, predugs
    else:
        return statistics
    

    
def main(folder):
    """ main routine of the program """
    # parse the command line arguments
    description = 'Gather statistics about the predugs in the experiments.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output', dest='output', type=str,
                        help='file to which the output statistics are written')
#     parser.add_argument('-p', '--plot', dest='plot',
#                         action='store_true', help='plots the result')
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='files to analyze')

    args = parser.parse_args()
    
    # get files to analyze
    filenames = args.files
    
    # analyze all the files
    parameters = {'logging/enabled': False}
    statistics = get_predug_statistics(filenames, parameters=parameters)
    
#     if args.plot:
#         # show the result if requested
#         plt.plot(ground[:, 0], ground[:, 1], 'b-', lw=3)
#         
#         # determine traces that are far from the average
#         dist = []
#         for k, trace in enumerate(traces):
#             ys = np.interp(ground[:, 0], trace[:, 0], trace[:, 1])
#             dist.append(np.linalg.norm(ground[:, 1] - ys))
#             if dist[-1] < 0.3:
#                 plt.plot(trace[:, 0], trace[:, 1], 'k-', alpha=0.2)
#                 if dist[-1] > 0.2:
#                     print 'Possibly bad ground profile', filenames[k]
#             else:
#                 plt.plot(trace[:, 0], trace[:, 1], 'r-', alpha=0.2)
#                 print 'Bad ground profile', filenames[k]
#         
#         traces = [t for k, t in enumerate(traces)
#                   if dist[k] < 0.3]
#         ground = curves.average_normalized_functions(traces)
#         
#         plt.plot(ground[:, 0], ground[:, 1], 'g-', lw=3)
#         plt.gca().invert_yaxis()
#         plt.show()
        
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
