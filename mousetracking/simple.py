'''
Created on Aug 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains convenience functions for scanning mouse videos.
'''

from __future__ import division

import os
import warnings

from .algorithm.parameters import PARAMETERS_DEFAULT, set_base_folder
from .algorithm.analyzer import Analyzer
from .algorithm.pass1 import FirstPass
from .algorithm.pass2 import SecondPass



def scan_video(name, video=None, parameters=None, passes=2, **kwargs):
    """ scans a single video """
    # initialize parameters dictionary
    if parameters is None:
        parameters = {}
    
    # check whether the video should be cropped
    crop_video = kwargs.pop('crop_video', True)
    
    # set extra parameters that were given
    if 'frames' in kwargs: # don't overwrite frames if given in parameters
        parameters['video/frames'] = kwargs.pop('frames', None)
    if 'crop' in kwargs: # don't overwrite cropping_rect if given in parameters
        parameters['video/cropping_rect'] = kwargs.pop('crop', None)
    if 'scale_length' in kwargs: # don't overwrite factor_length if given in parameters
        parameters['factor_length'] = kwargs.pop('scale_length', 1)
    if 'debug_output' in kwargs: # don't overwrite debug_output if given in parameters
        parameters['debug/output'] = kwargs.pop('debug_output', None)
    if kwargs:
        warnings.warn('There are unused kwargs: %s' % ', '.join(kwargs))
    
    # do first pass
    job = FirstPass(name, parameters=parameters)
    job.load_video(video, crop_video=crop_video)
    job.process_video()
    
    # do second pass
    if passes > 1:
        job = SecondPass.from_first_pass(job)
        job.process_data()
        job.produce_video()
    


def scan_video_in_folder(folder, name, parameters=None, **kwargs):
    """ scans a single video from a folder """
    
    # create parameter dictionary
    params = dict(PARAMETERS_DEFAULT).copy()
    params.update(parameters)
    
    # set the folder in the respective parameters
    params['base_folder'] = folder
        
    # scan the video
    return scan_video(name, parameters=params, **kwargs)



def load_results(name, parameters=None, cls=Analyzer, **kwargs):
    """ loads the results of a previously scanned video based on the
    provided name and the folder given in parameters """
    # set up the result structure
    return cls(name, parameters=parameters, read_data=True, **kwargs)



def load_result_file(result_file, parameters=None, **kwargs):
    """ loads the results of a simulation based on the result file """
    # read folder and name from result_file
    result_file = os.path.abspath(result_file)
    folder, filename = os.path.split(result_file)
    name = os.path.splitext(filename)[0]
    name = '_'.join(name.split('_')[:-1])

    # set parameters and load results    
    if parameters is None:
        parameters = {}
    parameters = set_base_folder(parameters, folder, include_default=True)
    return load_results(name, parameters, **kwargs)
    