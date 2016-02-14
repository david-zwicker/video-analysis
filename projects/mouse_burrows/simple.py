'''
Created on Aug 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains convenience functions for scanning mouse videos.
'''

from __future__ import division

import os
import warnings

import yaml

from .algorithm import FirstPass, SecondPass, ThirdPass, FourthPass
from .algorithm.parameters import PARAMETERS_DEFAULT
from .algorithm.analyzer import Analyzer



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
    if 'debug_output' in kwargs: # don't overwrite debug_output if given in parameters
        parameters['debug/output'] = kwargs.pop('debug_output', None)
    scale_length = kwargs.pop('scale_length', 1) 
    if kwargs:
        warnings.warn('There are unused kwargs: %s' % ', '.join(kwargs))
    
    # do first pass
    job = FirstPass(name, parameters=parameters)
    job.scale_parameters(factor_length=scale_length)
    job.load_video(video, crop_video=crop_video)
    job.process()
    
    # do second pass
    if passes > 1:
        job = SecondPass.from_first_pass(job)
        job.process()
    
    # do third pass
    if passes > 2:
        job = ThirdPass.from_second_pass(job)
        job.process()
    
    # do fourth pass
    if passes > 3:
        job = FourthPass.from_third_pass(job)
        job.process()
    


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



def load_result_file(result_file, parameters=None, do_logging=None, **kwargs):
    """ loads the results of a simulation based on the result file """
    if not result_file.endswith('_results.yaml'):
        raise ValueError('Invalid result filename.')
    
    # read folder and name from result_file
    result_file = os.path.abspath(result_file)
    folder, filename = os.path.split(result_file)
    name = filename[:-len('_results.yaml')]
    
    # read the paths from the yaml file
    with open(result_file, 'r') as infile:
        data = yaml.load(infile)
    result_folder = data['parameters']['output']['folder']   

    # infer base folder
    if result_folder.endswith('.') and not result_folder.endswith('..'):
        result_folder = result_folder[:-1]
    if result_folder.endswith(os.sep):
        result_folder = result_folder[:-1]
    if result_folder.startswith('.' + os.sep):
        result_folder = result_folder[2:]
    if not folder.endswith(result_folder):
        last_folder = os.path.split(folder)[1] 
        raise ValueError('Result file does not reside in the right folder. '
                         'File is in `%s`, but is expected in `%s`.' %
                         (last_folder, result_folder))
    if result_folder:
        base_folder = folder[:-len(result_folder)]
    else:
        base_folder = folder

    # adjust parameters
    if parameters is None:
        parameters = {}
    parameters['base_folder'] = base_folder
    parameters['output/folder'] = result_folder
    if do_logging is not None:
        parameters['logging/enabled'] = do_logging

    # load results
    return load_results(name, parameters, **kwargs)
    