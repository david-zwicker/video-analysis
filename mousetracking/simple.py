'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import os

from .parameters_default import PARAMETERS_DEFAULT
from .algorithm.data_handler import DataHandler
from .algorithm.pass1 import FirstPass
from .algorithm.pass2 import SecondPass


def scan_video(name, video=None, parameters=None, **kwargs):
    """ scans a single video """
    # initialize parameters dictionary
    params = PARAMETERS_DEFAULT.copy()
    if parameters is not None:
        params.update(parameters)
    
    # set extra parameters that were given
    if kwargs.get('frames', None) is not None:
        params['video/frames'] = kwargs.pop('frames')
    if kwargs.get('crop', None) is not None:
        params['video/cropping_rect'] = kwargs.pop('crop')

    # do first pass
    job = FirstPass(name, parameters=params,
                    debug_output=kwargs.get('debug_output', None))
    job.load_video(video)
    job.process_video()
    
    # do second pass
    job = SecondPass.from_first_pass(job)
    
    

def scan_video_in_folder(folder, name, parameters=None, **kwargs):
    """ scans a single video from a folder """
    
    if parameters is None:
        parameters = {}
    
    # set the folders
    pattern = parameters.get('video/filename_pattern', PARAMETERS_DEFAULT['video/filename_pattern'])
    parameters['video/filename_pattern'] = os.path.join(folder, pattern)
    parameters['output/result_folder'] = os.path.join(folder, 'results') 
    parameters['output/video/folder'] = os.path.join(folder, 'debug') 
    
    return scan_video(name, parameters=parameters, **kwargs)



def load_results(name, parameters=None):
    """ loads the results of a previously scanned video """
    # initialize parameters dictionary
    params = PARAMETERS_DEFAULT.copy()
    if parameters is not None:
        params.update(parameters)

    # set up the result structure
    results = DataHandler(name, parameters=params)
    results.read_data()
    return results

