'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import os

from .algorithm.data_handler import PARAMETERS_DEFAULT
from .algorithm.pass1 import FirstPass
from .algorithm.pass2 import SecondPass



def scan_video(video, name, parameters=None, **kwargs):
    """ scans a single video """
    if parameters is None:
        parameters = {}
    
    # set extra parameters that were given
    if kwargs.get('frames', None) is not None:
        parameters['video/frames'] = kwargs.pop('frames')
    if kwargs.get('crop', None) is not None:
        parameters['video/cropping_rect'] = kwargs.pop('crop')

    # do first pass
    job = FirstPass(name, parameters=parameters,
                    debug_output=kwargs.get('debug_output', None))
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
    
    return scan_video(None, name, parameters, **kwargs)
