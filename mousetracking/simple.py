'''
Created on Aug 19, 2014

@author: zwicker

This module contains convenience functions for scanning mouse videos.
'''

from __future__ import division

import os

from .algorithm.parameters_default import PARAMETERS_DEFAULT
from .algorithm.data_handler import DataHandler
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
    if kwargs.get('frames', None) is not None:
        parameters['video/frames'] = kwargs.pop('frames')
    if kwargs.get('crop', None) is not None:
        parameters['video/cropping_rect'] = kwargs.pop('crop')
        
    # do first pass
    job = FirstPass(name, parameters=parameters,
                    debug_output=kwargs.get('debug_output', None))
    job.load_video(video, crop_video=crop_video)
    job.process_video()
    
    # do second pass
    if passes > 1:
        job = SecondPass.from_first_pass(job)
        job.process_data()
        job.produce_video()
    

def scan_video_in_folder(folder, name, parameters=None, **kwargs):
    """ scans a single video from a folder """
    
    if parameters is None:
        parameters = {}
    
    # set the folder in the respective parameters
    for key in ('video/filename_pattern', 'output/result_folder',
                'output/video/folder_debug', 'logging/folder'):
        value = parameters.get(key, PARAMETERS_DEFAULT[key])
        if value:
            parameters[key] = os.path.join(folder, value)
        
    # scan the video
    return scan_video(name, parameters=parameters, **kwargs)



def load_results(name, parameters=None, cls=DataHandler, **kwargs):
    """ loads the results of a previously scanned video """
    # set up the result structure
    results = cls(name, parameters=parameters, **kwargs)
    results.read_data()
    return results

