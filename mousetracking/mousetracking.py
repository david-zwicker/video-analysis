'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

from pass1 import FirstPass
from pass2 import SecondPass


def scan_video_in_folder(folder, prefix='', parameters=None, debug_output=None, **kwargs):
    """ scans a single video from a folder """
    
    if parameters is None:
        parameters = {}
    
    # set extra parameters that were given
    if kwargs.get('frames', None) is not None:
        parameters['video/frames'] = kwargs['frames']
    if kwargs.get('crop', None) is not None:
        parameters['video/cropping_rect'] = kwargs['crop']

    # do first pass
    job = FirstPass(folder, prefix, parameters, debug_output)
    job.process_video()
    
    # do second pass
    job = SecondPass.from_first_pass(job)