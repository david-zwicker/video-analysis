'''
Created on Oct 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np

from ..simple import load_result_file


logger = logging.getLogger('mousetracking.utilities')


import matplotlib.pyplot as plt

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
            logger.warn('Data of `%s` could not be read', filename)
        
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
        plt.plot(points[:, 0], points[:, 1], 'k-')
        profiles.append(points)
        
    # average the profiles
    len_max = max(len(ps) for ps in profiles)
    xs = np.linspace(0, 1, len_max)
    ys = np.mean([np.interp(xs, ps[:, 0], ps[:, 1])
                  for ps in profiles], axis=0)
    profile_avg = np.c_[xs, ys] 
    
    if ret_traces:
        return profile_avg, profiles
    else:
        return profile_avg
    

