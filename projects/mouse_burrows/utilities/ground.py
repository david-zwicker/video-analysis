'''
Created on Oct 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

from projects.mouse_burrows import load_result_file
from video.analysis import curves


logger = logging.getLogger('mousetracking.utilities')


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
        profiles.append(points)
        
    # average the profiles
    profile_avg = curves.average_normalized_functions(profiles) 
    
    if ret_traces:
        return profile_avg, profiles
    else:
        return profile_avg
    

