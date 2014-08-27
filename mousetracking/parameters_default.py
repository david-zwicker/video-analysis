'''
Created on Aug 27, 2014

@author: zwicker

Provides a dictionary with default parameters for the mouse tracking
'''

import numpy as np

PARAMETERS_DEFAULT = {
    # filename pattern used to look for videos
    'video/filename_pattern': 'raw_video/*',
    # number of initial frames to not analyze
    'video/ignore_initial_frames': 0,
    # radius of the blur filter [in pixel]
    'video/blur_radius': 3,
    
    # where to write the log files to
    'logging/folder': None,
    
    # locations and properties of output
    'output/result_folder': './results/',
    'output/video/folder': './debug/',
    'output/video/extension': '.mov',
    'output/video/codec': 'libx264',
    'output/video/bitrate': '2000k',
    
    # thresholds for cage dimension [in pixel]
    'cage/width_min': 650,
    'cage/width_max': 800,
    'cage/height_min': 400,
    'cage/height_max': 500,
                               
    # how often are the color estimates adapted [in frames]
    'colors/adaptation_interval': 1000,
                               
    # determines the rate with which the background is adapted [in 1/frames]
    'background/adaptation_rate': 0.01,
    'explored_area/adaptation_rate': 1e-4,
    
    # spacing of the points in the ground profile [in pixel]
    'ground/point_spacing': 20,
    # adapt the ground profile only every number of frames [in frames]
    'ground/adaptation_interval': 100,
    # width of the ridge [in pixel]
    'ground/width': 5,
    
    # relative weight of distance vs. size of objects [dimensionless]
    'objects/matching_weigth': 0.5,
    # size of the window used for motion detection [in frames]
    'objects/matching_moving_window': 20,
    # threshold above which an objects is said to be moving [in pixels/frame]
    'objects/matching_moving_threshold': 10,
        
    # `mouse.intensity_threshold` determines how much brighter than the
    # background (usually the sky) has the mouse to be. This value is
    # measured in terms of standard deviations of the sky color
    'mouse/intensity_threshold': 1,
    # radius of the mouse model [in pixel]
    'mouse/model_radius': 25,
    # minimal area of a feature to be considered in tracking [in pixel^2]
    'mouse/min_area': 100,
    # maximal speed of the mouse [in pixel per frame]
    'mouse/max_speed': 30, 
    # maximal area change allowed between consecutive frames [dimensionless]
    'mouse/max_rel_area_change': 0.5,

    # how often are the burrow shapes adapted [in frames]
    'burrows/adaptation_interval': 100,
    # what is a typical radius of a burrow [in pixel]
    'burrows/radius': 10,
    # minimal area a burrow cross section has to have
    'burrows/min_area': 1000,
    'burrows/centerline_angle': np.pi/6,
    'burrows/centerline_segment_length': 25,
    # extra number of pixel around burrow outline used for fitting [in pixel]
    'burrows/fitting_margin': 20,
    # determines how much the burrow outline might be simplified. The quantity 
    # determines by what fraction the total outline length is allowed to change 
    'burrows/outline_simplification_threshold': 0.01,#0.005,
}