'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


# general tracking parameters 
parameters_tracking_default = {
    'input/frames': None,
    
    'output/zoom_factor': 2,
    'output/background': 'original', #< ('original', 'gradient', 'thresholded')
                               
    'gradient/blur_radius': 10, 
    'gradient/threshold': 0.05, 
                
#     'detection/statistics_window': 10,
    'detection/statistics_window': 50,
    'detection/statistics_threshold': 5,
    'detection/border_distance': 20,
    'detection/mask_size': 30,
    'detection/area_min': 50000,
    'detection/area_max': 500000,
    'detection/watershed_threshold': 0.2,
    
    'outline/blur_radius_initial': 20,
    'outline/max_iterations': 300,
    'outline/line_tension': 0,  
    'outline/bending_stiffness': 1e4, #< bending stiffness for the tail outline
    'outline/adaptation_rate': 1e0, #< rate with which the active snake adapts
    
    'measurement/spline_smoothing': 20, #< smoothing factor for measurement lines
    'measurement/line_offset': 0.5, #< determines position of the measurement line
    'measurement/line_scan_width': 30, #< width of the line scan along the measurement lines
}


# special tracking parameters for individual videos
parameters_tracking_special = {
    # video with two touching tails
    '20140804_bw_tl_edf': {
        'detection/watershed_threshold': 0.5 },
    # video with loads to particles in background
    '20140808_nub_e15-16_tl_edf': {
        'detection/statistics_window': 50,}, 
    # video with three tails
    '20140831_BW_E14-15_tl_edf': { 
        'detection/watershed_threshold': 0.4,},
}