'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


# general tracking parameters 
parameters_tracking = {
    'default': {
        'input/frames': None,
        'input/zoom_factor': 1,
        
        'output/zoom_factor': 2,
        'output/background': 'original', #< ('original', 'gradient', 'thresholded')
                                   
        'gradient/blur_radius': 10, 
        'gradient/threshold': 0.05, 
                    
    #     'detection/statistics_window': 10,
        'detection/statistics_window': 50,
        'detection/statistics_threshold': 3,
        'detection/shape_max_speed': 50,
        'detection/border_distance': 50,
        'detection/mask_size': 30,
        'detection/area_min': 100000,
        'detection/area_max': 500000,
        'detection/boundary_length_max': 500,
        'detection/every_frame': True,
        
        'contour/typical_width': 150,
        'contour/blur_radius_initial': 20,
        'contour/blur_radius': 20,
        'contour/border_anchor_distance': 100,
        'contour/max_iterations': 1000,
        'contour/line_tension': 0,  
        'contour/bending_stiffness': 1e4, #< bending stiffness for the tail outline
        'contour/adaptation_rate': 1e0, #< rate with which the active snake adapts
        
        'measurement/spline_smoothing': 20, #< smoothing factor for measurement lines
        'measurement/line_offset': 0.5, #< determines position of the measurement line
        'measurement/line_scan_width': 30, #< width of the line scan along the measurement lines
    },
    'fast': {
        'input/frames': None,
        'input/zoom_factor': 1/5,
        
        'output/zoom_factor': 1,
        'output/background': 'original', #< ('original', 'gradient', 'thresholded')
                                   
        'gradient/blur_radius': 10//5, 
        'gradient/threshold': 0.05, 
                    
    #     'detection/statistics_window': 10,
        'detection/statistics_window': 50//5,
        'detection/statistics_threshold': 3,
        'detection/shape_max_speed': 50//5,
        'detection/border_distance': 50//5,
        'detection/mask_size': 30//5,
        'detection/area_min': 50000//(5*5),
        'detection/area_max': 500000//(5*5),
        
        'contour/blur_radius_initial': 50//5,
        'contour/blur_radius_initial': 20//5,
        'contour/max_iterations': 300,
        'contour/line_tension': 0,  
        'contour/bending_stiffness': 1e4//(5*5), #< bending stiffness for the tail outline
        'contour/adaptation_rate': 1e0//5, #< rate with which the active snake adapts
        
        'measurement/spline_smoothing': 20//5, #< smoothing factor for measurement lines
        'measurement/line_offset': 0.5, #< determines position of the measurement line
        'measurement/line_scan_width': 30//5, #< width of the line scan along the measurement lines
    },
}


# special tracking parameters for individual videos
parameters_tracking_special = {
    '20140729_NUB_E14_edf': {
        'detection/statistics_threshold': 30},
    # video with two touching tails
    '20140804_bw_tl_edf': {
        'detection/statistics_threshold': 20,
        'detection/statistics_threshold': 2},
    # little contrast at second tail
    '20140808_nub_e15-16_tl_edf':{
        'detection/every_frame': False},
    # video with three tails
    '20140831_BW_E14-15_tl_edf': { 
        'detection/statistics_window': 30,
        'detection/statistics_threshold': 3,
        'detection/every_frame': False,
    },
}