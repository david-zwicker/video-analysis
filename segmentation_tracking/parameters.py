'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


# general tracking parameters 
parameters_tracking_default = {
    'gradient/blur_radius': 10, 
    'gradient/threshold': 0.05, 
                
    'detection/mask_size': 30,
    'detection/min_area': 50000,
    'detection/watershed_threshold': 0.2,
    
    'outline/blur_radius_initial': 20,
    'outline/max_iterations': 300,  
    'outline/bending_stiffness': 1e4, #< bending stiffness for the tail outline
    'outline/adaptation_rate': 1e0, #< rate with which the active snake adapts
    
    'measurement/spline_smoothing': 20, #< smoothing factor for measurement lines
    'measurement/line_offset': 0.5, #< determines position of the measurement line
    'measurement/line_scan_width': 30, #< width of the line scan along the measurement lines
}


# special tracking parameters for individual videos
parameters_tracking_special = {
    '20140831_BW_E14-15_tl_edf': {'gradient/threshold': 0.1,
                                  'outline/bending_stiffness': 1e5}
}