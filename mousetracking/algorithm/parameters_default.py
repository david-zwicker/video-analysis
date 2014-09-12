'''
Created on Aug 27, 2014

@author: zwicker

Provides a dictionary with default parameters for the mouse tracking.
This can also be seen as some kind of documentation of the available
parameters.
'''

from __future__ import division

from collections import namedtuple

class UNIT(object):
    FACTOR = 1
    FRACTION = 2
    LENGTH_PIXEL = 11
    AREA_PIXEL = 12
    TIME_FRAMES = 20
    RATE_FRAMES = 21
    SPEED_PIXEL_FRAME = 30
    

Parameter = namedtuple('Parameter',
                       ['key', 'default_value', 'unit', 'description'])


PARAMETERS = [
    # Video input
    Parameter('video/filename_pattern', 'raw_video/*.MTS', None,
              'Filename pattern used to look for videos'),
    Parameter('video/initial_adaptation_frames', 100, UNIT.TIME_FRAMES,
              'Number of initial frames to skip during analysis'),
    Parameter('video/blur_radius', 3, UNIT.LENGTH_PIXEL,
              'Radius of the blur filter to remove noise'),
    Parameter('video/frames', None, None,
              'Frames of the video which are analyzed [start and end index should be given]'),
    Parameter('video/cropping_rect', None, None,
              "Rectangle to which the video is cropped. This can be either four "
              "numbers [left, top, width, height] or some string like "
              "'upper left', 'lower right', etc."),
          
    # Logging    
    Parameter('logging/folder', './logging/', None,
              'Folder to which the log file is written'),
    Parameter('logging/level_stderr', 'INFO', None,
              'Level of messages to log to stderr [standard python logging levels]'),
    Parameter('logging/level_file', 'DEBUG', None,
              'Level of messages to log to file if folder is set '
              '[standard python logging levels]'),
            
    # Output
    Parameter('output/result_folder', './results/', None,
              'Folder to which the YAML and HDF5 result files are written'),
    Parameter('output/video/folder_debug', './debug/', None,
              'Folder to which debug videos are written'),
    Parameter('output/video/extension', '.mov', None,
              'File extension used for debug videos'),
    Parameter('output/video/codec', 'libx264', None,
              'ffmpeg video codec used for debug videos'),
    Parameter('output/video/bitrate', '2000k', None,
              'Bitrate used for debug videos'),
    
    # Cage
    Parameter('cage/width_min', 650, UNIT.LENGTH_PIXEL,
              'Minimal width of the cage. This is only used to make a '
              'plausibility test of the results.'),
    Parameter('cage/width_max', 800, UNIT.LENGTH_PIXEL,
              'Maximal width of the cage. This is only used to make a '
              'plausibility test of the results.'),
    Parameter('cage/height_min', 400, UNIT.LENGTH_PIXEL,
              'Minimal height of the cage. This is only used to make a '
              'plausibility test of the results.'),
    Parameter('cage/height_max', 500, UNIT.LENGTH_PIXEL,
              'Maximal height of the cage. This is only used to make a '
              'plausibility test of the results.'),
    Parameter('cage/frame_width', 25, UNIT.LENGTH_PIXEL,
              'Width of the cage frame'),
    Parameter('cage/linescan_width', 30, UNIT.LENGTH_PIXEL,
              'Width of the linescan use to detect the cage frame.'),
                
    # Colors               
    Parameter('colors/adaptation_interval', 1000, UNIT.TIME_FRAMES,
              'How often are the color estimates adapted'),

    # Background and explored area                             
    Parameter('background/adaptation_rate', 1e-2, UNIT.RATE_FRAMES,
              'Rate at which the background is adapted'),
    Parameter('explored_area/adaptation_rate_outside', 1e-3, UNIT.RATE_FRAMES,
              'Rate at which the explored area is adapted outside of burrows'),
    Parameter('explored_area/adaptation_rate_burrows', 1e-5, UNIT.RATE_FRAMES,
              'Rate at which the explored area is adapted inside burrows'),
    
    # Ground
    Parameter('ground/point_spacing', 20, UNIT.LENGTH_PIXEL,
              'Spacing of the support points describing the ground profile'),
    Parameter('ground/flat_top_fraction', 0.2, UNIT.FRACTION,
              'Fraction of total width where the top of the ground is flat'),
    Parameter('ground/adaptation_interval', 100, UNIT.TIME_FRAMES,
              'How often is the ground profile adapted'),
    Parameter('ground/width', 5, UNIT.LENGTH_PIXEL,
              'Width of the ground profile ridge'),
    Parameter('ground/snake_bending_energy', 5e5, UNIT.FACTOR,
              'Determines the stiffness of the snake model of the ground profile'),
    Parameter('ground/smoothing_sigma', 300, UNIT.TIME_FRAMES,
              'Standard deviation for Gaussian smoothing'),
    
    # Mouse 
    Parameter('mouse/intensity_threshold', 1, UNIT.FACTOR,
              'Determines how much brighter than the background (usually the '
              'sky) the mouse has to be. This value is measured in terms of '
              'standard deviations of the sky color'),
    Parameter('mouse/model_radius', 25, UNIT.LENGTH_PIXEL,
              'Radius of the mouse model'),
    Parameter('mouse/area_min', 100, UNIT.AREA_PIXEL,
              'Minimal area of a feature to be considered in tracking'),
    Parameter('mouse/area_mean', 700, UNIT.AREA_PIXEL,
              'Mean area of a mouse, which is used to score the mouse'),
    Parameter('mouse/speed_max', 30, UNIT.SPEED_PIXEL_FRAME,
              'Maximal speed of the mouse.'),
    Parameter('mouse/max_rel_area_change', 0.5, UNIT.FACTOR,
              'Maximal area change allowed between consecutive frames'),
    Parameter('tracking/weight', 0.5, UNIT.FACTOR,
              'Relative weight of distance vs. size of objects for matching them'),
    Parameter('tracking/moving_window', 20, UNIT.TIME_FRAMES,
              'Number of consecutive frames used for motion detection'),
    Parameter('tracking/moving_threshold', 15, UNIT.SPEED_PIXEL_FRAME,
              'Threshold speed above which an object is said to be moving'),
    Parameter('tracking/time_scale', 10, UNIT.TIME_FRAMES,
              'Time duration of not seeing the mouse after which we do not know where it is anymore'),
    Parameter('tracking/tolerated_overlap', 10, UNIT.TIME_FRAMES,
              'How much are two consecutive tracks allowed to overlap'),
    Parameter('tracking/initial_score_threshold', 1000, UNIT.FACTOR,
              'Initial threshold for building the tracking graph'),
    Parameter('tracking/end_node_interval', 1000, UNIT.TIME_FRAMES,
              'What time duration do we consider for start and end nodes'),
    Parameter('tracking/splitting_duration_min', 10, UNIT.TIME_FRAMES,
              'Track duration above which two overlapping tracks are split'),
        
    # Burrows
    Parameter('burrows/adaptation_interval', 100, UNIT.TIME_FRAMES,
              'How often are the burrow shapes adapted'),
    Parameter('burrows/cage_margin', 30, UNIT.LENGTH_PIXEL,
              'Margin of a potential burrow to the cage boundary'),
    Parameter('burrows/width', 20, UNIT.LENGTH_PIXEL,
              'What is a typical width of a burrow'),
    Parameter('burrows/width_min', 10, UNIT.LENGTH_PIXEL,
              'What is a minimal width of a burrow'),
    Parameter('burrows/area_min', 1000, UNIT.AREA_PIXEL,
              'Minimal area a burrow cross section has to have'),
    Parameter('burrows/ground_point_distance', 10, UNIT.LENGTH_PIXEL,
              'Maximal distance of ground profile to outline points that are '
              'considered exit points'),
    Parameter('burrows/centerline_segment_length', 25, UNIT.LENGTH_PIXEL,
              'Length of a segment of the center line of a burrow'),
    Parameter('burrows/curvature_radius_max', 50, UNIT.LENGTH_PIXEL,
              'Maximal radius of curvature the centerline is allowed to have'),
    Parameter('burrows/fitting_length_threshold', 75, UNIT.LENGTH_PIXEL,
              'Length above which burrows are refined by fitting'),
    Parameter('burrows/fitting_width_threshold', 40, UNIT.LENGTH_PIXEL,
              'Length below which burrows are refined by fitting'),
    Parameter('burrows/fitting_edge_width', 3, UNIT.LENGTH_PIXEL,
              'Width of the burrow edge used in the template for fitting'),
    Parameter('burrows/fitting_edge_R2min', -10, UNIT.FACTOR,
              'Minimal value of the Coefficient of Determination (R^2) above '
              'which the fit of a burrow edge is considered good enough and '
              'will be used'),
    Parameter('burrows/outline_simplification_threshold', 0.005, UNIT.FACTOR,
              'Determines how much the burrow outline might be simplified. '
              'The quantity determines by what fraction the total outline '
              'length is allowed to change'),
]

PARAMETERS_DEFAULT = {p.key: p.default_value for p in PARAMETERS}
