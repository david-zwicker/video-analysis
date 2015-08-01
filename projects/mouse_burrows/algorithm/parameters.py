'''
Created on Aug 27, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Provides a dictionary with default parameters for the mouse tracking.
This can also be seen as some kind of documentation of the available
parameters.
'''

from __future__ import division

from collections import namedtuple, defaultdict
import warnings

import numpy as np


# enum of different units that we use
class UNIT(object):
    FACTOR = 1
    FRACTION = 2
    FOLDER = 3
    SUBFOLDER = 4
    COLOR = 5
    BOOLEAN = 6
    INTEGER = 7
    STRING = 8
    LENGTH_PIXEL = 11
    LENGTH_CM = 12
    AREA_PIXEL = 15
    TIME_FRAMES = 20
    RATE_FRAMES = 21
    SPEED_PIXEL_FRAME = 30
    SPEED_CM_SEC = 31
    DEPRECATED = 100
    
    # create dictionary with parser functions
    parser = defaultdict(lambda: lambda val: val)

# initialize converters
UNIT.parser[UNIT.FACTOR] = float
UNIT.parser[UNIT.FRACTION] = float
UNIT.parser[UNIT.BOOLEAN] = bool
UNIT.parser[UNIT.INTEGER] = int
UNIT.parser[UNIT.LENGTH_PIXEL] = float
UNIT.parser[UNIT.LENGTH_CM] = float
UNIT.parser[UNIT.AREA_PIXEL] = float
UNIT.parser[UNIT.TIME_FRAMES] = float
UNIT.parser[UNIT.RATE_FRAMES] = float
UNIT.parser[UNIT.SPEED_PIXEL_FRAME] = float
UNIT.parser[UNIT.SPEED_CM_SEC] = float
    

# define a class that holds information about parameters 
Parameter = namedtuple('Parameter',
                       ['key', 'default_value', 'unit', 'description'])


# define all parameters that we support with associated information
PARAMETER_LIST = [
    # Basic parameters
    Parameter('base_folder', '.', UNIT.FOLDER,
              'Base folder in which all files are kept'),
    Parameter('factor_length', 1, UNIT.DEPRECATED, #UNIT.FACTOR,
              'A factor by which all length scales will be scaled.'
              'Deprecated since 2014-12-20. Instead, `scale_length` should be '
              'used, which will be processed when loading the parameters once'),
    Parameter('use_threads', True, UNIT.BOOLEAN,
              'Determines whether multithreading is used in analyzing the '
              'videos. Generally, multithreading should speed up the analysis, '
              'but this is not always the case, especially for small videos, '
              'where the thread overhead is large.'),
                  
    # Video input
    Parameter('video/filename_pattern', 'raw_video/*.MTS', UNIT.SUBFOLDER,
              'Filename pattern used to look for videos'),
    Parameter('video/initial_adaptation_frames', 100, UNIT.TIME_FRAMES,
              'Number of initial frames to skip during analysis'),
    Parameter('video/blur_method', 'gaussian', UNIT.STRING,
              'The method to be used for reducing noise in the video. The '
              'supported methods are `mean`, `gaussian`, `bilateral`, in '
              'increasing complexity, i.e. decreasing speed.'),
    Parameter('video/blur_radius', 3, UNIT.LENGTH_PIXEL,
              'Radius of the blur filter to remove noise'),
    Parameter('video/blur_sigma_color', 0, UNIT.COLOR,
              'Standard deviation in color space of the bilateral filter'),
    Parameter('video/frames', None, None,
              'Frames of the video which are analyzed [start and end index '
              'should be given]'),
    Parameter('video/frames_skip', 0, UNIT.TIME_FRAMES,
              'Number of frames that are skipped before starting the '
              'analysis. This value is only considered if `video/frames` '
              'is None.'),
    Parameter('video/cropping_rect', None, None,
              "Rectangle to which the video is cropped. This can be either four "
              "numbers [left, top, width, height] or some string like "
              "'upper left', 'lower right', etc."),
    Parameter('video/remove_water_bottle', True, UNIT.BOOLEAN,
              'Flag that indicates whether the water bottle should be removed '
              'from the video'),
    Parameter('video/water_bottle_template', 'water_bottle.png', None,
              'Name of the template for removing the water bottle from the '
              'background estimate.'),
    Parameter('video/water_bottle_region', [0.8, 1., 0., 0.3], None,
              'Defines the region [x_min, x_max, y_min, y_max] in which the '
              'upper left corner of the water bottle rectangle lies. The '
              'coordinates are given relative to the cage width and height. '
              'This is used to restrict the template matching to a sensible '
              'region.'),
          
    # Logging
    Parameter('logging/enabled',  True, UNIT.BOOLEAN,
              'Flag indicating whether logging is enabled'),
    Parameter('logging/folder', 'logging/', UNIT.SUBFOLDER,
              'Folder to which the log file is written'),
    Parameter('logging/level_stderr', 'INFO', None,
              'Level of messages to log to stderr [standard python logging levels]'),
    Parameter('logging/level_file', 'INFO', None,
              'Level of messages to log to file if folder is set '
              '[standard python logging levels]'),
            
    # Debug
    Parameter('debug/output', [], None,
              "List of identifiers determining what debug output is produced. "
              "Supported identifiers include 'video', 'explored_area', "
              "'background', 'difference', 'cage_estimate', "
              "'ground_estimate', 'explored_area_mask'."),
    Parameter('debug/use_multiprocessing', True, UNIT.BOOLEAN,
              'Flag indicating whether multiprocessing should be used to read '
              'and display videos'),
    Parameter('debug/folder', 'debug/', UNIT.SUBFOLDER,
              'Folder to which debug videos are written'), 
    Parameter('debug/window_position', None, None,
              'Position (x, y) of the top-left corner of the debug window'),
            
    # Output
    Parameter('output/folder', 'results/', UNIT.SUBFOLDER,
              'Folder to which the YAML and HDF5 result files are written'),
    Parameter('output/video/folder', 'results/', UNIT.SUBFOLDER,
              'Folder to which the result video is written'),
    Parameter('output/output_period', 1, UNIT.TIME_FRAMES,
              'How often are frames written to the output file or shown on the '
              'screen'),
    Parameter('output/video/enabled', True, UNIT.BOOLEAN,
              'Flag determining whether the final video should be produced'),
    Parameter('output/video/extension', '.mov', None,
              'File extension used for debug videos'),
    Parameter('output/video/codec', 'libx264', None,
              'ffmpeg video codec used for debug videos'),
    Parameter('output/video/bitrate', '2000k', None,
              'Bitrate used for debug videos'),
    Parameter('output/hdf5_compression', 'gzip', None,
              'Compression algorithm to be used for the HDF5 data. Possible '
              'options might be None, "gzip", "lzf", and "szip".'),
    # Output video
    Parameter('output/video/period', 100, UNIT.TIME_FRAMES,
              'How often are frames written to the output file'),
    Parameter('output/video/mouse_trail_length', 1000, UNIT.TIME_FRAMES,
              'How long is the trail indicating the mouse position in the past'),
    
    # Cage
    Parameter('cage/width_cm', 85.5, UNIT.LENGTH_CM,
              'Measured width of the cages/antfarms. The width is measured '
              'inside the cage, not including the frame.'),
    Parameter('cage/determine_boundaries', True, UNIT.BOOLEAN,
              'Flag indicating whether the cropping rectangle should be determined '
              'automatically. If False, we assume that the original video is '
              'already cropped'),
    Parameter('cage/restrict_to_largest_patch', True, UNIT.BOOLEAN,
              'Determines whether the cage analysis will be restricted to the '
              'largest patch in the first thresholded image.'),
    Parameter('cage/threshold_zscore', 0.5, UNIT.FACTOR,
              'Factor that determines the threshold for producing the binary '
              'image that is used to located the frame of the cage. The '
              'threshold is calculated according to the formula '
              'thresh = img_mean - factor*img_std, where factor is the factor'
              'determined here.'),
    Parameter('cage/refine_by_fitting', True, UNIT.BOOLEAN,
              'Flag determining whether the cage rectangle should be refined '
              'by using fitting to locate the cage boundaries.'),
    Parameter('cage/boundary_detection_bottom_estimate', 0.95, UNIT.FRACTION,
              'Fraction of the image height that is used to estimate the '
              'position of the bottom of the frame'),
    Parameter('cage/boundary_detection_thresholds', [0.7, 0.3, 0.7, 0.9], None,
              'Thresholds for the boundary detection algorithm. The four values '
              'are the fraction of bright pixels necessary to define the '
              'boundary for [left, top, right, bottom], respectively.'),
    Parameter('cage/width_min', 550, UNIT.LENGTH_PIXEL,
              'Minimal width of the cage. This is only used to make a '
              'plausibility test of the results'),
    Parameter('cage/width_max', 800, UNIT.LENGTH_PIXEL,
              'Maximal width of the cage. This is only used to make a '
              'plausibility test of the results'),
    Parameter('cage/height_min', 300, UNIT.LENGTH_PIXEL,
              'Minimal height of the cage. This is only used to make a '
              'plausibility test of the results'),
    Parameter('cage/height_max', 500, UNIT.LENGTH_PIXEL,
              'Maximal height of the cage. This is only used to make a '
              'plausibility test of the results'),
    Parameter('cage/rectangle_buffer', 5, UNIT.LENGTH_PIXEL,
              'Margin by which the estimated cage rectangle is enlarged '
              'before it is located by fitting.'),
    Parameter('cage/linescan_length', 50, UNIT.LENGTH_PIXEL,
              'Length of the line scan that is used to determine the cage '
              'boundary.'),
    Parameter('cage/linescan_width', 30, UNIT.LENGTH_PIXEL,
              'Width of the line scan use to extend the ground line to the '
              'cage frame.'),
    Parameter('cage/linescan_smooth', 5, UNIT.LENGTH_PIXEL,
              'Standard deviation used for smoothing the line scan profile'),
                
    # Colors               
    Parameter('colors/adaptation_interval', 1000, UNIT.TIME_FRAMES,
              'How often are the color estimates adapted'),
    Parameter('colors/std_min', 5, UNIT.COLOR,
              'Minimal standard deviation of sky and sand colors'),

    # Background and explored area                             
    Parameter('background/adaptation_rate', 1e-2, UNIT.RATE_FRAMES,
              'Rate at which the background is adapted'),
    Parameter('explored_area/adaptation_rate_outside', 1e-3, UNIT.RATE_FRAMES,
              'Rate at which the explored area is adapted outside of burrows'),
    Parameter('explored_area/adaptation_rate_burrows', 0, UNIT.RATE_FRAMES,
              'Rate at which the explored area is adapted inside burrows'),
    
    # Ground
    Parameter('ground/template', '', None,
              'Name of the ground template stored in the assets directory. '
              'If the template is not given or could not be found, an '
              'alternative method based on line scans is used.'),
    Parameter('ground/template_width_factors', np.arange(0.7, 1.01, 0.05), None,
              'Different factors to try for scaling the template width with '
              'respect to the cage width.'),
    Parameter('ground/template_aspect_factors', np.arange(0.7, 1.31, 0.1), None,
              'Different factors to try for scaling the template aspect ratio.'),
    Parameter('ground/template_width_fraction', 0.8, UNIT.FRACTION,
              'Fraction of the full template width that is used for matching.'),
    Parameter('ground/template_margin', 40, UNIT.LENGTH_PIXEL,
              'Margin on the top and the bottom of the template.'),
    Parameter('ground/point_spacing', 20, UNIT.LENGTH_PIXEL,
              'Spacing of the support points describing the ground profile'),
    Parameter('ground/linescan_length', 50, UNIT.DEPRECATED, #UNIT.LENGTH_PIXEL,
              'Length of the line scan used to determine the ground profile. '
              'Deprecated since 2014-12-19'),
    Parameter('ground/slope_detector_max_factor', 0.4, UNIT.FACTOR,
              'Factor important in the ridge detection step, where the ridge '
              'is roughly located by looking at vertical line scans and points '
              'with large slopes are located. The smaller this factor, the more '
              'such points are detected and the further up the profile is '
              'estimated to be'),
    Parameter('ground/length_max', 1500, UNIT.LENGTH_PIXEL,
              'Maximal length of the ground profile above which it is rejected'),
    Parameter('ground/curvature_energy_factor', 1, UNIT.DEPRECATED, #UNIT.FACTOR,
              'Relative strength of the curvature energy to the image energy '
              'in the snake model of the ground line.'
              'Deprecated since 2014-12-19.'),
    Parameter('ground/snake_energy_max', 10, UNIT.DEPRECATED, #UNIT.FACTOR,
              'Determines the maximal energy the snake is allowed to have. '
              'Deprecated since 2014-12-19'),
    Parameter('ground/slope_max', 3, UNIT.FRACTION,
              'Maximal slope of the side ridges'),
    Parameter('ground/frame_margin', 50, UNIT.LENGTH_PIXEL,
              'Width of the margin to the frame in which the ground profile is '
              'not determined'),
    Parameter('ground/grabcut_uncertainty_margin', 50, UNIT.LENGTH_PIXEL,
              'Width of the region around the estimated profile, in which '
              'the GrabCut algorithm may optimize'),
    Parameter('ground/active_snake_gamma', 1e-1, UNIT.FACTOR,
              'Time scale of the active snake evolution algorithm for finding '
              'the ground line. Too large gammas may lead to instabilities in '
              'the algorithm, while too small gammas may cause a very slow '
              'convergence.'),
    Parameter('ground/active_snake_beta', 1e6, UNIT.FACTOR,
              'Stiffness of the active snake evolution algorithm for finding '
              'the ground line. Larger values lead to straighter lines.'),
    Parameter('ground/adaptation_interval', 100, UNIT.TIME_FRAMES,
              'How often is the ground profile adapted'),
    Parameter('ground/ridge_width', 5, UNIT.LENGTH_PIXEL,
              'Width of the ground profile ridge'),
    Parameter('ground/smoothing_sigma', 1000, UNIT.TIME_FRAMES,
              'Standard deviation for Gaussian smoothing over time'),
    
    # Mouse and the associated tracking
    Parameter('mouse/intensity_threshold', 1, UNIT.FACTOR,
              'Determines how much brighter than the background (usually the '
              'sky) the mouse has to be. This value is measured in terms of '
              'standard deviations of the sky color'),
    Parameter('mouse/model_radius', 25, UNIT.LENGTH_PIXEL,
              'Radius of the mouse model'),
    Parameter('mouse/area_max', 5000, UNIT.AREA_PIXEL,
              'Maximal area of a feature to be considered in tracking'),
    Parameter('mouse/area_min', 100, UNIT.AREA_PIXEL,
              'Minimal area of a feature to be considered in tracking'),
    Parameter('mouse/area_mean', 700, UNIT.AREA_PIXEL,
              'Mean area of a mouse, which is used to score the mouse'),
    Parameter('mouse/speed_max', 30, UNIT.SPEED_PIXEL_FRAME,
              'Maximal speed of the mouse'),
    Parameter('mouse/max_rel_area_change', 0.5, UNIT.FACTOR,
              'Maximal area change allowed between consecutive frames'),
    Parameter('mouse/speed_smoothing_window', 25, UNIT.DEPRECATED,
              'Deprecated since 2014-11-29. Use '
              '`tracking/position_smoothing_window` instead.'),
    Parameter('mouse/moving_threshold_cm_sec', 5, UNIT.SPEED_CM_SEC,
              'The threshold value of the speed above which the mouse is '
              'considered to be moving.'),
    Parameter('mouse/moving_threshold_pixel_frame', None, UNIT.DEPRECATED,
              'Deprecated since 2014-12-01.'),
    Parameter('mouse/activity_smoothing_interval', 30*60*30, #< 30 minutes
              UNIT.TIME_FRAMES,
              'The standard deviation of the Gaussian that is used for '
              'smoothing temporal data that is associated with activity '
              'measurements.'),
    Parameter('mouse/digging_rate_time_min', 30*60, UNIT.TIME_FRAMES,
              'Minimal time span the mouse has to be digging before we '
              'calculate a digging rate.'),
                  
    Parameter('tracking/weight', 0.5, UNIT.FACTOR,
              'Relative weight of distance vs. size of objects for matching '
              'them'),
    Parameter('tracking/moving_window', 200, UNIT.TIME_FRAMES,
              'Number of consecutive frames used for motion detection'),
    Parameter('tracking/moving_threshold', 1, UNIT.SPEED_PIXEL_FRAME,
              'Threshold speed above which an object is said to be moving'),
    Parameter('tracking/object_count_max', 7, UNIT.INTEGER,
              'Maximal number of objects allowed in a single frame. If there '
              'are more objects, the entire frame is discarded'),
    Parameter('tracking/time_scale', 10, UNIT.TIME_FRAMES,
              'Time duration of not seeing the mouse after which we do not ' 
              'know where it is anymore'),
    Parameter('tracking/tolerated_overlap', 50, UNIT.TIME_FRAMES,
              'How much are two consecutive tracks allowed to overlap'),
    Parameter('tracking/initial_score_threshold', 1000, UNIT.FACTOR,
              'Initial threshold for building the tracking graph'),
    Parameter('tracking/score_threshold_max', 1e10, UNIT.FACTOR,
              'Maximal threshold above which the graph based tracking is '
              'aborted.'),
    Parameter('tracking/end_node_interval', 1000, UNIT.TIME_FRAMES,
              'What time duration do we consider for start and end nodes'),
    Parameter('tracking/splitting_duration_min', 10, None,
              'Track duration above which two overlapping tracks are split'),
    Parameter('tracking/maximal_gap', 10, UNIT.TIME_FRAMES,
              'Maximal gap length where we will use linear interpolation to ' 
              'determine the mouse position'),
    Parameter('tracking/maximal_jump', 50, UNIT.LENGTH_PIXEL,
              'Maximal distance between two tracks where we will use linear '
              'interpolation to determine the intermediated mouse positions.'),
    Parameter('tracking/position_smoothing_window', 5, UNIT.TIME_FRAMES,
              'The number of frames over which the mouse position is smoothed '
              'in order to calculate its velocity'),
    Parameter('tracking/mouse_distance_threshold', 500, UNIT.LENGTH_PIXEL,
              'Distance over which an object must move in order to call it a '
              'mouse. This is used to identify tracks which surely belong to '
              'mice. Graph matching is then used to fill in the gaps.'),
    Parameter('tracking/mouse_min_mean_speed', 0.5, UNIT.SPEED_PIXEL_FRAME,
              'Minimal average speed an object must have in order to be '
              'surely considered as a mouse. This is introduced to prevent '
              'stationary objects to be called a mouse.'),
    Parameter('tracking/max_track_count', 5000, UNIT.INTEGER,
              'Maximal number of tracks that can be connected. If there are '
              'more tracks, we throw out small tracks until the count '
              'decreased to the one given here.'),
        
    # Burrows
    Parameter('burrows/enabled_pass1', False, UNIT.BOOLEAN,
              'Whether burrows should be located in the first pass'),
    Parameter('burrows/enabled_pass3', True, UNIT.BOOLEAN,
              'Whether burrows should be located in the third pass'),
    Parameter('burrows/enabled_pass4', True, UNIT.BOOLEAN,
              'Whether burrows should be located in the fourth pass'),
    Parameter('burrows/adaptation_interval', 100, UNIT.TIME_FRAMES,
              'How often are the burrow shapes adapted'),
    Parameter('burrows/cage_margin', 30, UNIT.LENGTH_PIXEL,
              'Margin of a potential burrow to the cage boundary'),
    Parameter('burrows/width', 20, UNIT.LENGTH_PIXEL,
              'Typical width of a burrow'),
    Parameter('burrows/width_min', 10, UNIT.LENGTH_PIXEL,
              'Minimal width of a burrow'),
    Parameter('burrows/chunk_area_min', 50, UNIT.AREA_PIXEL,
              'Minimal area a burrow chunk needs to have in order to be '
              'considered.'),
    Parameter('burrows/area_min', 400, UNIT.AREA_PIXEL,
              'Minimal area a burrow cross section has to have'),
    Parameter('burrows/ground_point_distance', 10, UNIT.LENGTH_PIXEL,
              'Maximal distance of ground profile to outline points that are '
              'considered exit points'),
    Parameter('burrows/shape_threshold_distance', 50, UNIT.LENGTH_PIXEL,
              'Threshold value for the distance of burrow points from the '
              'ground points. If all points are closer than this threshold, '
              'the burrow is called a "wide burrow". Otherwise, the burrow '
              'will be treated as a "long burrow".'),
    Parameter('burrows/centerline_segment_length', 15, UNIT.LENGTH_PIXEL,
              'Length of a segment of the center line of a burrow'),
    Parameter('burrows/curvature_radius_max', 30, UNIT.LENGTH_PIXEL,
              'Maximal radius of curvature the centerline is allowed to have'),
    Parameter('burrows/grabcut_burrow_core_area_min', 500, UNIT.AREA_PIXEL,
              'Minimal area the sure region of the mask for the grab cut '
              'algorithm is supposed to have'),
    Parameter('burrows/fitting_length_threshold', 100, UNIT.LENGTH_PIXEL,
              'Length above which burrows are refined by fitting'),
    Parameter('burrows/fitting_width_threshold', 30, UNIT.LENGTH_PIXEL,
              'Width below which burrows are refined by fitting'),
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
    Parameter('burrows/simplification_threshold_area', 50, UNIT.AREA_PIXEL,
              'Burrow outline points are removed when the resulting effective '
              'change is below this threshold'),
    Parameter('burrows/chunk_dist_max', 30, UNIT.LENGTH_PIXEL,
              'Maximal distance between a burrow chunk and another structure '
              '(either another chunk or the ground line), such that the chunk '
              'is connected to the other structure.'),
    Parameter('burrows/image_statistics_window', 50, UNIT.LENGTH_PIXEL,
              'Half of the size of the window over which the statistics of '
              'the image are calculated.'),
    Parameter('burrows/image_statistics_overlap_threshold', 0.5, UNIT.FRACTION,
              'The threshold value of the allowed overlap of the background '
              'and foreground statistics. If the distributions overlap more '
              'than this value the point is considered to be background since '
              'it cannot be discriminated reliably.'),
    Parameter('burrows/activity_ignore_interval', 30*60*5, #< 5 minutes
              UNIT.TIME_FRAMES,
              'The time interval of the burrow trajectory that is ignored in '
              'the activity analysis. This is mainly done to circumvent '
              'problems with the initial predug.'),
    Parameter('burrows/activity_smoothing_interval', 30*60*30, #< 30 minutes
              UNIT.TIME_FRAMES,
              'The standard deviation of the Gaussian that is used for '
              'smoothing temporal data that is associated with activity '
              'measurements.'),
    Parameter('burrows/predug_analyze_time', 30*60, UNIT.TIME_FRAMES, #< 1 min
              'The time duration after burrow detection at which the predug is '
              'analyzed.'),
    Parameter('burrows/predug_area_threshold', 1000, UNIT.AREA_PIXEL,
              'The minimal area in pixels the burrow has to have in order to '
              'be considered as a predug.'),
                  
    Parameter('burrows/active_contour/blur_radius', 2, UNIT.LENGTH_PIXEL,
              'Blur radius of the active contour algorithm used for refining '
              'the burrow shape.'),
    Parameter('burrows/active_contour/stiffness', 1e4, UNIT.AREA_PIXEL,
              'Stiffness of the active contour algorithm used for refining the '
              'burrow shape.'),
    Parameter('burrows/active_contour/convergence_rate', 1e-2, UNIT.FACTOR,
              'Convergence rate of the active contour algorithm used for '
              'refining the burrow shape.'),
    Parameter('burrows/active_contour/max_iterations', 100, UNIT.FACTOR,
              'Maximal number of iterations of the active contour algorithm '
              'used for refining the burrow shape.'),
          
    # analysis after tracking
    Parameter('analysis/frames', None, None,
              'Frames of the video which are included in the report of the '
              'analysis [start and end index should be given]. If this is '
              'omitted, all analyzed frames are included'),
                  
    # Computation resources
    Parameter('project/symlink_folder', None, None,
              'If set, a symlink pointing to the base_folder will be created '
              'in this directory when a project is created.'),
    Parameter('resources/notification_email', 'dzwicker@seas.harvard.edu', None,
              'Email address of the user to be notified in case of problems.'),
    Parameter('resources/slurm_partition', 'general', None,
              'Name of the slurm partition to use for submitting jobs'),
    Parameter('resources/pass1/job_id', None, None, 'Job id of pass 1'),              
    Parameter('resources/pass1/cores', 3, UNIT.INTEGER, 'Number of cores for pass 1'),
    Parameter('resources/pass1/time', 50*60, None, 'Maximal computation minutes for pass 1'),
    Parameter('resources/pass1/memory', 1000, None, 'Maximal RAM per core for pass 1 [in MB]'),
    Parameter('resources/pass2/job_id', None, None, 'Job id of pass 2'),              
    Parameter('resources/pass2/cores', 1, UNIT.INTEGER, 'Number of cores for pass 2'),
    Parameter('resources/pass2/time', 25*60, None, 'Maximal computation minutes for pass 2'),
    Parameter('resources/pass2/memory', 8000, None, 'Maximal RAM per core for pass 2 [in MB]'),
    Parameter('resources/pass3/job_id', None, None, 'Job id of pass 3'),              
    Parameter('resources/pass3/cores', 2, UNIT.INTEGER, 'Number of cores for pass 3'),
    Parameter('resources/pass3/time', 30*60, None, 'Maximal computation minutes for pass 3'),
    Parameter('resources/pass3/memory', 1000, None, 'Maximal RAM per core for pass 3 [in MB]'),
    Parameter('resources/pass4/job_id', None, None, 'Job id of pass 4'),              
    Parameter('resources/pass4/cores', 2, UNIT.INTEGER, 'Number of cores for pass 4'),
    Parameter('resources/pass4/time', 25*60, None, 'Maximal computation minutes for pass 4'),
    Parameter('resources/pass4/memory', 2000, None, 'Maximal RAM per core for pass 4 [in MB]'),
]

# collect all parameters in a convenient dictionary
PARAMETERS = {p.key: p for p in PARAMETER_LIST}
# collect the default values of all parameters
PARAMETERS_DEFAULT = {p.key: p.default_value for p in PARAMETER_LIST
                      if p.unit != UNIT.DEPRECATED}



def set_base_folder(parameters, folder, include_default=False):
    """ changes the base folder of all folders given in the parameter
    dictionary.
    include_default is a flag indicating whether the default parameters
    describing folders should also be included. """
    
    warnings.warn("Base folder is a parameter now.", DeprecationWarning)
    
    # convert to plain dictionary if it is anything else
    parameters_type = type(parameters)
    if parameters_type != dict:
        parameters = parameters.to_dict(flatten=True)
    
    # add all default folders, which will be changed later
    if include_default:
        for p in PARAMETER_LIST:
            if p.unit == UNIT.SUBFOLDER and p.key not in parameters:
                parameters[p.key] = p.default_value

    # set the base folder    
    parameters['base_folder'] = folder
            
    # return the result as the original type 
    return parameters_type(parameters)



def scale_parameters(parameters, factor_length=1, factor_time=1):
    """ takes a DictXpath dictionary of parameters and scales them according to
    their unit and the given scale factors """
    # scale each parameter in the list
    for key in parameters.iterkeys(flatten=True):
        unit = PARAMETERS[key].unit
        if unit == UNIT.LENGTH_PIXEL:
            parameters[key] *= factor_length
        elif unit == UNIT.AREA_PIXEL:
            parameters[key] *= factor_length**2
        elif unit == UNIT.TIME_FRAMES:
            parameters[key] *= factor_time
        elif unit == UNIT.RATE_FRAMES:
            parameters[key] /= factor_time
        elif unit == UNIT.SPEED_PIXEL_FRAME:
            parameters[key] *= factor_length/factor_time

