'''
Created on Aug 27, 2014

@author: zwicker

Provides a dictionary with default parameters for the mouse tracking.
This can also be seen as some kind of documentation of the available
parameters.
'''


PARAMETERS_DEFAULT = {
    # filename pattern used to look for videos
    'video/filename_pattern': 'raw_video/*.MTS',
    # number of initial frames to skip during analysis
    'video/initial_adaptation_frames': 100,
    # radius of the blur filter to remove noise [in pixel]
    'video/blur_radius': 3,
    # frames of the video which are analyzed [start and end index should be given]
    'video/frames': None,
    # rectangle to which the video is cropped
    # this can be either four numbers [left, top, width, height] or some
    # string like 'upper left', 'lower right', etc.
    'video/cropping_rect': None,
        
    # folder to which the log file is written
    'logging/folder': './logging/',
    # level of messages to log to stderr [standard python logging levels]
    'logging/level_stderr': 'INFO',
    # level of messages to log to file if folder is set [standard python logging levels]
    'logging/level_file': 'DEBUG',
    
    # folder to which the YAML and HDF5 result files are written
    'output/result_folder': './results/',
    # folder to which debug videos are written
    'output/video/folder_debug': './debug/',
    # file extension used for debug videos
    'output/video/extension': '.mov',
    # ffmpeg video codec used for debug videos
    'output/video/codec': 'libx264',
    # bitrate used for debug videos
    'output/video/bitrate': '2000k',
    
    # thresholds for cage dimension [in pixel]
    # These are only used to make a plausibility test of the results.
    # Setting the min to 0 and max to a large number should still allow the
    # algorithm to find a sensible cage approximation
    'cage/width_min': 650,
    'cage/width_max': 800,
    'cage/height_min': 400,
    'cage/height_max': 500,
                               
    # how often are the color estimates adapted [in frames]
    'colors/adaptation_interval': 1000,
                               
    # the rate at which the background is adapted [in 1/frames]
    'background/adaptation_rate': 1e-2,
    # the rate at which the explored area is adapted [in 1/frames]
    'explored_area/adaptation_rate_outside': 1e-3,
    'explored_area/adaptation_rate_burrows': 1e-5,
    
    # spacing of the support points describing the ground profile [in pixel]
    'ground/point_spacing': 20,
    # how often is the ground profile adapted [in frames]
    'ground/adaptation_interval': 100,
    # width of the ground profile ridge [in pixel]
    'ground/width': 5,
    # standard deviation for Gaussian smoothing [in frames]
    'ground/smoothing_sigma': 300,
    
    # `mouse.intensity_threshold` determines how much brighter than the
    # background (usually the sky) has the mouse to be. This value is
    # measured in terms of standard deviations of the sky color
    'mouse/intensity_threshold': 1,
    # radius of the mouse model [in pixel]
    'mouse/model_radius': 25,
    # minimal area of a feature to be considered in tracking [in pixel^2]
    'mouse/area_min': 100,
    # mean area of a mouse, which is used to score the mouse 
    'mouse/area_mean': 700,
    # maximal speed of the mouse [in pixel per frame]
    # this is only used for the first-pass
    'mouse/speed_max': 30, 
    # maximal area change allowed between consecutive frames [dimensionless]
    'mouse/max_rel_area_change': 0.5,

    # relative weight of distance vs. size of objects for matching them [dimensionless]
    'tracking/weight': 0.5,
    # number of consecutive frames used for motion detection [in frames]
    'tracking/moving_window': 20,
    # threshold speed above which an object is said to be moving [in pixels/frame]
    'tracking/moving_threshold': 10*1e10,
    # time duration of not seeing the mouse after which we don't know where it is anymore [in frames]
    'tracking/time_scale': 10,
    # how much are two consecutive tracks allowed to overlap [in frames]
    'tracking/tolerated_overlap': 10,
    # initial threshold for building the tracking graph
    'tracking/initial_score_threshold': 1,
    # what time duration do we consider for start and end nodes [in frames]
    'tracking/end_node_interval': 1000,
        
    # how often are the burrow shapes adapted [in frames]
    'burrows/adaptation_interval': 100,
    # margin of a potential burrow to the cage boundary [in pixel]
    'burrows/cage_margin': 30,
    # what is a typical width of a burrow [in pixel]
    'burrows/width': 20,
    # what is a minimal width of a burrow [in pixel]
    'burrows/width_min': 10,
    # minimal area a burrow cross section has to have [in pixel^2]
    'burrows/area_min': 1000,
    # maximal distance of ground profile to outline points that are considered exit points [in pixel]
    'burrows/ground_point_distance': 10,
    # length of a segment of the center line of a burrow [in pixel]
    'burrows/centerline_segment_length': 25,
    # the maximal radius of curvature the centerline is allowed to have
    'burrows/curvature_radius_max': 50,
    # the eccentricity above which burrows are refined by fitting [dimensionless]
    'burrows/fitting_eccentricity_threshold': 0.98,
    # the length above which burrows are refined by fitting [in pixel]
    'burrows/fitting_length_threshold': 75,
    # width of the burrow edge used in the template for fitting
    'burrows/fitting_edge_width': 3,
    'burrows/fitting_edge_R2min': -10,
    # determines how much the burrow outline might be simplified. The quantity 
    # determines by what fraction the total outline length is allowed to change 
    'burrows/outline_simplification_threshold': 0.005,
}
