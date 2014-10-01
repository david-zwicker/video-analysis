#!/usr/bin/env python

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from mousetracking.parallel import scan_video_quadrants

# configure basic logging, which will be overwritten later
logging.basicConfig()

# define the parameters used for tracking
parameters = {TRACKING_PARAMETERS}  # @UndefinedVariable

# set job parameters
parameters.update({{
    'video/filename_pattern': "{VIDEO_FILE}",
    'base_folder': "{JOB_DIRECTORY}",
    'logging/folder': ".",
    'debug/folder': ".",
    'output/folder': ".",
    'output/video/folder': ".",
}})

# do the first pass scan
scan_video_quadrants(video=None, parameters=parameters, passes=1)
