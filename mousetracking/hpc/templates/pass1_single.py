#!/usr/bin/env python

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from mousetracking import scan_video

# configure basic logging, which will be overwritten later
logging.basicConfig()

# define the parameters used for tracking
parameters = {TRACKING_PARAMETERS}  # @UndefinedVariable

# set job parameters
parameters.update({{
    'video/filename_pattern': "{VIDEO_FILE}",
    'logging/folder': "{JOB_DIRECTORY}",
    'debug/folder': "{JOB_DIRECTORY}",
    'output/folder': "{JOB_DIRECTORY}",
    'output/video/folder': "{JOB_DIRECTORY}",
}})

# create file structure
open('_running_pass1', 'a').close()

try:
    # do the first pass scan
    scan_video("{NAME}", parameters=parameters, passes=1,
               debug_output={DEBUG_OUTPUT})  # @UndefinedVariable
finally:
    # remove temporary file
    os.remove('_running_pass1')
