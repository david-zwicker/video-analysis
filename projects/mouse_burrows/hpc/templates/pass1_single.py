#!/usr/bin/env python2

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from numpy import array  # @UnusedImport

from projects.mouse_burrows import scan_video
from projects.mouse_burrows.hpc.project import process_trials
from video.io.backend_ffmpeg import FFmpegError 

# configure basic logging, which will be overwritten later
logging.basicConfig()

# define the parameters used for tracking
parameters = {TRACKING_PARAMETERS}  # @UndefinedVariable

# set job parameters
job_id = sys.argv[1]
parameters.update({{
    'video/filename_pattern': "{VIDEO_FILE_TEMPORARY}",
    'base_folder': "{JOB_DIRECTORY}",
    'logging/folder': ".",
    'debug/folder': ".",
    'output/folder': ".",
    'output/video/folder': ".",
    'resources/pass1/job_id': job_id,
}})

# do the first pass scan
for trial in process_trials("{LOG_FILE}" % job_id, 10):
    try:
        scan_video("{NAME}", parameters=parameters, passes=1,
                   scale_length={SCALE_LENGTH}) # @UndefinedVariable
    except FFmpegError:
        print('FFmpeg error occurred! Repeat the analysis.')
    else:
        print('Analysis finished successfully.')
        break
