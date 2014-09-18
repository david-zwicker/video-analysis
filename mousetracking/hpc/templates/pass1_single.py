#!/usr/bin/env python
#@PydevCodeAnalysisIgnore

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser({FOLDER_CODE}))

from mousetracking import scan_video

# configure basic logging, which will be overwritten later
logging.basicConfig()

parameters = {TRACKING_PARAMETERS}

# set job parameters
parameters.update({
    'video/filename_pattern': {VIDEO_FILE},
    'logging/folder': '.',
    'debug/folder': '.',
    'output/folder': '.',
    'output/video/folder': '.',
})

# create file structure
open('_running_pass1', 'a').close()

try:
    # do the actual scan
    scan_video({NAME}, parameters=parameters,
               scale_length=720/1080,
               passes=1)
finally:
    # remove temporary file
    os.remove('_running_pass1')
