#!/usr/bin/env python2
#@PydevCodeAnalysisIgnore

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser({FOLDER_CODE}))

from mousetracking import load_results
from mousetracking.algorithm import SecondPass

# configure basic logging, which will be overwritten later
logging.basicConfig()

# set job parameters
parameters = {
    'logging/folder': '.',
    'debug/folder': '.',
    'output/folder': '.',
    'output/video/folder': '.',
    'cage/determine_boundaries': False
}

# create file structure
open('_running_pass2', 'a').close()

try:
    # do the actual scan
    results = load_results({NAME}, parameters, cls=SecondPass)
    results.process_data()
    results.produce_video()
finally:
    # remove temporary file
    os.remove('_running_pass2')
