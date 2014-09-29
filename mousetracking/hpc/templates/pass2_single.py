#!/usr/bin/env python2

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from mousetracking import load_results
from mousetracking.algorithm import SecondPass

# configure basic logging, which will be overwritten later
logging.basicConfig()

# set job parameters
parameters = {{'logging/folder': "{JOB_DIRECTORY}",
               'output/folder': "{JOB_DIRECTORY}",}}

# do the second pass scan
results = load_results("{NAME}", parameters, cls=SecondPass)
results.process_data()
if results.data['parameters/output/video/enabled']:
    results.produce_video()
