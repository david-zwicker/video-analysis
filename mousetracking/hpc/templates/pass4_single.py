#!/usr/bin/env python2

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from numpy import array  # @UnusedImport

from mousetracking.algorithm.pass4 import FourthPass
from mousetracking.hpc.project import process_trials

# configure basic logging, which will be overwritten later
logging.basicConfig()

# set specific parameters for this job
parameters = {SPECIFIC_PARAMETERS}  # @UndefinedVariable

# set job parameters
job_id = sys.argv[1]
parameters.update({{
    'base_folder': "{JOB_DIRECTORY}",
    'logging/folder': ".",
    'output/folder': ".",
    'resources/pass3/job_id': job_id,
}})

# do the second pass scan
for trial in process_trials("{LOG_FILE}" % job_id, 10):
    pass4 = FourthPass("{NAME}", parameters=parameters, read_data=True)
    pass4.process()