#!/usr/bin/env python2

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from numpy import array  # @UnusedImport

from mousetracking.algorithm.pass2 import SecondPass

# configure basic logging, which will be overwritten later
logging.basicConfig()

# set specific parameters for this job
parameters = {SPECIFIC_PARAMETERS}  # @UndefinedVariable

# set job parameters
parameters.update({{'base_folder': "{JOB_DIRECTORY}",
                    'logging/folder': ".",
                    'output/folder': ".",}})

# do the second pass scan
pass2 = SecondPass("{NAME}", parameters=parameters, read_data=True)
pass2.process()
pass2.process()
if pass2.data['parameters/output/video/enabled']:
    pass2.produce_video()
