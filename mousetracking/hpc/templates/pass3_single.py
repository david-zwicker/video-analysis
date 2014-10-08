#!/usr/bin/env python2

from __future__ import division

import sys
import os
import logging
sys.path.append(os.path.expanduser("{FOLDER_CODE}"))

from mousetracking.algorithm import ThirdPass

# configure basic logging, which will be overwritten later
logging.basicConfig()

# set job parameters
parameters = {{'base_folder': "{JOB_DIRECTORY}",
               'logging/folder': ".",
               'output/folder': ".",}}

# do the second pass scan
pass3 = ThirdPass("{NAME}", parameters=parameters, read_data=True)
pass3.process()