'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os
import pprint


class HPCProjectBase(object):
    """ class that manages a high performance computing project """
    # general information about the setup 
    machine_configuration = {'USER_EMAIL': 'dzwicker@seas.harvard.edu',
                             'PARTITION': 'general',
                             'PASS1_CORES': 4,
                             'PASS1_TIME': 20*60,
                             'PASS1_MEMORY': 1000,
                             'PASS2_CORES': 2,
                             'PASS2_TIME': 20*60,
                             'PASS2_MEMORY': 10000,}
    job_files = [] #< files that need to be set up for the project
    
    
    def __init__(self, video_file, result_folder, video_name=None,
                 tracking_parameters=None, debug_output=None, passes=2):
        """ initializes a project with all necessary information
        video_file is the filename of the video to scan
        result_folder is a general folder in which the results will be stored.
            Note that a subfolder will be used for all results
        video_name denotes a name associated with this video, which will be used
            to name folders and such. If no name is given, the filename is used.
        tracking_parameters is a dictionary that sets the parameters that are
            used for tracking.
        """
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.video_file = os.path.abspath(video_file)
        self.passes = passes

        # save tracking parameters
        if tracking_parameters is None:
            self.tracking_parameters = {}
        else:
            self.tracking_parameters = tracking_parameters
        
        # determine the name of the video
        if video_name is None:
            _, filename = os.path.split(video_file)
            fileparts = filename.split('.')
            self.name = '.'.join(fileparts[:-1])
        self.name = video_name
            
        result_folder = os.path.abspath(os.path.expanduser(result_folder))
        self.folder = os.path.join(result_folder, self.name)
        
        
    def get_tempalte(self, template):
        """ return the content of a chosen template """
        # check whether the template is given as an absolute path
        if os.path.isabs(template):
            filename = template
        else:
            filename = os.path.join(
                os.path.dirname(__file__),
                'templates/%s' % template
            )

        return open(filename).read()
        
    
    def setup(self):
        """ setup the project folder """
        # extract folder of current file
        this_folder, _ = os.path.split(__file__)
        folder_code = os.path.abspath(os.path.join(this_folder, '../..'))
        
        # setup information
        params = self.machine_configuration.copy()
        params['FOLDER_CODE'] = folder_code
        params['JOB_DIRECTORY'] = self.folder
        params['NAME'] = self.name
        params['VIDEO_FILE'] = self.video_file
        params['TRACKING_PARAMETERS'] = pprint.pformat(self.tracking_parameters)
        
        # ensure that the result folder exists
        try:
            os.makedirs(self.folder)
        except OSError:
            pass
        
        # set up job scripts
        for filename in self.job_files:
            script = self.get_tempalte(filename)
            script = script.format(**params)
            open(os.path.join(self.folder, filename), 'w').write(script)
            
        self.logger.info('Prepared project in %s', self.folder)
            
        
    def submit(self):
        """ submit the job to the cluster """
        raise NotImplementedError
