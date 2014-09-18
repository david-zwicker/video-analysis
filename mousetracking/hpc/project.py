'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os



class HPCProjectBase(object):
    """ class that manages a high performance computing project """
    # general information about the setup 
    machine_configuration = {'FOLDER_CODE': '~/Code/video-analysis',
                             'USER_EMAIL': 'dzwicker@seas.harvard.edu'}
    job_files = []
    
    
    def __init__(self, video_file, result_folder, video_name=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.video_file = video_file
        
        # determine parameters for the video
        if video_name is None:
            _, filename = os.path.split(video_file)
            fileparts = filename.split('.')
            self.name = '.'.join(fileparts[:-1])
        self.name = video_name
            
        self.folder = os.path.join(os.path.expanduser(result_folder), 
                                   self.name)
        
        
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
        # setup information
        params = self.machine_configuration.copy()
        params['JOB_DIRECTORY'] = self.folder
        params['VIDEO_FILE'] = self.video_file
        
        # ensure that the result folder exists
        try:
            os.makedirs(self.folder)
        except OSError:
            pass
        
        # set up job scripts
        for filename in self.job_files:
            script = self.get_tempalte(filename)
            script = script.format(params)
            open(os.path.join(self.folder, filename), 'w').write(script)
            
        self.logger.info('Prepared project in %s', self.folder)
            
        
    def submit(self):
        """ submit the job to the cluster """
        raise NotImplementedError
