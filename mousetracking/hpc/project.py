'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import os
import pprint

from ..algorithm.parameters import PARAMETERS_DEFAULT
from ..algorithm.data_handler import DataDict


class HPCProjectBase(object):
    """ class that manages a high performance computing project """
    job_files = [] #< files that need to be set up for the project
    
    
    def __init__(self, folder, name=None, parameters=None, passes=2):
        """ initializes a project with all necessary information """
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.folder = folder
        self.name = name
        self.passes = passes

        # save tracking parameters
        self.parameters = DataDict(PARAMETERS_DEFAULT)
        if parameters is not None:
            self.parameters.from_dict(parameters)
        
    
    @classmethod
    def create(cls, video_file, result_folder, video_name=None,
               parameters=None, passes=2):
        """ creates a new project from data
        video_file is the filename of the video to scan
        result_folder is a general folder in which the results will be stored.
            Note that a subfolder will be used for all results
        video_name denotes a name associated with this video, which will be used
            to name folders and such. If no name is given, the filename is used.
        parameters is a dictionary that sets the parameters that are
            used for tracking.
        """
        video_file = os.path.abspath(video_file)

        # determine the name of the video
        if video_name is None:
            _, filename = os.path.split(video_file)
            fileparts = filename.split('.')
            video_name = '.'.join(fileparts[:-1])
            
        # setup the project instance
        result_folder = os.path.abspath(os.path.expanduser(result_folder))
        folder = os.path.join(result_folder, video_name)
        project =  cls(folder, video_name, parameters, passes)
        
        # extract folder of current file
        this_folder, _ = os.path.split(__file__)
        folder_code = os.path.abspath(os.path.join(this_folder, '../..'))
        
        # setup general information
        tracking_parameters = project.parameters.to_dict(flatten=True)
        params = {'FOLDER_CODE': folder_code,
                  'JOB_DIRECTORY': project.folder,
                  'NAME': project.name,
                  'VIDEO_FILE': video_file,
                  'TRACKING_PARAMETERS': pprint.pformat(tracking_parameters)}
        
        # setup job resources
        resource_iter = project.parameters['resources'].iteritems(flatten=True)
        for key, value in resource_iter:
            params[key.upper()] = value
        
        # ensure that the result folder exists
        try:
            os.makedirs(project.folder)
        except OSError:
            pass
        
        # set up job scripts
        for filename in cls.job_files:
            script = project.get_template(filename)
            script = script.format(**params)
            open(os.path.join(project.folder, filename), 'w').write(script)
            
        project.logger.info('Prepared project in folder %s', project.folder)
        return project
        
        
    def get_template(self, template):
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
            
        
    def submit(self):
        """ submit the job to the cluster """
        raise NotImplementedError
