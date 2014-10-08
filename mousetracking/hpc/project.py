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
    
    files_job = tuple()      #< files that need to be set up for the project
    files_cleanup = tuple()  #< files that need to be deleted to clean the work folder
    default_passes = 3
    
    def __init__(self, folder, name=None, parameters=None, passes=None):
        """ initializes a project with all necessary information """
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.folder = folder
        self.name = name

        if passes is None:
            passes = self.default_passes
        if not hasattr(passes, '__iter__'):
            self.passes = range(1, passes + 1)
        else:
            self.passes = passes

        # save tracking parameters
        self.parameters = DataDict(PARAMETERS_DEFAULT)
        if parameters is not None:
            self.parameters.from_dict(parameters)
            
        
    def clean_workfolder(self, purge=False):
        """ clears the project folder """
        # determine which files to delete
        if purge:
            try:
                files_to_delete = os.listdir(self.folder)
            except OSError:
                files_to_delete = tuple()
        else:
            files_to_delete = []
            for p in self.passes:
                files_to_delete.extend(self.files_job[p] + self.files_cleanup[p])
            
        # iteratively delete these files
        for filename in files_to_delete:
            file_path = os.path.join(self.folder, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass


    @classmethod
    def create(cls, video_file, result_folder, video_name=None,
               parameters=None, passes=None, prepare_workfolder='auto' ):
        """ creates a new project from data
        video_file is the filename of the video to scan
        result_folder is a general folder in which the results will be stored.
            Note that a subfolder will be used for all results
        video_name denotes a name associated with this video, which will be used
            to name folders and such. If no name is given, the filename is used.
        parameters is a dictionary that sets the parameters that are
            used for tracking.
        passes is an integer which is 1, 2 or 3, indicating whether only the first
            tracking pass or also the second one should be initialized
        prepare_workfolder can be 'none', 'clean', or 'purge', which indicates
            increasing amounts of files that will be deleted before creating
            the project. If it is set to 'auto', the folder will be purged
            if a first pass run is requested.
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
        project = cls(folder, video_name, parameters, passes)
        
        if prepare_workfolder == 'auto':
            pass1_requested = (1 in project.passes)
            project.clean_workfolder(purge=pass1_requested)
        elif 'clean' in prepare_workfolder:
            project.clean_workfolder()
        elif 'purge' in prepare_workfolder:
            project.clean_workfolder(purge=True)
        
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
        for pass_id in project.passes:
            # add job files to parameters
            for k, filename in enumerate(cls.files_job[pass_id]):
                params['JOB_FILE_%d' % k] = filename
        
            # create the job scripts
            for filename in cls.files_job[pass_id]:
                script = project.get_template(filename)
                script = script.format(**params)
                open(os.path.join(project.folder, filename), 'w').write(script)
            
        # create symbolic link if requested
        symlink_folder = project.parameters['project/symlink_folder']
        if symlink_folder:
            dst = os.path.join(symlink_folder, project.name)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.symlink(project.folder, dst)
            except OSError as err:
                project.logger.warn('Symlink creation failed: %s', err)
            
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

