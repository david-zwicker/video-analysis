'''
Created on Sep 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import glob
import logging
import os
import pprint
import time

import numpy as np

from ..algorithm.parameters import PARAMETERS_DEFAULT
from data_structures.dict_xpath import DictXpath



def process_trials(logfile, max_iterations=10):
    """ returns an generator which yields the current trial number until the
    processing is finished. The finish condition is based on analyzing the
    logfile.
    max_iterations determines how many iterations are done at most"""
    for trial in xrange(max_iterations):
        yield trial

        # check for an error in the log file
        processing_finished = True
        try:
            for line in open(logfile, "r"):
                if 'FFmpeg encountered the following error' in line:
                    # sleep up to two minutes to get around weird race conditions
                    logging.info('Restarted the analysis since an FFmpeg error '
                                 'was encountered.')
                    time.sleep(np.random.randint(120))
                    processing_finished = False
        except IOError:
            # file likely does not exist => we assume no error 
            pass
        
        if processing_finished:
            break



class HPCProjectBase(object):
    """ class that manages a high performance computing project """
    
    files_job = tuple()      #< files that need to be set up for the project
    files_cleanup = tuple()  #< files that need to be deleted to clean the work folder
    file_parameters = '%s_results.yaml' #< pattern for the file where the parameters are stored
    file_log_pass = "log_pass%d_%%s.txt" #< pattern for the log file of each pass
    default_passes = 4
    
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
        self.parameters = DictXpath(PARAMETERS_DEFAULT)
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
                files_to_delete.extend(self.files_job[p])
                files_to_delete.extend(self.files_cleanup[p])
            
        # iteratively delete these files
        for pattern in files_to_delete:
            file_pattern = os.path.join(self.folder, pattern)
            for file_path in glob.iglob(file_pattern):
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass


    @classmethod
    def create(cls, video_file, result_folder, video_name=None,
               parameters=None, passes=None, prepare_workfolder='auto'):
        """ creates a new project from data
        video_file is the filename of the video to scan
        result_folder is a general folder in which the results will be stored.
            Note that a subfolder will be used for all results
        video_name denotes a name associated with this video, which will be used
            to name folders and such. If no name is given, the filename is used.
        parameters is a dictionary that sets the parameters that are
            used for tracking. There is a special parameter 'scale_length' that
            we look for, which is applied in the first pass only.
        passes is an integer which is 1, 2 or 3, indicating whether only the first
            tracking pass or also the second one should be initialized
        prepare_workfolder can be 'none', 'clean', or 'purge', which indicates
            increasing amounts of files that will be deleted before creating
            the project. If it is set to 'auto', the folder will be purged
            if a first pass run is requested.
        specific_parameters are extra parameters that 
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
        
        # prepare the project
        project.prepare_project(video_file, result_folder, video_name,
                                parameters, passes, prepare_workfolder)
        return project
        
        
    def prepare_project(self, video_file, result_folder, video_name=None,
                        parameters=None, passes=None,
                        prepare_workfolder='auto'):
        """ prepare the work directory by setting up all necessary files """
        
        if prepare_workfolder == 'auto':
            pass1_requested = (1 in self.passes)
            self.clean_workfolder(purge=pass1_requested)
        elif 'clean' in prepare_workfolder:
            self.clean_workfolder()
        elif 'purge' in prepare_workfolder:
            self.clean_workfolder(purge=True)
        
        # extract folder of current file
        this_folder, _ = os.path.split(__file__)
        folder_code = os.path.abspath(os.path.join(this_folder, '../..'))
        
        # setup tracking parameters
        tracking_parameters = self.parameters.to_dict(flatten=True)
        # extract the factor for the lengths and provide it separately
        scale_length = tracking_parameters.pop('scale_length', 1)
        scale_length = parameters.pop('scale_length', scale_length)
        # setup all variables that might be used in the templates
        params = {'FOLDER_CODE': folder_code,
                  'JOB_DIRECTORY': self.folder,
                  'NAME': self.name,
                  'VIDEO_FILE': video_file,
                  'TRACKING_PARAMETERS': pprint.pformat(tracking_parameters),
                  'SPECIFIC_PARAMETERS': pprint.pformat(parameters),
                  'SCALE_LENGTH': scale_length}
        
        # setup job resources
        resource_iter = self.parameters['resources'].iteritems(flatten=True)
        for key, value in resource_iter:
            params[key.upper()] = value
        
        # ensure that the result folder exists
        try:
            os.makedirs(self.folder)
        except OSError:
            pass
        
        # set up job scripts
        for pass_id in self.passes:
            # add job files to parameters
            for k, filename in enumerate(self.files_job[pass_id]):
                params['JOB_FILE_%d' % k] = filename
            params['LOG_FILE'] = os.path.join(self.folder, 
                                              self.file_log_pass % pass_id)
        
            # create the job scripts
            for filename in self.files_job[pass_id]:
                script = self.get_template(filename)
                script = script.format(**params)
                open(os.path.join(self.folder, filename), 'w').write(script)
            
        # create symbolic link if requested
        symlink_folder = self.parameters['project/symlink_folder']
        if symlink_folder:
            dst = os.path.join(symlink_folder, self.name)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.symlink(self.folder, dst)
            except OSError as err:
                self.logger.warn('Symlink creation failed: %s', err)
            
        self.logger.info('Prepared project in folder %s', self.folder)
        
        
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

