'''
Created on Aug 16, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

contains classes that manage data input and output 
'''

from __future__ import division

import datetime
import logging
import os
import subprocess

import yaml
try:
    import dateutil
except ImportError:
    dateutil = None

import objects
from .parameters import PARAMETERS, PARAMETERS_DEFAULT, UNIT, scale_parameters
from utils.files import change_directory, ensure_directory_exists
from utils.data_structures import (DictXpathLazy, LazyHDFValue,
                                   prepare_data_for_yaml)
from utils.misc import get_loglevel_from_name
from utils.cache import cached_property
from video.io import load_any_video
from video.filters import FilterCrop, FilterMonochrome


LOGGING_FILE_MODES = {'create': 'w', #< create new log file 
                      'append': 'a'} #< append to old log file


# find the file handle to /dev/null to dumb strings
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



class DataHandler(object):
    """ class that handles the data and parameters of mouse tracking """
    logging_mode = 'append'
    report_unknown_parameters = True

    # dictionary of data items that are stored in a separated HDF file
    # and will be loaded only on access
    hdf_values = {'pass1/ground/profile': objects.GroundProfileList,
                  'pass1/objects/tracks': objects.ObjectTrackList,
                  'pass1/burrows/tracks': objects.BurrowTrackList,
                  'pass2/ground_profile': objects.GroundProfileTrack,
                  'pass2/mouse_trajectory': objects.MouseTrack,
                  'pass3/burrows/tracks': objects.BurrowTrackList,
                  'pass4/burrows/tracks': objects.BurrowTrackList}

    # look up table for where to find folders in the parameter dictionary    
    folder_lut = {'results': 'parameters/output/folder',
                  'video': 'parameters/output/video/folder',
                  'logging': 'parameters/logging/folder',
                  'debug': 'parameters/debug/folder'}

    
    def __init__(self, name='', parameters=None, initialize_parameters=True,
                 read_data=False):
        self.name = name
        self.logger = logging.getLogger('mousetracking')

        # initialize the data handled by this class
        self.video = None
        self.data = DictXpathLazy()
        self.data.create_child('parameters')
        self.data['parameters'].from_dict(PARAMETERS_DEFAULT)
        self.parameters_user = parameters #< parameters with higher priority

        # folders must be initialized before the data is read
        if initialize_parameters:
            self.initialize_parameters(parameters)
            self.set_status('Initialized parameters')

        if read_data:
            # read_data internally initializes the parameters 
            self.read_data()
            self.set_status('Data from previous run has been read')


    def set_status(self, status):
        """ sets the status of the analysis """
        self.data['analysis-status'] = {'state': status}
        

    def check_parameters(self, parameters):
        """ checks whether the parameters given in the input do actually exist
        in this version of the code """
        unknown_params, deprecated_params = [], []
        for key in parameters:
            if key not in PARAMETERS:
                unknown_params.append(key)
            elif PARAMETERS[key].unit == UNIT.DEPRECATED:
                deprecated_params.append(key)

        if unknown_params:
            raise ValueError('Parameter(s) %s are not known.' % unknown_params)
        if deprecated_params:
            self.logger.warn('Parameter(s) %s are deprecated and will not be '
                             'used in the analysis.' % deprecated_params)
        

    def initialize_parameters(self, parameters=None):
        """ initialize parameters """
        if parameters is not None:
            if self.report_unknown_parameters:
                self.check_parameters(parameters)
            # update parameters with the given ones
            self.data['parameters'].from_dict(parameters)
            
        # create logger for this object
        self.logger = logging.getLogger(self.name)
        self.logger.handlers = []     #< reset list of handlers
        self.logger.propagate = False #< disable default logger
        logging_level_min = logging.CRITICAL
        
        if self.data['parameters/logging/enabled']:
            # add default logger to stderr
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s ' + self.name + 
                                          '%(levelname)8s: %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            level_stderr = self.data['parameters/logging/level_stderr']
            logging_level = get_loglevel_from_name(level_stderr)
            handler.setLevel(logging_level)
            self.logger.addHandler(handler) 
            logging_level_min = min(logging_level_min, logging_level)
        
            # setup handler to log to file
            logfile = self.get_filename('log.log', 'logging')
            logging_mode = LOGGING_FILE_MODES[self.logging_mode]
            handler = logging.FileHandler(logfile, mode=logging_mode)
            handler.setFormatter(formatter)
            level_file = self.data['parameters/logging/level_file']
            logging_level = get_loglevel_from_name(level_file)
            handler.setLevel(logging_level)
            self.logger.addHandler(handler)
            logging_level_min = min(logging_level_min, logging_level)
            
        self.logger.setLevel(logging_level_min)
        
        if self.data['parameters/debug/use_multiprocessing']:
            self.logger.debug('Analysis runs in process %d' % os.getpid())
            
        # setup mouse parameters as class variables
        # => the code is not thread-safe if different values for these 
        #        parameters are used in the same process
        moving_window = self.data.get('parameters/tracking/moving_window', None)
        if moving_window:
            objects.ObjectTrack.moving_window_frames = moving_window
        moving_threshold = self.data.get('parameters/tracking/moving_threshold', None)
        if moving_threshold:
            threshold = objects.ObjectTrack.moving_window_frames*moving_threshold
            objects.ObjectTrack.moving_threshold_pixel = threshold
        
        hdf5_compression = self.data.get('parameters/output/hdf5_compression', None)
        if hdf5_compression:
            LazyHDFValue.compression = hdf5_compression
            
            
    def scale_parameters(self, factor_length=1, factor_time=1):
        """ scales the parameters in length and time """
        scale_parameters(self.data['parameters'],
                        factor_length=factor_length,
                        factor_time=factor_time)
            
            
    @property
    def debug_enabled(self):
        """ return True if this is a debug run """
        return self.logger.isEnabledFor(logging.DEBUG)
            

    def get_folder(self, folder):
        """ makes sure that a folder exists and returns its path """
        base_folder = self.data['parameters/base_folder']
        try:
            # get parameter key from folder name
            data_key = self.folder_lut[folder]
            folder = os.path.join(base_folder, self.data[data_key])
        except KeyError: 
            self.logger.warn('Requested unknown folder `%s`.' % folder)

        folder = os.path.abspath(folder)
        ensure_directory_exists(folder)
        return folder


    def get_filename(self, filename, folder=None):
        """ returns a filename, optionally with a folder prepended """
        if self.name: 
            filename = self.name + '_' + filename
        else:
            filename = filename
        
        # check the folder
        if folder is None:
            return filename
        else:
            return os.path.join(self.get_folder(folder), filename)
      

    def get_code_status(self):
        """ returns a dictionary with information about the current version of
        the code in the local git repository """
        code_status = {}
        
        def get_output(cmd):
            try:
                output = subprocess.check_output(cmd, stderr=DEVNULL)
            except (OSError, subprocess.CalledProcessError):
                output = None
            return output        
        
        # go to root of project
        folder, _ = os.path.split(__file__)
        folder = os.path.abspath(os.path.join(folder, '..'))
        with change_directory(folder):
            # get number of commits
            commit_count = get_output(['git', 'rev-list', 'HEAD', '--count'])
            if commit_count is None:
                output = get_output(['git', 'rev-list', 'HEAD', '--count'])
                if output is not None:
                    commit_count = int(output.count('\n'))
            else:
                commit_count = int(commit_count.strip())
            code_status['commit_count'] = commit_count
    
            # get the current revision
            revision = get_output(['git', 'rev-parse', 'HEAD'])
            if revision is not None:
                revision = revision.splitlines()[0]
            code_status['revision'] = revision
            
            # get the date of the last change
            last_change = get_output(['git', 'show', '-s', r'--format=%ci'])
            if last_change is not None:
                last_change = last_change.splitlines()[0]
            code_status['last_change'] = last_change
        
        return code_status
    
    
    def load_video(self, video=None, crop_video=True, cropping_rect=None,
                   skip_frames=0):
        """ loads the video and applies a monochrome and cropping filter """
        # initialize the video
        if video is None:
            video_filename_pattern = os.path.join(self.data['parameters/base_folder'],
                                                  self.data['parameters/video/filename_pattern'])
            self.video = load_any_video(video_filename_pattern)
                
        else:
            self.video = video
            video_filename_pattern = None

        # save some data about the video
        video_info = {'frame_count': self.video.frame_count,
                      'size': '%d x %d' % tuple(self.video.size),
                      'fps': self.video.fps}
        try:
            video_info['filecount'] = self.video.filecount
        except AttributeError:
            video_info['filecount'] = 1
            
        self.data.create_child('video', video_info)
        self.data['video/filename_pattern'] = video_filename_pattern 

        # restrict the analysis to an interval of frames
        frames = self.data.get('parameters/video/frames', None)
        frames_skip = self.data.get('parameters/video/frames_skip', 0)
        if frames is None:
            frames = (frames_skip, self.video.frame_count)
        if skip_frames > 0:
            frames[0] += skip_frames
        if 0 < frames[0] or frames[1] < self.video.frame_count:
            self.video = self.video[frames[0]:frames[1]]
            
        video_info['frames'] = frames

        if cropping_rect is None:
            cropping_rect = self.data.get('parameters/video/cropping_rect', None)

        # restrict video to green channel if it is a color video
        color_channel = 'green' if self.video.is_color else None
        video_info['color_channel'] = color_channel

        if crop_video and cropping_rect is not None:
            if isinstance(cropping_rect, str):
                # crop according to the supplied string
                self.video = FilterCrop(self.video, region=cropping_rect,
                                        color_channel=color_channel)
            else:
                # crop to the given rect
                self.video = FilterCrop(self.video, rect=cropping_rect,
                                        color_channel=color_channel)
                
            video_info['cropping_rect'] = cropping_rect
                
        else: # user_crop is not None => use the full video
            if color_channel is None:
                self.video = self.video
            else:
                self.video = FilterMonochrome(self.video, color_channel)

            video_info['cropping_rect'] = None
        
        return video_info


    @cached_property
    def data_lastmodified(self):
        """ returns the time at which the data was last modified """
        # try reading the time stamp from the data
        try:
            last_update_str = self.data.get['analysis-status/updated_last']
        except (KeyError, TypeError):
            last_update = None
        else:
            # use dateutil if present, otherwise fall back to datetime
            if dateutil:
                last_update = dateutil.parser.parse(last_update_str)
            else:
                last_update = datetime.datetime.strptime(last_update_str,
                                                         "%Y-%m-%d %H:%M:%S")
        
        if not last_update:
            # try reading the last-modified timestamp from the yaml file
            filename = self.get_filename('results.yaml', 'results')
            try:
                ts = os.path.getmtime(filename)
                last_update = datetime.datetime.fromtimestamp(ts)
            except IOError:
                # the data has not been written and is thus brand new  
                last_update = datetime.datetime.now()
        
        return last_update
    
    
    def get_cage(self):
        """ returns an object representing the cage """
        try:
            cage_size = self.data['pass1/video/size']
            width, height = [int(v) for v in cage_size[0].split('x')]
            cage = objects.Cage(0, 0, width, height)
        except KeyError:
            cage = None
        return cage

            
    def write_data(self):
        """ writes the results to a file """

        self.logger.info('Started writing out all data.')

        # prepare writing the data
        main_result = self.data.copy()
        main_result['analysis-status/updated_last'] = str(datetime.datetime.now())
        
        # write large amounts of data to accompanying hdf file
        hdf_filename= self.get_filename('results.hdf5', 'results')
        for key, cls in self.hdf_values.iteritems():
            if key in main_result:
                # get the value, but don't load it from HDF file
                # This prevents unnecessary read/write cycles
                value = main_result.get_item(key, load_data=False)
                if not isinstance(value, LazyHDFValue):
                    assert cls == value.__class__
                    obj = cls.storage_class.create_from_data(key, value,
                                                             hdf_filename)
                    main_result[key] = obj
        
        # write the main result file to YAML
        filename = self.get_filename('results.yaml', 'results')
        with open(filename, 'w') as outfile:
            yaml.dump(prepare_data_for_yaml(main_result),
                      outfile,
                      default_flow_style=False,
                      indent=4) 
       
                        
    def read_data(self):
        """ read the data from result file """
        
        # read the main result file and copy data into internal dictionary
        filename = self.get_filename('results.yaml', 'results')
        self.logger.info('Read YAML data from %s', filename)
        
        with open(filename, 'r') as infile:
            yaml_content = yaml.load(infile)
            
        if yaml_content: 
            self.data.from_dict(yaml_content)
        else:
            raise ValueError('Result file is empty.')
        
        # initialize the parameters read from the YAML file
        # but overwrite the user supplied parameters before that
        self.initialize_parameters(self.parameters_user)
        
        # initialize the loaders for values stored elsewhere
        hdf_folder = self.get_folder('results')
        for key, data_cls in self.hdf_values.iteritems():
            if key in self.data:
                value = self.data.get_item(key, load_data=False) 
                storage_cls = data_cls.storage_class
                if isinstance(value, LazyHDFValue):
                    value.set_hdf_folder(hdf_folder)
                else:
                    lazy_loader = storage_cls.create_from_yaml_string(self.data[key],
                                                                      data_cls,
                                                                      hdf_folder)
                    self.data[key] = lazy_loader
        
        self.logger.info('Read previously calculated data from files.')

 
    def close(self):
        """ close all resources hold by this class.
        Currently, this is only the logging facility """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
        
    
    def __del__(self):
        self.close()
        

        