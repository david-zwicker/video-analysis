'''
Created on Aug 16, 2014

@author: zwicker
'''

from __future__ import division

import collections
import datetime
import logging
import os

import numpy as np
import yaml
import h5py

from .objects import ObjectTrack, GroundProfile, Burrow, BurrowTrack
from video.io import VideoFileStack
from video.filters import FilterCrop, FilterMonochrome
from video.utils import ensure_directory_exists, prepare_data_for_yaml



class DataHandler(object):
    """ class that handles the data and parameters of mouse tracking """


    def __init__(self, name='', parameters=None):

        # initialize tracking parameters        
        self.data = Data()
        self.data.create_child('parameters')

        # initialize additional properties
        self.name = name
        self.data['analysis-status'] = 'Initialized parameters'

        if parameters is not None:
            self.initialize_parameters(parameters)
        

    def initialize_parameters(self, parameters=None):
        """ initialize parameters """
        if parameters is not None:
            self.data['parameters'].from_dict(parameters)
            
        # set up logging
        self.logger = logging.getLogger(self.name)
        if self.data.get('parameters/logging/folder', None) is not None:
            logfile = self.get_filename('log.log',self.data['parameters/logging/folder'])
            handler = logging.FileHandler(logfile, mode='w')
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler) 
            
        # setup the mouse objects
        curvature_radius_max = self.data.get('parameters/burrows/curvature_radius_max', None)
        if curvature_radius_max:
            Burrow.curvature_radius_max = curvature_radius_max 
        centerline_segment_length = self.data.get('parameters/burrows/centerline_segment_length', None)
        if centerline_segment_length:
            Burrow.centerline_segment_length = centerline_segment_length
        ground_point_distance = self.data.get('parameters/burrows/ground_point_distance', None)
        if ground_point_distance:
            Burrow.ground_point_distance = ground_point_distance
            

    def get_folder(self, folder):
        """ makes sure that a folder exists and returns its path """
        if folder == 'results':
            folder = os.path.abspath(self.data['parameters/output/result_folder'])
        elif folder == 'debug':
            folder = os.path.abspath(self.data['parameters/output/video/folder'])
            
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
      

    def log_event(self, description):
        """ stores and/or outputs the time and date of the event given by name """
        event = str(datetime.datetime.now()) + ': ' + description 
        self.logger.info(event)
        
        # save the event in the result structure
        if 'event_log' not in self.data:
            self.data['event_log'] = []
        self.data['event_log'].append(event)

    
    def load_video(self, video=None):
        """ loads the video and applies a monochrome and cropping filter """
        # initialize the video
        if video is None:
            video_filename_pattern = os.path.join(self.data['parameters/video/filename_pattern'])
            self.video = VideoFileStack(video_filename_pattern)
        else:
            self.video = video
            
        # save some data about the video
        self.data.create_child('video/raw', {'frame_count': self.video.frame_count,
                                             'size': '%d x %d' % self.video.size,
                                             'fps': self.video.fps})
        try:
            self.data['video/raw/filecount'] = self.video.filecount
        except AttributeError:
            self.data['video/raw/filecount'] = 1
        
        # restrict the analysis to an interval of frames
        frames = self.data.get('parameters/video/frames', None)
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
        else:
            frames = (0, self.video.frame_count)
            
        cropping_rect = self.data.get('parameters/video/cropping_rect', None)

        if cropping_rect is None:
            # use the full video
            if self.video.is_color:
                # restrict video to green channel if it is a color video
                self.video = FilterMonochrome(self.video, 'g')
            else:
                self.video = self.video
                
        else: # user_crop is not None                
            # restrict video to green channel if it is a color video
            color_channel = 1 if self.video.is_color else None
            
            if isinstance(cropping_rect, str):
                # crop according to the supplied string
                self.video = FilterCrop(self.video, region=cropping_rect,
                                        color_channel=color_channel)
            else:
                # crop to the given rect
                self.video = FilterCrop(self.video, rect=cropping_rect,
                                        color_channel=color_channel)

            
    def write_data(self):
        """ writes the results to a file """

        self.log_event('Started writing out all data.')

        # prepare writing the data
        main_result = self.data.copy()
        
        # write large amounts of data to accompanying hdf file
        hdf_name = self.get_filename('results.hdf5')
        hdf_uri = self.get_filename('results.hdf5', 'results')
        with h5py.File(hdf_uri, 'w') as hdf_file:
            # write the ground profile                
            ground_profile = main_result['pass1/ground/profile']
            data = [obj.to_array() for obj in ground_profile]
            hdf_file.create_dataset('pass1/ground_profile', data=np.concatenate(data))
            hdf_file['pass1/ground_profile'].attrs['column_names'] = ground_profile[0].array_columns
            main_result['pass1/ground/profile'] = '@%s:pass1/ground_profile' % hdf_name
    
            # write out the object tracks
            for index, object_track in enumerate(main_result['pass1/objects/tracks']):
                object_track.save_to_hdf5(hdf_file, 'pass1/objects/%d' % index)
            if main_result['pass1/objects/tracks']:
                main_result['pass1/objects/tracks'] = '@%s:pass1/objects' % hdf_name
    
            # write out the burrow tracks
            for index, burrow_track in enumerate(main_result['pass1/burrows/data']):
                burrow_track.save_to_hdf5(hdf_file, 'pass1/burrows/%d' % index)
            if main_result['pass1/burrows/data']:
                main_result['pass1/burrows/data'] = '@%s:pass1/burrows' % hdf_name
        
        # write the main result file to YAML
        filename = self.get_filename('results.yaml', 'results')
        with open(filename, 'w') as outfile:
            yaml.dump(prepare_data_for_yaml(main_result),
                      outfile,
                      default_flow_style=False,
                      indent=4)       
            
            
    def load_object_collection_from_hdf(self, key, cls): 
        """ loads a list of data objects from the accompanied HDF file """
        
        # read the link
        assert self.data[key][0] == '@'
        data_str = self.data[key][1:] # strip the first character, which should be an @
        hdf_filename, dataset = data_str.split(':')
        
        # open the associated HDF5 file
        hdf_filepath = os.path.join(self.get_folder('results'), hdf_filename)
        with h5py.File(hdf_filepath, 'r') as hdf_file:
            # iterate over the data and create objects from it
            self.data[key] = []
            for subset in hdf_file[dataset].itervalues():
                print subset
                obj = cls.from_array(subset)
                self.data[key].append(obj)
                        
                        
    def load_object_list_from_hdf(self, key, cls): 
        """ load a data object from the accompanied HDF file """
        
        # read the link
        assert self.data[key][0] == '@'
        data_str = self.data[key][1:] # strip the first character, which should be an @
        hdf_filename, dataset = data_str.split(':')
        
        # open the associated HDF5 file
        hdf_filepath = os.path.join(self.get_folder('results'), hdf_filename)
        with h5py.File(hdf_filepath, 'r') as hdf_file:
            self.data[key] = []
            index, obj_data = None, None
            # iterate over the data and create objects from it
            for line in hdf_file[dataset]:
                if line[0] == index:
                    # append object to the current track
                    obj_data.append(line)
                else:
                    # save the track and start a new one
                    if obj_data:
                        self.data[key].append(cls.from_array(obj_data))
                    obj_data = [line]
                    index = line[0]
            
            if obj_data:
                self.data[key].append(cls.from_array(obj_data))
            
                        
    def read_data(self, load_from_hdf=True):
        """ read the data from result files.
        If load_from_hdf is False, the data from the HDF file is not loaded.
        """
        
        # read the main result file
        filename = self.get_filename('results.yaml', 'results')
        with open(filename, 'r') as infile:
            data = yaml.load(infile)
            
        # initialize the parameters read from the YAML file
        self.initialize_parameters()
        
        # copy the data into the internal data representation
        self.data.from_dict(data)
        
        # load additional data if requested
        if load_from_hdf:
            self.load_object_list_from_hdf('pass1/ground/profile', GroundProfile)            
            self.load_object_collection_from_hdf('pass1/objects/tracks', ObjectTrack)
            self.load_object_collection_from_hdf('pass1/burrows/data', BurrowTrack)

 
    #===========================================================================
    # DATA ANALYSIS
    #===========================================================================
        
    def mouse_underground(self, position):
        """ checks whether the mouse is under ground """
        ground_y = np.interp(position[0], self.ground[:, 0], self.ground[:, 1])
        return position[1] - self.params['mouse.model_radius']/2 > ground_y



class Data(collections.MutableMapping):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key:
    
    d = Data({'a': {'b': 1}})
    
    d['a/b']
    >>>> 1
    
    d['c/d'] = 2
    
    d
    >>>> {'a': {'b': 1}, 'c': {'d': 2}}
    """
    
    sep = '/'
    
    def __init__(self, data=None):
        self.data = {}
        if data is not None:
            self.from_dict(data)
    
    
    def __getitem__(self, key):
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            return self.data[parent][rest]
        else:
            return self.data[key]
        
        
    def __setitem__(self, key, value):
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            try:
                self.data[parent][rest] = value
            except KeyError:
                # create new child if it does not exists
                child = Data()
                child[rest] = value
                self.data[parent] = child
        else:
            self.data[key] = value
    
    
    def __delitem__(self, key):
        if self.sep in key:
            parent, rest = key.split(self.sep, 1)
            del self.data[parent][rest]
        else:
            del self.data[key]


    # Miscellaneous dictionary methods are just mapped to data
    def __len__(self): return len(self.data)
    def __iter__(self): return self.data.__iter__()
    def keys(self): return self.data.keys()
    def values(self): return self.data.values()
    def items(self): return self.data.items()
    def iterkeys(self): return self.data.iterkeys()
    def itervalues(self): return self.data.itervalues()
    def iteritems(self): return self.data.iteritems()
    def clear(self): self.data.clear()
           
            
    def __repr__(self):
        return 'Data(' + repr(self.data) + ')'


    def create_child(self, key, values=None):
        """ creates a child dictionary and fills it with values """
        self[key] = self.__class__(values)
        return self[key]


    def copy(self):
        """ makes a shallow copy of the data """
        res = Data()
        for key, value in self.iteritems():
            if isinstance(value, dict):
                value = value.copy()
            res[key] = value
        return res


    def from_dict(self, data):
        """ fill the object with data from a dictionary """
        for key, value in data.iteritems():
            if isinstance(value, dict):
                self[key] = Data(value)
            else:
                self[key] = value

            
    def to_dict(self):
        """ convert object to a nested dictionary structure """
        res = {}
        for key, value in self.iteritems():
            if isinstance(value, Data):
                value = value.to_dict()
            res[key] = value
        return res

    
    def pprint(self, *args, **kwargs):
        """ pretty print the current structure as nested dictionaries """
        from pprint import pprint
        pprint(self.to_dict(), *args, **kwargs)


        
        
        