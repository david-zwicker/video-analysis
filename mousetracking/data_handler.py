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

from .mouse_objects import ObjectTrack, GroundProfile, Burrow
from video.io import VideoFileStack
from video.filters import FilterCrop, FilterMonochrome
from video.utils import ensure_directory_exists, prepare_data_for_yaml


PARAMETERS_DEFAULT = {
    # filename pattern used to look for videos
    'video/filename_pattern': 'raw_video/*',
    # number of initial frames to not analyze
    'video/ignore_initial_frames': 0,
    # radius of the blur filter [in pixel]
    'video/blur_radius': 3,

    # settings for video output    
    'video/output/extension': '.mov',
    'video/output/codec': 'libx264',
    'video/output/bitrate': '2000k',
    
    # thresholds for cage dimension [in pixel]
    'cage/width_min': 650,
    'cage/width_max': 800,
    'cage/height_min': 400,
    'cage/height_max': 500,
                               
    # how often are the color estimates adapted [in frames]
    'colors/adaptation_interval': 1000,
                               
    # determines the rate with which the background is adapted [in 1/frames]
    'background/adaptation_rate': 0.01,
    'explored_area/adaptation_rate': 1e-4,
    
    # spacing of the points in the ground profile [in pixel]
    'ground/point_spacing': 20,
    # adapt the ground profile only every number of frames [in frames]
    'ground/adaptation_interval': 100,
    # width of the ridge [in pixel]
    'ground/width': 5,
    
    # relative weight of distance vs. size of objects [dimensionless]
    'objects/matching_weigth': 0.5,
    # size of the window used for motion detection [in frames]
    'objects/matching_moving_window': 20,
    # threshold above which an objects is said to be moving [in pixels/frame]
    'objects/matching_moving_threshold': 10,
        
    # `mouse.intensity_threshold` determines how much brighter than the
    # background (usually the sky) has the mouse to be. This value is
    # measured in terms of standard deviations of the sky color
    'mouse/intensity_threshold': 1,
    # radius of the mouse model [in pixel]
    'mouse/model_radius': 25,
    # minimal area of a feature to be considered in tracking [in pixel^2]
    'mouse/min_area': 100,
    # maximal speed of the mouse [in pixel per frame]
    'mouse/max_speed': 30, 
    # maximal area change allowed between consecutive frames [dimensionless]
    'mouse/max_rel_area_change': 0.5,

    # how often are the burrow shapes adapted [in frames]
    'burrows/adaptation_interval': 10, # 100
    # what is a typical radius of a burrow [in pixel]
    'burrows/radius': 10,
    # extra number of pixel around burrow outline used for fitting [in pixel]
    'burrows/fitting_margin': 20,
    # determines how much the burrow outline might be simplified. The quantity 
    # determines by what fraction the total outline length is allowed to change 
    'burrows/outline_simplification_threshold': 0.01#0.005,
}



class DataHandler(object):
    """ class that handles the data and parameters of mouse tracking """

    LARGE_DATA = {'pass1/ground/profile': GroundProfile,
                  'pass1/objects/tracks': ObjectTrack,
                  'pass1/burrows/data': Burrow}

    def __init__(self, folder, prefix='', parameters=None):

        # initialize tracking parameters        
        self.data = Data()
        self.data.create_child('parameters')
        self.data['parameters'].from_dict(PARAMETERS_DEFAULT)
        if parameters is not None:
            self.data['parameters'].from_dict(parameters)
            
        # initialize additional properties
        self.data['analysis-status'] = 'Initialized parameters'
        self.folder = folder
        self.prefix = prefix + '_' if prefix else ''


    def get_folder(self, folder):
        """ makes sure that a folder exists and returns its path """
        if folder == 'results':
            folder = os.path.join(self.folder, 'results')
        elif folder == 'debug':
            folder = os.path.join(self.folder, 'debug')
            
        ensure_directory_exists(folder)
        return folder


    def get_filename(self, filename, folder=None):
        """ returns a filename, optionally with a folder prepended """ 
        filename = self.prefix + filename
        
        # check the folder
        if folder is None:
            return filename
        else:
            return os.path.join(self.get_folder(folder), filename)
      

    def log_event(self, description):
        """ stores and/or outputs the time and date of the event given by name """
        event = str(datetime.datetime.now()) + ': ' + description 
        logging.info(event)
        
        # save the event in the result structure
        if 'event_log' not in self.data:
            self.data['event_log'] = []
        self.data['event_log'].append(event)

    
    def load_video(self):
        """ loads the video and applies a monochrome and cropping filter """
        
        # initialize video
        video_filename_pattern = self.data['parameters/video/filename_pattern']
        self.video = VideoFileStack(os.path.join(self.folder, video_filename_pattern))
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
                video_crop = FilterMonochrome(self.video, 'g')
            else:
                video_crop = self.video
                
        else: # user_crop is not None                
            # restrict video to green channel if it is a color video
            color_channel = 1 if self.video.is_color else None
            
            if isinstance(cropping_rect, str):
                # crop according to the supplied string
                video_crop = FilterCrop(self.video, quadrant=cropping_rect,
                                        color_channel=color_channel)
            else:
                # crop to the given rect
                video_crop = FilterCrop(self.video, rect=cropping_rect,
                                        color_channel=color_channel)

        return video_crop
            
            
    def write_data(self):
        """ writes the results to a file """

        self.log_event('Started writing out all data.')

        # prepare writing the data
        main_result = self.data.copy()
        hdf_name = self.get_filename('results.hdf5')
        hdf_file = h5py.File(self.get_filename('results.hdf5', 'results'), 'w')
        
        # write large data to HDF5
        for key, cls in self.LARGE_DATA.iteritems():
            # read column_names from the underlying class
            column_names = cls.array_columns
            
            # turn the list of objects into a numpy array
            if cls.index_columns > 0:
                # the first columns are enough to separate the data
                result = [obj.to_array() for obj in main_result[key]]
                
            else:
                # we have to add an extra index to separate the data later
                result = []
                for index, obj in enumerate(main_result[key]):
                    data = obj.to_array()
                    index_array = np.zeros((data.shape[0], 1), np.int32) + index
                    result.append(np.hstack((index_array, data)))
                    
                if column_names is not None:
                    column_names = ['Automatic Index'] + column_names                    
                
            if len(result) > 0:
                result = np.concatenate(result) 
        
                # write the numpy array to HDF5
                logging.debug('Writing dataset `%s` to file `%s`', key, hdf_name)
                dataset = hdf_file.create_dataset(key, data=result)
                if column_names is not None:
                    hdf_file[key].attrs['column_names'] = column_names
                
                # replace the original data with a reference to the HDF5 data
                main_result[key] = '@%s:%s' % (hdf_name, dataset.name.encode('ascii', 'replace'))
                
            else:
                main_result[key] = []
        
        # write the main result file to YAML
        filename = self.get_filename('results.yaml', 'results')
        with open(filename, 'w') as outfile:
            yaml.dump(prepare_data_for_yaml(main_result),
                      outfile,
                      default_flow_style=False,
                      indent=4)        
        
    
    def read_data(self):
        """ read the data from result files """
        
        # read the main result file
        filename = self.get_filename('results.yaml', 'results')
        with open(filename, 'r') as infile:
            data = yaml.load(infile)
        
        # copy the data into the internal data representation
        self.data.from_dict(data)
                    
        for key, cls in self.LARGE_DATA.iteritems():
            # read the link
            data_str = self.data[key][1:] # the first character should be an @
            hdf_filename, dataset = data_str.split(':')
            
            # open the associated HDF5 file
            hdf_filepath = os.path.join(self.get_folder('results'), hdf_filename)
            hdf_file = h5py.File(hdf_filepath, 'r')
            
            # check whether the first column has been prepended automatically
            if hdf_file[dataset].attrs['column_names'][0] == 'Automatic Index':
                data_start = 1    #< data starts at first column
                index_columns = 1 #< the first column is used as an index
            else:
                data_start = 0    #< all items are considered data 
                try:
                    # try to retrieve the length of the index
                    index_columns = cls.index_columns
                except AttributeError:
                    # otherwise, it defaults to the first column
                    index_columns = 1
            
            # iterate over the data and create objects from it
            self.data[key] = []
            index, obj_data = [], []
            for line in hdf_file[dataset]:
                if line[:index_columns] == index:
                    # append object to the current track
                    obj_data.append(line[data_start:])
                else:
                    # save the track and start a new one
                    if obj_data:
                        self.data[key].append(cls.from_array(obj_data))
                    obj_data = [line[data_start:]]
                    index = line[:index_columns]
            
            self.data[key].append(cls.from_array(obj_data))

 
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


        
        
        