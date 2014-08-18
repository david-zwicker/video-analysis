'''
Created on Aug 16, 2014

@author: zwicker
'''

from __future__ import division

import logging
import itertools
import os

import numpy as np
import yaml
import h5py

from video.io import VideoFileStack
from video.filters import FilterCrop, FilterMonochrome
from video.utils import NestedDict, ensure_directory_exists, prepare_data_for_yaml
from video.analysis.curves import point_distance


PARAMETERS_DEFAULT = {
    # number of initial frames to not analyze
    'video/ignore_initial_frames': 0,
    # radius of the blur filter [in pixel]
    'video/blur_radius': 3,
    
    # thresholds for cage dimension [in pixel]
    'cage/width_min': 650,
    'cage/width_max': 800,
    'cage/height_min': 400,
    'cage/height_max': 500,
                               
    # how often are the color estimates adapted [in frames]
    'colors/adaptation_interval': 1000,
                               
    # determines the rate with which the background is adapted [in 1/frames]
    'background/adaptation_rate': 0.01,
    
    # spacing of the points in the sand profile [in pixel]
    'sand_profile/point_spacing': 20,
    # adapt the sand profile only every number of frames [in frames]
    'sand_profile/adaptation_interval': 100,
    # width of the ridge [in pixel]
    'sand_profile/width': 5,
    
    # relative weight of distance vs. size of objects [dimensionless]
    'objects/matching_weigth': 0.5,
    # size of the window used for motion detection [in frames]
    'objects/matching_moving_window': 20,
        
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
    'burrows/adaptation_interval': 100,
    # what is a typical radius of a burrow [in pixel]
    'burrows/radius': 10
}


class DataHandler(object):
    """ class that handles the data and parameters of mouse tracking """

    video_filename_pattern = 'raw_video/*'
    
    def __init__(self, folder, prefix='', parameters=None, **kwargs):

        # initialize tracking parameters        
        self.data = NestedDict()
        self.data['parameters'].from_dict(PARAMETERS_DEFAULT)
        if parameters is not None:
            self.data['parameters'].update(parameters)

        # set extra parameters that were given
        self.data['video/frames_specified'] = kwargs.get('frames', None)
        self.data['video/region_specified'] = kwargs.get('crop', None)
            
        # initialize additional properties
        self.data['analysis-status'] = 'Initialized parameters'
        self.folder = folder
        self.prefix = prefix + '_' if prefix else ''
        
        #TODO check unused kwargs


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
      
    
    def load_video(self, cropping_rect=None):
        """ loads the video and applies a monochrome and cropping filter """
        
        # initialize video
        self.video = VideoFileStack(os.path.join(self.folder, self.video_filename_pattern))
        self.data['video/raw'].from_dict({'filename_pattern': self.video_filename_pattern,
                                          'frame_count': self.video.frame_count,
                                          'size': '%d x %d' % self.video.size,
                                          'fps': self.video.fps})
        
        # restrict the analysis to an interval of frames
        frames = self.data['video/frames_specified']
        if frames is not None:
            self.video = self.video[frames[0]:frames[1]]
        else:
            frames = (0, self.video.frame_count)
            
        if cropping_rect is None:
            # use the full video
            if self.video.is_color:
                # restrict video to green channel if it is a color video
                video_crop = FilterMonochrome(self.video, 'g')
            else:
                video_crop = self.video
                
            rect_given = [0, 0, self.video.size[0], self.video.size[1]]

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

            rect_given = video_crop.rect
            
        return video_crop
            
            
    def write_data(self):
        """ writes the results to a file """

        # prepare writing the data
        main_result = self.data.copy()
        hdf_name = self.get_filename('results.hdf5')
        hdf_file = h5py.File(self.get_filename('results.hdf5', 'results'), 'w')
        
        def write_object_list(key, has_index=False, column_names=None):
            """ helper function that writes a list of objects to HDF5.
            If has_index is True, the first column of the array returned by calling
            to_array() on each object serves as an index to distinguish the object.
            """
            # turn the list of objects into a numpy array
            if has_index:
                result = [obj.to_array() for obj in main_result[key]]
                
            else:
                result = []
                for index, obj in enumerate(main_result[key]):
                    data = obj.to_array()
                    index_array = np.zeros((len(obj), 1), np.int) + index
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
            main_result[key] = hdf_name + ':' + dataset.name.encode('ascii', 'replace')
        
        
        # write large data to HDF5
        write_object_list('pass1/sand_profile', True,
                          ['Frame ID', 'Position X', 'Position Y'])

        write_object_list('pass1/objects/tracks', False,
                          ['Frame ID', 'Position X', 'Position Y', 'Object Area'])
        
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
                    
        def read_object_list(key, cls=tuple):
            """ function that reads lists of objects from the linked HDF5 file """
            # read the link
            data_str = self.data[key]
            hdf_filename, dataset = data_str.split(':')
            
            # open the associated HDF5 file
            hdf_filepath = os.path.join(self.get_folder('results'), hdf_filename)
            hdf_file = h5py.File(hdf_filepath, 'r')
            
            # check whether the first column has been prepended automatically
            if hdf_file[dataset].attrs['column_names'][0] == 'Automatic Index':
                data_start = 1
            else:
                data_start = 0
            
            # iterate over the data and create objects from it
            self.data[key] = []
            index, obj_data = 0, []
            for line in hdf_file[dataset]:
                if line[0] == index:
                    # append object to the current track
                    obj_data.append(line[data_start:])
                else:
                    # save the track and start a new one
                    if obj_data:
                        self.data[key].append(cls.from_array(obj_data))
                    obj_data = [line[data_start:]]
                    index = line[0]
            
            self.data[key].append(cls.from_array(obj_data))


        read_object_list('pass1/sand_profile', SandProfile)
        read_object_list('pass1/objects/tracks', ObjectTrack)
        
 
    #===========================================================================
    # DATA ANALYSIS
    #===========================================================================
        
    def mouse_underground(self, position):
        """ checks whether the mouse is under ground """
        sand_y = np.interp(position[0], self.sand_profile[:, 0], self.sand_profile[:, 1])
        return position[1] - self.params['mouse.model_radius']/2 > sand_y


       
class Object(object):
    """ represents a single object by its position and size """
    __slots__ = ['pos', 'size'] #< save some memory
    
    def __init__(self, pos, size):
        self.pos = (int(pos[0]), int(pos[1]))
        self.size = size



class ObjectTrack(object):
    """ represents a time course of objects """
    # TODO: hold everything inside lists, not list of objects
    # TODO: speed up by keeping track of velocity vectors
    
    def __init__(self, time=None, obj=None, moving_window=20):
        self.times = [] if time is None else [time]
        self.objects = [] if obj is None else [obj]
        self.moving_window = moving_window
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(time=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(time=%d..%d)' % (self.times[0], self.times[-1])
        
    def __len__(self):
        return len(self.times)
    
    @property
    def last_pos(self):
        """ return the last position of the object """
        return self.objects[-1].pos
    
    @property
    def last_size(self):
        """ return the last size of the object """
        return self.objects[-1].size
    
    def predict_pos(self):
        """ predict the position in the next frame.
        It turned out that setting the current position is the best predictor.
        This is because mice are often stationary (especially in complicated
        tracking situations, like inside burrows). Additionally, when mice
        throw out dirt, there are frames, where dirt + mouse are considered 
        being one object, which moves the center of mass in direction of the
        dirt. If in the next frame two objects are found, than it is likely
        that the dirt would be seen as the mouse, if we'd predict the position
        based on the continuation of the previous movement
        """
        return self.objects[-1].pos
        
    def get_track(self):
        """ return a list of positions over time """
        return [obj.pos for obj in self.objects]

    def append(self, time, obj):
        """ append a new object with a time code """
        self.times.append(time)
        self.objects.append(obj)
        
    def is_moving(self):
        """ return if the object has moved in the last frames """
        pos = self.objects[-1].pos
        dist = sum(point_distance(pos, obj.pos)
                   for obj in self.objects[-self.moving_window:])
        return dist > 5*self.moving_window
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        return np.array([(time, obj.pos[0], obj.pos[1], obj.size)
                         for time, obj in itertools.izip(self.times, self.objects)],
                        dtype=np.int)

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        res = cls()
        res.times = [d[0] for d in data]
        res.objects = [Object(pos=(d[1], d[2]), size=d[3]) for d in data]
        return res



class SandProfile(object):
    """ dummy class representing a single sand profile at a certain point
    in time """
    
    def __init__(self, time, points):
        self.time = time
        self.points = points
        
    def __repr__(self):
        return 'SandProfile(time=%d, %d points)' % (self.time, len(self.points))
        
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        time_array = np.zeros((len(self.points), 1), np.int) + self.time
        return np.hstack((time_array, self.points))

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        data = np.asarray(data)
        return cls(data[0, 0], data[1:, :])
        
       
        
        
        