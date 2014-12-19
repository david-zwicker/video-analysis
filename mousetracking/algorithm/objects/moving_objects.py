'''
Created on Aug 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Holds classes that describe moving objects.
Note that we only identify the mouse in the second pass of the tracking. 
'''

from __future__ import division

import itertools

import numpy as np
from scipy.ndimage import filters

from data_structures.lazy_values import LazyHDFCollection
from video.analysis import curves
from data_structures.cache import cached_property

from .. import debug  # @UnusedImport



class MovingObject(object):
    """ represents a single object by its position and size.
    The label is used to distinguish different objects in the detection phase """
    __slots__ = ['pos', 'size', 'label'] #< save some memory
    
    def __init__(self, pos, size, label=None):
        self.pos = (int(pos[0]), int(pos[1]))
        self.size = size
        self.label = label
        
    
    def __repr__(self):
        if self.label:
            return 'MovingObject((%d, %d), %d, %s)' % (self.pos + (self.size, self.label))
        else:
            return 'MovingObject((%d, %d), %d)' % (self.pos + (self.size,))



class ObjectTrack(object):
    """ represents a time course of moving objects """
    column_names = ('Time', 'Position X', 'Position Y', 'Object Area')
    mouse_area_mean = 700
    
    moving_window_frames = 20
    moving_threshold_pixel = 20*10
    
    def __init__(self, times=None, objects=None):
        self.times = [] if times is None else times
        self.objects = [] if objects is None else objects
        
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(time=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(timespan=%d..%d)' % (self.times[0], self.times[-1])
        
        
    def __len__(self): return len(self.times)
    def __getitem__(self, *args): return self.objects.__getitem__(*args)
    
    
    @property
    def start(self): return self.times[0]
    @property
    def end(self): return self.times[-1]
    @property
    def duration(self): return self.times[-1] - self.times[0]
    @property
    def first(self): return self.objects[0]
    @property
    def last(self): return self.objects[-1]
    
    def __iter__(self):
        return itertools.izip(self.times, self.objects)

    
    @cached_property
    def mouse_score(self):
        """ return a score of how likely this trace represents a mouse
        The returned value ranges from 0 to 1
        """
        mean_area = np.mean([obj.size for obj in self.objects])
        area_score = np.exp(-2*(1 - mean_area/self.mouse_area_mean)**2)
        return area_score
    
        
    def get_pos(self, time):
        """ returns the position at a specific time """
        try:
            idx = self.times.index(time)
        except AttributeError:
            # assume that self.times is a numpy array
            idx = np.nonzero(self.times == time)[0][0]
        return self.objects[idx].pos
        
    
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
        
        
    def get_track(self, start=None, end=None, step=None):
        """ return a list of positions over time """
        s = slice(start, end, step)
        return [obj.pos for obj in self.objects[s]]


    def get_trajectory(self, smoothing=0):
        """ returns a numpy array of positions over time """
        trajectory = np.array([obj.pos for obj in self.objects])
        if smoothing:
            filters.gaussian_filter1d(trajectory, output=trajectory,
                                      sigma=smoothing, axis=0, mode='nearest')
        return trajectory


    def append(self, time, obj):
        """ append a new object with a time code """
        self.times.append(time)
        self.objects.append(obj)
        
        
    def is_moving(self):
        """ return if the object has moved in the last frames """
        dist = curves.curve_length(self.get_track(-self.moving_window_frames, None))
        return dist > self.moving_threshold_pixel
    
    
    def overlaps(self, other):
        """ returns True if the other ObjectTrack overlaps with the current one """
        s0, s1 = self.time[0], self.time[-1]
        o0, o1 = other[0], other[-1]
        return (s0 <= o1 and o0 <= s1)
    
    
    def split(self, split_times):
        """ splits the current track into chunks separated by the given split_times """
        split_indices = np.asarray(split_times) - self.start
        chunks = np.split(self.times, split_indices)
        idx, result = 0, []
        for chunk in chunks:
            track = ObjectTrack(chunk, self.objects[idx: idx+len(chunk)])
            result.append(track)
            idx += len(chunk)
        return result
    
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        return np.array([(time, obj.pos[0], obj.pos[1], obj.size)
                         for time, obj in itertools.izip(self.times, self.objects)],
                        dtype=np.int32)


    @classmethod
    def create_from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        objects = [MovingObject(pos=(d[1], d[2]), size=d[3]) for d in data]
        return cls([d[0] for d in data], objects)


    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        if key in hdf_file:
            del hdf_file[key]
        hdf_file.create_dataset(key, data=self.to_array(), track_times=True)


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.create_from_array(hdf_file[key])
       
       
       
class ObjectTrackList(list):
    """ organizes a list of ObjectTrack instances """
    item_class = ObjectTrack
    storage_class = LazyHDFCollection
    hdf_attributes = {'column_names': ObjectTrack.column_names}
    
    duration_min = 2 #< minimal duration of a track to be considered
    

    def __getitem__(self, item):
        result = super(ObjectTrackList, self).__getitem__(item)
        if isinstance(item, slice):
            return ObjectTrackList(result)
        else:
            return result
    

    def __getslice__(self, i, j):
        return ObjectTrackList(super(ObjectTrackList, self).__getslice__(i, j))

    
    def insert(self, index, item):
        if item.duration >= self.duration_min:
            super(ObjectTrackList, self).insert(index, item)
    
    
    def extend(self, items):
        super(ObjectTrackList, self).extend(item for item in items
                                            if item.duration >= self.duration_min)
    
    
    def append(self, item):
        if item.duration >= self.duration_min:
            super(ObjectTrackList, self).append(item)


    def insert_sorted(self, item, index_min=0):
        """ inserts a new item into the sorted list.
        Assumes that the internal list is already sorted.
        index_min can optionally indicate a minimal index beyond
        which the item will be insert. Supplying this option can
        increase the insertion process. 
        """
        if len(item) > 0:
            for k, track in enumerate(self[index_min:], index_min):
                if item.start <= track.start:
                    self.insert(k, item)
                    break
            else:
                self.append(item)


    def break_long_tracks(self, duration_cutoff, excluded_tracks=None):
        """ breaks apart long tracks and stores the chunks """
        if excluded_tracks is None:
            excluded_tracks = set()
        else:
            excluded_tracks = set(excluded_tracks)
        
        k1 = 0
        # iterate over changing list `self`
        while k1 < len(self):
            track1 = self[k1]
            if track1 in excluded_tracks or track1.duration < duration_cutoff:
                # track is excluded or too short => check next one
                k1 += 1
                continue
            
            # check against overlapping tracks
            for k2, track2 in enumerate(self[k1 + 1:], k1 + 1):
                if track2.start >= track1.end - duration_cutoff:
                    # there won't be any overlapping tracks 
                    break #< check the next track1
                if track2 in excluded_tracks:
                    continue #< skip this track
                if track2.duration >= duration_cutoff:
                    # both tracks are long and they overlap => split them
                    track1s, track2s = [], [] #< split tracks
                    if track1.start == track2.start:
                        if track1.end < track2.end:
                            track2s = track2.split([track1.end + 1])
                        elif track2.end < track1.end:
                            track1s = track1.split([track2.end + 1])
                        # else track1.end == track2.end and we don't do anything

                    # track1.start < track2.start, because tracks are sorted
                    elif track1.end < track2.end:
                        track1s = track1.split([track2.start])
                        track2s = track2.split([track1.end + 1])
                    elif track1.end == track2.end:
                        track1s = track1.split([track2.start])
                    else: # track1.end > track2.end
                        track1s = track1.split([track2.start, track2.end + 1])
                        
                    if track1s or track2s:
                        # delete tracks that have been split
                        if track2s: del self[k2] #< has to be deleted before k1!
                        if track1s: del self[k1]
                                
                        # insert the split tracks
                        for track in itertools.chain(track1s, track2s):
                            self.insert_sorted(track, index_min=k1)
                        
                        k1 -= 1 #< the track at k1 might have been replaced => check it again 
                        break #< check the next track1

            # check the next track
            k1 += 1


