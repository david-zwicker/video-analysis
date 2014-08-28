'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import numpy as np

from .data_handler import DataHandler
from video.analysis import curves


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, name='', parameters=None):
        super(SecondPass, self).__init__(name, parameters)
        self.initialize_parameters()
        self.params = self.data['parameters']
        self.log_event('Pass 2 - Started initializing the analysis.')
        

    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        # create the data and copy the data from first_pass
        obj = cls(first_pass.name)
        obj.data = first_pass.data
        obj.params = first_pass.data['parameters']
        obj.tracks = first_pass.tracks
        # initialize parameters
        obj.initialize_parameters()
        return obj


    def process_data(self):
        """ do the second pass of the analysis """
        self.find_mouse_track()
        #self.smooth_ground_profile()
        #self.produce_video()
        
        
    def find_mouse_track(self):
        """ identifies the mouse trajectory by connecting several object
        tracks """
        # retrieve data
        tracks = self.data['pass1/objects/tracks']

        # find the longest track as a basis for finding the complete track
        start = np.argmax([len(track) for track in tracks])

        def score_connection(left, right):
            """ scoring function that defines how well the two tracks match """
            obj_l, obj_r = left.objects[-1], right.objects[0] #< the respective objects
            
            # calculate the distance between new and old objects
            dist_score = curves.point_distance(obj_l.pos, obj_r.pos)
            # normalize distance to the maximum speed
            dist_score /= self.params['mouse/max_speed']
            # calculate the difference of areas between new and old objects
            area_score = abs(obj_l.size - obj_r.size)/(obj_l.size + obj_r.size)

            # build a combined score from this
            alpha = self.params['objects/matching_weigth']
            score = alpha*dist_score + (1 - alpha)*area_score
            
            # score should decrease for larger time differences
            dt = right.times[0] - left.times[-1] #< time difference
            time_scale = self.params['tracking/time_scale'] 
            return score*np.exp(-dt/time_scale)
            
            
        result = [track] #< list of final tracks
        score = {} #< dictionary of scores for potential matches
        for k, track in enumerate(tracks[start:], start):
            score[k] = score_connection(result[-1], track)
            
        return result
                
        
