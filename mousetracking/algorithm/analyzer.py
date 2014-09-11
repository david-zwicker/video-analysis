'''
Created on Sep 11, 2014

@author: zwicker

Contains a class that can be used to analyze results from the tracking
'''

import numpy as np

from data_handler import DataHandler


class Analyzer(DataHandler):
    
    def __init__(self, *args, **kwargs):
        super(Analyzer, self).__init__(*args, **kwargs)
        
        self.time_scale = self.data['video/analyzed/fps']
        
    
    def get_burrow_lengths(self):
        """ returns a list of burrows containing their length over time """
        burrow_tracks = self.data['pass1/burrows/tracks']
        results = []
        for burrow_track in burrow_tracks:
            times = np.asarray(burrow_track.times)/self.time_scale
            lenghts = [burrow.length for burrow in burrow_track.burrows]
            data = np.c_[times, lenghts]
            results.append(data)
                  
        return results
            