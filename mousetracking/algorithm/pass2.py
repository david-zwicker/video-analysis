'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

from .data_handler import DataHandler


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
        obj.pass1 = first_pass.data['pass1'] 
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
        pass