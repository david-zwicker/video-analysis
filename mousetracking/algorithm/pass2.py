'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

from .data_handler import DataHandler


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, name='', video=None, parameters=None):
        super(SecondPass, self).__init__(name, video, parameters)
        self.params = self.data['parameters']
        #self.pass1 = self.data['pass1']        
        self.log_event('Pass 2 - Started initializing the analysis.')
        

    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        obj = cls(first_pass.name, first_pass.video)
        obj.data = first_pass.data
        obj.params = first_pass.data['parameters']
        #obj.pass1 = first_pass.data['pass1'] 
        obj.tracks = first_pass.tracks       


    def process_data(self):
        """ do the second pass of the analysis """
        self.find_mouse_track()
        #self.smooth_ground_profile()
        #self.produce_video()
        
        
    def find_mouse_track(self):
        pass