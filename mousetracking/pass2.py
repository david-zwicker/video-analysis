'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

from .data_handler import DataHandler


class SecondPass(DataHandler):
    """ class containing methods for the second pass """
    
    def __init__(self, folder, prefix='', parameters=None):
        super(SecondPass, self).__init__(folder, prefix, parameters)
        self.params = self.data['parameters']
        self.pass1 = self.data['pass1']        


    @classmethod
    def from_first_pass(cls, first_pass):
        """ create the object directly from the first pass """
        obj = cls(first_pass.folder, first_pass.prefix)
        obj.data = first_pass.data
        obj.params = first_pass.data['parameters']
        obj.pass1 = first_pass.data['pass1'] 
        obj.tracks = first_pass.tracks
        obj.burrows = first_pass.burrows       


    def process_data(self):
        """ do the second pass of the analysis """
        self.find_mouse_track()
        #self.smooth_ground_profile()
        #self.produce_video()
        
        
    def find_mouse_track(self):
        pass