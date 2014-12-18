'''
Created on Dec 18, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import datetime

import cv2
import yaml

from .data_handler import DataHandler


class PassBase(DataHandler):
    """ base class for the actual video analysis passes """
    
    pass_id_replacements = {'first': '1',
                            'second': '2',
                            'third': '3',
                            'fourth': '4'}
    
    
    def __init__(self, *args, **kwargs):
        super(PassBase, self).__init__(*args, **kwargs)
        self.initialize_pass()


    def initialize_pass(self):
        """ initialize values necessary for this run """
        self.params = self.data['parameters']
        self.result = self.data.create_child(self.pass_name)
        self.result['code_status'] = self.get_code_status()
        self.debug = {}
        self._cache = {}
        if self.params['debug/output'] is None:
            self.debug_output = []
        else:
            self.debug_output = self.params['debug/output']
            
    
    def blur_image(self, image):
        """ returns a blurred version of the image """
        blur_sigma = self.params['video/blur_radius']
        blur_sigma_color = self.params['video/blur_sigma_color']

        if blur_sigma_color == 0:
            image_blurred = cv2.GaussianBlur(image, ksize=(0, 0),
                                             sigmaX=blur_sigma,
                                             borderType=cv2.BORDER_REPLICATE)
        
        else:
            image_blurred = cv2.bilateralFilter(image, d=int(blur_sigma),
                                                sigmaColor=blur_sigma_color,
                                                sigmaSpace=blur_sigma)
            
        return image_blurred
            
    
    def log_event(self, description):
        """ stores and/or outputs the time and date of the event given by name """
        self.logger.info(description)
        
        # save the event in the result structure
        if 'event_log' not in self.data:
            self.data['event_log'] = []
        event = str(datetime.datetime.now()) + ': ' + description 
        self.data['event_log'].append(event)

    
    def update_status(self, status_update):
        """ update the status file.
        The update is done by reading the file, updating it, and then writing it
        again. There is a possible race-condition if the surveillance system is
        doing this at the same time.
        """
        path = self.get_filename('status.yaml', 'logging')

        # read data
        try:
            with open(path, 'r') as fp:
                status = yaml.load(fp)
        except IOError:
            status = {}
            
        # update data
        status.update(status_update)
            
        # write data
        with open(path, 'w') as fp:
            yaml.dump(status, fp)
        
    
    def set_pass_status(self, **kwargs):
        """ update the status for the current pass """
        # make sure that we know the name of this pass
        pass_name = kwargs.pop('pass_name', self.pass_name)
        
        # set the current time      
        data = kwargs
        data['timestamp'] = str(datetime.datetime.now())
        
        # set the current job_id if it is given
        job_id = self.params.get('resources/%s/job_id' % pass_name, None)
        if job_id:
            data['job_id'] = job_id
        
        # write the updated status
        self.update_status({pass_name: data})
        
        