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

    passes = ['pass%d' % k for k in xrange(1, 5)]
    
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
        blur_method = self.params['video/blur_method'].lower()
        blur_sigma = self.params['video/blur_radius']
        blur_sigma_color = self.params['video/blur_sigma_color']

        if blur_method == 'mean':
            ksize = int(2*blur_sigma + 1)
            image_blurred = cv2.blur(image, ksize=(ksize, ksize),
                                     borderType=cv2.BORDER_REPLICATE)
        
        elif blur_method == 'gaussian':
            image_blurred = cv2.GaussianBlur(image, ksize=(0, 0),
                                             sigmaX=blur_sigma,
                                             borderType=cv2.BORDER_REPLICATE)
        
        elif blur_method == 'bilateral':
            image_blurred = cv2.bilateralFilter(image, d=int(blur_sigma),
                                                sigmaColor=blur_sigma_color,
                                                sigmaSpace=blur_sigma)
            
        else:
            raise ValueError('Unsupported blur method `%s`' % blur_method)
            
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
        
        # ensure integrity of the passes
        state_succ = None
        state_gen = (status[p_id] for p_id in self.passes if p_id in status) 
        for pass_state in state_gen:
            if state_succ:
                pass_state['state'] = state_succ
            elif pass_state['state'] != 'done':
                state_succ = '???'
            
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
        
        