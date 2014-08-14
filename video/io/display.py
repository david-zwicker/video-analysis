'''
Created on Aug 14, 2014

@author: zwicker
'''

from __future__ import division

from multiprocessing import Process, Pipe
import logging
import numpy as np

import cv2

try:
    import sharedmem
except ImportError:
    sharedmem = None
        
        
        
def _show_image_from_pipe(pipe, image_array, title):
    """ function that runs in a separate process to display images """
    while True:
        # read next command from pipe
        data = pipe.recv()
        
        if data == 'update':
            # update the image
            cv2.imshow(title, image_array)
            # check whether the user wants to abort
            if cv2.waitKey(1) & 0xFF in {27, ord('q')}:
                pipe.send('interrupt')
                break
                
        elif data == 'close':
            # break out of the loop if asked so
            break
        
        else:
            raise ValueError('Unknown command `%s`' % data)
        
    # cleanup
    pipe.close()
    cv2.destroyWindow(title)


        
class ImageShow(object):
    """ class that can show an image """
    
    def __init__(self, size, title='', multiprocessing=True):
        self.title = title
        self._data = None
        self._proc = None
        
        if multiprocessing:
            
            if sharedmem:
                try:
                    # create the pipe to talk to the child
                    self._pipe, pipe_child = Pipe(duplex=True)
                    # setup the shared memory area
                    self._data = sharedmem.empty(size, np.uint8)
                    # initialize the process that shows the image
                    self._proc = Process(target=_show_image_from_pipe,
                                         args=(pipe_child, self._data, title))
                    self._proc.start()
                    logging.debug('Started background process for displaying images')
                    
                except AssertionError:
                    logging.warn('Could not start a separate process to display images. '
                                 'The main process will thus be used.')
                
            else:
                logging.warn('Package sharedmem could not be imported and '
                             'images are thus shown using the main process.')

       
       
    def show(self, image):
        """ show an image.
        May raise KeyboardInterrupt, if the user opted to exit
        """ 
        if self._proc:
            # copy data to shared memory
            self._data[:] = image
            # tell the process to update window
            self._pipe.send('update')
            # check whether the user wants to quit
            if self._pipe.poll() and self._pipe.recv() == 'interrupt':
                raise KeyboardInterrupt

        else:
            # update the image
            cv2.imshow(self.title, image)
            # check whether the user wants to quit
            if cv2.waitKey(1) & 0xFF in {27, ord('q')}:
                raise KeyboardInterrupt
        
        
    def __del__(self):
        if self._proc:
            # shut down the process
            self._pipe.send('close')
            self._proc.join()
            self._pipe.close()
        else:
            # delete the opencv window
            cv2.destroyWindow(self.title)
