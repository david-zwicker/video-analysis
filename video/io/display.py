'''
Created on Aug 14, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys
import multiprocessing as mp
import logging

import numpy as np

import cv2

try:
    import sharedmem
except ImportError:
    sharedmem = None
        
        
logger = logging.getLogger('video.io')

        
def _show_image_from_pipe(pipe, image_array, title):
    """ function that runs in a separate process to display images """
    cv2.namedWindow(title)
    cv2.waitKey(10)

    show_image = True
    try:
        while show_image:
            # read next command from pipe
            command = pipe.recv()
            while pipe.poll():
                command = pipe.recv()
                if command == 'close':
                    break
            
            # process the last command
            if command == 'update':
                # update the image
                cv2.imshow(title, image_array)
                
                # check whether the user wants to abort
                while True:
                    # waitKey also handles other GUI events and we thus call it
                    # until everything is handled
                    key = cv2.waitKey(1)
                    if key & 0xFF in {27, ord('q')}:
                        pipe.send('interrupt')
                        show_image = False
                    elif key == -1:
                        break
    
            elif command == 'close':
                break
                    
            else:
                raise ValueError('Unknown command `%s`' % command)
            
    except KeyboardInterrupt:
        pipe.send('interrupt')
        
    # cleanup
    #pipe.close()
    cv2.destroyWindow(title)


        
class ImageShow(object):
    """ class that can show an image """
    
    def __init__(self, size, title='', multiprocessing=None):
        self.title = title
        self._proc = None
        
        # multiprocessing does not work in current MacOS OpenCV
        if multiprocessing is None:
            multiprocessing = (sys.platform != "darwin")
        
        if multiprocessing:
            
            if sharedmem:
                try:
                    # create the pipe to talk to the child
                    self._pipe, pipe_child = mp.Pipe(duplex=True)
                    # setup the shared memory area
                    self._data = sharedmem.empty(size, np.uint8)
                    # initialize the process that shows the image
                    self._proc = mp.Process(target=_show_image_from_pipe,
                                            args=(pipe_child, self._data, title))
                    self._proc.daemon = True
                    self._proc.start()
                    logger.debug('Started background process for displaying images')
                    
                except AssertionError:
                    logger.warn('Could not start a separate process to display images. '
                                'The main process will thus be used.')
                
            else:
                logger.warn('Package sharedmem could not be imported and '
                            'images are thus shown using the main process.')

       
       
    def show(self, image):
        """ show an image.
        May raise KeyboardInterrupt, if the user opted to exit
        """ 
        if image.ndim > 2:
            # reverse the color axis, to get BGR image required by OpenCV
            image = image[:, :, ::-1]
        
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
            cv2.imshow(self.title, image.astype(np.uint8))
            
            # check whether the user wants to abort
            while True:
                # waitKey also handles other GUI events and we thus call it
                # until everything is handled
                key = cv2.waitKey(1)
                if key & 0xFF in {27, ord('q')}:
                    raise KeyboardInterrupt
                elif key == -1:
                    break
        
        
    def close(self):
        if self._proc is not None:
            # shut down the process
            self._pipe.send('close')
            self._proc.join()
            self._pipe.close()
            self._proc = None
            
        else:
            # delete the opencv window
            cv2.destroyWindow(self.title)


    def __del__(self):
        self.close()
