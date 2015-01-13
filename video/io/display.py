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

        
def _show_image_from_pipe(pipe, image_array, title, position=None):
    """ function that runs in a separate process to display an image """
    cv2.namedWindow(title)
    if position is not None:
        cv2.moveWindow(title, position[0], position[1])
    cv2.waitKey(1)

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
                
            elif command == 'check_events':
                pass
                
            elif command == 'close':
                break
                    
            else:
                raise ValueError('Unknown command `%s`' % command)

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
            
    except KeyboardInterrupt:
        pipe.send('interrupt')
        
    # cleanup
    #pipe.close()
    cv2.destroyWindow(title)
    # work-around to handle GUI event loop 
    for _ in xrange(10):
        cv2.waitKey(1)


        
class ImageWindow(object):
    """ class that can show an image """
    
    def __init__(self, size, title='', output_period=1,
                 multiprocessing=True, position=None):
        """ initializes the video shower.
        size sets the width and the height of the image to be shown
        title sets the title of the window. This should be unique if multiple
            windows are used.
        output_period determines if frames are skipped during display.
            For instance, `output_period=10` only shows every tenth frame.
        multiprocessing indicates whether a separate process is used for
            displaying. If multiprocessing=None, multiprocessing is used
            for all platforms, except MacOX
        position determines the coordinates of the top left corner of the
            window that displays the image
        """
        self.title = title
        self.output_period = output_period
        self.this_frame = 0
        self._proc = None
        
        # multiprocessing does not work in current MacOS OpenCV
        if multiprocessing is None:
            multiprocessing = (sys.platform != "darwin")
        
        if multiprocessing:
            # open 
            if sharedmem:
                try:
                    # create the pipe to talk to the child
                    self._pipe, pipe_child = mp.Pipe(duplex=True)
                    # setup the shared memory area
                    self._data = sharedmem.empty(size, np.uint8)
                    # initialize the process that shows the image
                    self._proc = mp.Process(target=_show_image_from_pipe,
                                            args=(pipe_child, self._data,
                                                  title, position))
                    self._proc.daemon = True
                    self._proc.start()
                    logger.debug('Started process %d for displaying images' % self._proc.pid)
                    
                except AssertionError:
                    logger.warn('Could not start a separate process to display images. '
                                'The main process will thus be used.')
                
            else:
                logger.warn('Package sharedmem could not be imported and '
                            'images are thus shown using the main process.')
                
        if self._proc is None:
            # open window in this process
            cv2.namedWindow(title)
            if position is not None:
                cv2.moveWindow(title, position[0], position[1])
            cv2.waitKey(1) 
       
       
    def check_gui_events(self):
        """ checks whether the GUI sent any events to the window.
        The function raises a KeyboardInterrupt if the user wants to abort """  
        if self._proc:
            # check the viewer process for events
            if self._pipe.poll() and self._pipe.recv() == 'interrupt':
                raise KeyboardInterrupt
        else:
            # check the window for events
            while True:
                # waitKey also handles other GUI events and we thus call it
                # until everything is handled
                key = cv2.waitKey(1)
                if key & 0xFF in {27, ord('q')}:
                    raise KeyboardInterrupt
                elif key == -1:
                    break

       
    def show(self, image=None):
        """ show an image.
        May raise KeyboardInterrupt, if the user opted to exit
        """
        # check whether the current frame should be displayed 
        if image is not None and (self.this_frame % self.output_period) == 0:
            if image.ndim > 2:
                # reverse the color axis, to get BGR image required by OpenCV
                image = image[:, :, ::-1]
            
            if self._proc:
                # copy data to shared memory
                self._data[:] = image
                # and tell the process to update window
                self._pipe.send('update')
    
            else:
                # update the image in our window
                cv2.imshow(self.title, image.astype(np.uint8))
                    
        else:
            # image is not shown => still poll for events
            if self._proc:
                self._pipe.send('check_events')
        self.check_gui_events()
            
        # keep the internal frame count up to date 
        self.this_frame += 1
        
        
    def close(self):
        """ closes the window """
        if self._proc is not None:
            # shut down the process
            self._pipe.send('close')
            self._proc.join()
            self._pipe.close()
            self._proc = None
            
        else:
            # delete the opencv window
            cv2.destroyWindow(self.title)
            # work-around to handle GUI event loop 
            for _ in xrange(10):
                cv2.waitKey(1)


    def __del__(self):
        self.close()
