'''
Created on Aug 14, 2014

@author: zwicker
'''

from __future__ import division

import threading

import cv2


class _ImageShowThread(threading.Thread):
    """ TODO: maybe remove the threaded version, because it shows no speed up """
    def show_with_check(self, image):
        cv2.imshow('Debug video', image)
        if cv2.waitKey(1) & 0xFF in {27, ord('q')}:
            return True
        else:
            return False
        
        
        
class ImageShow(object):
    """ class that can show an image """
    
    def __init__(self, title='', threaded=False):
        self.title = title
        
        if threaded:
            self._thread = _ImageShowThread()
            self._thread.start()
        else:
            self._thread = None
       
       
    def show(self, image):
        """ show an image. """
        cv2.imshow(self.title, image)
        
        
    def show_with_check(self, image):
        """ show an image.
        The return is a boolean indicating whether to abort the process
        """ 
        if self._thread:
            return self._thread.show(image)
        else:
            cv2.imshow(self.title, image)
            if cv2.waitKey(1) & 0xFF in {27, ord('q')}:
                return True
            else:
                return False
        
        
    def __del__(self):
        if self._thread:
            self._thread.end()
        
