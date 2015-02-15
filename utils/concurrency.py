'''
Created on Feb 15, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import Queue as queue
import threading



class WorkerThread(object):
    """ class that launches a worker thread as a daemon that applies a given
    function """
    
    def __init__(self, function):
        """ initializes the worker thread with the supplied function that will
        be called subsequently """
        self.function = function
        self._result = None
        
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker_function,
                                        args=[self._q])
        self._thread.daemon = True
        self._thread.start()
        
        
    def _worker_function(self, q):
        """ event loop of the worker thread """ 
        while True:
            if q.get():
                self._result = self.function(*self._args, **self._kwargs)
            else:
                break
            q.task_done()
        
        
    def put(self, *args, **kwargs):
        """ starts the worker thread by passing the supplied arguments to it.
        Note that the arguments are not copied and should therefore not be
        modified while the thread is running in the background
        """
        if not self._q.empty():
            # wait until the last worker thread has finished
            self._q.join()
        self._args = args
        self._kwargs = kwargs
        self._q.put(True)
        
        
    def get(self):
        """ retrieves the result from the last job that was put """
        self._q.join()
        return self._result

