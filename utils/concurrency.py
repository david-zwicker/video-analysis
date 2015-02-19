'''
Created on Feb 15, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import threading



class WorkerThread(object):
    """ class that launches a worker thread as a daemon that applies a given
    function """
    
    def __init__(self, function, use_threads=True, synchronous=True):
        """ initializes the worker thread with the supplied function that will
        be called subsequently.
        `synchronous` is a flag determining whether the result from the worker
            thread will be synchronized with the input. If it is not, it can be
            that a call to `get` returns the result from a previous worker
            thread. Additionally, subsequent calls to `get` can return different
            results, even if no new calculation was initiated via `put`. 
        """
        self.function = function
        self.use_threads = use_threads
        self.synchronous = synchronous
        self._result = None
        
        if self.use_threads:
            self._event_start = threading.Event()
            self._event_finish = threading.Event()
            self._event_finish.set()
            self._thread = threading.Thread(target=self._worker_function,
                                            args=[self._event_start,
                                                  self._event_finish])
            self._thread.daemon = True
            self._thread.start()
        
        
    def _worker_function(self, event_start, event_finish):
        """ event loop of the worker thread """ 
        while True:
            # wait until starting signal is set
            event_start.wait()
            event_start.clear()
            # start the calculation
            self._result = self.function(*self._args, **self._kwargs)
            # signal that the calculation finished
            event_finish.set()
        
        
    def put(self, *args, **kwargs):
        """ starts the worker thread by passing the supplied arguments to it.
        Note that the arguments are not copied and should therefore not be
        modified while the thread is running in the background
        """
        # set the arguments for the call
        self._args = args
        self._kwargs = kwargs
        
        if self.use_threads:
            # wait until the worker is finished (in case it is still running)
            self._event_finish.wait()
            self._event_finish.clear()
            if self.synchronous:
                # reset the result variable if the result should be synchronized
                self._result = None
            # signal that the worker may begin
            self._event_start.set()
        
        else:
            # don't use threads and thus clear the result
            self._result = None
        
        
    def get(self):
        """ retrieves the result from the last job that was put """
        if self.use_threads:
            if self.synchronous or self._result is None:
                # wait until the worker finished
                self._event_finish.wait()
                # => the result is in self._result
                
        elif self._result is None:
            # calculate the result
            self._result = self.function(*self._args, **self._kwargs)
            
        # retrieve the result
        return self._result
