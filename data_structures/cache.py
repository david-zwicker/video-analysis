'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections



class DictFinite(collections.OrderedDict):
    """ cache with a limited number of items """
    
    default_capacity = 100    
    
    def __init__(self, *args, **kwargs):
        self.capacity = kwargs.pop('capacity', self.default_capacity)
        super(DictFinite, self).__init__(*args, **kwargs)


    def check_length(self):
        """ ensures that the dictionary does not grow beyond its capacity """
        while len(self) > self.capacity:
            self.popitem(last=False)
            

    def __setitem__(self, key, value):
        super(DictFinite, self).__setitem__(key, value)
        self.check_length()
        
        
    def update(self, values):
        super(DictFinite, self).update(values)
        self.check_length()
        
    

class cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"
                
    The data is stored in a dictionary named `_cache` attached to the instance
    of each object. The cache can thus be cleared by setting self._cache = {}

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.
    """

    def __init__(self, *args, **kwargs):
        """ setup the decorator """
        self.cache = None
        if len(args) > 0:
            if callable(args[0]):
                # called as a plain decorator
                self.__call__(*args, **kwargs)
            else:
                # called with arguments
                self.cache = args[0]
        else:
            # called with arguments
            self.cache = kwargs.pop('cache', self.cache)
            

    def __call__(self, func, doc=None, name=None):
        """ save the function to decorate """
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        return self


    def __get__(self, obj, owner):
        if obj is None:
            return self

        # load the cache structure
        if self.cache is None:
            try:
                cache = obj._cache
            except AttributeError:
                cache = obj._cache = {}
        else:
            try:
                cache = getattr(obj, self.cache)
            except:
                cache = {}
                print self.cache
                setattr(obj, self.cache, cache)
                
        # try to retrieve from cache or call and store result in cache
        try:
            value = cache[self.__name__]
        except KeyError:
            value = self.func(obj)
            cache[self.__name__] = value
        return value
    
