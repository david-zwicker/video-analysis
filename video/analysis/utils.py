'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division


class cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func


    def __get__(self, obj, type=None):  # @ReservedAssignment
        if obj is None:
            return self

        # try to retrieve from cache or call and store result in cache
        try:
            value = obj._cache[self.__name__]
        except KeyError:
            value = self.func(obj)
            obj._cache[self.__name__] = value
        except AttributeError:
            value = self.func(obj)
            obj._cache = {self.__name__: value}
        return value


