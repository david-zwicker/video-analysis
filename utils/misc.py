'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that would not fit into any other module in this
package.
'''

from __future__ import division

import logging
import warnings


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warn('Package tqdm could not be imported and progress bars are '
                 'thus not available.')


def display_progress(iterator, total=None, mininterval=5):
    """
    displays a progress bar when iterating
    """
    if tqdm is not None:
        return tqdm(iterator, total=total, leave=True, mininterval=mininterval)
    else:
        return iterator
    
    
    
def get_loglevel_from_name(name_or_int):
    """ converts a logging level name to the numeric representation """
    # see whether it is already an integer
    if isinstance(name_or_int, int):
        return name_or_int
    
    # convert it from the name
    level = logging.getLevelName(name_or_int.upper())
    if isinstance(level, int):
        return level
    else:
        raise ValueError('`%s` is not a valid logging level.' % name_or_int)



def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    Taken from http://code.activestate.com/recipes/391367-deprecated/
    """
    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc



def unique_based_on_id(data):
    """ returns a list with only unique items, where the uniqueness
    is determined from the id of the items. This can be useful in the
    case where the items cannot be hashed and a set can thus not be used. """
    result, seen = [], set()
    for item in data:
        if id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result

