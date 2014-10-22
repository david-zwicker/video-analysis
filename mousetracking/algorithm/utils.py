'''
Created on Sep 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Utility functions
'''

from __future__ import division

import contextlib
import logging
import os



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



def mean(values, empty=0):
    """ calculates mean of generator or iterator.
    Returns `empty` in case of an empty sequence """
    n, total = 0, 0.
    for value in values:
        total += value
        n += 1
    return total/n if n > 0 else empty



@contextlib.contextmanager
def change_directory(path):
    """
    A context manager which changes the directory to the given
    path, and then changes it back to its previous value on exit.
    Stolen from http://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def unique_based_on_id(data):
    """ returns a list with only unique items, where the uniqueness
    is determined from the id of the items """
    result, seen = [], set()
    for item in data:
        if id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result

