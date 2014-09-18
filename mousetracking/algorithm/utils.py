'''
Created on Sep 10, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

Utility functions
'''

from __future__ import division

import contextlib
import logging
import os



def get_loglevel_from_name(name):
    """ converts a logging level name to the numeric representation """
    level = logging.getLevelName(name.upper())
    if isinstance(level, int):
        return level
    else:
        raise ValueError('`%s` is not a valid logging level.' % name)



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
    yield
    os.chdir(prev_cwd)
