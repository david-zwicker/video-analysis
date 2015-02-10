'''
Created on Feb 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains functions that can be used to handle files and directories
'''

from __future__ import division

import contextlib
import os.path


# get parent directory
MAIN_DIRECTORY = os.path.dirname(os.path.dirname(__file__))


def get_full_path(*path):
    """ returns the path relative to the root directory.
    Inspired by http://codereview.stackexchange.com/a/20449 """
    return os.path.join(MAIN_DIRECTORY, *path)



@contextlib.contextmanager
def change_directory(path):
    """
    A context manager which changes the directory to the given
    path, and then changes it back to its previous value on exit.
    Copied from http://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)



def ensure_directory_exists(folder):
    """ creates a folder if it not already exists """
    try:
        os.makedirs(folder)
    except OSError:
        # assume that the directory already exists
        pass
