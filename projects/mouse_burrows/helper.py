'''
Created on Feb 22, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Useful functions for managing mouse burrow projects
'''

from __future__ import division

import re

# define the quadrants that appear in the experiments
QUADRANTS = {'5': ('UL', 'upper left'),
             '6': ('UR', 'upper right'),
             '7': ('DL', 'lower left'),
             '8': ('DR', 'lower right')}



def identifier_single(data):
    """ return the identifier of a single experiment """
    if 'id' in data:
        result = "%(index)s_" % data
    else:
        result = ""
    result += "%(date)s_%(mouse)s" % data
    return result



def parse_experiment_identifier(identifier):
    """ scans an experiment identifier and returns a list of the experiments
    """
    # split the identifier into tokens
    tokens = re.split('_|-', identifier)
    #print tokens
    
    # scan global data valid for all experiments
    data_glob = {}
    if 3 <= len(tokens[0]) <= 4:
        data_glob['id'] = tokens[0]
        tokens = tokens[1:]
    if len(tokens[0]) == 8:
        data_glob['date'] = tokens[0]
        tokens = tokens[1:]
    else:
        raise ValueError('Could not find any date in identifier.')
    
    # scan the remaining tokens for experiments
    results, data_exp = [], data_glob.copy()
    for token in tokens:
        # check for possible data
        if len(token) == 1 and token.isdigit():
            data_exp['cage'] = token
        elif 5 <= len(token) <= 7:
            data_exp['mouse'] = token
        # check whether we have enough data
        if 'cage' in data_exp and 'mouse' in data_exp:
            results.append(data_exp)
            data_exp = data_glob.copy()
            
    return results


    