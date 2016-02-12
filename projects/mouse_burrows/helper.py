'''
Created on Feb 22, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Useful functions for managing mouse burrow projects
'''

from __future__ import division

import copy
import re

# define the quadrants that appear in the experiments
QUADRANTS = {'5': {'short': 'UL', 'position': 'upper left'},
             '6': {'short': 'UR', 'position': 'upper right'},
             '7': {'short': 'DL', 'position': 'lower left'},
             '8': {'short': 'DR', 'position': 'lower right'}}

# define the quadrants that appear in the experiments when two cameras are used
QUADRANTS_HALF = {'5': {'short': 'U', 'position': 'upper'},
                  '6': {'short': 'U', 'position': 'upper'},
                  '7': {'short': 'D', 'position': 'lower'},
                  '8': {'short': 'D', 'position': 'lower'}}



def identifier_single(data, datesep='_', mousesep='_'):
    """ return the identifier of a single experiment """
    # check whether there is an id
    if 'id' in data:
        result = data['index'] + "_"
    else:
        result = ""
        
    # add the date
    result += data['date'] + datesep
        
    # check whether we are dealing with a single mouse or multiple
    if isinstance(data['mouse'], basestring):
        result += data['mouse']
    else:
        result += '_'.join(data['mouse'])
        
    return result



def parse_experiment_identifier(identifier, mouse_count=1):
    """ scans an experiment identifier and returns a list of the experiments
    """
    # split the identifier into tokens
    tokens = re.split('_|-', identifier)
    #print tokens
    
    # scan global data valid for all experiments
    data_glob = {'mouse': []}
    if 3 <= len(tokens[0]) <= 4:
        data_glob['id'] = tokens[0]
        tokens = tokens[1:]
    if len(tokens[0]) == 8:
        data_glob['date'] = tokens[0]
        tokens = tokens[1:]
    else:
        raise ValueError('Could not find any date in identifier.')
    
    # scan the remaining tokens for experiments
    results, data_exp = [], copy.deepcopy(data_glob)
    for token in tokens:
        # check for possible data
        if len(token) == 1 and token.isdigit():
            data_exp['cage'] = token
        elif 5 <= len(token):
            data_exp['mouse'].append(token)
                
        # check whether we have enough data
        if 'cage' in data_exp and len(data_exp['mouse']) >= mouse_count:
            if mouse_count == 1:
                data_exp['mouse'] = data_exp['mouse'][0]
            results.append(data_exp)
            data_exp = copy.deepcopy(data_glob)
            
    return results


    