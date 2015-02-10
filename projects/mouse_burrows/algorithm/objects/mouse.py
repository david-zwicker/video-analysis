'''
Created on Sep 11, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools
import re

import numpy as np
from scipy.ndimage import filters

from utils.data_structures import LazyHDFValue
from utils.math import contiguous_true_regions



class MouseStateCategory(object):
    """ class that organizes a single category of states """
    
    def __init__(self, name, symbols, states, length=None):
        """ initializes the category.
        name denotes the name of the category
        symbols is a string with a unique symbol per state
        states is a list of descriptions of the states
        length is an integer indicating the maximal capacity of the category
            This can be used to reserve room for adding additional states later
            without disturbing already calculated states.
        
        Note that we automatically add an extra symbol `?` which denotes an
        unknown state. This will be added as the first symbol and thus gets 
        the id zero.    
        """ 
        # prepare data
        self.name = name
        symbols = '?' + symbols
        states = ('?',) + tuple(states)
        self.possible_states = states
        
        # determine the length and do some checks
        assert len(symbols) == len(states)
        if length is None:
            self.length = len(symbols)
        else:
            self.length = length + 1
        assert len(symbols) <= self.length
        
        # create look up tables
        self.symbols = symbols.ljust(self.length, '?')
        self.states = {k: s for k, s in enumerate(states)}
        self.states.update({k: '?' for k in range(len(states), self.length)})
        self.states.update({sym: state for sym, state in zip(symbols, states)})
        self.ids = {s: k for k, s in enumerate(symbols)}
        self.ids.update({s: k for k, s in enumerate(states)})
    
    
        
class MouseStateConverter(object):
    """ class that manages the mapping of mouse states between different
    representations """
    
    def __init__(self, categories):
        """ initializes the converted.
        categories initializes the categories supported by the converted """
        self.categories = []
        self.factors = []
        self.max_id = 1
        for data in categories:
            category = MouseStateCategory(**data)
            self.categories.append(category)
            self.factors.append(self.max_id)
            self.max_id *= category.length
        
        
    def get_categories(self):
        """ return the the categories and their possible states """ 
        return [(c.name, c.possible_states[1:])
                for c in self.categories]
        
        
    def symbols_to_int(self, symbols):
        """ converts the symbol representation to the integer representation """
        return sum(fac * cat.ids.get(sym, 0)
                   for fac, cat, sym in itertools.izip(self.factors,
                                                       self.categories,
                                                       symbols))


    def int_to_symbols(self, value):
        """ converts the integer representation to the symbol representation """
        res = ''
        for cat in self.categories:
            res += cat.symbols[value % cat.length]
            value //= cat.length
        return res
    
    
    def dict_to_int(self, state):
        """ converts the dictionary representation to the integer representation """
        res = 0
        for fac, cat in itertools.izip(self.factors, self.categories):
            if cat.name in state:
                res += fac * cat.ids.get(state.pop(cat.name), 0)
                
        if state:
            raise RuntimeError('Not all information inside the state dictionary '
                               'could be converted to the other state '
                               'representation. Left over: %s' % state)
        return res
    
    
    def int_to_dict(self, value):
        """ converts the integer representation to the dictionary representation """
        res = {}
        for cat in self.categories:
            res[cat.name] = cat.states[value % cat.length]
            value //= cat.length
        return res
    

    def symbols_repr(self, symbols):
        """ returns a readable representation for the symbols """
        return ''.join(cat.states[sym]
                       for cat, sym in itertools.izip(self.categories, symbols)
                       if sym != '.' and sym != '?')


    def get_state_description(self):
        """ returns a string with a description of the different states """
        res = ['The state is a sum of integers with the following '
               'interpretation:']
        for fac, cat in itertools.izip(self.factors, self.categories):
            res.append('%s:' % cat.name)
            for key in sorted(cat.states.iterkeys()):
                state = cat.states[key]
                if isinstance(key, int) and state != '?':
                    res.append('  %d - %s' % (fac*key, state))
        return '\n'.join(res)
    
    
    def get_state_lookup_table(self, states):
        """ returns a dictionary that maps all possible mouse states onto the
        an integer. That integer gives the index in the list of supplied
        `states` patterns supplied to this function """ 
        re_states = [re.compile(pattern) for pattern in states]
        
        possible_states = [c.symbols for c in self.categories]
        lut = {}
        for state in itertools.product(*possible_states):
            key_symbols = ''.join(state)
            key_int = self.symbols_to_int(key_symbols)
            for k, pattern in enumerate(re_states):
                if pattern.match(key_symbols):
                    lut[key_int] = k
                    break
            else:
                lut[key_int] = None
                
        return lut

    

# create the mouse states used in this module
state_converter = MouseStateConverter((
    {'name': 'position_horizontal',
        'symbols': 'LR',
        'states': ('left', 'right'),
        'length': 9},
    {'name': 'location',
        'symbols': 'AHVDB',
        'states': ('air', 'hill', 'valley', 'dimple', 'burrow'),
        'length': 9},
    {'name': 'location_detail',
        'symbols': ' E',
        'states': ('general', 'end point'),
        'length': 9},
    {'name': 'dynamics',
        'symbols': 'SM',
        'states': ('stationary', 'moving'),
        'length': 9},
))



def state_symbols_match(pattern, value):
    """ returns True if the value matches the pattern, where '.' can be used
    as placeholders that match every state """
    return all(a == '.' or a == b
               for a, b in zip(pattern, value))
            
        

class MouseTrack(object):
    """ class that describes the mouse track """
    
    hdf_attributes = {'column_names': ('Position X', 'Position Y', 'Mouse State',
                                       'Index of closest ground point',
                                       'Distance from ground'),
                      'mouse_states': state_converter.get_state_description()}
    storage_class = LazyHDFValue
    
    def __init__(self, trajectory, states=None, ground_idx=None, ground_dist=None):
        self.pos = trajectory
        self.velocity = None
        
        # initialize arrays
        if states is not None:
            self.states = np.asarray(states, np.int)
        else:
            self.states = np.zeros(len(trajectory), np.int)
        if ground_idx is not None:
            self.ground_idx = np.asarray(ground_idx, np.double)
        else:
            self.ground_idx = np.zeros(len(trajectory), np.double) + np.nan
        if ground_dist is not None:
            self.ground_dist = np.asarray(ground_dist, np.double)
        else:
            self.ground_dist = np.zeros(len(trajectory), np.double) + np.nan
    
    
    def __repr__(self):
        return '%s(frames=%d)' % (self.__class__.__name__, len(self.pos))


    def trajectory_smoothed(self, sigma):
        """ returns the mouse trajectory smoothed with a Gaussian filter of
        standard deviation `sigma` """
        trajectory = np.empty_like(self.pos)
        trajectory.fill(np.nan)
        
        # smooth position
        indices = contiguous_true_regions(np.isfinite(self.pos[:, 0]))
        for start, end in indices:
            if end - start > 1:
                filters.gaussian_filter1d(self.pos[start:end, :],
                                          sigma, axis=0, mode='nearest',
                                          output=trajectory[start:end, :])

        return trajectory
        
    
    def calculate_velocities(self, sigma):
        """ calculates the velocity from smoothed positions """
        velocity = np.empty_like(self.pos)
        velocity.fill(np.nan)
        
        indices = contiguous_true_regions(np.isfinite(self.pos[:, 0]))
        for start, end in indices:
            if end - start > 1:
                # smooth position
                pos = filters.gaussian_filter1d(self.pos[start:end, :],
                                                sigma,  #< std of the filter
                                                axis=0, #< time axis
                                                mode='nearest')
                # differentiate to get velocity
                velocity[start:end, 0] = np.gradient(pos[:, 0])
                velocity[start:end, 1] = np.gradient(pos[:, 1])
        
        self.velocity = velocity
    
    
    def set_state(self, frame_id, state=None, ground_idx=None, ground_dist=None):
        """ sets the state of the mouse in frame frame_id """
        if state is not None:
            self.states[frame_id] = state_converter.dict_to_int(state)
        if ground_idx is not None:
            self.ground_idx[frame_id] = ground_idx
        if ground_dist is not None:
            self.ground_dist[frame_id] = ground_dist
    
    
    def to_array(self):
        # determine the full data set
        data = [self.pos, self.states[:, None],
                self.ground_idx[:, None], self.ground_dist[:, None]]
        data_count = len(data)
        # remove columns from the right if they do not contain data
        if not np.any(np.isfinite(self.ground_dist)):
            data_count -= 1
            if not np.any(np.isfinite(self.ground_idx)):
                data_count -= 1
                if not np.any(self.states != 0):
                    data_count -= 1
                    
        return np.concatenate(data[:data_count], axis=1)
    
    
    @classmethod
    def from_array(cls, data):
        data_len = data.shape[1]
        states = data[:, 2] if data_len > 2 else None
        ground_idx = data[:, 3] if data_len > 3 else None
        ground_dist = data[:, 4] if data_len > 4 else None
        return cls(data[:, :2], states, ground_idx, ground_dist)



def test_state_conversion():
    """ simple test function for the state conversion """
    for value in xrange(0, state_converter.max_id):
        # test conversion to symbol string
        symbols = state_converter.int_to_symbols(value)
        val = state_converter.symbols_to_int(symbols)
        if symbols != state_converter.int_to_symbols(val):
            raise AssertionError('Wrong symbol representation: %d != %s' % (value, symbols))

        # test conversion to dictionary
        state = state_converter.int_to_dict(value)
        val = state_converter.dict_to_int(state)
        if state != state_converter.int_to_dict(val):
            raise AssertionError('Wrong symbol representation: %d != %s' % (value, state))
        
    print('The test was successful.') 



if __name__ == '__main__':
    print('Test the state to value conversion')
    test_state_conversion()
    