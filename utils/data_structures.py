'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains several data structures and functions for manipulating them
'''

from __future__ import division

import collections
import datetime
import os
import sys
import warnings

import numpy as np
import h5py



def transpose_list_of_dicts(data, missing=None):
    """ turns a list of dictionaries into a dictionary of lists, filling in
    `missing` for items that are not available in some dictionaries """
    result = {}
    result_len = 0
    keys = set()
    
    # iterate through the whole list and add items one by one
    for item in data:
        # add the items to the result dictionary
        for k, v in item.iteritems():
            try:
                result[k].append(v)
            except KeyError:
                keys.add(k)
                result[k] = [missing] * result_len + [v]
                
        # add missing items
        for k in keys - set(item.keys()):
            result[k].append(missing)
                
        result_len += 1
    
    return result



def save_dict_to_csv(data, filename, first_columns=None, **kwargs):
    """ function that takes a dictionary of lists and saves it as a csv file """
    if first_columns is None:
        first_columns = []

    # sort the columns 
    sorted_index = {c: k for k, c in enumerate(sorted(data.keys()))}
    def column_key(col):
        """ helper function for sorting the columns in the given order """
        try:
            return first_columns.index(col)
        except ValueError:
            return len(first_columns) + sorted_index[col]
    sorted_keys = sorted(data.keys(), key=column_key)
        
    # create a data table and indicated potential units associated with the data
    # in the header
    table = collections.OrderedDict()
    for key in sorted_keys:
        value = data[key]
        
        # check if value has a unit
        if hasattr(value, 'units'):
            # value is single item with unit
            key += ' [%s]' % value.units
            value = value.magnitude
            
        elif len(value) > 0 and any(hasattr(v, 'units') for v in value):
            # value is a list with at least one unit attached to it
            
            try:
                # get list of units ignoring empty items
                units = set(str(item.units)
                            for item in value
                            if item is not None)
            except AttributeError:
                # one item did not have a unit
                for k, item in enumerate(value):
                    if not hasattr(item, 'units'):
                        print([val[k] for val in data.values()])
                        raise AttributeError('Value `%s = %s` does not have '
                                             'any units' % (key, item))
                raise
            
            # make sure that the units are all the same
            assert len(units) == 1
            
            # construct key and values
            key += ' [%s]' % value[0].units
            value = [item.magnitude if item is not None else None
                     for item in value]
            
        table[key] = value

    # create a pandas data frame to save data to CSV
    import pandas as pd
    pd.DataFrame(table).to_csv(filename, **kwargs)




def get_chunk_size(shape, num_elements):
    """ tries to determine an optimal chunk size for an array with a given 
    shape by chunking the longest axes first """
    chunks = list(shape)
    while np.prod(chunks) > num_elements:
        dim_long = np.argmax(chunks) #< get longest dimension
        chunks[dim_long] = 1 #< temporary set to one for np.prod 
        chunks[dim_long] = max(1, num_elements // np.prod(chunks))
    return tuple(chunks)
    


class OmniContainer(object):
    """ helper class that acts as a container that contains everything """
    def __bool__(self, key):
        return True
    
    def __contains__(self, key):
        return True
    
    def __delitem__(self, key):
        pass
    
    def __repr__(self):
        return '%s()' % self.__class__.__name__
    
    

class LazyLoadError(RuntimeError):
    """ exception that can be thrown if lazy-loading failed """
    pass



class LazyValue(object):
    """ base class that represents a value that is only loaded when it is
    accessed """
    def load(self):
        raise NotImplementedError
    


class LazyHDFValue(LazyValue):
    """ class that represents a value that is only loaded from HDF when it is
    accessed """
    chunk_elements = 10000
    compression = None
    

    def __init__(self, data_cls, key, hdf_filename):
        self.data_cls = data_cls
        self.key = key
        self.hdf_filename = hdf_filename
        

    def __repr__(self):
        return '%s(data_cls=%s, key="%s", hdf_filename="%s")' % (
                    self.__class__.__name__, self.data_cls.__name__,
                    self.key, self.hdf_filename)
        
        
    def set_hdf_folder(self, hdf_folder):
        """ replaces the folder of the hdf file """
        hdf_name = os.path.basename(self.hdf_filename)
        self.hdf_filename = os.path.join(hdf_folder, hdf_name)
        
        
    def get_yaml_string(self):
        """ returns a representation of the object as a single string, which
        is useful for referencing the object in YAML """
        hdf_name = os.path.basename(self.hdf_filename)
        return '@%s:%s' % (hdf_name, self.key)
        
        
    @classmethod
    def create_from_yaml_string(cls, value, data_cls, hdf_folder):
        """ create an instance of the class from the yaml string and additional
        information """

        # consistency check
        if value[0] != '@':
            raise RuntimeError('Item with lazy loading does not start with `@`')
        
        # read the link
        data_str = value[1:] # strip the first character, which should be an @
        hdf_name, key = data_str.split(':')
        hdf_filename = os.path.join(hdf_folder, hdf_name)
        return cls(data_cls, key, hdf_filename)
        
    
    @classmethod    
    def create_from_data(cls, key, data, hdf_filename):
        """ store the data in a HDF file and return the storage object """
        data_cls = data.__class__
        with h5py.File(hdf_filename, 'a') as hdf_file:
            # delete possible previous key to have a clean storage
            if key in hdf_file:
                del hdf_file[key]
                
            # save actual data as an array
            data_array = np.asarray(data.to_array())
            if cls.compression is None or data_array.size < cls.chunk_elements:
                hdf_file.create_dataset(key, data=data_array, track_times=True)
            else:
                chunks = get_chunk_size(data_array.shape, cls.chunk_elements)
                hdf_file.create_dataset(key, data=data_array, track_times=True,
                                        chunks=chunks, compression=cls.compression)
                
            # add attributes to describe data 
            hdf_file[key].attrs['written_on'] = str(datetime.datetime.now())
            if hasattr(data_cls, 'hdf_attributes'):        
                for attr_key, attr_value in data_cls.hdf_attributes.iteritems():
                    hdf_file[key].attrs[attr_key] = attr_value
            
        return cls(data_cls, key, hdf_filename)
    
        
    def load(self):
        """ load the data and return it """
        # open the associated HDF5 file and read the data
        with h5py.File(self.hdf_filename, 'r') as hdf_file:
            data = hdf_file[self.key][:]  #< copy data into RAM
            result = self.data_cls.from_array(data)
        
        # create object
        return result



class LazyHDFCollection(LazyHDFValue):
    """ class that represents a collection of values that are only loaded when they are accessed """

    @classmethod    
    def create_from_data(cls, key, data, hdf_filename):
        """ store the data in a HDF file and return the storage object """
        data_cls = data.__class__

        # save a collection of objects to hdf
        with h5py.File(hdf_filename, 'a') as hdf_file:
            # reset the whole structure if it is there
            if key in hdf_file:
                del hdf_file[key]
                
            # create group in case data is empty
            hdf_file.create_group(key)

            # write all objects as individual datasets            
            key_format = '{}/%0{}d'.format(key, len(str(len(data))))
            for index, obj in enumerate(data):
                obj.save_to_hdf5(hdf_file, key_format % index)
    
            hdf_file[key].attrs['written_on'] = str(datetime.datetime.now())
            if hasattr(data_cls, 'hdf_attributes'):        
                for attr_key, attr_value in data_cls.hdf_attributes.iteritems():
                    hdf_file[key].attrs[attr_key] = attr_value

        return cls(data_cls, key, hdf_filename)
    
        
    def load(self):
        """ load the data and return it """
        # open the associated HDF5 file and read the data
        item_cls = self.data_cls.item_class
        with h5py.File(self.hdf_filename, 'r') as hdf_file:
            # iterate over the data and create objects from it
            data = hdf_file[self.key]
            if data:
                result = self.data_cls(item_cls.from_array(data[index][:])
                                       for index in sorted(data.keys()))
                # here, we have to use sorted() to iterate in the correct order 
            else: # empty dataset
                result = self.data_cls()
                
        return result



def prepare_data_for_yaml(data):
    """ recursively converts all numpy types to their closest python equivalents """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, collections.MutableMapping):
        return {k: prepare_data_for_yaml(v) for k, v in data.iteritems()}
    elif isinstance(data, (list, tuple)):
        return [prepare_data_for_yaml(v) for v in data]
    elif isinstance(data, LazyHDFValue):
        return data.get_yaml_string()
    elif data is not None and not isinstance(data, (bool, int, float, list, basestring)):
        warnings.warn('Encountered unknown instance of `%s` in YAML preparation' %
                      data.__class__)
    return data



class NestedDict(collections.MutableMapping):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key:
    
    d = NestedDict({'a': {'b': 1}})
    
    d['a/b']
    >>>> 1
    
    d['a/c'] = 2
    
    d
    >>>> {'a': {'b': 1, 'c': 2}}
    """
    
    def __init__(self, data=None, sep='/', dict_class=dict):
        """ initialize the NestedDict object
        `data` is a dictionary that is used to fill the current object
        `sep` determines the separator used for accessing different levels of
            the structure
        `dict_class` is the dictionary class that will handle items under the
            hood and for instance determines how items are iterated over
        """

        # store details about the dictionary
        self.sep = sep
        self.dict_class = dict_class

        # set data
        self.data = self.dict_class()
        if data is not None:
            self.from_dict(data)


    def get_item(self, key):
        """ returns the item identified by `key`.
        If load_data is True, a potential LazyValue gets loaded """
        try:
            if isinstance(key, basestring) and self.sep in key:
                # sub-data is accessed
                child, grandchildren = key.split(self.sep, 1)
                try:
                    value = self.data[child].get_item(grandchildren)
                except AttributeError:
                    raise KeyError(key)
            else:
                value = self.data[key]
        except KeyError:
            raise KeyError(key)

        return value

    
    def __getitem__(self, key):
        return self.get_item(key)
        
        
    def __setitem__(self, key, value):
        """ writes the item into the dictionary """
        if isinstance(key, basestring) and self.sep in key:
            # sub-data is written
            child, grandchildren = key.split(self.sep, 1)
            try:
                self.data[child][grandchildren] = value
            except KeyError:
                # create new child if it does not exists
                child_node = self.__class__()
                child_node[grandchildren] = value
                self.data[child] = child_node
            except TypeError:
                raise TypeError('Can only use Xpath assignment if all children '
                                'are NestedDict instances.')
                
        else:
            self.data[key] = value
    
    
    def __delitem__(self, key):
        """ deletes the item identified by key """
        try:
            if isinstance(key, basestring) and self.sep in key:
                # sub-data is deleted
                child, grandchildren = key.split(self.sep, 1)
                try:
                    del self.data[child][grandchildren]
                except TypeError:
                    raise KeyError(key)
    
            else:
                del self.data[key]
        except KeyError:
            raise KeyError(key)


    def __contains__(self, key):
        """ returns True if the item identified by key is contained in the data """
        if isinstance(key, basestring) and self.sep in key:
            child, grandchildren = key.split(self.sep, 1)
            try:
                return child in self.data and grandchildren in self.data[child]
            except TypeError:
                return False

        else:
            return key in self.data


    # Miscellaneous dictionary methods are just mapped to data
    def __len__(self): return len(self.data)
    def __iter__(self): return self.data.__iter__()
    def keys(self): return self.data.keys()
    def values(self): return self.data.values()
    def items(self): return self.data.items()
    def clear(self): self.data.clear()


    def itervalues(self, flatten=False):
        """ an iterator over the values of the dictionary
        If flatten is true, iteration is recursive """
        for value in self.data.itervalues():
            if flatten and isinstance(value, NestedDict):
                # recurse into sub dictionary
                for v in value.itervalues(flatten=True):
                    yield v
            else:
                yield value 
                
                
    def iterkeys(self, flatten=False):
        """ an iterator over the keys of the dictionary
        If flatten is true, iteration is recursive """
        if flatten:
            for key, value in self.data.iteritems():
                if isinstance(value, NestedDict):
                    # recurse into sub dictionary
                    try:
                        prefix = key + self.sep
                    except TypeError:
                        raise TypeError('Keys for NestedDict must be strings '
                                        '(`%s` is invalid)' % key)
                    for k in value.iterkeys(flatten=True):
                        yield prefix + k
                else:
                    yield key
        else:
            for key in self.iterkeys():
                yield key


    def iteritems(self, flatten=False):
        """ an iterator over the (key, value) items
        If flatten is true, iteration is recursive """
        for key, value in self.data.iteritems():
            if flatten and isinstance(value, NestedDict):
                # recurse into sub dictionary
                try:
                    prefix = key + self.sep
                except TypeError:
                    raise TypeError('Keys for NestedDict must be strings '
                                    '(`%s` is invalid)' % key)
                for k, v in value.iteritems(flatten=True):
                    yield prefix + k, v
            else:
                yield key, value 

            
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.data))


    def create_child(self, key, values=None):
        """ creates a child dictionary and fills it with values """
        self[key] = self.__class__(values)
        return self[key]


    def copy(self):
        """ makes a shallow copy of the data """
        res = self.__class__()
        for key, value in self.iteritems():
            if isinstance(value, (dict, NestedDict)):
                value = value.copy()
            res[key] = value
        return res


    def from_dict(self, data):
        """ fill the object with data from a dictionary """
        for key, value in data.iteritems():
            if isinstance(value, dict):
                if key in self and isinstance(self[key], NestedDict):
                    # extend existing NestedDict instance
                    self[key].from_dict(value)
                else:
                    # create new NestedDict instance
                    self[key] = self.__class__(value)
            else:
                # store simple value
                self[key] = value

            
    def to_dict(self, flatten=False):
        """ convert object to a nested dictionary structure.
        If flatten is True a single dictionary with complex keys is returned.
        If flatten is False, a nested dictionary with simple keys is returned """
        res = self.dict_class()
        for key, value in self.iteritems():
            if isinstance(value, NestedDict):
                value = value.to_dict(flatten)
                if flatten:
                    for k, v in value.iteritems():
                        try:
                            res[key + self.sep + k] = v
                        except TypeError:
                            raise TypeError('Keys for NestedDict must be strings '
                                            '(`%s` or `%s` is invalid)' % (key, k))
                else:
                    res[key] = value
            else:
                res[key] = value
        return res

    
    def pprint(self, *args, **kwargs):
        """ pretty print the current structure as nested dictionaries """
        from pprint import pprint
        pprint(self.to_dict(), *args, **kwargs)



class LazyNestedDict(NestedDict):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key.
    Additionally, this class supports loading lazy values if they are accessed
    """
    
    def get_item(self, key, load_data=True):
        """ returns the item identified by `key`.
        If load_data is True, a potential LazyValue gets loaded """
        try:
            if isinstance(key, basestring) and self.sep in key:
                # sub-data is accessed
                child, grandchildren = key.split(self.sep, 1)
                try:
                    value = self.data[child].get_item(grandchildren, load_data)
                except AttributeError:
                    raise KeyError(key)
            else:
                value = self.data[key]
        except KeyError:
            raise KeyError(key)

        # load lazy values
        if load_data and isinstance(value, LazyValue):
            try:
                value = value.load()
            except KeyError as err:
                # we have to relabel KeyErrors, since they otherwise shadow
                # KeyErrors raised by the item actually not being in the NestedDict
                # This then allows us to distinguish between items not found in
                # NestedDict (raising KeyError) and items not being able to load
                # (raising LazyLoadError)
                err_msg = ('Cannot load item `%s`.\nThe original error was: %s'
                           % (key, err)) 
                raise LazyLoadError, err_msg, sys.exc_info()[2] 
            self.data[key] = value #< replace loader with actual value
            
        return value
    