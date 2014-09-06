'''
Created on Sep 5, 2014

@author: zwicker
'''

import os

import h5py



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



class LazyHDFValue(object):
    """ class that represents a value that is only loaded when it is accessed """

    def __init__(self, data_cls, key, hdf_folder, hdf_name):
        self.data_cls = data_cls
        self.key = key
        self.hdf_folder = hdf_folder
        self.hdf_name = hdf_name
        
        
    def __repr__(self):
        return '%s(data_cls=%s, key="%s", hdf_folder="%s", hdf_name="%s")' % (
                    self.__class__.__name__, self.data_cls.__name__,
                    self.key, self.hdf_folder, self.hdf_name)
        
        
    @property
    def to_string(self):
        return '@%s:%s' % (self.hdf_name, self.key)
        
        
    @classmethod
    def create_from_string(cls, value, data_cls, hdf_folder):
        # consistency check
        if value[0] != '@':
            raise RuntimeError('Item with lazy loading does not start with `@`')
        
        # read the link
        data_str = value[1:] # strip the first character, which should be an @
        hdf_name, key = data_str.split(':')
        return cls(data_cls, key, hdf_folder, hdf_name)
        
    
    @classmethod    
    def create_from_data(cls, key, data, hdf_name, hdf_file):
        """ store the data in the file and return the storage object """
        data_cls = data.__class__
        hdf_file.create_dataset(key, data=data.to_array())
        if hasattr(data_cls, 'hdf_attributes'):        
            for attr_key, attr_value in data_cls.hdf_attributes.iteritems():
                hdf_file[key].attrs[attr_key] = attr_value
            
        return cls(data_cls, key, None, hdf_name)
    
        
    def load(self):
        """ load the data and return it """
        if self.hdf_folder is None:
            raise RuntimeError('Folder of the HDF file is unknown and data cannot be loaded.')
        
        # open the associated HDF5 file and read the data
        hdf_filepath = os.path.join(self.hdf_folder, self.hdf_name)
        with h5py.File(hdf_filepath, 'r') as hdf_file:
            data = hdf_file[self.key]
            result = self.data_cls.from_array(data)
        
        # create object
        return result



class LazyHDFCollection(LazyHDFValue):
    """ class that represents a collection of values that are only loaded when they are accessed """
   
    @classmethod    
    def create_from_data(cls, key, data, hdf_name, hdf_file):
        """ store the data in the file and return the storage object """
        data_cls = data.__class__

        # save a collection of objects to hdf
        key_format = '{}/%0{}d'.format(key, len(str(len(data))))
        for index, obj in enumerate(data):
            obj.save_to_hdf5(hdf_file, key_format % index)

        if hasattr(data_cls, 'hdf_attributes'):        
            for attr_key, attr_value in data_cls.hdf_attributes.iteritems():
                hdf_file[key].attrs[attr_key] = attr_value

        return cls(data_cls, key, None, hdf_name)
    
        
    def load(self):
        """ load the data and return it """
        if self.hdf_folder is None:
            raise RuntimeError('Folder of the HDF file is unknown and data cannot be loaded.')
                
        # open the associated HDF5 file and read the data
        hdf_filepath = os.path.join(self.hdf_folder, self.hdf_name)
        with h5py.File(hdf_filepath, 'r') as hdf_file:
            # iterate over the data and create objects from it
            data = hdf_file[self.key]
            result = self.data_cls.load_list(data)
                
        return result

