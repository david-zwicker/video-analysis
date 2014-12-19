'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import sys

from .lazy_values import LazyValue



class LazyLoadError(RuntimeError):
    pass



class DictXpath(collections.MutableMapping):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key:
    
    d = DictXpath({'a': {'b': 1}})
    
    d['a/b']
    >>>> 1
    
    d['a/c'] = 2
    
    d
    >>>> {'a': {'b': 1, 'c': 2}}
    """
    
    sep = '/'
    
    def __init__(self, data=None):
        # set data
        self.data = {}
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
                                'are DictXpath instances.')
                
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
            if flatten and isinstance(value, DictXpath):
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
                if isinstance(value, DictXpath):
                    # recurse into sub dictionary
                    try:
                        prefix = key + self.sep
                    except TypeError:
                        raise TypeError('Keys for DictXpath must be strings '
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
            if flatten and isinstance(value, DictXpath):
                # recurse into sub dictionary
                try:
                    prefix = key + self.sep
                except TypeError:
                    raise TypeError('Keys for DictXpath must be strings '
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
            if isinstance(value, (dict, DictXpath)):
                value = value.copy()
            res[key] = value
        return res


    def from_dict(self, data):
        """ fill the object with data from a dictionary """
        for key, value in data.iteritems():
            if isinstance(value, dict):
                if key in self and isinstance(self[key], DictXpath):
                    # extend existing DictXpath structure
                    self[key].from_dict(value)
                else:
                    # create new DictXpath structure
                    self[key] = self.__class__(value)
            else:
                # store simple value
                self[key] = value

            
    def to_dict(self, flatten=False):
        """ convert object to a nested dictionary structure.
        If flatten is True a single dictionary with complex keys is returned.
        If flatten is False, a nested dictionary with simple keys is returned """
        res = {}
        for key, value in self.iteritems():
            if isinstance(value, DictXpath):
                value = value.to_dict(flatten)
                if flatten:
                    for k, v in value.iteritems():
                        try:
                            res[key + self.sep + k] = v
                        except TypeError:
                            raise TypeError('Keys for DictXpath must be strings '
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



class DictXpathLazy(DictXpath):
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
                # KeyErrors raised by the item actually not being in the DictXpath
                # This then allows us to distinguish between items not found in
                # DictXpath (raising KeyError) and items not being able to load
                # (raising LazyLoadError)
                err_msg = ('Cannot load item `%s`.\nThe original error was: %s'
                           % (key, err)) 
                raise LazyLoadError, err_msg, sys.exc_info()[2] 
            self.data[key] = value #< replace loader with actual value
            
        return value
    