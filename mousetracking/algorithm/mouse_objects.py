'''
Created on Aug 19, 2014

@author: zwicker
'''

from __future__ import division

import itertools

import numpy as np
import cv2
import shapely
import shapely.geometry as geometry

from video.analysis import curves, regions
from video.analysis.regions import corners_to_rect, expand_rectangle, get_enclosing_outline

import debug


# monkey patch shapely.geometry to get compatibility with older shapely versions
if not hasattr(geometry, 'LinearRing'):
    geometry.LinearRing = geometry.polygon.LinearRing



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


    def __get__(self, obj, type=None):
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



class Object(object):
    """ represents a single object by its position and size """
    __slots__ = ['pos', 'size', 'label'] #< save some memory
    
    def __init__(self, pos, size, label=None):
        self.pos = (int(pos[0]), int(pos[1]))
        self.size = size
        self.label = label



class ObjectTrack(object):
    """ represents a time course of objects """
    # TODO: hold everything inside lists, not list of objects
    # TODO: speed up by keeping track of velocity vectors
    
    array_columns = ['Time', 'Position X', 'Position Y', 'Object Area']
    
    def __init__(self, time=None, obj=None, moving_window=20, moving_threshold=10):
        self.times = [] if time is None else [time]
        self.objects = [] if obj is None else [obj]
        self.moving_window = moving_window
        self.moving_threshold = moving_threshold*moving_window
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(span=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(span=%d..%d)' % (self.times[0], self.times[-1])
        
    def __len__(self):
        return len(self.times)
    
    @property
    def last_pos(self):
        """ return the last position of the object """
        return self.objects[-1].pos
    
    @property
    def last_size(self):
        """ return the last size of the object """
        return self.objects[-1].size
    
    def predict_pos(self):
        """ predict the position in the next frame.
        It turned out that setting the current position is the best predictor.
        This is because mice are often stationary (especially in complicated
        tracking situations, like inside burrows). Additionally, when mice
        throw out dirt, there are frames, where dirt + mouse are considered 
        being one object, which moves the center of mass in direction of the
        dirt. If in the next frame two objects are found, than it is likely
        that the dirt would be seen as the mouse, if we'd predict the position
        based on the continuation of the previous movement
        """
        return self.objects[-1].pos
        
    def get_track(self):
        """ return a list of positions over time """
        return [obj.pos for obj in self.objects]

    def append(self, time, obj):
        """ append a new object with a time code """
        self.times.append(time)
        self.objects.append(obj)
        
    def is_moving(self):
        """ return if the object has moved in the last frames """
        pos = self.objects[-1].pos
        dist = sum(curves.point_distance(pos, obj.pos)
                   for obj in self.objects[-self.moving_window:])
        return dist > self.moving_threshold
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        return np.array([(time, obj.pos[0], obj.pos[1], obj.size)
                         for time, obj in itertools.izip(self.times, self.objects)],
                        dtype=np.int32)

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        res = cls()
        res.times = [d[0] for d in data]
        res.objects = [Object(pos=(d[1], d[2]), size=d[3]) for d in data]
        return res


    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        hdf_file.create_dataset(key, data=self.to_array())
        hdf_file[key].attrs['column_names'] = self.array_columns


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.from_array(hdf_file[key])
        



class GroundProfile(object):
    """ dummy class representing a single ground profile at a certain point
    in time """
    
    array_columns = ['Time', 'Position X', 'Position Y']
    
    def __init__(self, time, points):
        self.time = time
        self.points = points
        
    def __repr__(self):
        return 'GroundProfile(time=%d, %d points)' % (self.time, len(self.points))
        
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        time_array = np.zeros((len(self.points), 1), np.int32) + self.time
        return np.hstack((time_array, self.points))

    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        data = np.asarray(data)
        return cls(data[0, 0], data[1:, :])
   
   
   
class Burrow(object):
    """ represents a single burrow """
    
    # parameters influencing how the centerline is determined
    centerline_angle = np.pi/6 #< TODO: turn this parameter into a radius of curvature
    centerline_segment_length = 25
    
    ground_point_distance = 10
    
    
    def __init__(self, outline, centerline=None, length=None, refined=False):
        """ initialize the structure using points on its outline """
        self._outline = np.asarray(outline, np.double)
        self.centerline = centerline
        self.length = length
        self.refined = refined
        self._cache = {}


    def copy(self):
        return Burrow(self.outline.copy())

        
    def __len__(self):
        return len(self.outline)
        
        
    def __repr__(self):
        polygon = self.polygon
        center = polygon.centroid
        return ('Burrow(center=(%d, %d), area=%s, points=%d)' %
                (center.x, center.y, polygon.area, len(self)))


    @property
    def outline(self):
        return self._outline

    
    @outline.setter
    def outline(self, value):
        self._outline = value
        # reset cache
        self.centerline = None
        self._cache = {}
        
        
    @cached_property
    def polygon(self):
        """ return the polygon of the burrow outline """
        return geometry.Polygon(np.asarray(self.outline, np.double))    
    
    
    @property
    def area(self):
        """ return the area of the burrow shape """
        return self.polygon.area
    
    
    @property
    def is_valid(self):
        return len(self.outline) > 3
    
    
    @cached_property
    def eccentricity(self):
        """ return the eccentricity of the burrow shape
        The eccentricity will be between 0 and 1, corresponding to a circle
        and a straight line, respectively.
        """
        m = cv2.moments(np.asarray(self.outline, np.uint8))
        a, b, c = m['mu20'], -m['mu11'], m['mu02']
        e1 = (a + c) + np.sqrt(4*b**2 + (a - c)**2)
        e2 = (a + c) - np.sqrt(4*b**2 + (a - c)**2)
        if e1 == 0:
            return 0
        else:
            return np.sqrt(1 - e2/e1)
    
                
    def contains(self, point):
        """ returns True if the point is inside the burrow """
        return self.polygon.contains(geometry.Point(point))
    
    
    def intersects(self, polygon):
        """ returns True if polygon intersects the burrow """
        try:
            return not self.polygon.intersection(polygon).is_empty
        except shapely.geos.TopologicalError:
            return False
    
    
    def simplify_outline(self, tolerance=0.1):
        """ simplifies the outline """
        outline = geometry.LineString(self.outline)
        tolerance = tolerance*outline.length
        outline = outline.simplify(tolerance, preserve_topology=True)
        self.outline = np.array(outline.coords, np.double)
    
    
    def get_bounding_rect(self, margin=0):
        """ returns the bounding rectangle of the burrow """
        bounds = self.polygon.bounds
        bound_rect = corners_to_rect(bounds[:2], bounds[2:])
        return expand_rectangle(bound_rect, margin)
    
    
    def extend_outline(self, extension_polygon, simplify_threshold):
        """ extends the outline of the burrow to also enclose the object given
        by polygon """
        # get the union of the burrow and the extension
        burrow = self.polygon.union(extension_polygon)
        
        # determine the outline of the union
        outline = get_enclosing_outline(burrow)
        
        outline = outline.simplify(simplify_threshold*outline.length)

        self.outline = np.asarray(outline, np.int32)

    
    def get_centerline(self, ground):
        """ determine the centerline, given the outline and the ground profile.
        The ground profile is used to determine the burrow exit. """
        
        if self.centerline is not None:
            return self.centerline
        
        # get the ground line 
        ground_line = geometry.LineString(np.array(ground, np.double))
        
        # reparameterize the burrow outline to locate the burrow exit reliably
        outline = curves.make_curve_equidistant(self.outline, 10)
        outline = np.asarray(outline, np.double)

        # calculate the distance of each outline point to the ground
        dist = np.array([ground_line.distance(geometry.Point(p)) for p in outline])
        
        # get points at the burrow exit (close to the ground profile)
        indices = (dist < self.ground_point_distance)
        if np.any(indices):
            p_exit = outline[indices, :].mean(axis=0)
        else:
            p_exit = np.argmin(outline)
        p_exit = curves.get_projection_point(ground_line, p_exit)
            
        # get the two points closest to the exit point
        dist = np.linalg.norm(outline - p_exit, axis=1)
        k1 = np.argmin(dist)
        dist[k1] = np.inf
        k2 = np.argmin(dist)
        p1, p2 = outline[k1], outline[k2]
        # get the points such that p1 is left of p2
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        
        # send out rays perpendicular to the ground profile
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) + np.pi/2
        p_anchor = (p_exit[0] + 5*np.cos(angle), p_exit[1] + 5*np.sin(angle))
        ray_length = np.inf
        outline_poly = geometry.LinearRing(self.outline)
        
        centerline = [p_exit]
        while True:
            dist_max, point_max = 0, None
            # try some rays distributed around `angle`
            for a in np.linspace(angle - self.centerline_angle,
                                 angle + self.centerline_angle, 16):
                
                p_far = (p_anchor[0] + 1000*np.cos(a),
                         p_anchor[1] + 1000*np.sin(a))
                
                p_hit, dist_hit = regions.get_ray_hitpoint(p_anchor, p_far,
                                                           outline_poly, ret_dist=True)
                if dist_hit > dist_max:
                    dist_max = dist_hit
                    point_max = p_hit
                    angle = a
                        
            # abort if the search was not successful
            if point_max is None:
                break
            
            # get the length of the longest ray
            ray_length = curves.point_distance(p_anchor, point_max)
            if ray_length > self.centerline_segment_length:
                # continue shooting out rays
                p_anchor = (p_anchor[0] + self.centerline_segment_length*np.cos(angle),
                            p_anchor[1] + self.centerline_segment_length*np.sin(angle))
                centerline.append(p_anchor)
            else:
                centerline.append(point_max)
                break
                    
        # save results                    
        self.centerline = centerline
        self.length = curves.curve_length(centerline)
        return centerline
            
        
    def get_length(self, ground):
        """ calculates the centerline and returns its length """
        self.get_centerline(ground)
        return self.length
            
    
    def to_array(self):
        """ converts the internal representation to a single array """
        return np.concatenate((np.array([[self.length, 0]], np.double),
                               np.asarray(self.outline, np.double)))
        

    @classmethod
    def from_array(cls, data):
        """ creates a burrow track from a single array """
        return cls(outline=data[1:], length=data[0][0])
        
        
        
class BurrowTrack(object):
    array_columns = ['Time', 'Position X', 'Position Y']
    
    def __init__(self, time=None, burrow=None):
        self.times = [] if time is None else [time]
        self.burrows = [] if burrow is None else [burrow]
        
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'BurrowTrack([])'
        elif len(self.times) == 1:
            return 'BurrowTrack(span=%d)' % (self.times[0])
        else:
            return 'BurrowTrack(span=%d..%d)' % (self.times[0], self.times[-1])
        
        
    def __len__(self):
        return len(self.times)
    
    
    @property
    def last(self):
        """ return the last position of the object """
        return self.burrows[-1]
    
    
    @property
    def last_seen(self):
        return self.times[-1]
    
    
    def append(self, time, burrow):
        """ append a new burrow with a time code """
        self.times.append(time)
        self.burrows.append(burrow)
    
    
    def to_array(self):
        """ converts the internal representation to a single array
        useful for storing the data """
        res = []
        for time, burrow in itertools.izip(self.times, self.burrows):
            burrow_data = burrow.to_array()
            time_array = np.zeros((len(burrow_data), 1), np.int32) + time
            res.append(np.hstack((time_array, burrow_data)))
        if res:
            return np.vstack(res)
        else:
            return []
        
        
    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        burrow_track = cls()
        burrow_data = None
        time_cur = -1
        for d in data:
            print d
            if d[0] != time_cur:
                if burrow_data is not None:
                    burrow_track.append(time_cur, Burrow.from_array(burrow_data))
                time_cur = d[0]
                burrow_data = [d[1:]]
            else:
                burrow_data.append(d[1:])

        if burrow_data is not None:
            burrow_track.append(time_cur, Burrow.from_array(burrow_data))

        return burrow_track
    
 
    def save_to_hdf5(self, hdf_file, key):
        """ save the data of the current burrow to an HDF5 file """
        hdf_file.create_dataset(key, data=self.to_array())
        hdf_file[key].attrs['column_names'] = self.array_columns
        hdf_file[key].attrs['remark'] = (
            'Each burrow is represented by its outline saved as a list of points '
            'of the format (Time, X, Y), where all points with the same Time belong '
            'to the same burrow. However, the first entry always contains the '
            'burrow length and is not part of the burrow outline'
        ) 


    @classmethod
    def create_from_hdf5(cls, hdf_file, key):
        """ creates a burrow track from data in a HDF5 file """
        return cls.from_array(hdf_file[key])
   
    
class RidgeProfile(object):
    """ represents a ridge profile to compare it against an image in fitting """
    
    def __init__(self, size, profile_width=1):
        """ initialize the structure
        size is half the width of the region of interest
        profile_width determines the blurriness of the ridge
        """
        self.size = size
        self.ys, self.xs = np.ogrid[-size:size+1, -size:size+1]
        self.width = profile_width
        self.image = None
        
        
    def set_data(self, image, angle):
        """ sets initial data used for fitting
        image denotes the data we compare the model to
        angle defines the direction perpendicular to the profile 
        """
        
        self.image = image - image.mean()
        self.image_std = image.std()
        self._sina = np.sin(angle)
        self._cosa = np.cos(angle)
        
        
    def get_difference(self, distance):
        """ calculates the difference between image and model, when the 
        model is moved by a certain distance in its normal direction """ 
        # determine center point
        px =  distance*self._cosa
        py = -distance*self._sina
        
        # determine the distance from the ridge line
        dist = (self.ys - py)*self._sina - (self.xs - px)*self._cosa
        
        # apply sigmoidal function
        model = np.tanh(dist/self.width)
     
        return np.ravel(self.image - 1.5*self.image_std*model)


    