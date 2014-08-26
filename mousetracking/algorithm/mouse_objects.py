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

from video.analysis import curves
from video.analysis.regions import corners_to_rect, expand_rectangle, get_enclosing_outline

import debug



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
    index_columns = 0
    
    def __init__(self, time=None, obj=None, moving_window=20, moving_threshold=10):
        self.times = [] if time is None else [time]
        self.objects = [] if obj is None else [obj]
        self.moving_window = moving_window
        self.moving_threshold = moving_threshold*moving_window
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'ObjectTrack([])'
        elif len(self.times) == 1:
            return 'ObjectTrack(time=%d)' % (self.times[0])
        else:
            return 'ObjectTrack(time=%d..%d)' % (self.times[0], self.times[-1])
        
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



class GroundProfile(object):
    """ dummy class representing a single ground profile at a certain point
    in time """
    
    array_columns = ['Time', 'Position X', 'Position Y']
    index_columns = 1
    
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
    """ represents a single burrow to compare it against an image in fitting """
    
    array_columns = ['Position X', 'Position Y']
    index_columns = 0 #< there could be multiple burrows at each time point
    # Hence, time can not be used as an index
    
    def __init__(self, outline, time=None):
        """ initialize the structure
        """
        self.outline = np.asarray(outline, np.double)
        self.time = time
        
        # internal caches used for fitting
#         self.image = image
#         self.mask = mask
#         self._angles = None #< internal cache
        self._color_burrow = None
        self._color_sand = None
#         self._model = None
        
        
#     def clear_cache(self):
#         self.image = None
#         self.mask = None
#         self._angles = None
#         self._color_burrow = None
#         self._color_sand = None
#         self._model = None 


    def copy(self):
        return Burrow(self.outline.copy())

        
    def __len__(self):
        return len(self.outline)
        
        
    def __repr__(self):
        polygon = self.polygon
        center = polygon.centroid
        return 'Burrow(center=(%d, %d), area=%s, points=%d)' % \
                            (center.x, center.y, polygon.area, len(self))
        
    
    @property
    def polygon(self):
        return geometry.Polygon(np.asarray(self.outline, np.double))    
    
    
    @property
    def area(self):
        return self.polygon.area
    
    
    @property
    def is_valid(self):
        return len(self.outline) > 3
    
    
    @property
    def eccentricity(self):
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

        # debug.show_shape(burrow, polygon, outline)
        
#         # find indices of the anchor points
#         i1 = outline.index([int(self.outline[ 0][0]), int(self.outline[ 0][1])])
#         i2 = outline.index([int(self.outline[-1][0]), int(self.outline[-1][1])])
#         i1, i2 = min(i1, i2), max(i1, i2)
#         
#         # figure out in what direction we have to go around the polygon
#         if i2 - i1 > len(outline)//2:
#             # the right outline goes from i1 .. i2
#             self.outline = outline[i1:i2+1]
#         else:
#             # the right outline goes from i2 .. -1 and start 0 .. i1
#             self.outline = outline[i2:] + outline[:i1]  
     

#     def show_image(self, mark_points):
#         # draw burrow
#         image = self.image.copy()
#         cv2.drawContours(image, np.array([self.outline], np.int32), -1, 255, 1)
#         for k, p in enumerate(self.outline):
#             color = 255 if mark_points[k] else 128
#             cv2.circle(image, (int(p[0]), int(p[1])), 3, color, thickness=-1)
#         debug.show_image(image)
    
    
    def get_centerline(self, ground):
        ground_line = geometry.LineString(np.array(ground, np.double))
        
        outline = curves.make_curve_equidistant(self.outline, 10)        
        
        outline = np.asarray(outline, np.double)
        
        dist = np.array([ground_line.distance(geometry.Point(p)) for p in outline])
        
        # estimate the burrow exit point
        indices = (dist < 25)
        if np.any(indices):
            p_surface = outline[indices, :].mean(axis=0)
        else:
            p_surface = np.argmin(outline)
        k0 = np.argmin(np.linalg.norm(outline - p_surface, axis=1))
        kl = kr = k0
        l = len(outline)
        centerline = []
        while kr - kl < l:
            p = ((outline[kl%l][0] + outline[kr%l][0])/2,
                 (outline[kl%l][1] + outline[kr%l][1])/2)
            centerline.append(p)
            kr += 1
            kl -= 1
        
#        p_deep = self.outline[np.argmax(dist)]
        
        return centerline #(p_deep, p_surface)
        
    
    
    def to_array(self):
        """ converts the internal representation to a single array """
        return np.asarray(self.outline, np.int32)


    @classmethod
    def from_array(cls, data):
        return cls(outline=data)
        
        
    
class BurrowLine(object):
    
    def __init__(self, centerline, widths):
        self.centerline = centerline
        self.widths = widths
        
    
    def copy(self):
        # np.array is called to make sure we get copies of the data
        return BurrowLine(np.array(self.centerline), np.array(self.widths))
        
        
    def __len__(self):
        return len(self.widths)
       
    
    @property
    def outline(self):
        """ constructs the burrow outline from the centerline and the widths """
        centerline = self.centerline
        line1, line2 = [], []
        
        # iterate and build the two side lines
        for k, p in enumerate(centerline):
            if k == 0:
                p1, p2 = centerline[0], centerline[1]
            elif k == len(self)-1:
                p1, p2 = centerline[-2], centerline[-1]
            else:
                p1, p2 = centerline[k-1], centerline[k+1]
            
            w = self.widths[k]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx**2 + dy**2)

#             # get first point
#             if k == 0:        
#                 line2.append((p[0] - w*dx/dist, p[1] - w*dy/dist))
            
            if k > 0:
                line1.append((p[0] + w*dy/dist, p[1] - w*dx/dist))
                line2.append((p[0] - w*dy/dist, p[1] + w*dx/dist))

        # construct the outline
        return line1[::-1] + [centerline[0]] + line2
        
        
    @property
    def polygon(self):
        """ returns the burrow outline polygon """
        return geometry.Polygon(np.array(self.outline, np.double))

    

class BurrowTrack(object):
    array_columns = ['Time', 'Position X', 'Position Y']
    index_columns = 0 #< there could be multiple burrows at each time point
    
    def __init__(self, time=None, burrow=None):
        self.times = [] if time is None else [time]
        self.burrows = [] if burrow is None else [burrow]
        
        
    def __repr__(self):
        if len(self.times) == 0:
            return 'BurrowTrack([])'
        elif len(self.times) == 1:
            return 'BurrowTrack(time=%d)' % (self.times[0])
        else:
            return 'BurrowTrack(time=%d..%d)' % (self.times[0], self.times[-1])
        
        
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
            time_array = np.zeros((len(burrow), 1), np.int32) + time
            res.append(np.hstack((time_array, burrow.to_array())))
        if res:
            return np.vstack(res)
        else:
            return []


    @classmethod
    def from_array(cls, data):
        """ constructs an object from an array previously created by to_array() """
        burrow_track = cls()
        outline = []
        time_cur = -1
        for d in data:
            if d[0] != time_cur:
                if outline is not None:
                    burrow_track.append(time_cur, outline)
                time_cur = d[0]
                outline = [d[1:]]
            else:
                outline.append(d[1:])

        if outline is not None:
            burrow_track.append(time_cur, outline)

        return burrow_track
    
    
    
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


    