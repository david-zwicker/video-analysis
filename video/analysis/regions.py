'''
Created on Aug 4, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from collections import defaultdict
import itertools

import cv2
import numpy as np
from scipy import ndimage, interpolate
from shapely import geometry, geos

import curves
from active_contour import ActiveContour
from data_structures.cache import cached_property
from external import simplify_polygon_visvalingam as simple_poly 

from .. import debug  # @UnusedImport


def corners_to_rect(p1, p2):
    """ creates a rectangle from two corner points.
    The points are both included in the rectangle.
    """
    xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0]) 
    ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
    return (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)



def rect_to_corners(rect, count=2):
    """ returns `count` corner points for a rectangle.
    These points are included in the rectangle.
    count determines the number of corners. 2 and 4 are allowed values.
    """
    p1 = (rect[0], rect[1])
    p2 = (rect[0] + rect[2] - 1, rect[1] + rect[3] - 1)
    if count == 2:
        return p1, p2
    elif count == 4:
        return p1, (p2[0], p1[1]), p2, (p1[0], p2[1])
    else:
        raise ValueError('count must be 2 or 4 (cannot be %d)' % count)



def rect_to_slices(rect):
    """ creates slices for an array from a rectangle """
    slice_x = slice(rect[0], rect[2] + rect[0])
    slice_y = slice(rect[1], rect[3] + rect[1])
    return slice_y, slice_x



def get_overlapping_slices(t_pos, t_shape, i_shape, anchor='center', ret_rect=False):
    """ calculates slices to compare parts of two images with each other
    i_shape is the shape of the larger image in which a smaller image of
    shape t_shape will be placed.
    Here, t_pos specifies the position of the smaller image in the large one.
    The input variables follow this convention:
        t_pos = (x-position, y-position)
        t_shape = (template-height, template-width)
        i_shape = (image-height, image-width)
    """
    
    # get dimensions to determine center position         
    t_top  = t_shape[0]//2
    t_left = t_shape[1]//2
    if anchor == 'center':
        pos = (t_pos[0] - t_left, t_pos[1] - t_top)
    elif anchor == 'upper left':
        pos = t_pos
    else:
        raise ValueError('Unknown anchor point: %s' % anchor)

    # get the dimensions of the overlapping region        
    h = min(t_shape[0], i_shape[0] - pos[1])
    w = min(t_shape[1], i_shape[1] - pos[0])
    if h <= 0 or w <= 0:
        raise RuntimeError('Template and image do not overlap')
    
    # get the leftmost point in both images
    if pos[0] >= 0:
        i_x, t_x = pos[0], 0
    elif pos[0] <= -t_shape[1]:
        raise RuntimeError('Template and image do not overlap')
    else: # pos[0] < 0:
        i_x, t_x = 0, -pos[0]
        w += pos[0]
        
    # get the upper point in both images
    if pos[1] >= 0:
        i_y, t_y = pos[1], 0
    elif pos[1] <= -t_shape[0]:
        raise RuntimeError('Template and image do not overlap')
    else: # pos[1] < 0:
        i_y, t_y = 0, -pos[1]
        h += pos[1]
        
    # build the slices used to extract the information
    slices= ((slice(t_y, t_y + h), slice(t_x, t_x + w)),  # slice for the template
             (slice(i_y, i_y + h), slice(i_x, i_x + w)))  # slice for the image
    
    if ret_rect:
        return slices, (i_x, i_y, w, h)
    else:
        return slices



def find_bounding_box(mask):
    """ finds the rectangle, which bounds a white region in a mask.
    The rectangle is returned as [left, top, width, height]
    Currently, this function only works reliably for connected regions 
    """

    # find top boundary
    top = 0
    while not np.any(mask[top, :]):
        top += 1
    # top contains the first non-empty row
    
    # find bottom boundary
    bottom = top + 1
    try:
        while np.any(mask[bottom, :]):
            bottom += 1
    except IndexError:
        bottom = mask.shape[0]
    # bottom contains the first empty row

    # find left boundary
    left = 0
    while not np.any(mask[:, left]):
        left += 1
    # left contains the first non-empty column
    
    # find right boundary
    try:
        right = left + 1
        while np.any(mask[:, right]):
            right += 1
    except IndexError:
        right = mask.shape[1]
    # right contains the first empty column
    
    return (left, top, right - left, bottom - top)


       
def expand_rectangle(rect, amount=1):
    """ expands a rectangle by a given amount """
    return (rect[0] - amount, rect[1] - amount, rect[2] + 2*amount, rect[3] + 2*amount)
    
       
       
def get_largest_region(mask, ret_area=False):
    """ returns a mask only containing the largest region """
    # find all regions and label them
    labels, num_features = ndimage.measurements.label(mask)

    # find the label of the largest region
    label_max = np.argmax(
        ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
    ) + 1
    
    return labels == label_max



def get_external_contour(points, resolution=None):
    """ takes a list of `points` defining a linear ring, which can be self-intersecting,
    and returns an approximation to the external contour """
    if resolution is None:
        # determine resolution from minimal distance of consecutive points
        dist_min = np.inf
        for p1, p2 in itertools.izip(np.roll(points, 1, axis=0), points):
            dist = curves.point_distance(p1, p2)
            if dist > 0:
                dist_min = min(dist_min, dist)
        resolution = 0.5*dist_min
        
        # limit the resolution such that there are at most 2048 points
        dim_max = np.max(np.ptp(points, axis=0)) #< longest dimension
        resolution = max(resolution, dim_max/2048)

    # build a linear ring with integer coordinates
    ps_int = np.array(np.asarray(points)/resolution, np.int)
    ring = geometry.LinearRing(ps_int)

    # get the image of the linear ring by plotting it into a mask
    x_min, y_min, x_max, y_max = ring.bounds
    shape = ((y_max - y_min) + 3, (x_max - x_min) + 3)
    x_off, y_off = int(x_min - 1), int(y_min - 1)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [ps_int], 255, offset=(-x_off, -y_off))

    # find the contour of this mask to recover the exterior contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(x_off, y_off))
    return np.array(np.squeeze(contours))*resolution



def get_enclosing_outline(polygon):
    """ gets the enclosing contour of a (possibly complex) polygon """
    polygon = regularize_polygon(polygon)

    # get the contour
    try:
        contour = polygon.boundary
    except ValueError:
        # return empty feature since `polygon` was not a valid polygon
        return geometry.LinearRing()
    
    if isinstance(contour, geometry.multilinestring.MultiLineString):
        largest_polygon = None
        # find the largest polygon, which should be the enclosing contour
        for line in contour:
            poly = geometry.Polygon(line)
            if largest_polygon is None or poly.area > largest_polygon.area:
                largest_polygon = poly
        if largest_polygon is None:
            contour = geometry.LinearRing()
        else:
            contour = largest_polygon.boundary
    return contour
    
    
    
def regularize_polygon(polygon):
    """ regularize a shapely polygon using polygon.buffer(0) """
    area_orig = polygon.area #< the result should have a similar area
    
    # try regularizing polygon using the buffer(0) trick
    result = polygon.buffer(0)
    if isinstance(result, geometry.MultiPolygon):
        # retrieve the result with the largest area
        result = max(result, key=lambda obj: obj.area)
        
    # check the resulting area
    if result.area < 0.5*area_orig:
        # the polygon was likely complex and the buffer(0) trick did not work
        # => we use a more reliable but slower method
        contour = get_external_contour(polygon.exterior.coords)
        result = geometry.Polygon(contour)
    
    return result



def regularize_linear_ring(linear_ring):
    """ regularize a list of points defining a contour """
    polygon = geometry.Polygon(linear_ring)
    regular_polygon = regularize_polygon(polygon)
    if regular_polygon.is_empty:
        return geometry.LinearRing() #< empty linear ring
    else:
        return regular_polygon.exterior



def regularize_contour_points(contour):
    """ regularize a list of points defining a contour """
    if len(contour) >= 3:
        polygon = geometry.Polygon(np.asarray(contour, np.double))
        regular_polygon = regularize_polygon(polygon)
        if regular_polygon.is_empty:
            return [] #< empty list of points
        else:
            contour = regular_polygon.exterior.coords
    return contour



def simplify_contour(contour, threshold):
    """ simplifies a contour based on its area.
    Single points are removed if the area change of the resulting polygon
    is smaller than `threshold`. 
    """
    if isinstance(contour, geometry.LineString):
        return simple_poly.simplify_line(contour, threshold)
    elif isinstance(contour, geometry.LinearRing):
        return simple_poly.simplify_ring(contour, threshold)
    elif isinstance(contour, geometry.Polygon):
        return simple_poly.simplify_polygon(contour, threshold)
    else:
        # assume contour are coordinates of a linear ring
        ring = geometry.LinearRing(contour)
        ring = simple_poly.simplify_ring(ring, threshold)
        if ring is None:
            return None
        else:
            return ring.coords[:-1]



def get_intersections(geometry1, geometry2):
    """ get intersection points between two (line) geometries """
    # find the intersections between the ray and the burrow contour
    try:
        inter = geometry1.intersection(geometry2)
    except geos.TopologicalError:
        return []
    
    # process the result
    if inter is None or inter.is_empty:
        return []    
    elif isinstance(inter, geometry.Point):
        # intersection is a single point
        return [inter.coords[0]]
    elif isinstance(inter, geometry.MultiPoint):
        # intersection contains multiple points
        return [p.coords[0] for p in inter]
    else:
        # intersection contains objects of lines
        # => we cannot do anything sensible and thus return nothing
        return []

    
    
def get_ray_hitpoint(point_anchor, point_far, line_string, ret_dist=False):
    """ returns the point where a ray anchored at point_anchor hits the polygon
    given by line_string. The ray extends out to point_far, which should be a
    point beyond the polygon.
    If ret_dist is True, the distance to the hit point is also returned.
    """
    # define the ray
    ray = geometry.LineString((point_anchor, point_far))
    
    # find the intersections between the ray and the burrow contour
    try:
        inter = line_string.intersection(ray)
    except geos.TopologicalError:
        inter = None
    
    # process the result    
    if isinstance(inter, geometry.Point):
        if ret_dist:
            # also return the distance
            dist = curves.point_distance(inter.coords[0], point_anchor)
            return inter.coords[0], dist
        else:
            return inter.coords[0]

    elif inter is not None and not inter.is_empty:
        # find closest intersection if there are many points
        dists = [curves.point_distance(p.coords[0], point_anchor) for p in inter]
        k_min = np.argmin(dists)
        if ret_dist:
            return inter[k_min].coords[0], dists[k_min]
        else:
            return inter[k_min].coords[0]
        
    else:
        # return empty result
        if ret_dist:
            return None, np.nan
        else:
            return None
        
    

def get_ray_intersections(point_anchor, angles, polygon, ray_length=1000):
    """ shoots out rays from point_anchor in different angles and determines
    the points where polygon is hit.
    """
    points = []
    for angle in angles:
        point_far = (point_anchor[0] + ray_length*np.cos(angle),
                     point_anchor[1] + ray_length*np.sin(angle))
        point_hit = get_ray_hitpoint(point_anchor, point_far, polygon)
        points.append(point_hit)
    return points


    
def get_farthest_ray_intersection(point_anchor, angles, polygon, ray_length=1000):
    """ shoots out rays from point_anchor in different angles and determines
    the farthest point where polygon is hit.
    Returns the hit point, its distance to point_anchor and the associated
    angle
    """
    point_max, dist_max, angle_max = None, 0, None
    # try some rays distributed around `angle`
    for angle in angles:
        point_far = (point_anchor[0] + ray_length*np.cos(angle),
                     point_anchor[1] + ray_length*np.sin(angle))
        point_hit, dist_hit = get_ray_hitpoint(point_anchor, point_far,
                                               polygon, ret_dist=True)
        if dist_hit > dist_max:
            dist_max = dist_hit
            point_max = point_hit
            angle_max = angle
    return point_max, dist_max, angle_max



def triangle_area(a, b, c):
    """ returns the area of a triangle with sides a, b, c 
    Note that the input could also be numpy arrays
    """
    # use Heron's formula to calculate triangle area
    s = (a + b + c)/2
    radicand = s*(s - a)*(s - b)*(s - c)
    
    if isinstance(radicand, np.ndarray):
        i = (radicand > 0)
        np.sqrt(radicand[i], out=radicand[i])
        # sometimes rounding errors produce small negative quantities
        radicand[~i] = 0
        return radicand

    else:
        # input is plain number
        if radicand > 0:
            return np.sqrt(radicand)
        else:
            # sometimes rounding errors produce small negative quantities
            return 0



def make_distance_map(mask, start_points, end_points=None):
    """
    fills a binary region of the array `mask` with new values.
    The values are based on the distance to the start points `start_points`,
    which must lie in the domain.
    If end_points are supplied, the functions stops when any of these
    points is reached.
    The function does not return anything but rather modifies the mask itself
    """
    if end_points is None:
        end_points = set()
    else:
        end_points = set(end_points)

    SQRT2 = np.sqrt(2)
    
    # initialize the shape
    ymax, xmax = mask.shape[:2]
    stack = defaultdict(set)
    stack[2] = set(start_points) #< initialize the stack with the start points

    # loop until all points are filled
    while stack:
        # get next distance to consider
        dist = min(stack.keys())
        # iterate through all points with the minimal distance
        for x, y in stack.pop(dist):
            # check whether x, y is a valid point that can be filled
            # Note that we only write and check each point once. This is valid
            # since we fill points one after another and can thus ensure that
            # we write the closest points first. We tested that changing the
            # condition to mask[x, y] > dist does not change the result
            if 0 <= x < xmax and 0 <= y < ymax and mask[y, x] == 1:
                mask[y, x] = dist
                
                # finish if we found an end point
                if (x, y) in end_points:
                    return 
                
                # add all surrounding points to the stack
                stack[dist + 1] |= set(((x - 1, y), (x + 1, y),
                                        (x, y - 1), (x, y + 1)))

                stack[dist + SQRT2] |= set(((x - 1, y - 1), (x + 1, y - 1),
                                            (x - 1, y + 1), (x + 1, y + 1)))



MASK = np.zeros((3, 3))
MASK[1, :] = MASK[:, 1] = 1
MASK[0, 0] = MASK[2, 0] = MASK[0, 2] = MASK[2, 2] = np.sqrt(2)

def shortest_path_in_distance_map(distance_map, end_point):
    """ finds and returns the shortest path in the distance map `distance_map` that
    leads from the given `end_point` to a start point (defined by having the
    minimal distance value in the map) """
    # make sure points outside the shape are not included in the distance
    distance_map[distance_map <= 1] = np.iinfo(distance_map.dtype).max
    
    xmax = distance_map.shape[1] - 1
    ymax = distance_map.shape[0] - 1
    x, y = end_point
    points = [end_point]
    d = distance_map[y, x]

    # iterate through path until we reached the minimum
    while True:
        if 0 < x < xmax and 0 < y < ymax:
            # find point with minimal distance in surrounding
            surrounding = (distance_map[y-1:y+2, x-1:x+2] - d) / MASK
            dy, dx = np.unravel_index(surrounding.argmin(), (3, 3))
            # get new coordinates
            x += dx - 1
            y += dy - 1
            # check whether the new point is viable
            if distance_map[y, x] < d:
                # distance decreased
                d = distance_map[y, x]
            elif distance_map[y, x] == d:
                # distance stayed constant
                if (x, y) in points:
                    # we already saw this point
                    break
            else:
                # we reached a minimum and will thus stop
                break            
            points.append((x, y))
        else:
            # we reached the border
            break
    
    return points



class Rectangle(object):
    """ a class for handling rectangles """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    @classmethod
    def from_points(cls, p1, p2):
        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
        y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
        return cls(x1, y1, x2 - x1, y2 - y1)
    
    @classmethod
    def from_list(cls, data):
        return cls(*data)

    def to_list(self):
        return [self.x, self.y, self.width, self.height]
    
    def copy(self):
        return self.__class__(self.x, self.y, self.width, self.height)
        
    def __repr__(self):
        return ("Rectangle(x=%g, y=%g, width=%g, height=%g)"
                % (self.x, self.y, self.width, self.height))
            
    @property
    def data(self):
        return self.x, self.y, self.width, self.height
    
    @property
    def data_int(self):
        return (int(self.x), int(self.y),
                int(self.width), int(self.height))
    
    @property
    def left(self):
        return self.x
    @left.setter
    def left(self, value):
        self.x = value
    
    @property
    def right(self):
        return self.x + self.width
    @right.setter
    def right(self, value):
        self.width = value - self.x
    
    @property
    def top(self):
        return self.y
    @top.setter
    def top(self, value):
        self.y = value
    
    @property
    def bottom(self):
        return self.y + self.height
    @bottom.setter
    def bottom(self, value):
        self.height = value - self.y        

    def set_corners(self, p1, p2):
        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
        y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
        self.x = x1
        self.y = y1
        self.width = x2 - x1
        self.height = y2 - y1 
            
    @property
    def corners(self):
        return (self.x, self.y), (self.x + self.width, self.y + self.height)
    @corners.setter
    def corners(self, ps):
        self.set_corners(ps[0], ps[1])
    
    @property
    def contour(self):
        x2, y2 = self.x + self.width, self.y + self.height
        return ((self.x, self.y), (x2, self.y),
                (x2, y2), (self.x, y2))
    
    @property
    def slices(self):
        slice_x = slice(self.x, self.x + self.width)
        slice_y = slice(self.y, self.y + self.height)
        return slice_x, slice_y

    @property
    def p1(self):
        return (self.x, self.y)
    @p1.setter
    def p1(self, p):
        self.set_corners(p, self.p2)
           
    @property
    def p2(self):
        return (self.x + self.width, self.y + self.height)
    @p2.setter
    def p2(self, p):
        self.set_corners(self.p1, p)
        
    def buffer(self, amount):
        """ dilate the rectangle by a certain amount in all directions """
        self.x -= amount
        self.y -= amount
        self.width += 2*amount
        self.height += 2*amount
    
    def intersection(self, other):
        """ return the intersection between this rectangle and the other """
        left = max(self.left, other.left) 
        right = min(self.right, other.right)
        top = max(self.top, other.top)
        bottom = min(self.bottom, other.bottom)
        return Rectangle.from_points((left, top), (right, bottom))
        
    @property
    def area(self):
        return self.width * self.height
    
    
    def points_inside(self, points):
        """ returns a boolean array indicating which of the points are inside
        this rectangle """
        return ((self.left <= points[:, 0]) & (points[:, 0] <= self.right) &
                (self.top  <= points[:, 1]) & (points[:, 1] <= self.bottom))



class Polygon(object):
    """ class that represents a single polygon """
    
    def __init__(self, contour):
        if len(contour) < 3:
            raise ValueError("Polygon must have at least three points.")
        self.contour = np.asarray(contour, np.double)


    def copy(self):
        return Polygon(self.contour.copy())


    @property
    def contour(self):
        return self._contour
    
    @contour.setter 
    def contour(self, points):
        """ set the contour of the burrow.
        `point_list` can be a list/array of points or a shapely LinearRing
        """ 
        if points is None:
            self._contour = None
        else:
            if isinstance(points, geometry.LinearRing):
                ring = points
            else:
                ring = geometry.LinearRing(points)
                
            # make sure that the contour is given in clockwise direction
            self._contour = np.array(points, np.double)
            if ring.is_ccw:
                self._contour = self._contour[::-1]
            
        self._cache = {} #< reset cache
        
        
    @cached_property
    def contour_ring(self):
        """ return the linear ring of the burrow contour """
        return geometry.LinearRing(self.contour)
    
        
    @cached_property
    def polygon(self):
        """ return the polygon of the burrow contour """
        return geometry.Polygon(self.contour)
    
    
    @cached_property
    def centroid(self):
        return np.array(self.polygon.centroid)
    
    
    @cached_property
    def position(self):
        return np.array(self.polygon.representative_point())
    
    
    @cached_property
    def area(self):
        """ return the area of the burrow shape """
        return self.polygon.area
    
    
    @cached_property
    def eccentricity(self):
        """ return the eccentricity of the burrow shape
        The eccentricity will be between 0 and 1, corresponding to a circle
        and a straight line, respectively.
        """
        m = cv2.moments(np.asarray(self.contour, np.uint8))
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
    
    
    def get_bounding_rect(self, margin=0):
        """ returns the bounding rectangle of the burrow """
        bounds = geometry.MultiPoint(self.contour).bounds
        bound_rect = corners_to_rect(bounds[:2], bounds[2:])
        if margin:
            bound_rect = expand_rectangle(bound_rect, margin)
        return np.asarray(bound_rect, np.int)
            
        
    def get_mask(self, margin=0, dtype=np.uint8, ret_offset=False):
        """ builds a mask of the burrow """
        # prepare the array to store the mask into
        rect = self.get_bounding_rect(margin=margin)
        mask = np.zeros((rect[3], rect[2]), dtype)

        # draw the burrow into the mask
        contour = np.asarray(self.contour, np.int)
        offset = (-rect[0], -rect[1])
        cv2.fillPoly(mask, [contour], color=1, offset=offset)
        
        if ret_offset:
            return mask, (-offset[0], -offset[1])
        else:
            return mask
        
        
    def get_centerline_estimate(self, end_points=None):
        """ determines an estimate to a center line of the polygon
        `end_points` can either be None, a single Point, or two points.
        """
        
        def _find_point_connection(p1, p2=None, maximize_distance=False):
            """ estimate centerline between the one or two points """
            mask, offset = self.get_mask(margin=1, dtype=np.int32,
                                         ret_offset=True)
            p1 = (p1[0] - offset[0], p1[1] - offset[1])
            if maximize_distance or p2 is None:
                dist_prev = 0 if maximize_distance else np.inf
                # iterate until second point is found
                while True:
                    # make distance map starting from point p1
                    distance_map = mask.copy()
                    make_distance_map(distance_map, start_points=(p1,))
                    # find point farthest point away from p1
                    idx_max = np.unravel_index(distance_map.argmax(),
                                               distance_map.shape)
                    dist = distance_map[idx_max]
                    p2 = idx_max[1], idx_max[0]
                    # print 'p1', p1, 'p2', p2
                    if dist <= dist_prev:
                        break
                    dist_prev = dist
                    # take farthest point as new start point
                    p1 = p2
            else:
                # locate the centerline between the two given points
                p2 = (p2[0] - offset[0], p2[1] - offset[1])
                distance_map = mask
                make_distance_map(distance_map,
                                  start_points=(p1,),end_points=(p2,))
                
            # find path between p1 and p2
            path = shortest_path_in_distance_map(distance_map, p2)
            return curves.translate_points(path, *offset)
        
        
        if end_points is None:
            # determine both end points
            path = _find_point_connection(np.array(self.position),
                                          maximize_distance=True)
        else:
            end_points = np.squeeze(end_points)
            if end_points.shape == (2, ):
                # determine one end point
                path = _find_point_connection(end_points,
                                              maximize_distance=False)
            elif end_points.shape == (2, 2):
                # both end points are already determined
                path = _find_point_connection(end_points[0], end_points[1])
            else:
                raise TypeError('`end_points` must have shape (2,) or (2, 2)')
            
        return path
        
        
    def get_centerline_optimized(self, alpha=1e3, beta=1e6, gamma=0.01,
                                 spacing=20, max_iterations=1000):
        """ determines the center line of the polygon using an active contour
        algorithm """
        # use an active contour algorithm to find centerline
        ac = ActiveContour(blur_radius=1, alpha=alpha, beta=beta,
                           gamma=gamma, closed_loop=False)
        ac.max_iterations = max_iterations

        # set the potential from the  distance map
        mask, offset = self.get_mask(1, ret_offset=True)
        potential = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 5)
        ac.set_potential(potential)
        
        # initialize the centerline from the estimate
        points = self.get_centerline_estimate()
        points = curves.make_curve_equidistant(points, spacing=spacing)        
        points = curves.translate_points(points, -offset[0], -offset[1])
        # anchor the end points
        anchor = np.zeros(len(points), np.bool)
        anchor[0] = anchor[-1] = True
        # find the best contour
        points = ac.find_contour(points, anchor, anchor)
        
        points = curves.make_curve_equidistant(points, spacing=spacing)        
        return curves.translate_points(points, *offset)
        

    def get_centerline_smoothed(self, points=None, skip_length=90):
        """ determines the center line of the polygon using an active contour
        algorithm. If `points` are given, they are used for getting the
        smoothed centerline. Otherwise, we determine the optimized centerline 
        """
        if points is None:
            points = self.get_centerline_optimized()
        
        spacing = 10
        length = curves.curve_length(points)
        
        # get the points to interpolate
        points = curves.make_curve_equidistant(points, spacing=spacing)
        skip_points = skip_length // spacing
        points = points[skip_points:-skip_points]
        
        # do spline fitting to smooth the line
        try:
            tck, _ = interpolate.splprep(np.transpose(points), k=3, s=length)
        except ValueError:
            pass
        else:
            # extend the center line in both directions to make sure that it
            # crosses the outline
            overshoot = 5*skip_length #< absolute overshoot
            num_points = (length + 2*overshoot)/spacing
            overshoot /= length #< overshoot relative to total length
            s = np.linspace(-overshoot, 1 + overshoot, num_points)
            points = interpolate.splev(s, tck)
            points = zip(*points) #< transpose list
        
            # restrict center line to burrow shape
            cline = geometry.LineString(points).intersection(self.polygon)
            
            if isinstance(cline, geometry.MultiLineString):
                points = max(cline, key=lambda obj: obj.length).coords
            else:
                points = np.array(cline.coords)
        
        return points
    
    
    def get_centerline(self, method='smoothed', **kwargs):
        """ get the centerline of the polygon """
        if method == 'smoothed':
            return self.get_centerline_smoothed(**kwargs)
        elif method == 'optimized':
            return self.get_centerline_optimized(**kwargs)
        elif method == 'estimate':
            return self.get_centerline_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`' % method)
        