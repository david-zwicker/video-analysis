'''
Created on Aug 4, 2014

@author: zwicker
'''

from __future__ import division

import operator

import numpy as np
import scipy.ndimage as ndimage
import shapely.geometry as geometry

import curves


def corners_to_rect(p1, p2):
    """ creates a rectangle from two corner points.
    The points are both included in the rectangle.
    """
    xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0]) 
    ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
    return (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)


def rect_to_corners(rect):
    """ returns two corner points for a rectangle.
    Both these points are included in the rectangle.
    """
    p1 = (rect[0], rect[1])
    p2 = (rect[0] + rect[2] - 1, rect[1] + rect[3] - 1)
    return p1, p2


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
    
       
def get_largest_region(mask):
    """ returns a mask only containing the largest region """
    # find all regions and label them
    labels, num_features = ndimage.measurements.label(mask)

    # find the label of the largest region
    label_max = np.argmax(
        ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
    ) + 1
    
    return labels == label_max


def get_enclosing_outline(polygon):
    """ gets the enclosing outline of a (possibly complex) polygon """
    # get the outline
    outline = polygon.boundary
    
    if isinstance(outline, geometry.multilinestring.MultiLineString):
        largest_polygon = None
        # find the largest polygon, which should be the enclosing outline
        for line in outline:
            poly = geometry.Polygon(line)
            if largest_polygon is None or poly.area > largest_polygon.area:
                largest_polygon = poly
        outline = largest_polygon.boundary
    return outline
    
    
def regularize_polygon(polygon):
    """ regularizes a shapely polygon using polygon.buffer(0) """
    # regularize polygon
    polygon = polygon.buffer(0)
    # retrieve the result with the largest area
    if isinstance(polygon, geometry.MultiPolygon):
        polygon = max(polygon, key=operator.attrgetter('area'))
    return polygon


def regularize_contour(contour):
    """ regularizes a list of points defining a contour """ 
    polygon = geometry.Polygon(contour)
    regular_polygon = regularize_polygon(polygon)
    if polygon is not regular_polygon:
        contour = regular_polygon.exterior.coords
    return contour

    
def get_ray_hitpoint(point_anchor, point_far, line_string, ret_dist=False):
    """ returns the point where a ray anchored at point_anchor hits the polygon
    given by line_string. The ray extends out to point_far, which should be a
    point beyond the polygon.
    If ret_dist is True, the distance to the hit point is also returned.
    """
    
    ray = geometry.LineString((point_anchor, point_far))
    # find the intersections between the ray and the burrow outline
    inter = line_string.intersection(ray)
    if isinstance(inter, geometry.Point):
        # check whether this points is farther away than the last match
        if ret_dist:
            dist = curves.point_distance(inter.coords[0], point_anchor)
            return inter.coords[0], dist
        else:
            return inter.coords[0]

    elif not inter.is_empty:
        # find closest intersection if there are many points
        dists = [curves.point_distance(p.coords[0], point_anchor) for p in inter]
        k_min = np.argmin(dists)
        if ret_dist:
            return inter[k_min].coords[0], dists[k_min]
        else:
            return inter[k_min].coords[0]
        
    else:
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

