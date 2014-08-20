'''
Created on Aug 4, 2014

@author: zwicker
'''

import numpy as np
import scipy.ndimage as ndimage
import shapely
import shapely.geometry as geometry


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


def get_overlapping_slices(t_pos, i_shape, t_shape, anchor='center'):
    """ calculates slices to compare parts of two images with each other
    i_shape is the shape of the larger image in which a smaller image of
    shape t_shape will be placed.
    Here, t_pos specifies the position of the smaller image in the large one.
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
    return ((slice(i_y, i_y + h), slice(i_x, i_x + w)),  # slice for the image
            (slice(t_y, t_y + h), slice(t_x, t_x + w)))  # slice for the template


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
    

