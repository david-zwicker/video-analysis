'''
Created on Aug 4, 2014

@author: zwicker
'''

import numpy as np
import scipy.ndimage as ndimage


def find_bounding_rect(mask):
    """ finds the rectangle, which bounds a white region in a mask.
    The rectangle is returned as [top, left, bottom, right]
    Currently, this function only works reliably for connected regions 
    """

    # find top boundary
    top = 0
    while not np.any(mask[top, :]):
        top += 1
    
    # find bottom boundary
    bottom = top + 1
    try:
        while np.any(mask[bottom, :]):
            bottom += 1
    except IndexError:
        bottom = mask.shape[0] - 1

    # find left boundary
    left = 0
    while not np.any(mask[:, left]):
        left += 1
    
    # find right boundary
    try:
        right = left + 1
        while np.any(mask[:, right]):
            right += 1
    except IndexError:
        right = mask.shape[1] - 1
    
    return np.array([top, left, bottom, right])

       
def get_largest_region(mask):
    """ returns a mask only containing the largest region """
    # find all regions and label them
    labels, num_features = ndimage.measurements.label(mask)

    # find the label of the largest region
    label_max = np.argmax(
        ndimage.measurements.sum(labels, labels, index=range(1, num_features + 1))
    ) + 1
    
    return labels == label_max