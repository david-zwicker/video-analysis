'''
Created on Aug 4, 2014

@author: zwicker
'''

import itertools
import logging
import numpy as np


def add_overlay(background, overlay, mask):
    """
    adds another overlay as an overlay
    """
    
    # check properties of the supplied videos
    if not background.size == overlay.size == mask.size:
        raise ValueError('Videos currently must be of the same size.')
    
    if not background.frame_count == overlay.frame_count == mask.frame_count:
        raise ValueError('Videos currently must have the same length.')
    
    if background.is_color != overlay.is_color:
        raise ValueError('Background and Overlay must have same number of color channels.')

    logging.info('Adding overlay to a video.')

    # create copy of the background video
    result = background.copy()

    # iterate over all videos
    for r_frame, o_frame, m_frame in itertools.izip(result, overlay, mask):
        if background.is_color:
            r_frame[m_frame, :]= o_frame[m_frame, :]
        else:
            r_frame[m_frame]= o_frame[m_frame]
        
    return result