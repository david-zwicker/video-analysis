'''
Created on Feb 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, widgets

from utils import graphics
from video.analysis.shapes import Rectangle



class RegionPicker(object):
    """ class that allows to pick a region in a given image """
    
    def __init__(self, ax=None):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
            
        self.width = self.ax.get_xlim()[1]
        self.height = self.ax.get_ylim()[1]
            
        # create the widget for selecting the range
        useblit = graphics.backend_supports_blitting()
        self.selector = \
                widgets.RectangleSelector(self.ax, self.select_callback,
                                          drawtype='box',
                                          useblit=useblit,
                                          button=[1]) # left button
            
        # the rectangle marking the selected area
        self.selected = None
        self.selected_marker = patches.Rectangle((0, -5), 0, 0, color='y',
                                                 alpha=0.5)
        self.ax.add_patch(self.selected_marker)
        
        
    def select_callback(self, eclick, erelease):
        """ callback for changing the selection range
        eclick and erelease are the press and release events """
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata
        
        # determine chosen range
        left = int(min(x1, x2) + 0.5)
        right = int(np.ceil(max(x1, x2) + 0.5))

        top = int(min(y1, y2) + 0.5)
        bottom = int(np.ceil(max(y1, y2) + 0.5))
        
        
        self.selected = Rectangle.from_points((x1, y1), (x2, y2))
        self.selected_marker.set_x(left - 0.5)
        self.selected_marker.set_width(right - left)
        self.selected_marker.set_y(top - 0.5)
        self.selected_marker.set_height(bottom - top)
        self.ax.figure.canvas.draw() #< update the graphics


    def show(self):
        """ show the picker and block until the window is closed. Returns the 
        selected rectangle """
        plt.show()
        return self.selected
        
