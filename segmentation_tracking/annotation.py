'''
Created on Jan 28, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os

import yaml
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from video.analysis import curves


class TackingAnnotations(object):
    """ class managing annotations for a tracking object
    TODO: implement locking mechanism to avoid race conditions
    """
    
    # data base file
    database = os.path.join(os.path.split(__file__)[0],
                            'annotations',
                            'tails.yaml')
    
    
    def __init__(self, video_name):
        self.name = video_name
        
        
    def __get__(self):
        """ return the annotations stored for this video """
        db = yaml.load(open(self.database, 'r'))
        if db:
            return db.get(self.name, {})
        else:
            return {}
    
    
    def __getitem__(self, key):
        """ return a property of the annotations stored for this video """
        annotations = self.__get__()
        return annotations.get(key, None)
        
        
    def __set__(self, value):
        """ stores annotations for this video """
        db = yaml.load(open(self.database, 'r'))
        db[self.name] = value
        yaml.dump(db, open(self.database, 'w'))
        
    
    def __setitem__(self, key, value):
        """ sets a property of the annotations stored for this video """
        annotations = self.__get__()
        annotations[key] = value
        self.__set__(annotations)
        
        
        
class SegmentPicker(object):
    """ class that provides a GUI for defining segments on images """
    
    len_min = 10 #< minimal segment length in pixel
    
    def __init__(self, frame, features, segments=None):
        self.frame = frame
        self.features = features
        if segments is None:
            self.segments = []
        else:
            self.segments = segments


    def run(self):
        """ runs the segment picker """
        # show the images
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                                           sharey=True, squeeze=True)
        ax_img, ax_feat = self.axes
        
        imshow_args = {'interpolation': 'none', 'aspect': 1,
                       'cmap': plt.get_cmap('gray')}
        
        ax_img.imshow(self.frame, **imshow_args)
        ax_img.set_title('First frame of video')
        ax_img.set_autoscale_on(False)

        ax_feat.imshow(self.features, **imshow_args)
        ax_feat.set_title('Automatic feature extraction')
        ax_feat.set_autoscale_on(False)
        
        # internal data 
        self.result = 'cancel'
        self.bounds = self.frame.shape
        self._ax_segments = [[] for _ in xrange(len(self.axes))]
        self.point = None
        self._ax_points = None
        
        # initialize data
        if self.segments:
            segments = self.segments
            self.segments = []
            for segment in segments:
                self._add_segment(segment)

        self.msg()

        # add buttons
        ax_ok = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_cancel = plt.axes([0.81, 0.05, 0.1, 0.075])
        bn_ok = Button(ax_ok, 'Ok')
        bn_ok.on_clicked(self.clicked_ok)
        bp_cancel = Button(ax_cancel, 'Cancel')
        bp_cancel.on_clicked(self.clicked_cancel)
        
        # initialize the interaction with the image
        self.fig.canvas.mpl_connect('button_press_event', self.handle_input)
        
        # process result
        plt.show()
        return self.result
        
    
    def clicked_ok(self, event):
        if event.button == 1:
            self.result = 'ok'
            plt.close()

        
    def clicked_cancel(self, event):
        if event.button == 1:
            self.result = 'cancel'
            plt.close()

        
    def _set_point(self, x, y):
        """ sets the anchor point """
        self.point = [x, y]
        self._ax_points = [
            ax.plot(x, y, 'ro', ms=5)[0]
            for ax in self.axes
        ]


    def _remove_point(self):
        """ removes the anchor point """
        self.point = None
        for ax_points in self._ax_points:
            ax_points.remove()
            
            
    def _add_segment(self, segment):
        """ add a segment """
        self.segments.append(segment)
        coords = np.array(segment)
        for k, ax in enumerate(self.axes):
            l, = ax.plot(coords[:, 0], coords[:, 1], 'r', lw=2)
            self._ax_segments[k].append(l)
            
            
    def _remove_segment(self):
        """ remove last segment """
        for ax_segments in self._ax_segments:
            ax_segments.pop().remove()
        self.segments.pop()


    def msg(self, text=None):
        """ output message to stdout """
        num_seg = len(self.segments)
        if text:
            print('%s - %d segment(s) defined.' % (text, num_seg))
        if self.point:
            plt.suptitle('%d segments and start point defined.' % num_seg)
        else:
            plt.suptitle('%d segments defined.' % num_seg)


    def handle_input(self, event):
        """ handles the user input """
        if not event.inaxes:
            print('Please click inside the images.')
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.bounds[0] and 0 <= y < self.bounds[1]):
            print('Please click inside the images.')
            return 

        if event.button == 1:
            # left button clicked => add new object
            if self.point:
                if curves.point_distance(self.point, (x, y)) < self.len_min:
                    self.msg("Line too short; choose a different end point")
                else:
                    self._add_segment([self.point, [x, y]])
                    self._remove_point()
                    self.msg('Added new segment')
                
            else:
                self._set_point(x, y)
                self.msg('Defined start point')
                # update the graphics
                
            self.fig.canvas.draw()

        elif event.button == 3:
            # right button clicked => delete last input
            if self.point:
                self._remove_point()
                self.msg('Removed start point')

            elif self.segments:
                self._remove_segment()
                self.msg('Removed last segment')
                
            self.fig.canvas.draw()
            