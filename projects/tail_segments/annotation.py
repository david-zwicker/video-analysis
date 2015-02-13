'''
Created on Jan 28, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import os
import UserDict

import yaml
import numpy as np
from shapely import geometry

import matplotlib.pyplot as plt
from matplotlib import widgets

from utils import graphics



class TackingAnnotations(UserDict.DictMixin):
    """ class managing annotations for a tracking object
    """
    
    # data base file
    database = os.path.join(os.path.abspath(os.path.split(__file__)[0]),
                            'annotations', 'tails.yaml')
    
    
    def __init__(self, video_name):
        self.name = video_name
        
        
    def __get__(self):
        """ return the annotations stored for this video """
        with open(self.database, 'r') as fp:
            db = yaml.load(fp)
        if db:
            return db.get(self.name, {})
        else:
            return {}
    
    
    def __getitem__(self, key):
        """ return a property of the annotations stored for this video """
        annotations = self.__get__()
        return annotations.get(key, None)
        
        
    def __set__(self, value):
        """ stores annotations for this video.
        Note that this function is not thread save and there is a potential
        race-condition where the database could become inconsistent
        """
        with open(self.database, 'r') as fp:
            db = yaml.load(fp)
        db[self.name] = value
        with open(self.database, 'w') as fp:
            yaml.dump(db, fp)
        
    
    def __setitem__(self, key, value):
        """ sets a property of the annotations stored for this video """
        annotations = self.__get__()
        annotations[key] = value
        self.__set__(annotations)
        
        
    def setdefault(self, key, value):
        annotations = self.__get__()
        result = annotations.setdefault(key, value)
        if result == value:
            # annotations have been changed
            self.__set__(annotations)
        return result
        
        
        
class SegmentPicker(object):
    """ class that provides a GUI for defining segments on images """
    
    len_min = 10    #< minimal segment length in pixel
    pickradius = 20 #< picker radius around line
    lineprops = {'lw': 2, 'color': 'r'}
    
    def __init__(self, frame, features, segments=None):
        self._frame = frame
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
        plt.subplots_adjust(bottom=0.2)
        ax_img, ax_feat = self.axes
        
        imshow_args = {'interpolation': 'none', 'aspect': 1,
                       'cmap': plt.get_cmap('gray')}
        
        ax_img.imshow(self._frame, **imshow_args)
        ax_img.set_title('First _frame of video')
        ax_img.set_autoscale_on(False)

        ax_feat.imshow(self.features, **imshow_args)
        ax_feat.set_title('Automatic feature extraction')
        ax_feat.set_autoscale_on(False)
        
        # internal data 
        self.active = True
        self.result = 'cancel'
        self.bounds = self._frame.shape
        self._ax_segments = [[] for _ in xrange(len(self.axes))]
        
        # initialize data
        if self.segments:
            segments = self.segments
            self.segments = []
            for segment in segments:
                self._add_segment(segment)
                
                
        # drawtype is 'box' or 'line' or 'none'
        useblit = graphics.backend_supports_blitting()
        self.selectors = [
            widgets.RectangleSelector(ax, self.select_callback,
                                      drawtype='line',
                                      lineprops=self.lineprops,
                                      useblit=useblit,
                                      button=[1], # don't use middle button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels')
            for ax in (ax_img, ax_feat)]

        # add buttons
        ax_active = plt.axes([0.5, 0.05, 0.1, 0.075])
        check = widgets.CheckButtons(ax_active, ['active'], [self.active])
        check.on_clicked(self.clicked_check)

        ax_ok = plt.axes([0.7, 0.05, 0.1, 0.075])
        bn_ok = widgets.Button(ax_ok, 'Ok')
        bn_ok.on_clicked(self.clicked_ok)

        ax_cancel = plt.axes([0.81, 0.05, 0.1, 0.075])
        bp_cancel = widgets.Button(ax_cancel, 'Cancel')
        bp_cancel.on_clicked(self.clicked_cancel)
        
        self.msg()

        # initialize the interaction with the image
        self.fig.canvas.mpl_connect('button_press_event', self.click_image)
        
        # process result
        plt.show()
        return self.result


    def msg(self, text=None):
        """ output message to stdout """
        num_seg = len(self.segments)
        if text:
            print('%s - %d segment(s) defined.' % (text, num_seg))
            
        if self.active:
            lock_msg = ''
        else:
            lock_msg = ' [Locked]'
            
        title = '%d segments defined.' % num_seg
        plt.suptitle(title + lock_msg)
                
    
    def clicked_check(self, label):
        if label == 'active':
            self.active = not self.active
            if self.active:
                self.msg('Enabled editing')
            else:
                self.msg('Locked editing')
            for selector in self.selectors:
                selector.set_active(self.active)
            self.fig.canvas.draw() #< update the graphics
                
    
    def clicked_ok(self, event):
        if event.button == 1:
            self.result = 'ok'
            plt.close()

        
    def clicked_cancel(self, event):
        if event.button == 1:
            self.result = 'cancel'
            plt.close()
    
    
    def select_callback(self, eclick, erelease):
        """ eclick and erelease are the press and release events """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self._add_segment([[x1, y1], [x2, y2]])

            
    def _add_segment(self, segment):
        """ add a segment """
        coords = np.array(segment)
        for k, ax in enumerate(self.axes):
            l, = ax.plot(coords[:, 0], coords[:, 1], **self.lineprops)
            self._ax_segments[k].append(l)
        self.fig.canvas.draw() #< update the graphics
        self.segments.append(segment)
        self.msg('Added segment')
                    
            
    def _remove_segment(self, idx=-1):
        """ remove last segment """
        for ax_segments in self._ax_segments:
            ax_segments.pop(idx).remove()
        self.fig.canvas.draw() #< update the graphics
        self.segments.pop(idx)
        self.msg('Removed segment')


    def click_image(self, event):
        """ handles the user input """
        if not self.active or event.button != 3:
            return

        # check whether the click occurred in one of the images
        for ax in self.axes:
            if event.inaxes == ax:
                break
        else:
            return
        
        # determine picked point
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < self.bounds[0] and 0 <= y < self.bounds[1]):
            print('Please click inside the images.')
            return

        if self.segments:
            # find out which segment should be removed
            p = geometry.Point((x, y))
            dist = [geometry.LineString(segment).distance(p)
                    for segment in self.segments]
            idx = np.argmin(dist)
            
            if dist[idx] < self.pickradius:
                self._remove_segment(idx)
            else:
                self.msg('Right-click on a segment to remove it')
    
