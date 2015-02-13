'''
Created on Dec 22, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib import patches, widgets, ticker

from utils import graphics

from video import debug  # @UnusedImport



class Kymograph(object):
    """ class representing a single kymograph """
    
    align_window = 100
    feature_scale = 31
    blur_radius = 1   
    
    def __init__(self, linescans, normalize=True):
        """ create a kymograph from line scans """
        if normalize:
            self.data = [(line - np.mean(line))/np.std(line)
                         for line in linescans]
        else:
            self.data = [line for line in linescans]
        self.offsets = np.zeros(len(linescans), np.int)
        
    
    def _make_image(self, offsets):
        """ produces an image where all the line scans are left aligned """
        offsets = (offsets - self.offsets.min()).astype(np.int)
        
        # determine maximal length of a single line scan
        size = max(len(d) for d in self.data)
        size += offsets.max()
        
        # collect the full data
        img = np.zeros((len(self.data), size)) + np.nan
        
        # iterate over all time points
        for l in xrange(len(self.data)):
            start = offsets[l]
            img[l, start:start + len(self.data[l])] = self.data[l]
            
        return img 
    

    def _get_alignment_score(self, data, ret_profile=False):
        """ calculates a score for a particular alignment """
        # blur the buffer to filter some noise
        idx_nan = np.isnan(data)
        data = cv2.GaussianBlur(data, (0, 0), self.blur_radius)
        data[idx_nan] = np.nan
    
        # perform time average to find stationary features
        time_average = np.nanmean(data, axis=0)
    
        # filter spatially to locate features on specific scale
        kernel = np.ones(self.feature_scale, np.double)/self.feature_scale
        space_average = np.convolve(time_average, kernel, mode='same')
    
        # find spatial points where enough points contribute
        valid_points = (idx_nan.sum(axis=0) < 10)
    
        # judge the strength of the features by standard deviation
        score = np.nanstd(time_average[valid_points] - space_average[valid_points])
        
        if ret_profile:
            return score, time_average[valid_points]
        else:
            return score 
           
           
    def get_image(self):
        """ produces an image with the current alignment """
        return self._make_image(self.offsets)


    def get_alignment_score(self, ret_profile=False):
        """ returns the score of the current alignment
        If `ret_profile` is True, the time averaged profile is also returned
        """
        img = self.get_image()
        return self._get_alignment_score(img, ret_profile=ret_profile)


    def align_left(self):
        """ aligns the data at the left side """
        self.offsets[:] = 0

    
    def align_features_linearly(self):
        """ align features roughly by trying linear transformations """
        # prepare image to work with
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = self.align_window
        
        # initialize buffer that will be used
        data = np.empty((height, width + 2*window), np.double)

        score_best, offset_best = 0, None
        for offset in xrange(-window, window + 1):
            offsets = offset * np.linspace(0, 1, height)
        
            # prepare an array to hold the data
            height, width = image.shape
            data[:] = np.nan
            for y, offset in enumerate(offsets):
                s = int(window + offset)
                data[y, s:s + width] = image[y, :]
                            
            # judge this offset
            score = self._get_alignment_score(data)
            
            if score > score_best:
                score_best = score
                offset_best = offset
            
        offsets = offset_best * np.linspace(0, 1, height)
        self.offsets += offsets.astype(np.int)
        
        
    def align_features_individually(self, max_dist=1):
        """ run through all lines individually and try to move them """
        image = self._make_image(self.offsets)
        height, width = image.shape
        window = self.align_window

        # prepare an array to hold the data
        data = np.empty((height, width + 2*window), np.double) + np.nan
        data[:, window:-window] = image

        # set distances that we test to [-max_dist, ..., -1, 1, ..., max_dist]
        offset_values = np.arange(-max_dist, max_dist)
        offset_values[offset_values >= 0] += 1

        score = self._get_alignment_score(data)
        
        # initialize buffer that will be used
        data_best = data.copy()
        improved, score_best = 1, score
        while improved > 0:
            improved = 0
            for y in xrange(height):
                for offset in offset_values: #< try left and right shift
                    if offset < 0:
                        data[y, :offset] = data_best[y, -offset:] #< shift row to left
                    else:
                        data[y, offset:] = data_best[y, :-offset] #< shift row to right
                    score = self._get_alignment_score(data)
                    if score > score_best:
                        improved += 1
                        score_best = score
                        self.offsets[y] += offset
                        data_best = data.copy()
                        break
                else:
                    # reset row if no better state was found
                    data[y, :] = data_best[y, :]
                    
            logging.debug('Improved %d lines.' % improved)
    
            
    def align_features(self, mode='individually'):
        """ align features in the kymograph """
        self.align_features_linearly()
        if mode == 'individually':
            self.align_features_individually(1)
            


class KymographAligner(object):
    """ class that provides a GUI for aligning kymographs """
    
    # margin at either side in pixels
    window_margin = 50
    # dictionary defining what displacement keys cause  
    key_offset_mapping = {'a': -1, 'A': -5,
                          'd': +1, 'D': +5}

    
    def __init__(self, kymograph, title='Kymograph'):
        self.kymograph = kymograph
        self.title = title


    def run(self):
        """ runs the kymograph aligner """
        # show the images
        self.fig = plt.figure()
        self.ax = plt.gca()
        plt.subplots_adjust(bottom=0.2)
        
        # prepare image to work with
        image = self.kymograph._make_image(self.kymograph.offsets)
        height, width = image.shape
        window = self.window_margin
        self.data = np.empty((height, width + 2*window), np.double) + np.nan
        self.data[:, window:width + window] = image
        
        self.image = self.ax.imshow(self.data, interpolation='none',
                                    aspect='auto', cmap=plt.get_cmap('gray'))
        self.ax.set_title(self.title)
        
        # internal data 
        self.active = True
        self.result = 'cancel'
        self._ax_points = None
        
        # create the widget for selecting the range
        useblit = graphics.backend_supports_blitting()
        self.selector = \
            widgets.RectangleSelector(self.ax, self.select_callback,
                                      drawtype='box',
                                      useblit=useblit,
                                      button=[1]) # only use the left button
            
        # the rectangle marking the selected area
        self.selected = slice(0, 0)
        self.selected_marker = patches.Rectangle((0, -5), self.data.shape[1], 0,
                                                 color='y', alpha=0.5)
        self.ax.add_patch(self.selected_marker)

        # add buttons
        ax_align = plt.axes([0.5, 0.05, 0.1, 0.075])
        bn_align = widgets.Button(ax_align, 'Align All')
        bn_align.on_clicked(self.clicked_align)

        ax_ok = plt.axes([0.7, 0.05, 0.1, 0.075])
        bn_ok = widgets.Button(ax_ok, 'Save')
        bn_ok.on_clicked(self.clicked_ok)

        ax_cancel = plt.axes([0.81, 0.05, 0.1, 0.075])
        bp_cancel = widgets.Button(ax_cancel, 'Cancel')
        bp_cancel.on_clicked(self.clicked_cancel)
        
        # initialize the interaction with the image
        self.fig.canvas.mpl_connect('key_release_event', self.key_callback)
        
        # process result
        plt.show()
        return self.result


    def _update_image(self):
        """ update the image that is shown from the internal data """
        score = self.kymograph._get_alignment_score(self.data)
        self.ax.set_title('%s [Score: %g]' % (self.title, score))
        
        self.image.set_data(self.data)
        self.fig.canvas.draw() #< update the graphics

    
    def clicked_align(self, event):
        """ callback for the align button """
        if event.button == 1:
            self.kymograph.align_features_individually(3)
            img = self.kymograph.get_image()
            x = (self.data.shape[1] - img.shape[1]) // 2
            self.data[:] = np.nan
            self.data[:, x:x + img.shape[1]] = img
            self._update_image()
                
    
    def clicked_ok(self, event):
        """ callback for the ok button """
        if event.button == 1:
            self.result = 'ok'
            plt.close()

        
    def clicked_cancel(self, event):
        """ callback for the cancel button """
        if event.button == 1:
            self.result = 'cancel'
            plt.close()
            
            
    def select_callback(self, eclick, erelease):
        """ callback for changing the selection range
        eclick and erelease are the press and release events """
        y1, y2 = eclick.ydata, erelease.ydata
        # determine chosen range
        top = int(min(y1, y2) + 0.5)
        bottom = int(np.ceil(max(y1, y2) + 0.5))
        self.selected = slice(top, bottom)
        self.selected_marker.set_y(top - 0.5)
        self.selected_marker.set_height(bottom - top)
        self.fig.canvas.draw() #< update the graphics
        print('Chose interval [%d, %d]' % (top, bottom))
        
        
    def key_callback(self, event):
        """ callback for events when keys are released """
        if event.key in self.key_offset_mapping:
            # a key in the mapping has been pressed
            dx = self.key_offset_mapping[event.key]
            if dx < 0:
                # move selected range to the left
                self.data[self.selected, :dx] = self.data[self.selected, -dx:]
            else:
                # move selected range to the left
                self.data[self.selected, dx:] = self.data[self.selected, :-dx]
            self.kymograph.offsets[self.selected] += dx
            self._update_image()
            
        elif event.key == 'q':
            # close the GUI
            self.result = 'cancel'
            plt.close()
            
        else:
            print('Key %s released' % event.key)
            
            

class KymographPlotter(object):
    """ class used for plotting kymographs """
    
    interpolation = 'nearest'
    lineprops = {'color': 'b'} #< defines the line style for measure_lines
    
    
    def __init__(self, kymograph, title='', length_scale=1, time_scale=1, 
                 use_tex=None):
        """ initializes the plotter with a `kymograph` and a `title`.
        `length_scale` sets the length of a single pixel in micrometer
        `time_scale` sets the length of a single pixel in minutes
        `use_tex` determines whether tex is used for outputting values 
        """
        self.kymograph = kymograph
        self.title = title
        self.length_scale = length_scale
        self.time_scale = time_scale
    
        # setup the plotting
        if use_tex is not None:
            plt.rcParams['text.usetex'] = use_tex

        self.fig = plt.figure()
        self.ax = plt.gca()
    
        # create image and determine the length and time scales
        img = self.kymograph.get_image()
        distance = img.shape[1] * self.length_scale
        duration = img.shape[0] * self.time_scale 
        extent = (0, distance, 0, duration)
        
        # plot image in gray scale
        self.ax.imshow(img, extent=extent, aspect='auto',
                       interpolation=self.interpolation, origin='lower',
                       cmap=plt.get_cmap('gray'))
        
        # use a time format for the y axis
        def hours_minutes(value, pos):
            """ formatting function """
            return '%d:%02d' % divmod(value, 60)
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(2*60))
        self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(hours_minutes))
        self.ax.invert_yaxis()
        
        self.ax.set_xlim(0, distance)
        self.ax.set_ylim(duration, 0)
        
        # label image
        if plt.rcParams['text.usetex']:
            self.ax.set_xlabel(r'Distance from posterior end [$\unit{\upmu m}$]')
        else:
            self.ax.set_xlabel(u'Distance from posterior end [\u00b5m]')
        self.ax.set_ylabel(r'Time [h:m]')
        
        self.ax.set_title(self.title)
            
            
    def close(self):
        """ close the figure """
        plt.close(self.fig)
            
            
    def measure_lines(self):
        """ shows an interface for measuring lines """
        # create the widget for selecting the range
        useblit = graphics.backend_supports_blitting()
        self.selector = \
            widgets.RectangleSelector(self.ax, self.select_callback,
                                      drawtype='line', lineprops=self.lineprops,
                                      useblit=useblit, button=[1])
            
        self.line = self.ax.plot([-1, -1], [-1, -1], **self.lineprops)[0]
            
        plt.show()


    def select_callback(self, eclick, erelease):
        """ callback for changing the selection range
        eclick and erelease are the press and release events """
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata
        self.line.set_xdata((x1, x2))
        self.line.set_ydata((y1, y2))

        # calculate the speed of the line
        if plt.rcParams['text.usetex']:
            fmts = [r"Distance: $\unit[%(distance)d]{\upmu m}$",
                    r"Time: $\unit[%(time)d]{min}$",
                    r"Speed: $\unitfrac[%(speed)g]{\upmu m}{min}$"]
        else:
            fmts = [u"Distance: %(distance)d \u00b5m",
                    u"Time: %(time)d min",
                    u"Speed: %(speed)g \u00b5m/min"]
        fmtstr = ', '.join(fmts)

        dx = x1 - x2
        dy = y1 - y2
        if dy < 0:
            dx, dy = -dx, -dy
        self.ax.set_title(fmtstr % {'distance':dx, 'time':dy, 'speed':dx/dy})
        self.fig.suptitle(self.title)

        self.fig.canvas.draw() #< update the graphics
        
