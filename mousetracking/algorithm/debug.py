'''
Created on Aug 8, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

module that contains several functions useful when debugging the algorithm
'''

from __future__ import division

import functools
import itertools

import numpy as np

import cv2


# define exported functions such that loaded modules are not leaked into 
# the importing space
__all__ = ['show_image', 'show_shape', 'show_tracking_graph',
           'get_grabcut_image', 'print_filter_chain', 'save_frame_from_video']



def get_subplot_shape(num_plots=1):
    """ calculates an optimal subplot shape for a given number of images """
    if num_plots <= 2:
        num_rows = 1
    elif num_plots <= 6:
        num_rows = 2
    elif num_plots <= 12:
        num_rows = 3
    else:
        num_rows = 4    
    num_cols = int(np.ceil(num_plots/num_rows))
    return num_rows, num_cols



def _ax_format_coord(x, y, image):
    """ returns a string usable for formating the status line """ 
    col = int(x + 0.5)
    row = int(y + 0.5)
    if 0 <= col < image.shape[1] and 0 <= row < image.shape[0]:
        z = image[row, col]
        return 'x=%1.2f, y=%1.2f, z=%1.5g' % (x, y, z)
    else:
        return 'x=%1.2f, y=%1.2f' % (x, y)



def show_image(*images, **kwargs):
    """ shows a collection of images using matplotlib and waits for the user
    to continue """
    import matplotlib.pyplot as plt

    # determine the number of rows and columns to show
    num_rows, num_cols = get_subplot_shape(len(images))
    
    # get the color scale
    if kwargs.pop('equalize_colors', False):
        vmin, vmax = np.inf, -np.inf
        for image in images:
            vmin = min(vmin, image.min())    
            vmax = max(vmax, image.max())
    else:
        vmin, vmax = None, None    
        
    # apply mask if requested
    mask = kwargs.pop('mask', None)
    if mask is not None:
        images = [np.ma.array(image, mask=~mask) for image in images]
    
    # see if all the images have the same dimensions
    try:
        share_axes = (len(set(image.shape[:2] for image in images)) == 1)
    except AttributeError:
        # there is something else than a np.ndarray in the list
        share_axes = False
    
    # choose the color map and color scaling
    plt.gray()
    if kwargs.pop('lognorm', False):
        from matplotlib.colors import LogNorm
        vmin = max(vmin, 1e-4)
        norm = LogNorm(vmin, vmax)
    else:
        norm = None
    
    # plot all the images
    for k, image in enumerate(images):
        # create the axes
        if share_axes:
            # share axes with the first subplot
            if k == 0:
                ax = plt.subplot(num_rows, num_cols, k + 1)
                share_axes = ax
            else:
                ax = plt.subplot(num_rows, num_cols, k + 1,
                                 sharex=share_axes, sharey=share_axes)
        else:
            ax = plt.subplot(num_rows, num_cols, k + 1)
            
        # plot the image
        if isinstance(image, np.ndarray):
            img = ax.imshow(image, interpolation='nearest',
                            vmin=vmin, vmax=vmax, norm=norm)
            # add the colorbar
            if image.min() != image.max():
                # recipe from http://stackoverflow.com/a/18195921/932593
                from mpl_toolkits.axes_grid1 import make_axes_locatable  # @UnresolvedImport
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.05)
                try:
                    plt.colorbar(img, cax=cax)
                except DeprecationWarning:
                    # we don't care about these in the debug module
                    pass
            
            # adjust the mouse over effects
            ax.format_coord = functools.partial(_ax_format_coord, image=image)
            
        elif len(image) == 2:
            # assume it's a vector field plot
            u, v = image
            #max_len = np.hypot(u, v).max()
            
            ax.quiver(u[::-10, ::10], -v[::-10, ::10], pivot='tip',
                      angles='xy', scale_units='xy')
            plt.axis('equal')
            
        else:
            raise ValueError('Unsupported image type')
        
    # show the images and wait for user input
    plt.show()
    if kwargs.get('wait_for_key', True):
        raw_input('Press enter to continue...')
    
    
    
def show_shape(*shapes, **kwargs):
    """ plots several shapes """
    import matplotlib.pyplot as plt
    import shapely.geometry as geometry
    import descartes
    
    background = kwargs.get('background', None)
    wait_for_key = kwargs.get('wait_for_key', True)
    mark_points = kwargs.get('mark_points', False)
    
    # set up the plotting
    plt.figure()
    ax = plt.gca()
    colors = itertools.cycle('b g r c m y k'.split(' '))
    
    # plot background, if applicable
    if background is not None:
        axim = ax.imshow(background, origin='upper',
                         interpolation='nearest', cmap=plt.get_cmap('gray'))
        # adjust the mouse over effects
        ax.format_coord = functools.partial(_ax_format_coord, image=background)
        if background.min() != background.max():
            # recipe from http://stackoverflow.com/a/18195921/932593
            from mpl_toolkits.axes_grid1 import make_axes_locatable  # @UnresolvedImport
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            try:
                plt.colorbar(axim, cax=cax)
            except DeprecationWarning:
                # we don't care about these in the debug module
                pass

    # iterate through all shapes and plot them
    for shape in shapes:
        color = kwargs.get('color', colors.next())
        line_width = kwargs.get('lw', 3)
        
        if isinstance(shape, (geometry.Point, geometry.point.Point)):
            ax.plot(shape.x, shape.y, 'o', color=color, ms=20)
        
        elif isinstance(shape, geometry.MultiPoint):
            coords = np.array([(p.x, p.y) for p in shape])
            ax.plot(coords[:, 0], coords[:, 1], 'o', color=color, ms=4)
        
        elif isinstance(shape, geometry.Polygon):
            patch = descartes.PolygonPatch(shape,
                                           ec=kwargs.get('ec', 'none'),
                                           fc=color, alpha=0.5)
            ax.add_patch(patch)
            if mark_points:
                ax.plot(shape.xy[0], shape.xy[1], 'o',
                        markersize=2*line_width, color=color)
            
        elif isinstance(shape, geometry.LineString):
            ax.plot(shape.xy[0], shape.xy[1], color=color, lw=line_width)
            if mark_points:
                ax.plot(shape.xy[0], shape.xy[1], 'o',
                        markersize=2*line_width, color=color)
            
        elif isinstance(shape, geometry.multilinestring.MultiLineString):
            for line in shape:
                ax.plot(line.xy[0], line.xy[1], color=color, lw=line_width)
                if mark_points:
                    ax.plot(line.xy[0], line.xy[1], 'o',
                            markersize=2*line_width, color=color)
            
        else:
            raise ValueError("Don't know how to plot %r" % shape)
        
    # adjust image axes
    if background is None:
        ax.invert_yaxis()
        ax.margins(0.1)
        ax.autoscale_view(tight=False, scalex=True, scaley=True)
    else:
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
    
    plt.show()
    if wait_for_key:
        raw_input('Press enter to continue...')
           

    
def show_tracking_graph(graph, path=None, **kwargs):
    """ displays a representation of the tracking graph """
    import matplotlib.pyplot as plt
    
    # plot the known chunks
    for node, data in graph.nodes_iter(data=True):
        color = 'r' if data['highlight'] else 'g'
        plt.plot([node.start, node.end],
                 [node.first.pos[0], node.last.pos[0]],
                 color, lw=(4 + 10*node.mouse_score))
        
    try:
        max_weight = max(data['cost']
                         for _, _, data in graph.edges_iter(data=True))
    except ValueError:
        max_weight = 1
    
    if kwargs.get('plot_edges', False):
        for (a, b, d) in graph.edges_iter(data=True):
            plt.plot([a.end, b.start],
                     [a.last.pos[0], b.first.pos[0]],
                     color=str(d['cost']/max_weight), lw=1)
        
    # plot the actual graph
    if kwargs.get('plot_graph', True):
        if path is not None:
            node_prev = None
            for node in path:
                plt.plot([node.start, node.end],
                         [node.first.pos[0], node.last.pos[0]],
                         'b', lw=2)
                if node_prev is not None:
                    plt.plot([node_prev.end, node.start],
                             [node_prev.last.pos[0], node.first.pos[0]],
                             'b', lw=2)
                node_prev = node

    # show plot
    plt.xlabel('Time in Frames')
    plt.ylabel('X Position')
    plt.margins(0, 0.1)
    plt.show()
    if kwargs.get('wait_for_key', True):
        raw_input('Press enter to continue...')



def get_grabcut_image(mask):
    """ returns an image from a mask that was prepared for the grab cut
    algorithm, where the foreground is bright and the background is dark """
    image = np.zeros_like(mask, np.uint8)
    c = 255//4
    image[mask == cv2.GC_BGD   ] = 1*c 
    image[mask == cv2.GC_PR_BGD] = 2*c
    image[mask == cv2.GC_PR_FGD] = 3*c
    image[mask == cv2.GC_FGD   ] = 4*c
    return image
     

    
def print_filter_chain(video):
    """ prints information about a filter chain """
    # print statistics of current video
    print(str(video))
    
    # go up one level
    try:
        print_filter_chain(video._source)
    except AttributeError:
        pass
    
    
    
def save_frame_from_video(video, outfile):
    """ save the next video frame to outfile """
    # get frame
    pos = video.get_frame_pos()
    frame = video.next()
    video.set_frame_pos(pos)
    
    # turn it into image
    from PIL import Image  # @UnresolvedImport
    im = Image.fromarray(frame)
    im.save(outfile)
    