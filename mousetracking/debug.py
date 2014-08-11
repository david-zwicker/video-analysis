'''
Created on Aug 8, 2014

@author: zwicker
'''

from __future__ import division

from mpl_toolkits.axes_grid1 import make_axes_locatable  # @UnresolvedImport


def show_image(*images, **kwargs):
    """ shows the image using matplotlib and waits for the user to continue """
    import matplotlib.pyplot as plt

    num_plots = len(images)
    for k, image in enumerate(images):
        plt.subplot(1, num_plots, k + 1)
        plt.imshow(image, interpolation='none')
        plt.gray()
        
        # recipe from http://stackoverflow.com/a/18195921/932593
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        
    plt.show()
    if kwargs.pop('wait_for_key', True):
        raw_input('Press enter to continue...')
    
    
def print_filter_chain(video):
    """ prints information about a filter chain """
    # print statistics of current video
    line = str(video)
    if video._is_iterating:
        line += ', is iterating'
    print(line)
    
    # go up one level
    try:
        print_filter_chain(video._source)
    except AttributeError:
        pass
    