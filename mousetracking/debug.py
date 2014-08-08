'''
Created on Aug 8, 2014

@author: zwicker
'''

from __future__ import division


def show_image(image):
    """ shows the image using matplotlib and waits for the user to continue """
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.colorbar()
    plt.show()
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
    