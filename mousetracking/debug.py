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