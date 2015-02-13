'''
Created on Feb 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt


def backend_supports_blitting():
    """ returns a flag indicating whether the current backend supports blitting.
    """
    # retrieve or initialize cache
    try:
        blitting_support = backend_supports_blitting._cache
    except AttributeError:
        blitting_support = backend_supports_blitting._cache = {}
    
    # get currently active backend
    backend = mpl.get_backend().lower()
    if backend not in blitting_support:
        # test blitting support by looking for method
        fig = plt.figure()
        m1 = hasattr(fig.canvas, 'copy_from_bbox')
        m2 = hasattr(fig.canvas, 'restore_region')
        blitting_support[backend] = (m1 and m2)
        plt.close(fig)

    return blitting_support[backend]
