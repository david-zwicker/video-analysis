'''
Created on Feb 12, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import matplotlib as mpl


def backend_supports_blit():
    """ returns a flag indicating whether the current backend supports blitting.
    Ideally this function should check the actual code for the support, but
    currently it only checks whether the backend is macosx, which does not
    support blitting """
    return mpl.get_backend().lower() != 'macosx'