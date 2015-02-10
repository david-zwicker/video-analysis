"""
This package contains classes that represent objects which can be tracked
""" 

from projects.mouse_burrows import Burrow, BurrowTrack, BurrowTrackList
from projects.mouse_burrows import GroundProfile, GroundProfileList, GroundProfileTrack
from mousetracking.algorithm.objects.moving_objects import (MovingObject,
                                                            ObjectTrack,
                                                            ObjectTrackList)
from projects.mouse_burrows import MouseTrack

# define a simple object representing the cage rectangle
from video.analysis.shapes import Rectangle
Cage = Rectangle