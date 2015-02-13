"""
This package provides representations of videos with a unified interface for
accessing their data.

Colors are always represented by float values ranging from 0 to 1.
Two formats are supported:
    A single value indicates a grey scale
    Three values indicate RGB values
"""

from .base import VideoFork
from .memory import VideoMemory
from .file import (VideoFile, VideoFileStack, VideoImageStack, VideoFileWriter, 
                   show_video, load_any_video, write_video)
from .computed import VideoGaussianNoise
from .composer import VideoComposer
from .display import ImageWindow
