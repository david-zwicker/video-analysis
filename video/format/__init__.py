"""
This package provides representations of videos with a unified interface for
accessing their data
"""

from memory import VideoMemory
from file import VideoFile, VideoStack

__all__ = [VideoMemory, VideoFile, VideoStack]