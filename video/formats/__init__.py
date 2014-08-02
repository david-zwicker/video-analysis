"""
This package provides representations of videos with a unified interface for
accessing their data
"""

from .memory import VideoMemory
from .file import VideoFile, VideoFileStack, VideoImageStack

__all__ = ['VideoMemory', 'VideoFile', 'VideoFileStack', 'VideoImageStack']