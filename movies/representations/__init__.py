"""
This package collects different backends that can be used to read and write movies
"""

from memory import MemoryMovie
from file import MovieFile, MovieBatch

__all__ = [MemoryMovie, MovieFile, MovieBatch]