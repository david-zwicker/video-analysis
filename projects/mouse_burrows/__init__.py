"""
This package contains routines for tracking digging mouse
"""
# enable the faulthandler to detect low-level crashes
try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from .simple import (scan_video, scan_video_in_folder, load_results,
                     load_result_file)