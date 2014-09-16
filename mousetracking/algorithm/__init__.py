"""
package that contains several classes used in the tracking of the mouse
"""

from .parameters import PARAMETERS_DEFAULT, set_base_folder, scale_parameters
from .data_handler import DataHandler
from .pass1 import FirstPass
from .pass2 import SecondPass