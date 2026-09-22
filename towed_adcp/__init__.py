"""
towed_adcp: processing of underway, bottom-tracking ADCP data (bronze -> silver -> gold -> platinum).

Implements steps 3 to 6 of Valle-Levinson (2024), "Collection and processing of underway,
bottom-tracking ADCP data", Treatise on Estuarine and Coastal Science, 2nd ed., vol. 2, 207-218.
"""

from .io import read_winriver_ascii, save_matrix, load_matrix, to_datetime
from .bronze import build_bronze, profiles_from_bronze, BRONZE_COLUMNS
from .transects import (local_xy, lonlat_from_xy, repetitions_by_vertices, repetitions_from_file,
                        transect_geometry)
from .silver import joyce_calibration, build_silver, SILVER_COLUMNS
from .gold import build_gold, save_gold, load_gold
from .platinum import lsqfit, fit_metrics, build_platinum, principal_axis, rotate_gold, save_platinum, load_platinum

__version__ = '0.1.0'
__author__ = 'Liesvy Valladares'
