"""
Step 3 of the chapter: the bronze matrix.

Stacks the usable profiles of all files, one row per bin, with 11 columns (Fig. 5a of
Valle-Levinson, 2024).
"""

import glob

import numpy as np

from .io import read_winriver_ascii

BRONZE_COLUMNS = ['time', 'lat', 'lon', 'u', 'v', 'bin_depth', 'depth', 'u_bt', 'v_bt', 'backscatter', 'temperature']
# 0 time (decimal day), 1 lat, 2 lon, 3 u, 4 v (cm/s), 5 bin depth (m), 6 water-column depth (m),
# 7 u_bt, 8 v_bt (cm/s), 9 backscatter (dB or counts), 10 temperature (deg C)


def build_bronze(files, err_max=10.0, q_max=100.0, vbt_min=10.0, pg_min=None, verbose=True):
    """Read one or more WinRiver ASCII files and build the bronze matrix.

    `files` is a list of paths or a glob pattern (e.g. 'data/*_ASC.TXT').
    Returns (bronze, meta); meta['time_base'] = (year, month) origin of the decimal day.
    """
    if isinstance(files, str):
        files = sorted(glob.glob(files))
    if not files:
        raise FileNotFoundError('no input files found')
    blocks, meta = [], None
    for fn in files:
        b, m = read_winriver_ascii(fn, err_max, q_max, vbt_min, pg_min,
                                   time_base=None if meta is None else meta['time_base'], verbose=verbose)
        blocks.append(b)
        if meta is None:
            meta = dict(time_base=m['time_base'], n_profiles=0, n_valid_bins=0, n_ok_bins=0, files=[])
        meta['n_profiles'] += m['n_profiles']; meta['n_valid_bins'] += m['n_valid_bins']
        meta['n_ok_bins'] += m['n_ok_bins']; meta['files'].append(fn)
    bronze = np.vstack(blocks)
    if verbose:
        print(f'bronze: {bronze.shape[0]} rows x {bronze.shape[1]} columns, '
              f'{len(np.unique(bronze[:, 0]))} profiles with data')
    return bronze, meta


def profiles_from_bronze(bronze):
    """One row per profile (the first bin of each time). Useful for the track, bottom track,
    water depth and temperature, which repeat in every bin of a profile."""
    _, ip = np.unique(bronze[:, 0], return_index=True)
    return bronze[ip]
