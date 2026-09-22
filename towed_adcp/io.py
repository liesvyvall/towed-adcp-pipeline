"""
Reader for the WinRiver classic ASCII output, plus time and file helpers.

File layout (WinRiver II "Classic ASCII output"): one configuration line (7 fields) at the top,
then, for every profile, 6 header lines followed by a matrix with 13 columns and one row per bin.
See Fig. 4 of Valle-Levinson (2024).
"""

import os
from datetime import datetime, timedelta

import numpy as np

INVALID = -30000            # WinRiver flags bad values with -32768


def decimal_day(year, month, day, hour, minute, second, base=None):
    """Decimal day of the month: 19.5 is day 19 at 12:00.

    If `base` = (year, month) is an earlier month, the day keeps counting (32.1 is the 1st of the
    next month), so a survey that crosses a month boundary is not broken.
    """
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute)) + timedelta(seconds=float(second))
    if base is None:
        base = (int(year), int(month))
    return (dt - datetime(base[0], base[1], 1)).total_seconds()/86400 + 1


def to_datetime(t, time_base):
    """Decimal day (scalar or array) to datetime / datetime64 array."""
    origin = datetime(time_base[0], time_base[1], 1)
    if np.ndim(t) == 0:
        return origin + timedelta(days=float(t) - 1)
    return np.datetime64(origin) + ((np.asarray(t, float) - 1)*86400*1e6).astype('timedelta64[us]')


def read_winriver_ascii(fn, err_max=10.0, q_max=100.0, vbt_min=10.0, pg_min=None, time_base=None, verbose=True):
    """Read one WinRiver classic ASCII file and return its bronze matrix.

    Parameters
    ----------
    fn : path to the file.
    err_max : maximum |error velocity| (cm/s) to keep a bin (the chapter uses 10).
    q_max : maximum |discharge| to keep a bin (the chapter uses 100).
    vbt_min : minimum bottom-track speed (cm/s); slower profiles are dropped as idle or drifting
              (the chapter suggests 0.1-0.2 m/s).
    pg_min : minimum percent good per bin (the chapter uses 70). None disables the test; some
             exports report 0 in every bin.
    time_base : (year, month) used as the origin of the decimal day. None uses the first profile.

    Returns
    -------
    bronze : ndarray (n_bins, 11) with the columns listed in BRONZE_COLUMNS.
    meta : dict with time_base, number of profiles read and bin counts.
    """
    tok = open(fn).read().split()
    i = 7                                           # first line (7 fields) is the ADCP configuration
    rows, n_prof, n_val, n_ok = [], 0, 0, 0
    while i < len(tok):
        h1 = [float(s) for s in tok[i:i+13]]; i += 13   # yy mm dd hh mm ss cc, ensemble, n ens, pitch, roll, heading, temp
        h2 = [float(s) for s in tok[i:i+12]]; i += 12   # bottom track E, N, W, err, ..., depth per beam (last 4)
        i += 5                                           # distances made good
        h4 = [float(s) for s in tok[i:i+5]];  i += 5    # lat, lon, GPS velocity E, N, distance
        i += 9                                           # discharge by zone
        nr = int(tok[i]); i += 6                         # number of bins and units
        d = np.array(tok[i:i+13*nr], dtype=float).reshape(nr, 13); i += 13*nr
        n_prof += 1
        year = int(h1[0]); year += 2000 if year < 100 else 0
        if time_base is None:
            time_base = (year, int(h1[1]))

        tim = decimal_day(year, h1[1], h1[2], h1[3], h1[4], h1[5] + h1[6]/100, base=time_base)
        ubt, vbt, tem = h2[0], h2[1], h1[12]
        dep = np.mean(h2[8:12])                          # mean depth of the 4 beams
        val = d[:, 3] > INVALID                          # bins with a velocity
        n_val += val.sum()
        if ubt < INVALID or np.hypot(ubt, vbt) < vbt_min:  # no bottom track, or boat not moving
            continue
        ok = val & (np.abs(d[:, 6]) < err_max) & (np.abs(d[:, 12]) < q_max)
        if pg_min is not None:
            ok &= d[:, 11] > pg_min
        n_ok += ok.sum()
        if ok.sum() == 0:
            continue
        n = ok.sum()
        bsc = d[ok, 7:11].mean(axis=1)                   # mean backscatter of the 4 beams
        rows.append(np.column_stack([np.full(n, tim), np.full(n, h4[0]), np.full(n, h4[1]),
                                     d[ok, 3], d[ok, 4], d[ok, 0], np.full(n, dep),
                                     np.full(n, ubt), np.full(n, vbt), bsc, np.full(n, tem)]))
    if verbose:
        print(f'{os.path.basename(fn)}: {n_prof} profiles, {n_ok}/{n_val} bins with velocity pass the filters '
              f'({100*n_ok/max(n_val, 1):.0f} %)')
    bronze = np.vstack(rows) if rows else np.zeros((0, 11))
    meta = dict(time_base=time_base, n_profiles=n_prof, n_valid_bins=int(n_val), n_ok_bins=int(n_ok))
    return bronze, meta


def save_matrix(path, matrix, **meta):
    """Save a matrix (bronze or silver) to .npz together with scalar metadata."""
    np.savez(path, matrix=matrix, **{k: np.asarray(v) for k, v in meta.items()})


def load_matrix(path):
    """Load what save_matrix wrote: returns (matrix, metadata)."""
    z = np.load(path, allow_pickle=True)
    meta = {k: z[k].tolist() if z[k].ndim == 0 else z[k] for k in z.files if k != 'matrix'}
    if 'time_base' in meta:
        meta['time_base'] = tuple(int(x) for x in np.atleast_1d(meta['time_base']))
    return z['matrix'], meta
