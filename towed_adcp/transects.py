"""
Transect repetitions: when each pass starts and ends.

The chapter needs the start and end time of every repetition (the `timtran` file). There are two
ways to get them here:

1. `repetitions_from_file`: read the times from a file (the `timtran1.mat` of the chapter, or a
   CSV with columns transect, t_start, t_end).
2. `repetitions_by_vertices`: detect them from the track when the survey is a closed circuit that
   is repeated the same way every time. You give the vertices (the turning points) and the order
   in which they are visited; each leg between two consecutive vertices is a transect.

Both return the same table (DataFrame) with columns
    lap, transect, t_start, t_end, dur_min, d_start, d_end, ok
that the silver and gold steps use.
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

NM = 1852.0     # meters per minute of arc


def local_xy(lon, lat, lon0, lat0):
    """Local coordinates in meters (x east, y north), as in the chapter:
    1' of latitude = 1852 m; 1' of longitude = 1852 m x cos(lat)."""
    return (np.asarray(lon) - lon0)*60*NM*np.cos(np.deg2rad(lat0)), (np.asarray(lat) - lat0)*60*NM


def lonlat_from_xy(x, y, lon0, lat0):
    return lon0 + np.asarray(x)/(60*NM*np.cos(np.deg2rad(lat0))), lat0 + np.asarray(y)/(60*NM)


def transect_geometry(transects, vertices=None, lon0=None, lat0=None):
    """Normalize the transect definitions into a dict name -> geometry in meters.

    A transect can be given as:
      - [A, B]: names of two entries of `vertices` (dict name -> (lon, lat)); the gold distance is
        the projection on the A -> B axis.
      - {'A': (lon, lat), 'B': (lon, lat)}: same thing with explicit coordinates.
      - {'origin': (lon, lat)}: a straight line as in the chapter; the gold distance is the distance
        to the origin (gold.m).
    The local origin (lon0, lat0) defaults to the first vertex or the first origin.
    """
    geo = {}
    if lon0 is None:
        if vertices:
            lon0, lat0 = next(iter(vertices.values()))
        else:
            first = next(iter(transects.values()))
            lon0, lat0 = first['origin'] if 'origin' in first else first['A']
    for name, spec in transects.items():
        if isinstance(spec, (list, tuple)):
            A, B = vertices[spec[0]], vertices[spec[1]]
            names = (spec[0], spec[1])
        elif 'A' in spec:
            A, B = spec['A'], spec['B']
            names = ('A', 'B')
        else:
            A, B, names = spec['origin'], None, ('origin', None)
        xa, ya = local_xy(A[0], A[1], lon0, lat0)
        g = dict(A=tuple(A), xa=float(xa), ya=float(ya), names=names, lon0=lon0, lat0=lat0, mode='origin')
        if B is not None:
            xb, yb = local_xy(B[0], B[1], lon0, lat0)
            L = float(np.hypot(xb - xa, yb - ya))
            g.update(B=tuple(B), xb=float(xb), yb=float(yb), L=L, ex=float((xb - xa)/L), ey=float((yb - ya)/L), mode='axis')
        geo[name] = g
    return geo


def _table(rows):
    cols = ['lap', 'transect', 't_start', 't_end', 'dur_min', 'd_start', 'd_end', 'ok']
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def repetitions_by_vertices(profiles, vertices, transects, r_detect=120.0, min_sep=120, r_max=100.0,
                            dur_max=20.0, gap_max=60.0, verbose=True):
    """Detect the repetitions from the track of a circuit that is repeated.

    Parameters
    ----------
    profiles : bronze matrix with one row per profile (profiles_from_bronze).
    vertices : dict name -> (lon, lat) of the turning points of the circuit.
    transects : dict name -> [start vertex, end vertex], in sailing order. The start vertex of the
                first transect marks the beginning of every lap.
    r_detect : radius (m) to count a pass by the lap-start vertex.
    min_sep : minimum separation between passes by that vertex, in number of profiles.
    r_max : maximum distance (m) to both vertices for a repetition to be accepted.
    dur_max : maximum duration (min) of a repetition (so it is close to synoptic).
    gap_max : maximum gap (s) between profiles inside the repetition (file changes).

    How it works: the distance from every profile to every vertex is computed; local minima of the
    distance to the start vertex mark the beginning of each lap; inside each lap, the time of closest
    approach to each of the other vertices is found in order, and each pair of consecutive vertices
    bounds one repetition of that transect.
    """
    tp, lat, lon = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    lon0, lat0 = next(iter(vertices.values()))
    xp, yp = local_xy(lon, lat, lon0, lat0)
    dist = {}
    for k, (lo, la) in vertices.items():
        vx, vy = local_xy(lo, la, lon0, lat0)
        dist[k] = np.hypot(xp - vx, yp - vy)

    order = [spec[0] for spec in transects.values()]           # vertices in sailing order
    v0 = order[0]
    passes, _ = find_peaks(-dist[v0], height=-r_detect, distance=min_sep)
    if verbose:
        print(f'{len(passes)} passes by {v0} -> {len(passes) - 1} laps')

    rows = []
    for k in range(len(passes) - 1):
        i0, i1 = passes[k], passes[k + 1]
        idx = [i0]
        for v in order[1:]:                                     # closest approach to each vertex, searching forward
            idx.append(idx[-1] + int(dist[v][idx[-1]:i1].argmin()))
        idx.append(i1)
        for m, (name, spec) in enumerate(transects.items()):
            A, B = spec[0], spec[1]
            ia, ib = idx[m], idx[m + 1]
            dur = (tp[ib] - tp[ia])*1440
            gap = np.diff(tp[ia:ib + 1]).max()*86400 if ib > ia else 0.0
            ok = bool(dist[A][ia] < r_max and dist[B][ib] < r_max and dur < dur_max and gap < gap_max)
            rows.append(dict(lap=k, transect=name, t_start=tp[ia], t_end=tp[ib], dur_min=dur,
                             d_start=dist[A][ia], d_end=dist[B][ib], ok=ok))
    reps = _table(rows)
    if verbose and len(reps):
        print(reps.groupby('transect', sort=False).agg(n=('ok', 'sum'), mean_dur_min=('dur_min', 'mean')).round(1))
        print(f'{int(reps.ok.sum())} valid repetitions out of {len(reps)}')
    return reps


def repetitions_from_file(path, transects=None, transect=None, verbose=True):
    """Read the start and end times of each repetition from a file, like the chapter's `timtran`.

    Accepted formats (times in the same unit as the bronze matrix: decimal day of the month):
      - .mat with a variable `ti` of 2N values alternating start, end, start, end, ... (chapter).
      - .txt / .csv with a single column alternating start, end, ...
      - .txt / .csv with two columns: start, end.
      - .csv with a header and columns transect, t_start, t_end (and optionally lap).
    If the file has no transect name, every repetition is assigned to `transect` (or to the only
    entry of `transects`).
    """
    ext = os.path.splitext(path)[1].lower()
    df = None
    if ext == '.mat':
        from scipy.io import loadmat
        m = loadmat(path)
        key = [k for k in m if not k.startswith('__')][0]
        vals = np.asarray(m[key], float).ravel()
        t0, t1 = vals[0::2], vals[1::2]
    else:
        try:
            df = pd.read_csv(path, sep=None, engine='python')
            if not {'t_start', 't_end'} <= set(df.columns):
                df = None
        except Exception:
            df = None
        if df is None:
            vals = np.loadtxt(path, ndmin=2)
            if vals.shape[1] == 1:
                vals = vals.ravel(); t0, t1 = vals[0::2], vals[1::2]
            else:
                t0, t1 = vals[:, 0], vals[:, 1]
    if df is None:
        df = pd.DataFrame({'t_start': t0, 't_end': t1})
    if 'transect' not in df.columns:
        if transect is None:
            if transects is None or len(transects) != 1:
                raise ValueError('the file has no transect name: pass `transect` or define a single transect')
            transect = next(iter(transects))
        df['transect'] = transect
    if 'lap' not in df.columns:
        df['lap'] = df.groupby('transect').cumcount()
    df['dur_min'] = (df.t_end - df.t_start)*1440
    df['d_start'] = np.nan; df['d_end'] = np.nan
    df['ok'] = df.t_end > df.t_start
    reps = df[['lap', 'transect', 't_start', 't_end', 'dur_min', 'd_start', 'd_end', 'ok']].reset_index(drop=True)
    if verbose:
        print(f'{len(reps)} repetitions read from {os.path.basename(path)} '
              f'({reps.transect.nunique()} transect(s), mean duration {reps.dur_min.mean():.1f} min)')
    return reps
