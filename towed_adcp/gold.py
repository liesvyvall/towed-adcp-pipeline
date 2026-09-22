"""
Step 5 of the chapter: the gold matrices, a regular (distance, depth) grid per repetition.

For each transect and repetition, the triads (distance, bin depth, value) are interpolated with
linear Delaunay triangulation (scipy.interpolate.griddata, the same as Matlab's griddata) onto a
fixed grid. The distance is the projection on the A -> B axis of the transect ('axis' mode) or the
distance to an origin ('origin' mode, as in gold.m).

Each transect becomes a dict with:
    dii (nd,)  grid distances (m)                   hii (nz,)  grid depths (m)
    u, v, t, b (nz, nd, N)  corrected east/north velocity, time and backscatter
    bottom (nd,)  median bed of the transect         bottom_rep (nd, N)  bed of each repetition
    t_start (N,)  start of each repetition           L, ex, ey, xa, ya, lon0, lat0  geometry
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from .transects import local_xy


def distance_along(x, y, g):
    """Distance of each point (x, y) according to the transect geometry."""
    if g['mode'] == 'axis':
        return (x - g['xa'])*g['ex'] + (y - g['ya'])*g['ey']
    return np.hypot(x - g['xa'], y - g['ya'])


def build_gold(silver, reps, geometry, dy=10.0, dz=0.25, z_min=0.5, z_max=None, L=None,
               min_points=50, mask_below_bottom=True, verbose=True):
    """Build the gold matrices of all transects.

    Parameters
    ----------
    silver : silver matrix.
    reps : table of valid repetitions (the one returned by build_silver).
    geometry : dict name -> geometry (transect_geometry).
    dy, dz : horizontal (m) and vertical (m) grid spacing. Use dy >= dt x ship speed and
             dz >= bin size.
    z_min, z_max : vertical limits of the grid. z_max=None uses the deepest measured depth.
    L : horizontal length of the grid for 'origin' transects (m); None uses the largest observed
        distance. In 'axis' mode the A -> B length is used.
    min_points : minimum number of bins in a repetition to interpolate it.
    mask_below_bottom : set NaN at nodes below the bed of that repetition.
    """
    if z_max is None:
        z_max = np.nanmax(silver[:, 6])
    hii = np.arange(z_min, z_max + dz/2, dz)
    ts = silver[:, 0]
    gold = {}
    for name, g in geometry.items():
        rr = reps[reps.transect == name].sort_values('t_start')
        N = len(rr)
        if N == 0:
            continue
        x_all, y_all = local_xy(silver[:, 2], silver[:, 1], g['lon0'], g['lat0'])
        d_all = distance_along(x_all, y_all, g)
        Lg = g['L'] if g['mode'] == 'axis' else (L if L is not None else float(np.nanmax(d_all)))
        dii = np.arange(0, Lg + dy/2, dy)
        DI, HI = np.meshgrid(dii, hii)
        U, V, T, B = [np.full((len(hii), len(dii), N), np.nan) for _ in range(4)]
        bottom = np.full((len(dii), N), np.nan)
        for k, r in enumerate(rr.itertuples()):
            ii = (ts >= r.t_start) & (ts <= r.t_end)
            if ii.sum() < min_points:
                continue
            b = silver[ii]
            d = d_all[ii]
            pts = np.column_stack([d, b[:, 5]])
            try:
                for M, c in zip((U, V, T, B), (3, 4, 0, 7)):
                    M[:, :, k] = griddata(pts, b[:, c], (DI, HI), method='linear')
            except QhullError:
                continue
            # bed of this repetition: water-column depth against distance (one value per profile)
            _, ip = np.unique(b[:, 0], return_index=True)
            o = np.argsort(d[ip])
            bottom[:, k] = np.interp(dii, d[ip][o], b[ip, 6][o], left=np.nan, right=np.nan)
            if mask_below_bottom:
                below = HI > bottom[:, k][None, :]
                for M in (U, V, T, B):
                    M[:, :, k][below] = np.nan
        gold[name] = dict(dii=dii, hii=hii, u=U, v=V, t=T, b=B, bottom=np.nanmedian(bottom, axis=1),
                          bottom_rep=bottom, t_start=rr.t_start.values, L=Lg, mode=g['mode'],
                          xa=g['xa'], ya=g['ya'], ex=g.get('ex', np.nan), ey=g.get('ey', np.nan),
                          lon0=g['lon0'], lat0=g['lat0'], names=g['names'])
        if verbose:
            print(f'{name:14s} L = {Lg:5.0f} m  gold: {U.shape}  nodes with data: {100*np.isfinite(U).mean():.0f} %')
    return gold


def save_gold(path, gold):
    """Save all gold matrices to one .npz (keys <transect>__<field>)."""
    out = {}
    for name, g in gold.items():
        for k, v in g.items():
            out[f'{name}__{k}'] = np.asarray(v) if not isinstance(v, tuple) else np.asarray(v, dtype=object)
    np.savez(path, transects=np.array(list(gold)), **out)


def load_gold(path):
    z = np.load(path, allow_pickle=True)
    gold = {}
    for name in z['transects']:
        g = {}
        for key in z.files:
            if key.startswith(f'{name}__'):
                v = z[key]
                g[key.split('__', 1)[1]] = v.item() if v.ndim == 0 else v
        gold[str(name)] = g
    return gold
