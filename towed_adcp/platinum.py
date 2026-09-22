"""
Step 6 of the chapter: the platinum matrix, a least-squares fit at every node of the gold grid.

At each node the series of N repetitions is approximated by a residual plus M harmonic bands of
known frequency (eq. 4 of the chapter):

    u_m = u_r + sum_i u_0i sin(w_i t + phi_i) = u_r + sum_i [u_1i sin(w_i t) + u_2i cos(w_i t)]

Minimizing the squared error gives the system B = A X: with G the matrix whose columns are
[1, sin(w_1 t), cos(w_1 t), sin(w_2 t), ...], A = G^T G (the sums of sines and cosines of
lsqfit.m) and B = G^T u_m. Amplitude u_0i = sqrt(u_1i^2 + u_2i^2), phase phi_i = atan2(u_2i, u_1i).

The platinum matrix of each transect has shape (nz, nd, 2M+1): residual, then amplitude and phase
of each band.
"""

import numpy as np
import pandas as pd


def angular_frequencies(periods_h):
    """Angular frequencies in rad/day from periods in hours (dict or list)."""
    p = np.array(list(periods_h.values()) if isinstance(periods_h, dict) else periods_h, float)
    return 2*np.pi/(p/24)


def lsqfit(t, u, w):
    """Least-squares fit u = ur + sum a_i sin(w_i t + phi_i) (lsqfit.m for any number of bands).

    t in days, w in rad/day. Returns ([ur, a_1, phi_1, a_2, phi_2, ...], fitted series).
    """
    G = np.column_stack([np.ones_like(t)] + [f(wi*t) for wi in w for f in (np.sin, np.cos)])
    A = G.T @ G                      # coefficient matrix (the sums), same as lsqfit.m
    B = G.T @ u                      # measurement vector
    c = np.linalg.solve(A, B)
    out = [c[0]]
    for i in range(len(w)):
        u1, u2 = c[1 + 2*i], c[2 + 2*i]
        out += [np.hypot(u1, u2), np.arctan2(u2, u1)]
    return np.array(out), G @ c


def reconstruct(coef, t, w):
    """Series u(t) from the lsqfit coefficients (residual, amplitudes and phases)."""
    u = np.full_like(np.asarray(t, float), coef[0])
    for i, wi in enumerate(w):
        u = u + coef[1 + 2*i]*np.sin(wi*np.asarray(t) + coef[2 + 2*i])
    return u


def fit_metrics(u, uf):
    """R^2, Willmott skill and RMSE between observed `u` and fitted `uf`."""
    r2 = 1 - np.sum((u - uf)**2)/np.sum((u - u.mean())**2)
    skill = 1 - np.sum((u - uf)**2)/np.sum((np.abs(uf - u.mean()) + np.abs(u - u.mean()))**2)
    return r2, skill, np.sqrt(np.mean((u - uf)**2))


def fit_fields(fields, tg, w, n_min):
    """Node-by-node fit of several (nz, nd, N) fields that share the time matrix `tg`.

    Returns (coef, quality, n): coef[c] (nz, nd, 2M+1), quality[c] (nz, nd, 3) = R^2, skill, RMSE
    and n (nz, nd) good values per node. A node is fitted when it has more than n_min values.
    """
    names = list(fields)
    nz, nd, N = fields[names[0]].shape
    coef = {c: np.full((nz, nd, 1 + 2*len(w)), np.nan) for c in names}
    qual = {c: np.full((nz, nd, 3), np.nan) for c in names}
    n = np.zeros((nz, nd), int)
    for i in range(nz):
        for j in range(nd):
            jj = np.isfinite(tg[i, j])
            for c in names:
                jj &= np.isfinite(fields[c][i, j])
            n[i, j] = jj.sum()
            if jj.sum() <= n_min:
                continue
            for c in names:
                cc, fit = lsqfit(tg[i, j, jj], fields[c][i, j, jj], w)
                coef[c][i, j] = cc
                qual[c][i, j] = fit_metrics(fields[c][i, j, jj], fit)
    return coef, qual, n


def build_platinum(gold, periods_h, n_min=None, fields=('u', 'v'), verbose=True):
    """Platinum matrices of all transects.

    periods_h : band periods in hours, e.g. {'D1': 23.93, 'D2': 12.42, 'D4': 6.21}.
    n_min : minimum number of values per node; None = half the number of repetitions.
    fields : gold fields to fit ('u', 'v', 'b', ...).
    """
    w = angular_frequencies(periods_h)
    bands = list(periods_h) if isinstance(periods_h, dict) else [f'B{i+1}' for i in range(len(w))]
    platinum = {}
    for name, g in gold.items():
        N = g['u'].shape[2]
        nm = int(N/2) if n_min is None else n_min
        coef, qual, n = fit_fields({c: g[c] for c in fields}, g['t'], w, nm)
        p = dict(n=n, bands=bands, periods_h=np.array(list(periods_h.values()) if isinstance(periods_h, dict) else periods_h))
        for c in fields:
            p[c] = coef[c]; p[f'quality_{c}'] = qual[c]
        platinum[name] = p
        if verbose:
            txt = '  '.join(f'{c}: R2 {np.nanmean(qual[c][:, :, 0]):.2f} RMSE {np.nanmean(qual[c][:, :, 2]):.1f}' for c in fields)
            print(f'{name:14s} fitted nodes: {(n > nm).sum():4d}   {txt}')
    return platinum


def principal_axis(silver):
    """Angle (rad, counterclockwise from east, pointing north) of the principal axis of the
    depth-averaged current, and the fraction of the variance it explains."""
    mean = pd.DataFrame({'t': silver[:, 0], 'u': silver[:, 3], 'v': silver[:, 4]}).groupby('t').mean()
    val, vec = np.linalg.eigh(np.cov(mean.u, mean.v))
    e = vec[:, np.argmax(val)]
    if e[1] < 0:
        e = -e
    return float(np.arctan2(e[1], e[0])), float(val.max()/val.sum())


def rotate_gold(gold, theta):
    """Copy of the gold matrices with 'u', 'v' rotated to the axis `theta` (rad from east):
    'u' becomes the along-axis component (positive in the direction of theta) and 'v' the
    cross-axis component (positive 90 degrees to the right of the axis)."""
    ea = np.array([np.cos(theta), np.sin(theta)])
    et = np.array([np.sin(theta), -np.cos(theta)])
    out = {}
    for name, g in gold.items():
        h = dict(g)
        h['u'] = g['u']*ea[0] + g['v']*ea[1]
        h['v'] = g['u']*et[0] + g['v']*et[1]
        h['theta'] = theta
        out[name] = h
    return out


def save_platinum(path, platinum):
    out = {}
    for name, p in platinum.items():
        for k, v in p.items():
            out[f'{name}__{k}'] = np.asarray(v)
    np.savez(path, transects=np.array(list(platinum)), **out)


def load_platinum(path):
    z = np.load(path, allow_pickle=True)
    plat = {}
    for name in z['transects']:
        p = {}
        for key in z.files:
            if key.startswith(f'{name}__'):
                v = z[key]
                p[key.split('__', 1)[1]] = v.tolist() if v.dtype.kind in 'US' else (v.item() if v.ndim == 0 else v)
        plat[str(name)] = p
    return plat
