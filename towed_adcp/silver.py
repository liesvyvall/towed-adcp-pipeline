"""
Step 4 of the chapter: the silver matrix, compass calibration per repetition (Joyce, 1989;
Pollard and Read, 1989).

For each repetition the bottom track (u_bt, v_bt) is compared with the ship velocity from GPS
(u_sh, v_sh, from consecutive fixes) to get a rotation alpha and a scaling 1+beta:

    tan(alpha) = <u_bt v_sh - v_bt u_sh> / <u_bt u_sh + v_bt v_sh>
    1 + beta   = sqrt(<u_sh^2 + v_sh^2> / <u_bt^2 + v_bt^2>)

    u_c = (1+beta) (u cos(alpha) - v sin(alpha))
    v_c = (1+beta) (u sin(alpha) + v cos(alpha))

The silver matrix has 9 columns: the first 7 of the bronze matrix (with u, v corrected),
backscatter and temperature (Fig. 5b of the chapter).
"""

import numpy as np
import pandas as pd

SILVER_COLUMNS = ['time', 'lat', 'lon', 'u', 'v', 'bin_depth', 'depth', 'backscatter', 'temperature']


def joyce_calibration(profiles):
    """alpha (rad) and 1+beta of one repetition.

    `profiles`: rows of the bronze matrix with one row per profile (time, lat, lon, ..., u_bt, v_bt).
    """
    t, lat, lon, ubt, vbt = profiles[:, 0], profiles[:, 1], profiles[:, 2], profiles[:, 7], profiles[:, 8]
    dt = np.gradient(t)*86400
    vsh = np.gradient(lat)*60*1.852e5/dt                                  # ship velocity from GPS, cm/s
    ush = np.gradient(lon)*60*np.cos(np.deg2rad(lat[0]))*1.852e5/dt
    alpha = np.arctan(np.mean(ubt*vsh - vbt*ush)/np.mean(ubt*ush + vbt*vsh))
    beta = np.sqrt(np.mean(ush**2 + vsh**2)/np.mean(ubt**2 + vbt**2))
    return float(alpha), float(beta)


def rotate(u, v, alpha, beta):
    """Equation (3) of the chapter."""
    return beta*(u*np.cos(alpha) - v*np.sin(alpha)), beta*(u*np.sin(alpha) + v*np.cos(alpha))


def build_silver(bronze, reps, min_profiles=5, verbose=True):
    """Build the silver matrix: one Joyce correction per valid repetition in `reps`.

    Returns (silver, reps) where `reps` keeps only the valid repetitions and gains the columns
    alpha and beta. Repetitions with fewer than `min_profiles` profiles are dropped.
    """
    t = bronze[:, 0]
    blocks, rows = [], []
    for r in reps[reps.ok].itertuples():
        ii = (t >= r.t_start) & (t <= r.t_end)
        b = bronze[ii]
        _, ip = np.unique(b[:, 0], return_index=True)
        if len(ip) < min_profiles:
            continue
        alpha, beta = joyce_calibration(b[ip])
        p = np.zeros((len(b), 9))
        p[:, :7] = b[:, :7]
        p[:, 3], p[:, 4] = rotate(b[:, 3], b[:, 4], alpha, beta)
        p[:, 7:] = b[:, 9:]
        blocks.append(p)
        rows.append(dict(r._asdict(), alpha=alpha, beta=beta))
    silver = np.vstack(blocks)
    reps_ok = pd.DataFrame(rows).drop(columns='Index').reset_index(drop=True)
    if verbose:
        print(f'silver: {silver.shape[0]} rows x 9 columns, {len(reps_ok)} repetitions')
        print(reps_ok.groupby('transect', sort=False)[['alpha', 'beta']].agg(['median', 'std']).round(3))
    return silver, reps_ok
