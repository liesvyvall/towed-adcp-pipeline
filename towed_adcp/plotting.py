"""
Figures for the pipeline (journal style: Times New Roman, cmocean colormaps and grays).

Every function returns the figure; if `save` is given, the figure is written at 300 dpi.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean.cm as cmo
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .io import to_datetime
from .transects import local_xy, lonlat_from_xy
from .platinum import angular_frequencies, lsqfit, fit_metrics, reconstruct

BED, BED_LINE = '0.84', '0.2'      # fill and line of the bed in the sections
LETTERS = 'abcdefghijklmnop'


def set_style():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
        'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10.5,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'axes.linewidth': 0.8, 'axes.edgecolor': '0.25', 'axes.labelcolor': '0.1', 'text.color': '0.1',
        'xtick.color': '0.25', 'ytick.color': '0.25', 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'legend.frameon': False, 'axes.grid': False,
        'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


# ------------------------------------------------------------------ helpers
def label(ax, s, x=0.02, y=0.96):
    ax.text(x, y, s, transform=ax.transAxes, va='top', ha='left', fontweight='bold', fontsize=12,
            bbox=dict(fc='w', ec='none', alpha=0.8, pad=1.5))


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, color='0.85', lw=0.5)
    ax.set_axisbelow(True)


def time_axis(ax, fmt='%d-%b\n%H:%M', hours=(0, 6, 12, 18)):
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(hours)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def colorbar(mappable, ax, lab, **kw):
    cb = plt.colorbar(mappable, ax=ax, label=lab, **kw)
    cb.outline.set_linewidth(0.6)
    return cb


def transect_colors(names):
    return dict(zip(names, cmo.phase(np.linspace(0, 0.85, len(names)))))


def pretty(name):
    return name.replace('_', '→')


def nice_max(x, q=99):
    """Percentile q rounded up to a round number (1, 2, 2.5, 5 x 10^n)."""
    v = np.nanpercentile(x, q)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10**np.floor(np.log10(v))
    return float(next(m*e for m in (1, 2, 2.5, 5, 10) if m*e >= v))


def load_coast(path):
    """Coastline from a shapefile/GeoJSON through geopandas (optional). Returns None on failure."""
    if path is None:
        return None
    try:
        import geopandas as gpd
        return gpd.read_file(path)
    except Exception as e:                       # geopandas missing, PROJ database not found, etc.
        print(f'warning: could not read the coastline ({e}); maps are drawn without it')
        return None


def map_axes(ax, extent, coast=None, tick=0.002):
    """Lon/lat map axes with the right aspect ratio and no offset in the tick labels."""
    if coast is not None:
        coast.plot(ax=ax, color='0.9', edgecolor='0.4', lw=0.6, zorder=0)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1/np.cos(np.deg2rad(0.5*(extent[2] + extent[3]))))
    ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.MultipleLocator(tick))
        axis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))


def default_extent(profiles, margin=0.15):
    lo, la = profiles[:, 2], profiles[:, 1]
    dlo, dla = lo.max() - lo.min(), la.max() - la.min()
    return (lo.min() - margin*dlo, lo.max() + margin*dlo, la.min() - margin*dla, la.max() + margin*dla)


def section(ax, g, field, levels, cmap, bottom=True, sidelobe=0.86, z_max=None):
    """Filled contours of a (nz, nd) field with the bed in gray and the side-lobe band hatched."""
    cf = ax.contourf(g['dii'], g['hii'], field, levels, cmap=cmap, extend='both')
    z_max = g['hii'].max() if z_max is None else z_max
    if bottom:
        f = g['bottom']; ok = np.isfinite(f)
        ax.fill_between(g['dii'][ok], sidelobe*f[ok], f[ok], facecolor='none', hatch='////', edgecolor='0.7', lw=0, zorder=3)
        ax.fill_between(g['dii'][ok], f[ok], z_max + 1, color=BED, lw=0, zorder=3)
        ax.plot(g['dii'][ok], f[ok], color=BED_LINE, lw=1.2, zorder=4)
    ax.set_ylim(z_max, g['hii'].min() - 0.1); ax.set_xlim(0, g['L'])
    return cf


def _save(fig, save):
    if save:
        fig.savefig(save)
    return fig


# ------------------------------------------------------------------ bronze
def plot_trajectory(profiles, time_base, coast=None, extent=None, save=None):
    """Ship track colored by water-column depth and by hours since the start."""
    extent = extent or default_extent(profiles)
    tp = profiles[:, 0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 8.5))
    for ax, c, cmap, lab, let in zip(axs, (profiles[:, 6], (tp - tp.min())*24), (cmo.deep, cmo.thermal),
                                     ('Water-column depth (m)',
                                      f'Hours since start ({to_datetime(tp.min(), time_base):%d-%b %H:%M})'), 'ab'):
        map_axes(ax, extent, coast)
        sc = ax.scatter(profiles[:, 2], profiles[:, 1], c=c, s=4, lw=0, cmap=cmap, rasterized=True)
        colorbar(sc, ax, lab, shrink=0.75, pad=0.03)
        label(ax, f'({let})')
    axs[1].set_ylabel('')
    fig.suptitle('Towed ADCP track', y=0.93)
    return _save(fig, save)


def plot_bronze_series(bronze, time_base, save=None):
    """Time series of every bronze column (Fig. 6 of the chapter)."""
    tf = to_datetime(bronze[:, 0], time_base)
    fig, axs = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    axs = axs.ravel()
    panels = [(2, 'Longitude (°)'), (1, 'Latitude (°)'), (3, '$u$ (cm/s)'), (4, '$v$ (cm/s)'),
              (9, 'Backscatter'), (6, 'Depth (m)'), (10, 'Temperature (°C)')]
    sc = None
    for ax, (col, lab), let in zip(axs, panels, 'abcdefg'):
        if col in (3, 4, 9):
            sc = ax.scatter(tf, bronze[:, col], c=bronze[:, 5], s=1, lw=0, cmap=cmo.deep, rasterized=True)
        elif col == 6:
            ax.plot(tf, bronze[:, 5], '.', ms=1, color='0.8', rasterized=True, label='bins')
            ax.plot(tf, bronze[:, col], '.', ms=1, color=cmo.deep(0.8), rasterized=True, label='water column')
            ax.invert_yaxis(); ax.legend(markerscale=8, loc='upper right')
        else:
            ax.plot(tf, bronze[:, col], '.', ms=1, color=cmo.deep(0.7), rasterized=True)
        ax.set_ylabel(lab); label(ax, f'({let})'); clean(ax)
        if col in (1, 2):
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    ax = axs[7]
    ax.plot(tf, np.hypot(bronze[:, 7], bronze[:, 8]), '.', ms=1, color=cmo.deep(0.7), rasterized=True)
    ax.set_ylabel('Ship speed (cm/s)'); label(ax, '(h)'); clean(ax)
    for ax in axs[6:]:
        time_axis(ax)
    colorbar(sc, axs.tolist(), 'Bin depth (m)', shrink=0.4, pad=0.01)
    return _save(fig, save)


def plot_bronze_profiles(bronze, bin_size=0.2, save=None):
    """2-D histograms of u, v and backscatter against bin depth."""
    z0 = np.nanmin(bronze[:, 5]) - bin_size/2
    edges = np.arange(z0, np.nanmax(bronze[:, 5]) + bin_size, bin_size)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, col, lab, let in zip(axs, (3, 4, 9), ('$u$ (cm/s)', '$v$ (cm/s)', 'Backscatter'), 'abc'):
        h = ax.hist2d(bronze[:, col], bronze[:, 5], bins=(80, edges), cmap=cmo.dense, cmin=1, rasterized=True)
        ax.set_xlabel(lab); label(ax, f'({let})')
        colorbar(h[3], ax, 'Count', pad=0.02)
    axs[0].set_ylabel('Bin depth (m)'); axs[0].invert_yaxis()
    return _save(fig, save)


# ------------------------------------------------------------------ repetitions
def plot_repetitions(profiles, reps, time_base, vertices=None, coast=None, extent=None, lap=None, hours=3, save=None):
    """Map with one lap colored by transect, and latitude against time for the first hours."""
    extent = extent or default_extent(profiles)
    tp = profiles[:, 0]
    reps = reps[reps.ok]
    names = list(dict.fromkeys(reps.transect))
    col = transect_colors(names)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), gridspec_kw=dict(width_ratios=[1, 1.6]))
    map_axes(ax, extent, coast)
    ax.plot(profiles[:, 2], profiles[:, 1], '.', ms=1, color='0.78', rasterized=True, zorder=1)
    if lap is None:
        lap = reps.lap.value_counts().index[0] if len(reps) else 0
    for r in reps[reps.lap == lap].itertuples():
        ii = (tp >= r.t_start) & (tp <= r.t_end)
        ax.plot(profiles[ii, 2], profiles[ii, 1], '.', ms=4, color=col[r.transect], label=pretty(r.transect))
    if vertices:
        for v, (lo, la) in vertices.items():
            ax.plot(lo, la, 'o', ms=7, mfc='w', mec='0.15', mew=1.2, zorder=6)
            ax.annotate(v, (lo, la), xytext=(7, 5), textcoords='offset points', fontsize=11, fontweight='bold',
                        bbox=dict(fc='w', ec='none', alpha=0.8, pad=1), zorder=6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, title=f'lap {lap}'); label(ax, '(a)')

    ii = tp < tp[0] + hours/24
    ax2.plot(to_datetime(tp[ii], time_base), profiles[ii, 1], '.', ms=2, color='0.78')
    for r in reps[reps.t_start < tp[0] + hours/24].itertuples():
        jj = ii & (tp >= r.t_start) & (tp <= r.t_end)
        ax2.plot(to_datetime(tp[jj], time_base), profiles[jj, 1], '.', ms=4, color=col[r.transect])
        ax2.axvline(to_datetime(r.t_start, time_base), color='0.3', lw=0.6, ls='--')
    ax2.set_ylabel('Latitude (°)'); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    ax2.set_xlabel(f'{to_datetime(tp[0], time_base):%d-%b-%Y}'); clean(ax2); label(ax2, '(b)')
    ax2.set_title(f'First {hours} h: dashed lines mark the start of each repetition')
    return _save(fig, save)


# ------------------------------------------------------------------ silver
def plot_alpha_beta(reps, time_base, save=None):
    """Joyce's alpha and beta for every repetition, with the usual ranges of the chapter shaded."""
    names = list(dict.fromkeys(reps.transect))
    col = transect_colors(names)
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for name in names:
        r = reps[reps.transect == name]
        tf = to_datetime(r.t_start.values, time_base)
        axs[0].plot(tf, r.alpha, 'o', ms=5.5, mec='w', mew=0.5, color=col[name], label=pretty(name))
        axs[1].plot(tf, r.beta - 1, 'o', ms=5.5, mec='w', mew=0.5, color=col[name])
    axs[0].axhspan(-0.2, 0.2, color='0.93', zorder=0); axs[1].axhspan(-0.03, 0.03, color='0.93', zorder=0)
    axs[0].set_ylim(-0.23, 0.23)
    axs[0].set_ylabel(r'$\alpha$ (rad)'); axs[1].set_ylabel(r'$\beta$')
    axs[0].legend(ncol=min(len(names), 6), loc='upper center'); label(axs[0], '(a)'); label(axs[1], '(b)')
    for ax in axs:
        ax.axhline(0, color='0.3', lw=0.8); clean(ax)
    time_axis(axs[1])
    return _save(fig, save)


def plot_correction(bronze, reps, transect, time_base, save=None):
    """Velocities before and after the Joyce correction on one transect (Fig. 7 of the chapter)."""
    from .silver import rotate
    t = bronze[:, 0]
    b, p = [], []
    for r in reps[reps.transect == transect].itertuples():
        bb = bronze[(t >= r.t_start) & (t <= r.t_end)]
        b.append(bb); p.append(np.column_stack(rotate(bb[:, 3], bb[:, 4], r.alpha, r.beta)))
    b, p = np.vstack(b), np.vstack(p)
    tf = to_datetime(b[:, 0], time_base)
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, c, lab in zip(axs, (3, 4), ('$u$ (cm/s)', '$v$ (cm/s)')):
        ax.plot(tf, b[:, c], '.', ms=2.5, color='0.55', label='bronze', rasterized=True)
        ax.plot(tf, p[:, c - 3], '.', ms=1.5, color=cmo.thermal(0.55), label='silver (corrected)', rasterized=True)
        ax.set_ylabel(lab)
    axs[2].plot(tf, p[:, 0] - b[:, 3], '.', ms=2, color=cmo.balance(0.2), label='$u_c - u$', rasterized=True)
    axs[2].plot(tf, p[:, 1] - b[:, 4], '.', ms=2, color=cmo.balance(0.8), label='$v_c - v$', rasterized=True)
    axs[2].set_ylabel('Correction (cm/s)'); axs[2].legend(markerscale=6)
    axs[0].legend(markerscale=6); axs[0].set_title(f'Transect {pretty(transect)}: before and after the compass correction')
    for ax, let in zip(axs, 'abc'):
        label(ax, f'({let})'); clean(ax)
    time_axis(axs[2])
    return _save(fig, save)


# ------------------------------------------------------------------ gold
def plot_gold_panel(g, name, time_base, field='v', levels=None, n_panels=12, save=None):
    """`n_panels` repetitions of one gold field spread over the whole survey."""
    N = g[field].shape[2]
    sel = np.linspace(0, N - 1, min(n_panels, N)).round().astype(int)
    if levels is None:
        m = nice_max(np.abs(g[field]), 99)
        levels = np.linspace(-m, m, 25)
    ncol = 4 if len(sel) > 6 else 3
    nrow = int(np.ceil(len(sel)/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.3*nrow), sharex=True, sharey=True, layout='constrained')
    axs = np.atleast_1d(axs).ravel()
    cf = None
    for ax, k in zip(axs, sel):
        cf = section(ax, g, g[field][:, :, k], levels, cmo.balance)
        ax.set_title(f'rep. {k}: {to_datetime(g["t_start"][k], time_base):%d-%b %H:%M}', fontsize=11.5)
    for ax in axs[len(sel):]:
        ax.axis('off')
    for ax in axs[-ncol:]: ax.set_xlabel('Distance (m)')
    for ax in axs[::ncol]: ax.set_ylabel('Depth (m)')
    fig.suptitle(f'Gold matrix, transect {pretty(name)}: ${field}_c$')
    colorbar(cf, axs.tolist(), f'${field}_c$ (cm/s)', shrink=0.6, pad=0.01)
    return _save(fig, save)


# ------------------------------------------------------------------ platinum
def plot_node_fit(g, name, periods_h, time_base, i=None, j=None, z=1.5, d=None, field='v', save=None):
    """Series at one gold node with fits that add one band at a time (Fig. 9 of the chapter)."""
    if i is None:
        i = int(np.argmin(np.abs(g['hii'] - z)))
    if j is None:
        j = int(np.argmin(np.abs(g['dii'] - (g['L']/2 if d is None else d))))
    t1, u1 = g['t'][i, j], g[field][i, j]
    jj = np.isfinite(u1) & np.isfinite(t1)
    t1, u1 = t1[jj], u1[jj]
    tt = np.linspace(t1.min(), t1.max(), 500)
    bands = list(periods_h)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(to_datetime(t1, time_base), u1, 'o', ms=6.5, color='0.2', mec='w', mew=0.8,
            label=f'gold ({jj.sum()} repetitions)', zorder=5)
    # the band closest to semidiurnal goes first, then the others are added one by one
    order = [bands[np.argmin(np.abs(np.array(list(periods_h.values())) - 12.42))]]
    order += [b for b in bands if b not in order]
    for k in range(len(order)):
        sel = order[:k + 1]
        w = angular_frequencies({b: periods_h[b] for b in sel})
        cc, uf = lsqfit(t1, u1, w)
        r2 = fit_metrics(u1, uf)[0]
        ax.plot(to_datetime(tt, time_base), reconstruct(cc, tt, w), lw=1.2 + k, color=cmo.thermal(0.3 + 0.5*k/max(len(order) - 1, 1)),
                label=f'{" + ".join(sel)}:  $R^2$ = {r2:.2f}, residual = {cc[0]:+.1f} cm/s')
    ax.axhline(0, color='0.3', lw=0.8); clean(ax); ax.legend()
    ax.set_ylabel(f'${field}_c$ (cm/s)'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
    ax.set_title(f'Fit at one node of transect {pretty(name)} (y = {g["dii"][j]:.0f} m, z = {g["hii"][i]:.2f} m)')
    return _save(fig, save)


def hour_of_max(phase, T_h, t_ref):
    """Hour of the maximum of a sin(wt + phi), in hours after t_ref (decimal day), modulo T."""
    w = 2*np.pi/(T_h/24)
    return (((np.pi/2 - phase)/w - t_ref) % (T_h/24))*24


def plot_platinum_panel(g, p, name, time_base, fields=('u', 'v'), main_band=None, amp_levels=None,
                        res_levels=None, save=None):
    """Residual and amplitude of every band per component, plus the hour of the maximum of the
    main band and the R^2 of the fit for the last component. Contour levels adapt to the data
    (all components together) unless given."""
    bands, periods = list(p['bands']), np.asarray(p['periods_h'], float)
    if main_band is None:
        main_band = bands[int(np.argmin(np.abs(periods - 12.42)))]
    kb = bands.index(main_band)
    if res_levels is None:
        m = nice_max(np.abs(np.stack([p[c][:, :, 0] for c in fields])), 98)
        res_levels = np.linspace(-m, m, 21)
    if amp_levels is None:
        m = nice_max(np.stack([p[c][:, :, 1 + 2*k] for c in fields for k in range(len(bands))]), 99)
        amp_levels = {b: np.linspace(0, m, 21) for b in bands}
    rows = [('residual', 0, res_levels, cmo.balance, 'cm/s')]
    rows += [(f'{b} amplitude', 1 + 2*k, amp_levels[b], cmo.amp, 'cm/s') for k, b in enumerate(bands)]
    nr = len(rows) + 1
    fig, axs = plt.subplots(nr, len(fields), figsize=(6.5*len(fields), 3*nr), sharex=True, sharey=True, layout='constrained')
    axs = np.atleast_2d(axs)
    for kf, (tit, k, niv, cmap, uni) in enumerate(rows):
        cf = None
        for kc, c in enumerate(fields):
            ax = axs[kf, kc]
            cf = section(ax, g, p[c][:, :, k], np.asarray(niv), cmap)
            if k == 0 and c == fields[-1]:
                ax.contour(g['dii'], g['hii'], p[c][:, :, 0], [0], colors='0.25', linewidths=1)
            label(ax, f'${c}_c$: {tit}')
        colorbar(cf, axs[kf].tolist(), uni, pad=0.02, aspect=12)
    # last row: hour of the maximum of the main band, and R^2 of the last component
    c = fields[-1]; T = periods[kb]
    t_ref = np.floor(np.nanmin(g['t_start']))
    hm = np.where(p[c][:, :, 1 + 2*kb] > 5, hour_of_max(p[c][:, :, 2 + 2*kb], T, t_ref), np.nan)
    cf = section(axs[-1, 0], g, hm, np.linspace(0, T, 26), cmo.phase)
    colorbar(cf, axs[-1, 0], f'h after {to_datetime(t_ref, time_base):%d-%b %H:%M}', pad=0.02, aspect=12,
             ticks=np.arange(0, T + 0.1, 2))
    cf = section(axs[-1, -1], g, p[f'quality_{c}'][:, :, 0], np.arange(0.5, 1.001, 0.025), cmo.tempo)
    colorbar(cf, axs[-1, -1], '$R^2$', pad=0.02, aspect=12, ticks=np.arange(0.5, 1.01, 0.1))
    label(axs[-1, 0], f'${c}_c$: hour of {main_band} maximum'); label(axs[-1, -1], f'${c}_c$: $R^2$ of the fit')
    for ax in axs[:, 0]: ax.set_ylabel('Depth (m)')
    for ax in axs[-1]: ax.set_xlabel(f'Distance from {g["names"][0]} (m)')
    fig.suptitle(f'Platinum matrix, transect {pretty(name)}')
    return _save(fig, save)


def plot_residual_sections(gold, platinum, field='v', lab='Residual $v_c$ (cm/s)', levels=None,
                           zero_line=True, save=None):
    """Residual of one field on every transect, in a single figure."""
    names = list(gold)
    if levels is None:
        m = nice_max(np.abs(np.concatenate([platinum[n][field][:, :, 0].ravel() for n in names])), 98)
        levels = np.linspace(-m, m, 21)
    ncol = 2 if len(names) > 1 else 1
    nrow = int(np.ceil(len(names)/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(7*ncol, 3.7*nrow), layout='constrained')
    axs = np.atleast_1d(axs).ravel()
    cf = None
    for ax, name, let in zip(axs, names, LETTERS):
        g, r = gold[name], platinum[name][field][:, :, 0]
        cf = section(ax, g, r, levels, cmo.balance)
        if zero_line:
            ax.contour(g['dii'], g['hii'], r, [0], colors='0.25', linewidths=1)
        label(ax, f'({let}) {pretty(name)}')
        ax.set_xlabel(f'Distance from {g["names"][0]} (m)'); ax.set_ylabel('Depth (m)')
    for ax in axs[len(names):]:
        ax.axis('off')
    colorbar(cf, axs[:len(names)].tolist(), lab, shrink=0.6, pad=0.01)
    return _save(fig, save)


def _layer_mean(p, c, j, where, hii, z_top=1.5, n_min=30, n_bottom=4):
    col = p[c][:, j, 0]
    ok = np.isfinite(col) & (p['n'][:, j] >= n_min)
    if ok.sum() < 3:
        return np.nan
    i = np.where(ok)[0]
    sel = i[hii[i] <= z_top] if where == 'top' else i[-n_bottom:]
    return col[sel].mean() if len(sel) else np.nan


def plot_residual_map(gold, platinum, profiles, coast=None, extent=None, band=None, arrow_scale=0.0025/20,
                      ellipse_scale=0.0009/30, ref_arrow=10, ref_ellipse=30, save=None):
    """Residual near the surface and near the bed (arrows) and the ellipses of one band near the
    surface, on the map. Only meaningful for transects in 'axis' mode."""
    extent = extent or default_extent(profiles)
    aspect = 1/np.cos(np.deg2rad(0.5*(extent[2] + extent[3])))
    names = list(gold); col = transect_colors(names)
    fig, axs = plt.subplots(1, 3, figsize=(20, 9), layout='constrained')
    for ax in axs:
        map_axes(ax, extent, coast); ax.plot(profiles[:, 2], profiles[:, 1], '.', ms=0.5, color='0.82', rasterized=True, zorder=1)
    Q = {}
    for name in names:
        g, p = gold[name], platinum[name]
        hii = g['hii']
        bands = list(p['bands']); periods = np.asarray(p['periods_h'], float)
        kb = bands.index(band) if band else int(np.argmin(np.abs(periods - 12.42)))
        lon0, lat0 = g['lon0'], g['lat0']

        def lonlat(d):
            if g['mode'] == 'axis':
                return lonlat_from_xy(g['xa'] + d*g['ex'], g['ya'] + d*g['ey'], lon0, lat0)
            return lonlat_from_xy(g['xa'] + d, np.full_like(d, g['ya']), lon0, lat0)   # origin mode: local x axis
        for ax, where, c in ((axs[0], 'top', cmo.balance(0.15)), (axs[1], 'bottom', cmo.balance(0.85))):
            jj = np.arange(0, len(g['dii']), 2)
            ur = np.array([_layer_mean(p, 'u', j, where, hii) for j in jj])
            vr = np.array([_layer_mean(p, 'v', j, where, hii) for j in jj])
            lo, la = lonlat(g['dii'][jj])
            Q[where] = ax.quiver(lo, la, ur*arrow_scale*aspect, vr*arrow_scale, angles='xy', scale_units='xy', scale=1,
                                 color=c, width=0.004, zorder=4)
        tt = np.linspace(0, 2*np.pi, 60)
        for j in range(0, len(g['dii']), 4):
            ok = np.isfinite(p['u'][:, j, 0]) & (hii <= 1.5)
            if ok.sum() < 2:
                continue
            au = p['u'][ok, j, 1 + 2*kb].mean(); av = p['v'][ok, j, 1 + 2*kb].mean()
            fu = np.arctan2(np.sin(p['u'][ok, j, 2 + 2*kb]).mean(), np.cos(p['u'][ok, j, 2 + 2*kb]).mean())
            fv = np.arctan2(np.sin(p['v'][ok, j, 2 + 2*kb]).mean(), np.cos(p['v'][ok, j, 2 + 2*kb]).mean())
            lo1, la1 = lonlat(np.array([g['dii'][j]]))
            axs[2].plot(lo1 + au*np.sin(tt + fu)*ellipse_scale*aspect, la1 + av*np.sin(tt + fv)*ellipse_scale, '-',
                        color=col[name], lw=1, zorder=4)
            axs[2].plot(lo1, la1, '.', color=col[name], ms=3, zorder=5)
    for ax, where in zip(axs[:2], ('top', 'bottom')):
        ax.quiverkey(Q[where], 0.78, 0.06, ref_arrow*arrow_scale*aspect, f'{ref_arrow} cm/s', coordinates='axes', labelpos='S', color='0.15')
    lo1, la1 = extent[1] - 0.12*(extent[1] - extent[0]), extent[2] + 0.12*(extent[3] - extent[2])
    tt = np.linspace(0, 2*np.pi, 60)
    axs[2].plot(lo1 + ref_ellipse*np.sin(tt)*ellipse_scale*aspect, la1 + ref_ellipse*np.cos(tt)*ellipse_scale, '-', color='0.15', lw=1.2)
    axs[2].text(lo1, la1 - ref_ellipse*ellipse_scale*1.3, f'{ref_ellipse} cm/s', ha='center', fontsize=10)
    for ax, tit, let in zip(axs, ('Residual, surface layer (z ≤ 1.5 m)', 'Residual, deepest meter measured',
                                  'Main-band ellipses, surface layer'), 'abc'):
        ax.set_title(tit); label(ax, f'({let})')
    for name in names:
        axs[2].plot([], [], '-', color=col[name], label=pretty(name))
    axs[2].legend(loc='upper right', fontsize=9.5)
    return _save(fig, save)


# ------------------------------------------------------------------ surface temperature
def plot_surface_temperature_maps(profiles, time_base, reps=None, gold=None, window_h=4, cell=25.0, grid=10.0,
                                  max_dist=120.0, coast=None, extent=None, save=None):
    """Maps of the transducer temperature averaged over `window_h`-hour windows, with contours and
    one common color scale (min and max of the averaged fields). The last panel shows the
    temperature series with the windows and, when `gold` is given, the tide estimated from the
    water depth."""
    extent = extent or default_extent(profiles)
    tp = profiles[:, 0]
    lon0, lat0 = profiles[0, 2], profiles[0, 1]
    xp, yp = local_xy(profiles[:, 2], profiles[:, 1], lon0, lat0)
    t0 = np.floor(tp.min()*24)/24
    edges = np.arange(t0, tp.max() + window_h/24, window_h/24)
    xg = np.arange(xp.min() - 80, xp.max() + 80, grid); yg = np.arange(yp.min() - 80, yp.max() + 80, grid)
    XG, YG = np.meshgrid(xg, yg)
    LOG, LAG = lonlat_from_xy(XG, YG, lon0, lat0)
    water = np.ones(XG.shape, bool)
    if coast is not None:
        try:
            import shapely
            water = ~shapely.contains_xy(coast.geometry.union_all(), LOG, LAG)
        except Exception:
            pass
    fields = []
    for k in range(len(edges) - 1):
        ii = (tp >= edges[k]) & (tp < edges[k + 1])
        if ii.sum() < 10:
            fields.append(np.full(XG.shape, np.nan)); continue
        cells = pd.DataFrame({'x': np.round(xp[ii]/cell)*cell, 'y': np.round(yp[ii]/cell)*cell,
                              'T': profiles[ii, 10]}).groupby(['x', 'y']).mean().reset_index()
        Tg = griddata(cells[['x', 'y']].values, cells['T'].values, (XG, YG), method='linear')
        dmin = cKDTree(cells[['x', 'y']].values).query(np.column_stack([XG.ravel(), YG.ravel()]))[0].reshape(XG.shape)
        Tg[(dmin > max_dist) | ~water] = np.nan
        fields.append(Tg)
    vmin, vmax = np.nanmin(fields), np.nanmax(fields)
    levels = np.linspace(vmin, vmax, 25)
    n = len(fields)
    ncol = 4 if n + 1 > 6 else 3
    nrow = int(np.ceil((n + 1)/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2*ncol, 6*nrow), layout='constrained')
    axs = np.atleast_1d(axs).ravel()
    cf = None
    for k, (ax, Tg) in enumerate(zip(axs, fields)):
        map_axes(ax, extent, coast)
        ax.plot(profiles[:, 2], profiles[:, 1], '.', ms=0.4, color='0.8', rasterized=True, zorder=1)
        if np.isfinite(Tg).any():
            cf = ax.contourf(LOG, LAG, Tg, levels, cmap=cmo.thermal, zorder=2)
            cs = ax.contour(LOG, LAG, Tg, levels[::4], colors='0.15', linewidths=0.5, zorder=3)
            ax.clabel(cs, fmt='%.1f', fontsize=8, inline=True)
        ax.set_title(f'{to_datetime(edges[k], time_base):%d-%b %H:%M} – {to_datetime(min(edges[k+1], tp.max()), time_base):%H:%M}')
        label(ax, f'({LETTERS[k]})')
        if k % ncol: ax.set_ylabel('')
        if k < n - ncol: ax.set_xlabel('')
    colorbar(cf, axs[:n].tolist(), 'Surface temperature (°C)', shrink=0.6, pad=0.01)
    ax = axs[n]
    ax.plot(to_datetime(tp, time_base), profiles[:, 10], '.', ms=1.5, color='0.6', rasterized=True, label='profiles')
    for k in range(len(edges) - 1):
        ii = (tp >= edges[k]) & (tp < edges[k + 1])
        if ii.sum() == 0:
            continue
        ax.hlines(profiles[ii, 10].mean(), to_datetime(edges[k], time_base), to_datetime(min(edges[k + 1], tp.max()), time_base),
                  color=cmo.thermal(0.6), lw=3, label=f'{window_h}-h mean' if k == 0 else None)
        if k % 2:
            ax.axvspan(to_datetime(edges[k], time_base), to_datetime(min(edges[k + 1], tp.max()), time_base), color='0.93', zorder=0)
    ax.set_ylabel('Temperature (°C)'); ax.set_ylim(vmin - 0.6, vmax + 0.6); clean(ax); time_axis(ax)
    ax.legend(loc='lower left', markerscale=5); label(ax, f'({LETTERS[n]})')
    if reps is not None and gold is not None:
        eta, teta = [], []
        for r in reps.itertuples():
            if r.transect not in gold:
                continue
            g = gold[r.transect]
            ii = (tp >= r.t_start) & (tp <= r.t_end)
            x, y = local_xy(profiles[ii, 2], profiles[ii, 1], g['lon0'], g['lat0'])
            d = (x - g['xa'])*g['ex'] + (y - g['ya'])*g['ey'] if g['mode'] == 'axis' else np.hypot(x - g['xa'], y - g['ya'])
            eta.append(np.nanmean(profiles[ii, 6] - np.interp(d, g['dii'], g['bottom']))); teta.append(r.t_start)
        ax2 = ax.twinx()
        ax2.plot(to_datetime(np.array(teta), time_base), eta, '.', ms=3, color=cmo.deep(0.75), label='tide')
        ax2.set_ylabel('Tide (m)', color=cmo.deep(0.75)); ax2.tick_params(axis='y', colors=cmo.deep(0.75))
        ax2.spines['top'].set_visible(False); ax2.set_ylim(-1.3, 0.9); ax2.legend(loc='lower right')
    for ax in axs[n + 1:]:
        ax.axis('off')
    return _save(fig, save)
