"""
Run the whole pipeline (bronze -> silver -> gold -> platinum) from a YAML configuration file.

    python scripts/run_pipeline.py examples/config_closed_circuit.yml

See examples/*.yml for the options. Outputs (matrices as .npz, the repetition table as .csv and
the figures) go to the `output` folder named in the config.
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import towed_adcp as ta                                             # noqa: E402
from towed_adcp import plotting as pl                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Towed ADCP processing: bronze, silver, gold and platinum matrices')
    ap.add_argument('config', help='YAML configuration file')
    ap.add_argument('--no-figures', action='store_true', help='skip the figures')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(args.config))
    path = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    out = path(cfg.get('output', 'output'))
    os.makedirs(out, exist_ok=True)
    figs = os.path.join(out, 'figs'); os.makedirs(figs, exist_ok=True)
    draw = cfg.get('figures', True) and not args.no_figures

    # ---- bronze
    cb = cfg.get('bronze', {})
    files = cfg['files']
    files = [path(f) for f in files] if isinstance(files, list) else sorted(glob.glob(path(files)))
    bronze, meta = ta.build_bronze(files, err_max=cb.get('err_max', 10), q_max=cb.get('q_max', 100),
                                   vbt_min=cb.get('vbt_min', 10), pg_min=cb.get('pg_min'))
    time_base = meta['time_base']
    ta.save_matrix(os.path.join(out, 'bronze.npz'), bronze, time_base=time_base)
    profiles = ta.profiles_from_bronze(bronze)

    # ---- repetitions
    cr = cfg['repetitions']
    vertices = {k: tuple(v) for k, v in cr.get('vertices', {}).items()}
    transects = cr['transects']
    if cr.get('method', 'vertices') == 'vertices':
        reps = ta.repetitions_by_vertices(profiles, vertices, transects, r_detect=cr.get('r_detect', 120),
                                          min_sep=cr.get('min_sep', 120), r_max=cr.get('r_max', 100),
                                          dur_max=cr.get('dur_max_min', 20), gap_max=cr.get('gap_max_s', 60))
    else:
        reps = ta.repetitions_from_file(path(cr['file']), transects=transects, transect=cr.get('transect'))
    geometry = ta.transect_geometry(transects, vertices or None)

    # ---- silver
    silver, reps = ta.build_silver(bronze, reps)
    ta.save_matrix(os.path.join(out, 'silver.npz'), silver, time_base=time_base)
    reps.to_csv(os.path.join(out, 'repetitions.csv'), index=False)

    # ---- gold
    cg = cfg.get('gold', {})
    gold = ta.build_gold(silver, reps, geometry, dy=cg.get('dy', 10), dz=cg.get('dz', 0.25),
                         z_min=cg.get('z_min', 0.5), z_max=cg.get('z_max'), L=cg.get('L'))
    ta.save_gold(os.path.join(out, 'gold.npz'), gold)

    # ---- platinum
    cp = cfg.get('platinum', {})
    periods = cp.get('periods_h', {'D1': 23.93, 'D2': 12.42, 'D4': 6.21})
    platinum = ta.build_platinum(gold, periods, n_min=cp.get('n_min'))
    ta.save_platinum(os.path.join(out, 'platinum.npz'), platinum)

    # ---- rotation to the channel axis (optional)
    crot = cfg.get('rotate')
    gold_rot = plat_rot = None
    if crot:
        if crot.get('method', 'principal_axis') == 'principal_axis':
            theta, frac = ta.principal_axis(silver)
            print(f'principal axis of the current: heading {(90 - np.rad2deg(theta)) % 360:.1f} deg '
                  f'({100*frac:.0f} % of the variance)')
        else:
            theta = np.deg2rad(90 - crot['heading_deg'])          # compass heading -> angle from east
        gold_rot = ta.rotate_gold(gold, theta)
        plat_rot = ta.build_platinum(gold_rot, periods, n_min=cp.get('n_min'))
        ta.save_platinum(os.path.join(out, 'platinum_channel.npz'), plat_rot)
        with open(os.path.join(out, 'channel_axis.txt'), 'w') as f:
            f.write(f'theta_rad {theta}\nheading_deg {(90 - np.rad2deg(theta)) % 360}\n')

    # ---- figures
    if draw:
        pl.set_style()
        coast = pl.load_coast(path(cfg['coast'])) if cfg.get('coast') else None
        pl.plot_trajectory(profiles, time_base, coast, save=os.path.join(figs, '01_track.png'))
        pl.plot_bronze_series(bronze, time_base, save=os.path.join(figs, '02_bronze_series.png'))
        pl.plot_bronze_profiles(bronze, save=os.path.join(figs, '03_bronze_profiles.png'))
        pl.plot_repetitions(profiles, reps, time_base, vertices or None, coast, save=os.path.join(figs, '04_repetitions.png'))
        pl.plot_alpha_beta(reps, time_base, save=os.path.join(figs, '05_joyce_alpha_beta.png'))
        t_ref = reps.groupby('transect').alpha.median().abs().idxmax()
        pl.plot_correction(bronze, reps, t_ref, time_base, save=os.path.join(figs, f'06_correction_{t_ref}.png'))
        for name, g in gold.items():
            pl.plot_gold_panel(g, name, time_base, save=os.path.join(figs, f'07_gold_{name}_v.png'))
        name0 = next(iter(gold))
        pl.plot_node_fit(gold[name0], name0, periods, time_base, save=os.path.join(figs, '09_node_fit.png'))
        for name in gold:
            pl.plot_platinum_panel(gold[name], platinum[name], name, time_base, save=os.path.join(figs, f'10_platinum_{name}.png'))
        if all(g['mode'] == 'axis' for g in gold.values()):
            pl.plot_residual_map(gold, platinum, profiles, coast, save=os.path.join(figs, '11_residual_map.png'))
        if plat_rot is not None:
            pl.plot_residual_sections(gold_rot, plat_rot, 'u', 'Along-channel residual (cm/s)', save=os.path.join(figs, '12_residual_along.png'))
            pl.plot_residual_sections(gold_rot, plat_rot, 'v', 'Cross-channel residual (cm/s)', zero_line=False, save=os.path.join(figs, '13_residual_cross.png'))
        pl.plot_surface_temperature_maps(profiles, time_base, reps, gold, coast=coast, save=os.path.join(figs, '14_surface_temperature.png'))
        import matplotlib.pyplot as plt
        plt.close('all')
    print(f'\ndone: outputs in {out}')


if __name__ == '__main__':
    main()
