"""K33-HG per-frame spoon error plot (c3 with spin vs final no-spin). usage: hgplot.py OUT.png"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = 'cw/out/seg_'; rows = [('HGA', 'c3', 'c9'), ('HGB', 'c3', 'c8')]
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
for a, (v, o, n) in zip(ax, rows):
    d0 = np.load(f'{B}{v}{o}.npz', allow_pickle=False)[f'{o}_sp_deg']; d1 = np.load(f'{B}{v}{n}.npz', allow_pickle=False)[f'{n}_sp_deg']
    a.plot(d0, color='0.65', lw=1.2, label=f'{v}_{o} (spin): mean {d0.mean():.1f}, peak {d0.max():.0f}'); a.plot(d1, color='tab:blue', lw=1.6, label=f'{v}_{n} (no spin): mean {d1.mean():.1f}, peak {d1.max():.0f}')
    a.axhline(30, color='tab:red', ls='--', lw=1); a.axvspan(155, 205, color='tab:orange', alpha=0.1); a.set_ylabel('spoon error vs own target (deg)'); a.set_title(f'recon_{v}_ep33fix', loc='left'); a.legend(loc='upper right', frameon=False); a.set_ylim(0, 185)
ax[-1].set_xlabel('frame (20 fps); red = 30 deg limit, shaded = f155-205'); fig.savefig(sys.argv[1], dpi=130)
