"""Does the ink read as the human's? Not part of the pass rule, which was set earlier. lift_kept: of reference pen-up frames during writing, how often this pen is also up. ink_cm: mean distance from this pen's ink to the human's ink. stray: share of this pen's ink more than 3 mm from any human ink."""
import sys, numpy as np, wmetric as W
RF = W.RF; s, e = np.flatnonzero(RF['on'])[[0, -1]]; up = ~RF['on'][s:e + 1]; ri = RF['tip'][RF['on']]
for name in sys.argv[1:]:
    d = np.load(name); Q = d['q'] if hasattr(d, 'files') else d; pen = Q[:RF['n'], -7:]; t = W.tip_of(pen); h = (t - W.S) @ W.N; ob = np.abs(h - W.H0) < .005; ink = t[ob]
    if not len(ink): print(name, 'no ink'); continue
    D = np.linalg.norm(ink[:, None] - ri[None], axis=2).min(1); print(name, 'pen-up frames in writing', int(up.sum()), 'lift_kept', round(float((~ob[s:e + 1])[up].mean()), 3), 'ink_cm', round(float(D.mean() * 100), 2), 'stray', round(float((D > .003).mean()), 3), 'total ink cm', round(float(np.linalg.norm(np.diff(t, axis=0), axis=1)[ob[:-1] & ob[1:]].sum() * 100), 1))
