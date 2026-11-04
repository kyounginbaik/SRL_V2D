"""hypothetical spoon-position floors (ep5) from a saved (ach, ref) pair in scorer-aligned frames."""
import sys, numpy as np
from scipy.ndimage import gaussian_filter1d as gf
z = np.load(sys.argv[1]); a = z['ach'][:, 1, :3] * 100; r = z['ref'][:, 1, :3] * 100
M = lambda x: float(np.linalg.norm(x - r, axis=1).mean())
sh = lambda x, l: np.r_[x[l:], np.repeat(x[-1:], l, 0)]
e = a - r
print('actual', round(M(a), 3))
for s in (1, 2, 3, 4): print(f'S1 follow G_{s}(target) (no other error)', round(M(gf(r, s, axis=0, mode="nearest")), 3))
for s in (4, 6, 10): print(f'S2 actual minus slow error G_{s}(e)', round(M(a - gf(e, s, axis=0, mode="nearest")), 3))
a1 = sh(a, 1); e1 = a1 - r; print('S3 lag removed', round(M(a1), 3), '+ slow error G_4 removed', round(M(a1 - gf(e1, 4, axis=0, mode='nearest')), 3))
for g in (0.4, 0.5, 0.6, 0.75, 1.0):
    for s in (4, 8): print(f'S4 perfect slow (G_{s}) + stirring gain {g} zero lag', round(M(gf(r, s, axis=0, mode="nearest") + g * (r - gf(r, s, axis=0, mode="nearest"))), 3))
