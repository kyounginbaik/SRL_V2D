"""R33: hybrid residual table = BASE table with frames [t0, t1) taken from DONOR (linear cross-fade of RAMP s at both ends).
usage: r33hybrid.py BASE.npz DONOR.npz t0 t1 ramp OUT.npz"""
import sys
import numpy as np
b, dn = np.load(sys.argv[1])['table'], np.load(sys.argv[2])['table']; t0, t1, rp = map(float, sys.argv[3:6]); fps = 20.0
t = np.arange(len(b)) / fps; w = np.clip(np.minimum((t - t0) / rp + 1, (t1 - t) / rp + 1), 0, 1) if rp > 0 else ((t >= t0) & (t < t1)).astype(float)
out = b + w[:, None] * (dn[:len(b)] - b); np.savez_compressed(sys.argv[6], table=out); print(sys.argv[6], 'max |change|', float(np.abs(out - b).max()), 'frames changed', int((w > 0).sum()))
