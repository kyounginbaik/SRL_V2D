"""R33: end-of-clip table fixes for M1_C (right hand only; channels 0-7). usage: r33endfix.py BASE_TABLE OUT_DIR
V1 zero: right channels ramp to 0 over 15.8-16.2 s | V2 hold: right channels held at their 15.8 s value afterwards
V3 smooth: Gaussian sigma 3 frames on right channels from 14.5 s (blend in over 0.3 s) | V4 smooth+hold: V3 then V2"""
import sys, numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
b = np.load(sys.argv[1])['table']; out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); n = len(b); t = np.arange(n) / 20.0; RC = slice(0, 8)
def zero(x):
    x = x.copy(); w = np.clip((t - 15.8) / 0.4, 0, 1)[:, None]; x[:, RC] *= (1 - w); return x
def hold(x):
    x = x.copy(); k = int(15.8 * 20); x[k:, RC] = x[k, RC]; return x
def smooth(x):
    x = x.copy(); s = gaussian_filter1d(x[:, RC], 3, axis=0, mode='nearest'); w = np.clip((t - 14.5) / 0.3, 0, 1)[:, None]; x[:, RC] = x[:, RC] * (1 - w) + s * w; return x
for name, x in (('V1_zero', zero(b)), ('V2_hold', hold(b)), ('V3_smooth', smooth(b)), ('V4_smooth_hold', hold(smooth(b)))):
    np.savez_compressed(out / f'{name}.npz', table=x); print(name, 'max change', float(np.abs(x - b).max()), 'first changed frame', int(np.argmax(np.abs(x - b).max(1) > 1e-9)))
