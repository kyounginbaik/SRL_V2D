"""R33: stronger-grip tables. Adds a constant squeeze offset (finger flex + thumb channels) to a residual table while each object is
held: left hand (cup) f30-f320, right hand (spoon) f55-f300, smoothstep ramps of 0.3 s. usage: r33squeeze.py BASE.npz OUT.npz dL dR"""
import sys
import numpy as np
b = np.load(sys.argv[1])['table'].copy(); dL, dR = float(sys.argv[3]), float(sys.argv[4]); T = len(b); f = np.arange(T)
def win(a, e, r=6):
    w = np.clip(np.minimum((f - a) / r, (e - f) / r), 0, 1); return w * w * (3 - 2 * w)
b[:, 14:16] += dL * win(30, 320)[:, None]; b[:, 6:8] += dR * win(55, 300)[:, None]
np.savez_compressed(sys.argv[2], table=b); print(sys.argv[2], 'left', dL, 'right', dR)
