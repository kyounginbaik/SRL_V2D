import sys, numpy as np
from scipy.ndimage import gaussian_filter1d as gf
z = np.load(sys.argv[1]); a = z['ach'][:, 1, :3] * 100; r = z['ref'][:, 1, :3] * 100; e = a - r; s = gf(e, 10, axis=0, mode='nearest')
print('slow error G_10 per 0.5 s window (x,y,z cm) and |e| mean:')
for k in range(0, len(e), 10): print(f'{k/20:5.1f}', s[k:k+10].mean(0).round(2), round(float(np.linalg.norm(e[k:k+10], axis=1).mean()), 2))
print('mean |slow|', np.linalg.norm(s, axis=1).mean().round(3), 'mean slow comp', s.mean(0).round(3), 'rms per axis', np.sqrt((s**2).mean(0)).round(3))
