"""GT-free: is frame 0 of each object an outlier against its own early static window (world frame)?
usage: f0static.py BUNDLE EP [N=24]"""
import sys, numpy as np
from scipy.spatial.transform import Rotation as R
B, ep = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
d = np.load(f'{B}/episode_{ep:06d}/object_poses.npz'); W = d['world_from_object']; names = [str(n) for n in d['names']]
for s, nm in enumerate(names):
    p = W[:N, s, :3, 3]; r = R.from_matrix(W[:N, s, :3, :3])
    step = np.linalg.norm(np.diff(p, axis=0), axis=1) * 100
    static = int(np.argmax(np.r_[step, 99] > 0.5)) + 1  # frames before the first > 5 mm step
    k = max(static, 3)
    mean = r[1:k].mean()
    ang = (r * mean.inv()).magnitude() * 57.2958
    dp = np.linalg.norm(p - p[1:k].mean(0), axis=1) * 100
    print(f'ep{ep} {nm}: static f0..f{k - 1}; f0 vs mean(f1..f{k - 1}) {ang[0]:.2f} deg / {dp[0]:.2f} cm; '
          f'f1..f{k - 1} spread p90 {np.percentile(ang[1:k], 90):.2f} deg / {np.percentile(dp[1:k], 90):.2f} cm; obs f0 {bool(d["object_observed"][0, s])}')
    print('   rot from f0 (deg) f1..f12', np.round((r[1:13] * r[0].inv()).magnitude() * 57.2958, 2))
