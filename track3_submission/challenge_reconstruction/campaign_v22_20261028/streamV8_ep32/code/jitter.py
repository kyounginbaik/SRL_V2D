"""V8: floor from target jitter: error of a gaussian-smoothed target spoon vs the raw target (per 0.5 s window and windows 0-4.5/4.5-9/9-12.9)."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d as gf
T = np.load('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5/episode_000032/wooden_spoon_trajectory.npz')['T_world_obj']
N = len(T); p = T[:, :3, 3]; q = R.from_matrix(T[:, :3, :3]).as_quat()
for k in range(1, N):
    if np.dot(q[k], q[k - 1]) < 0: q[k] = -q[k]
print('frames', N)
for s in (1, 2, 3, 5, 8):
    ps = gf(p, s, axis=0, mode='nearest'); qs = gf(q, s, axis=0, mode='nearest'); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    e = np.linalg.norm(ps - p, axis=1) * 100; dg = np.degrees((R.from_quat(qs).inv() * R.from_quat(q)).magnitude())
    w = lambda a, b: f'{e[int(a*20):int(b*20)].mean():.2f}/{dg[int(a*20):int(b*20)].mean():.1f}'
    print(f'sigma {s}: all {e.mean():.2f}/{dg.mean():.1f}  0-4.5 {w(0,4.5)}  4.5-9 {w(4.5,9)}  6.5-8 {w(6.5,8)}  9-12.9 {w(9,12.9)}')
