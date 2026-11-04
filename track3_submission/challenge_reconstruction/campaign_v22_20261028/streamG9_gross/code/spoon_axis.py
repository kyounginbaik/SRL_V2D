"""Decompose ours-vs-compose spoon orientation differences into roll (about the spoon long axis) vs tilt of the long axis."""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
G = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross'
for tg, td in [('c21', '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target'), ('c27', '/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27')]:
    m = trimesh.load(f'{td}/episode_000003/wooden_spoon_mesh_metric.obj', force='mesh'); v = m.vertices - m.vertices.mean(0)
    w, U = np.linalg.eigh(v.T @ v); ax = U[:, -1]
    for ep in [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]:
        z = np.load(f'{G}/out/pf/v6_{tg}/ep{ep}.npz'); i = list(z['names']).index('wooden_spoon')
        qa, qr = z['ach'][:, i, 3:], z['ref'][:, i, 3:]
        D = R.from_quat(qr).inv() * R.from_quat(qa)       # ref-body -> ach-body difference (in ref body frame)
        tilt = np.degrees(np.arccos(np.clip(np.abs(D.apply(ax) @ ax), 0, 1)))   # long-axis direction change (sign-free)
        axdir = np.degrees(np.arccos(np.clip(D.apply(ax) @ ax, -1, 1)))         # signed: >90 = handle/bowl swapped
        tot = np.degrees(D.magnitude())
        n = len(tot); seg = []
        bad = tot > 90; k = 0
        while k < n:
            if bad[k]:
                j = k
                while j < n and bad[j]: j += 1
                seg.append(f'{k}-{j-1}'); k = j
            else: k += 1
        print(f'{tg} ep{ep:2d} n={n} ori mean {tot.mean():5.1f} | >90: {bad.mean():.2f} | axis-swap(>90) {np.mean(axdir>90):.2f} | tilt mean {tilt.mean():4.1f} | roll-type >90 frames {np.mean(bad & (tilt<30)):.2f} | pos {z["pos"][:, i].mean():.1f} | segs {seg[:8]}')
