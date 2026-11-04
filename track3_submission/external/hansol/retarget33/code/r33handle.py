"""R33: spoon handle and bowl cross-section (thickness x width, cm) along the long axis, new CD-O mesh vs G18 scene mesh."""
import numpy as np, trimesh
from pathlib import Path
M = {'new CD-O': '/data1/hansol/meshes_cdo_20261104/wooden_spoon.ply',
     'G18 scene': '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033/wooden_spoon.obj'}
for lab, p in M.items():
    m = trimesh.load(p, force='mesh', process=False); P = trimesh.sample.sample_surface(m, 40000, seed=0)[0]
    c = P.mean(0); w, U = np.linalg.eigh(np.cov((P - c).T)); q = (P - c) @ U; s = q[:, 2]; lo, hi = s.min(), s.max(); L = hi - lo
    wid = lambda k: np.ptp(q[k, 1]) if k.any() else 0
    bowl_hi = wid(s > hi - 0.3 * L) > wid(s < lo + 0.3 * L)
    print(f'== {lab}: length {100*L:.1f} cm (bowl at the {"+" if bowl_hi else "-"} end)')
    for frac in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95):
        t = (lo + frac * L) if bowl_hi else (hi - frac * L); k = np.abs(s - t) < 0.01
        print(f'   {100*frac:3.0f}% from handle tip: thickness {100*np.ptp(q[k,0]):4.2f} cm, width {100*np.ptp(q[k,1]):4.2f} cm')
