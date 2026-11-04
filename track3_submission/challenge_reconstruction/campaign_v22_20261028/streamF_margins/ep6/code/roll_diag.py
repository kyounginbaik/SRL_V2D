"""roll diagnosis: spoon orientation of new reference vs old (v21) reference and vs the phase-6 compose world; rotation axis in spoon body frame.
usage: roll_diag.py EP OLD_EPD [OTHER_EPD]"""
import sys, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
ep = int(sys.argv[1]); new = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep{ep}/world/episode_compose/episode_{ep:06d}')
def load(d):
    t = np.load(Path(d) / 'reference_trajectory.npz'); return t['object_positions'], t['object_quaternions_wxyz']
pn, qn = load(new); po, qo = load(sys.argv[2])
v = np.asarray(trimesh.load(f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_{ep:06d}/assets/wooden_spoon.obj', force='mesh', process=False).vertices)
c = v.mean(0); u, s, vt = np.linalg.svd(v - c); ax = vt[0]; print('spoon long axis (body)', ax.round(3), 'extent', (np.ptp((v - c) @ vt.T, 0) * 100).round(1))
i = 1
Rn = R.from_quat(qn[:, i][:, [1, 2, 3, 0]]); Ro = R.from_quat(qo[:, i][:, [1, 2, 3, 0]])
rel = Ro.inv() * Rn; rv = rel.as_rotvec(); ang = np.degrees(np.linalg.norm(rv, axis=1))
for f in range(0, len(ang), 10):
    a = rv[f] / max(1e-9, np.linalg.norm(rv[f])); print(f, 'ang %.0f' % ang[f], 'axis.long %.2f' % abs(a @ ax), 'z_new %.3f z_old %.3f' % (pn[f, i, 2], po[f, i, 2]),
        'roll_new_vs_f0 %.0f' % np.degrees((Rn[0].inv() * Rn[f]).magnitude()), 'roll_old_vs_f0 %.0f' % np.degrees((Ro[0].inv() * Ro[f]).magnitude()))
