"""Score executed trajectory npz files (object_positions, object_quaternions_wxyz) vs compose, per-window. usage: score_traj.py NPZ..."""
import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code'); import perframe as P
from scipy.spatial.transform import Rotation as R
for f in sys.argv[1:]:
    z = np.load(f); names = [str(x) for x in z['object_names']] if 'object_names' in z else ['wooden_spoon']; i = names.index('wooden_spoon')
    p = z['object_positions'][:407, i]; q = z['object_quaternions_wxyz'][:407, i][:, [1, 2, 3, 0]]
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, 3] = p; T[:, :3, :3] = R.from_quat(q).as_matrix()
    pe, oe = P.errors(T); print(f'{pe.mean():.2f} cm {oe.mean():.1f} deg  n={len(p)} | ' + ' '.join(f'{oe[a:a+40].mean():.0f}' for a in range(0, 407, 40)) + f'  {f[-110:]}')
