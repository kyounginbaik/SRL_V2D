"""usage: check_ref.py EP EPISODE_DIR MESH_DIR : scorer-aligned error of the episode dir's object reference (tracking_reference qpos) vs target_final_v5 (validation of the world transform)"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import blib as B
ep, epd, md = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
q = np.load(epd / 'tracking_reference.npz')['qpos']
P = {}
for n, c in (('blue_cup', 56), ('wooden_spoon', 63)):
    T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, :3] = R.from_quat(q[:, c + 3:c + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = q[:, c:c + 3]; P[n] = T
reg = B.register(ep, {n: md / f'{n}.obj' for n in B.NAMES}); e = B.per_frame(ep, P, reg)
print({n: (round(float(e[n][0].mean()), 3), round(float(e[n][1].mean()), 2), 'max', round(float(e[n][0].max()), 3), round(float(e[n][1].max()), 2)) for n in B.NAMES})
