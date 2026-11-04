"""P8: per-window spoon error decomposition (body-frame rotvec of achieved^-1 target, frame-0 aligned; x = long axis) for an evaluated rollout dir.
usage: roll_diag.py EXEC_or_EVAL_DIR [ep]"""
import sys, os, json, numpy as np, mujoco
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/code'); import tlibK as T
from scipy.spatial.transform import Rotation as R
p = sys.argv[1]; ep = int(sys.argv[2]) if len(sys.argv) > 2 else 35
if os.path.exists(p + '/execution'): p += '/execution'
m = mujoco.MjModel.from_xml_path(p + '/scene_physics.xml'); s = int(m.joint('wooden_spoon_free').qposadr[0]); q = np.load(p + '/trajectory.npz')['qpos']
Ta = np.tile(np.eye(4), (len(q), 1, 1)); Ta[:, :3, :3] = R.from_quat(q[:, s + 3:s + 7], scalar_first=True).as_matrix(); Ta[:, :3, 3] = q[:, s:s + 3]
pe, re = T.errors(ep, Ta); X = T.registration(ep)[0]; Tb = T.target(ep)[:len(q)] @ X; A = Tb[0] @ np.linalg.inv(Ta[0]); Tal = A @ Ta
rv = np.degrees((R.from_matrix(Tal[:, :3, :3]).inv() * R.from_matrix(Tb[:, :3, :3])).as_rotvec())
# target roll about own x relative to frame 0, achieved same
def roll(Tm): return np.degrees((R.from_matrix(Tm[0, :3, :3]).inv() * R.from_matrix(Tm[:, :3, :3])).as_rotvec())
print('mean %.3f %.2f' % (pe.mean(), re.mean()))
for i in range(0, len(q), 10):
    sl = slice(i, i + 10); print(f'f{i:3d} {i/20:5.1f}s pos {pe[sl].mean():5.2f} rot {re[sl].mean():5.1f}  body-rv {rv[sl].mean(0).round(0)}  tgt-rel {roll(Tb)[sl].mean(0).round(0)} ach-rel {roll(Tal)[sl].mean(0).round(0)}')
