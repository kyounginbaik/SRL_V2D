"""ep36 tgt_s5 = FINAL target (target_final_v14_g17) with only the 15.0 s jump removed: f297-308 replaced by a cubic position spline over
f287-296 / f309-318 and a rotation slerp f296 -> f309. Everything else is FINAL. Then the canonical-camera cross-check (s4check) incl. s5."""
import shutil, numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
E = 'episode_000036'; SRC = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17') / E
T = np.load(SRC / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); Tj = T.copy(); J0, J1 = 297, 308
good = list(range(J0 - 10, J0)) + list(range(J1 + 1, J1 + 11)); cs = CubicSpline(good, T[good, :3, 3], axis=0)
sl = Slerp([J0 - 1, J1 + 1], R.from_matrix(np.stack([T[J0 - 1, :3, :3], T[J1 + 1, :3, :3]])))
for t in range(J0, J1 + 1): Tj[t, :3, 3] = cs(t); Tj[t, :3, :3] = sl(t).as_matrix()
O = Path('/data1/hansol/spoon3537/targets/tgt_s5') / E
if O.exists(): shutil.rmtree(O)
shutil.copytree(SRC, O); np.savez_compressed(O / 'wooden_spoon_trajectory.npz', T_world_obj=Tj)
v = lambda X: np.linalg.norm(np.diff(X[:, :3, 3], axis=0), axis=1) * 2000
print('max speed cm/s FINAL %.0f  s5 %.0f; changed frames %d-%d' % (v(T).max(), v(Tj).max(), J0, J1))
