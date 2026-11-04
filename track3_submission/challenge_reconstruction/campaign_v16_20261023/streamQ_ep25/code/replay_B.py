"""Replay a teacher's commands in a frame-B scene (board-1 body frame half-turned about its normal) and compare with the teacher's
own execution (board-1 poses mapped back by Ry180). usage: replay_B.py TEACHER_DIR SCENE_B OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
import mujoco
t, scene, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=False)
z = {k: v for k, v in np.load(t / 'reference.npz', allow_pickle=False).items()}; m = mujoco.MjModel.from_xml_path(scene); adr = int(m.joint('wooden_piece_1_free').qposadr[0])
Q = np.asarray(z['qpos'], float).copy(); qRy = R.from_matrix(np.diag([-1., 1., -1.]))
Q[:, adr + 3:adr + 7] = (R.from_quat(Q[:, adr + 3:adr + 7], scalar_first=True) * qRy).as_quat(scalar_first=True); z['qpos'] = Q; np.savez(out / 'reference.npz', **z)
ex = execute_reference(scene, str(out / 'reference.npz'), out / 'execution'); a = np.load(ex['trajectory']); b = np.load(json.loads((t / 'result.json').read_text())['execution']['trajectory'])
qa = R.from_quat(a['object_quaternions_wxyz'][:, 0], scalar_first=True) * qRy.inv(); qb = R.from_quat(b['object_quaternions_wxyz'][:, 0], scalar_first=True)
dp = np.linalg.norm(a['object_positions'] - b['object_positions'], axis=-1); dr1 = np.degrees((qa * qb.inv()).magnitude())
for k in [1, 5, 20, 40, 60, 100, 200, 300, 355]: print(k, 'b1 %.4f cm %.3f deg | b2 %.4f cm' % (dp[k, 0] * 100, dr1[k], dp[k, 1] * 100))
