"""K33-HG: spoon orientation relative to the right palm, sim rollout vs target. usage: inhand.py VAR RUN"""
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
V, N = sys.argv[1], sys.argv[2]
C = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/K33_K33_cw{V}fix_{V}_{N}/episode_000033/backend/evaluation/execution')
z = np.load(C / 'trajectory.npz', allow_pickle=False); m = mujoco.MjModel.from_xml_path(str(C / 'scene_physics.xml')); d = mujoco.MjData(m)
bs = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(m.nbody)]; print([b for b in bs if 'right' in b and ('palm' in b or 'hand' in b or 'wrist' in b or 'rot' in b)][:10])
pb = bs.index('robot_right_right_hand_C_MC'); print('palm body', bs[pb])
Rp = []
for f in range(len(z['qpos'])):
    d.qpos[:] = z['qpos'][f]; mujoco.mj_kinematics(m, d); Rp.append(d.xmat[pb].reshape(3, 3).copy())
Rp = R.from_matrix(np.array(Rp)); k = [str(x) for x in z['object_names']].index('wooden_spoon'); qs = z['object_quaternions_wxyz'][:, k]; Rs = R.from_quat(qs[:, [1, 2, 3, 0]]); qr = None
jn = [str(x) for x in z['joint_names']]; j = next(i for i, x in enumerate(jn) if 'spoon' in x); Rr = R.from_quat(z['reference'][:, j + 3:j + 7][:, [1, 2, 3, 0]])
sim = Rp.inv() * Rs; ref = Rp.inv() * Rr; a = 155
print('frame sim_inhand_vs_f155  ref_inhand_vs_f155  sim_vs_ref_inhand')
for f in list(range(100, 150, 10)) + list(range(150, 200, 4)) + list(range(200, 348, 20)):
    print(f, round(np.degrees((sim[a].inv() * sim[f]).magnitude())), round(np.degrees((ref[a].inv() * ref[f]).magnitude())), round(np.degrees((ref[f].inv() * sim[f]).magnitude())))
