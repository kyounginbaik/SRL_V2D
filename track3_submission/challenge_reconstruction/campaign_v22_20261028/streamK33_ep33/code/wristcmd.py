"""K33-HG: right wrist world rotation and commanded rotation joints around the spin; residual-table size. usage: wristcmd.py VAR RUN TABLE"""
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
V, N, TB = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/'
def load(tag):
    C = Path(BASE + tag + '/episode_000033/backend/evaluation/execution'); z = np.load(C / 'trajectory.npz', allow_pickle=False); m = mujoco.MjModel.from_xml_path(str(C / 'scene_physics.xml')); d = mujoco.MjData(m)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot_right_right_hand_C_MC'); Rp = []
    for f in range(len(z['qpos'])):
        d.qpos[:] = z['qpos'][f]; mujoco.mj_kinematics(m, d); Rp.append(d.xmat[b].reshape(3, 3).copy())
    return z, R.from_matrix(np.array(Rp))
z, Rp = load(f'K33_K33_cw{V}fix_{V}_{N}'); z0, Rp0 = load('H13_ep33_b1')
an = [str(x) for x in z['actuator_names']]; ri = [i for i, a in enumerate(an) if 'right' in a and 'rot_' in a]; print([an[i] for i in ri])
jn = [str(x) for x in z['joint_names']]; rj = [i for i, a in enumerate(jn) if 'right_right_rot' in a]
t = np.load(TB, allow_pickle=False)['table']
ks = jn.index(next(x for x in jn if 'spoon' in x)); Rs = R.from_quat(z['qpos'][:, ks + 3:ks + 7][:, [1, 2, 3, 0]])
print('f  palmrate  palm_vs_f155  spoonworld_vs_f155 | ctrl rot deg | qpos rot deg | v29 palm_vs_f155 | table right rot deg')
for f in list(range(150, 186, 2)):
    print(f, round(np.degrees((Rp[f].inv() * Rp[f + 1]).magnitude())), round(np.degrees((Rp[155].inv() * Rp[f]).magnitude())), round(np.degrees((Rs[155].inv() * Rs[f]).magnitude())), np.degrees(z['ctrl'][f, ri]).round().tolist(), np.degrees(z['qpos'][f, rj]).round().tolist(), round(np.degrees((Rp0[155].inv() * Rp0[f]).magnitude())), np.degrees(t[f, 3:6]).round().tolist())
print('max|table| pos cm', (np.abs(t[:, :3]).max(0) * 100).round(1), 'rot deg', np.degrees(np.abs(t[:, 3:6]).max(0)).round(0), 'sq', np.abs(t[:, 6:8]).max(0).round(2))
print('spoonrate v29', [round(float(x)) for x in np.degrees((R.from_quat(z0['qpos'][:, ks + 3:ks + 7][:, [1, 2, 3, 0]])[150:185].inv() * R.from_quat(z0['qpos'][:, ks + 3:ks + 7][:, [1, 2, 3, 0]])[151:186]).magnitude())])
