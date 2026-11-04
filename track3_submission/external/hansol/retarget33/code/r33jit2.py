import json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d as gf
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
c9 = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']
m = mujoco.MjModel.from_xml_path(c9['scene']); d = mujoco.MjData(m); sb = m.body('wooden_spoon').id; cb = m.body('blue_cup').id
def poses(Q, b):
    Rs, ps = [], []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); Rs.append(d.xmat[b].reshape(3, 3).copy()); ps.append(d.xpos[b].copy())
    return R.from_matrix(np.array(Rs)), np.array(ps)
def jit(Rr, p, s=1.5):
    q = Rr.as_quat(); q[1:] *= np.sign(np.sum(q[1:] * q[:-1], axis=1))[:, None]; q = np.cumprod(np.r_[[1.0], np.sign(np.sum(q[1:] * q[:-1], axis=1))])[:, None] * Rr.as_quat()
    qs = gf(q, s, axis=0); Rs = R.from_quat(qs / np.linalg.norm(qs, axis=1, keepdims=True))
    return np.degrees((Rs.inv() * Rr).magnitude()), 100 * np.linalg.norm(p - gf(p, s, axis=0), axis=1)
q2 = np.load('/data1/hansol/retarget33/runs/eval_R2/eval.npz'); qc = np.load(c9['trajectory'])['qpos']
t7 = np.load('/data1/hansol/retarget33/targets/R33_HGA7b/episode_000033/wooden_spoon_trajectory.npz')['T_world_obj']
for lab, Q in (('R2', q2['Q']), ('HGA_c9', qc)):
    Rr, p = poses(Q, sb); a, b = jit(Rr, p); held = slice(50, 300)
    print(f'{lab}: spoon jitter (deviation from its own 1.5-frame smooth) rot mean {a[held].mean():.2f} deg, p95 {np.percentile(a[held],95):.1f}; pos mean {b[held].mean():.3f} cm (f50-300, held)')
Rt = R.from_matrix(t7[:, :3, :3]); a, b = jit(Rt, t7[:, :3, 3]); print(f'7b target: rot {a[50:300].mean():.2f} deg, pos {b[50:300].mean():.3f} cm')
e = q2['wooden_spoon_deg']; print(f'R2 per-frame spoon error vs 7b: mean {e.mean():.1f} deg; smoothed-error proxy: mean of 1.5-frame-smoothed error {gf(e,1.5).mean():.1f} (jitter in the error itself {np.abs(e-gf(e,1.5)).mean():.1f})')
