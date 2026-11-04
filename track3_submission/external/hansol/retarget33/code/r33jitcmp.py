"""R33: jitter / grip comparison of ep33 rollouts (GT-free). Per run: spoon angular-velocity change |d omega| (robot vs its target),
spoon rotation change relative to the right palm while held (f55-300), jitter amplitude = deviation from the run's own 1.5-frame
Gaussian-smoothed track. usage: r33jitcmp.py"""
import json
import numpy as np
import mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d as gf

K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
T = Path('/data1/hansol/retarget33')
c9 = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']
m = mujoco.MjModel.from_xml_path(c9['scene']); d = mujoco.MjData(m)
sb = m.body('wooden_spoon').id; pb = m.body('robot_right_right_hand_C_MC').id


def track(Q):
    Rs, Rp, ps = [], [], []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        Rs.append(d.xmat[sb].reshape(3, 3).copy()); Rp.append(d.xmat[pb].reshape(3, 3).copy()); ps.append(d.xpos[sb].copy())
    return R.from_matrix(np.array(Rs)), R.from_matrix(np.array(Rp)), np.array(ps)


def omega(Ra):
    return (Ra[1:] * Ra[:-1].inv()).as_rotvec()


def smooth_dev(Ra, p):
    q = Ra.as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    qs = gf(q, 1.5, axis=0); Rs = R.from_quat(qs / np.linalg.norm(qs, axis=1, keepdims=True))
    return np.degrees((Rs.inv() * Ra).magnitude()), 100 * np.linalg.norm(p - gf(p, 1.5, axis=0), axis=1)


runs = [('R2 (7b target, uploaded)', T / 'runs/eval_R2/eval.npz', T / 'targets/R33_HGA7b'),
        ('R4A (r6 target, no smoothness)', T / 'runs/eval_R4A/eval.npz', T / 'targets/R33_HGA_r6'),
        ('R5 (r6 target, smoothness hinge)', T / 'runs/eval_R5/eval.npz', T / 'targets/R33_HGA_r6')]
H = slice(55, 300)
print(f"{'run':34s} {'spoon cm/deg':>13s} {'|dw| robot/target':>18s} {'in-palm mean':>12s} {'>5deg':>6s} {'>15deg':>7s} {'jit rot':>8s} {'jit pos':>8s}")
for lab, ev, tg in runs:
    z = np.load(ev); Q = z['Q']
    Tt = np.load(tg / 'episode_000033/wooden_spoon_trajectory.npz')['T_world_obj'][:len(Q)]
    Rs, Rp, ps = track(Q); Rt = R.from_matrix(Tt[:, :3, :3])
    dw = np.degrees(np.linalg.norm(np.diff(omega(Rs), axis=0), axis=1)); dwt = np.degrees(np.linalg.norm(np.diff(omega(Rt), axis=0), axis=1))
    rel = Rp.inv() * Rs; wr = np.degrees((rel[1:] * rel[:-1].inv()).magnitude())
    ja, jp = smooth_dev(Rs, ps)
    print(f"{lab:34s} {z['wooden_spoon_pos'].mean():5.2f}/{z['wooden_spoon_deg'].mean():5.1f}  {dw[H].mean():7.2f} / {dwt[H].mean():5.2f}   "
          f"{wr[H].mean():9.2f}   {int((wr[H] > 5).sum()):5d} {int((wr[H] > 15).sum()):6d}   {ja[H].mean():6.2f}  {jp[H].mean():6.3f}")
print('held window f55-300; |dw| = frame-to-frame change of spoon angular velocity (deg/frame^2); in-palm = spoon rotation change relative to the right palm (deg/frame);'
      ' jit = mean deviation from own 1.5-frame smooth (deg, cm). Spoon cm/deg vs each run\'s own target.')
