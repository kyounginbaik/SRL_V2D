"""Jitter / grip diagnosis of physics rollouts (GT-free): per frame, spoon angular speed and its change (robot vs target), spoon-in-right-palm
relative rotation change (slip / rattle), palm angular speed. usage: r33jitter.py"""
import json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
c9 = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']
m = mujoco.MjModel.from_xml_path(c9['scene']); d = mujoco.MjData(m)
names = [m.body(i).name for i in range(m.nbody)]
cand = [n for n in names if 'right' in n.lower() and ('palm' in n.lower() or 'hand' in n.lower() or 'wrist' in n.lower())]
palm = next((n for n in cand if 'palm' in n.lower()), cand[0] if cand else None); print('right palm body:', palm, '| candidates', cand[:6])
sb = m.body('wooden_spoon').id; pb = m.body(palm).id
def track(Q):
    Rs, Rp, ps = [], [], []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        Rs.append(d.xmat[sb].reshape(3, 3).copy()); Rp.append(d.xmat[pb].reshape(3, 3).copy()); ps.append(d.xpos[sb].copy())
    return R.from_matrix(np.array(Rs)), R.from_matrix(np.array(Rp)), np.array(ps)
ang = lambda Ra: np.degrees((Ra[1:] * Ra[:-1].inv()).magnitude())            # deg / frame
def stats(lab, Q, Tt):
    Rs, Rp, ps = track(Q); w = ang(Rs); rel = Rp.inv() * Rs; wr = ang(rel); wp = ang(Rp)
    Rt = R.from_matrix(Tt[:len(Q), :3, :3]); wt = ang(Rt)
    acc = np.abs(np.diff(w)); acct = np.abs(np.diff(wt))
    print(f'\n== {lab}: spoon deg/frame robot mean {w.mean():.1f} (p95 {np.percentile(w,95):.1f}, max {w.max():.0f}) | target mean {wt.mean():.1f} (p95 {np.percentile(wt,95):.1f}, max {wt.max():.0f})')
    print(f'   change of speed |d omega| robot {acc.mean():.2f} vs target {acct.mean():.2f} deg/frame^2 | spoon-in-palm rotation change mean {wr.mean():.1f}, frames >5 deg {int((wr>5).sum())}, >15 deg {int((wr>15).sum())} | palm speed mean {wp.mean():.1f}')
    print('   per 40 frames  [spoon robot | target | in-palm | palm] deg/frame:')
    for a in range(0, len(w), 40):
        b = min(a + 40, len(w)); print(f'     f{a:3d}-{b:3d}: {w[a:b].mean():5.1f} | {wt[a:b].mean():5.1f} | {wr[a:b].mean():5.1f} | {wp[a:b].mean():5.1f}')
q2 = np.load('/data1/hansol/retarget33/runs/eval_R2/eval.npz')['Q']; t7 = np.load('/data1/hansol/retarget33/targets/R33_HGA7b/episode_000033/wooden_spoon_trajectory.npz')['T_world_obj']
qc = np.load(c9['trajectory'])['qpos']; tc = np.load(K / 'targets/K33_cwHGAfix/episode_000033/wooden_spoon_trajectory.npz')['T_world_obj']
print('note: target speeds are in the target world frame; a constant frame difference does not change frame-to-frame speeds')
stats('R2 (ours) vs 7b target', q2, t7); stats('HGA_c9 (team) vs its 11/02 target', qc, tc)
