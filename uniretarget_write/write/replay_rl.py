"""Replay a Newton RL eval rollout in our MuJoCo scene, kinematically: finger-pen overlap (same contact rule as the planner) and a 50 Hz qpos file for render_w.py.
Usage: replay_rl.py EVAL.npz WORLD OUT.npz   (WORLD 0..3, the worlds with full body records)
Newton and MuJoCo colliders differ, so overlap here is an estimate in the planner's collider set."""
import sys, numpy as np, mujoco, wmetric as W
from scipy.spatial.transform import Rotation as R, Slerp
from planner import base_from, XML
src, w, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d0 = np.load(src); q = d0['q'][:, w].astype(float); b = d0['b'][:, w].astype(float); o = d0['o'][:, w, 0].astype(float)
jl = [str(x) for x in d0['joint_labels']]; qs = d0['joint_q_start']; bl = [str(x) for x in d0['body_labels']]
T = len(q); fps = round((T + 1) / 11.28); tp = (np.arange(T) + 1) / fps
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand_C_MC')
hidx = bl.index('right_sharpa_wave/right_hand_C_MC')
fmap = []
for j in range(6, 28):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j); k = jl.index('right_sharpa_wave/' + nm); fmap.append((m.jnt_qposadr[j], qs[k]))
pid = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]
PG = {i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '').startswith('right_object_') and 'visual' not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}; BG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'board')
# check: Newton fingertip bodies vs MuJoCo after replay
tips = [n for n in ('right_thumb_DP', 'right_index_DP', 'right_middle_DP') if 'right_sharpa_wave/' + n in bl]
t50 = np.arange(int(tp[-1] / .02)) * .02; t50 = t50[t50 >= tp[0]]
def lerp(x): return np.stack([np.interp(t50, tp, x[:, i]) for i in range(x.shape[1])], 1)
Hp = lerp(b[:, hidx, :3]); Hr = Slerp(tp, R.from_quat(b[:, hidx, 3:7]))(t50); Pp = lerp(o[:, :3]); Pr = Slerp(tp, R.from_quat(o[:, 3:7]))(t50)
F = lerp(np.stack([q[:, s] for _, s in fmap], 1))
Q, PEN, TIPERR = [], [], []; qq = np.zeros(m.nq)
for i in range(len(t50)):
    X = np.eye(4); X[:3, :3] = Hr[i].as_matrix(); X[:3, 3] = Hp[i]
    for (a, _), v in zip(fmap, F[i]): qq[a] = v
    qq[:6] = base_from(m, d, hb, X, qq.copy()); qq[pid:pid + 3] = Pp[i]; qq[pid + 3:pid + 7] = Pr[i].as_quat()[[3, 0, 1, 2]]
    d.qpos[:] = qq; mujoco.mj_forward(m, d); Q.append(qq.copy())
    ds = [d.contact[c].dist for c in range(d.ncon) if (d.contact[c].geom1 in PG) != (d.contact[c].geom2 in PG) and BG not in (d.contact[c].geom1, d.contact[c].geom2)]; PEN.append(min(ds) if ds else np.nan)
    if i % 10 == 0 and tips:
        k = int(np.argmin(np.abs(tp - t50[i])))
        TIPERR.append(max(np.linalg.norm(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] - b[k, bl.index('right_sharpa_wave/' + n), :3]) for n in tips))
PEN = np.array(PEN); pad = int(round(tp[0] / .02)); Q = np.array(Q); Q = np.r_[np.tile(Q[0], (pad, 1)), Q]
np.savez(out, q=Q)
print(dict(world=w, frames=len(Q), overlap_p5_cm=round(float(-np.nanpercentile(PEN, 5)) * 100, 2), overlap_med_cm=round(float(-np.nanmedian(PEN)) * 100, 2), contact_frames=round(float(np.isfinite(PEN).mean()), 2), fk_check_cm=round(float(np.max(TIPERR)) * 100, 2) if TIPERR else None))
