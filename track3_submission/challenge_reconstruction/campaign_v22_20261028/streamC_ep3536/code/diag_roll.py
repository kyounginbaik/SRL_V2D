"""diagnose a plan: per-frame spoon body rotvec vs target, spoon-in-palm rotation change vs base rollout. usage: diag_roll.py BASE PARAMS_JSON [frames]"""
import os, sys, json, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cemC
from scipy.spatial.transform import Rotation as R
w = cemC.Worker(sys.argv[1]); m = w.m
def run(p):
    q = dict(cemC.plan36.DEFAULT); q.update(p); Q, _ = w.rollout(q); return Q
bid = [m.body(i).name for i in range(m.nbody)]; palm = 'robot_right_right_hand_C_MC'
print('palm body', palm)
def poses(Q):
    d = mujoco.MjData(m); out = []
    for q in Q:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        out.append((R.from_quat(d.xquat[m.body('wooden_spoon').id][[1, 2, 3, 0]]), R.from_quat(d.xquat[m.body(palm).id][[1, 2, 3, 0]])))
    return out
Q0 = run(dict(gr=0, gp=0)); Q1 = run(json.loads(sys.argv[2])); P0, P1 = poses(Q0), poses(Q1)
Rt = R.from_matrix(w.Tc[:, :3, :3])
fr = range(0, len(Q0), 4) if len(sys.argv) < 4 else [int(x) for x in sys.argv[3].split(',')]
for f in fr:
    s0, h0 = P0[f]; s1, h1 = P1[f]
    print(f, 'base-vs-tgt', np.round(np.degrees((Rt[f].inv() * s0).as_rotvec())).astype(int), 'new-vs-tgt', np.round(np.degrees((Rt[f].inv() * s1).as_rotvec())).astype(int),
          'palm new-vs-base(spoon body)', np.round(np.degrees((s0.inv() * h1 * h0.inv() * s0).as_rotvec())).astype(int),
          'in-hand change(spoon body)', np.round(np.degrees(((h1.inv() * s1).inv() * (h0.inv() * s0)).as_rotvec() * -1)).astype(int))
