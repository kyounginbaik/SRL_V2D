"""U8: decompose the late-window spoon orientation error (achieved vs target in the physics world, our body frame) and test whether the
target spoon pose is feasible against the physical rack (penetration depth when the spoon is placed at its target pose, achieved rack)."""
import os, sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
ep, actor, meshes, npz = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
env = mres.Env(dict(ep=ep, actor=actor, meshes=meshes)); Q = np.load(npz)['Q']
P = env.poses(Q); e, A = mres.L.per_frame(ep, P, env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(ep, env.reg, A, env.tgt)
n = 'wooden_spoon'; T = len(Q); a = env.oa[n]
Ra = R.from_matrix(P[n][:, :3, :3]); Rt = R.from_matrix(tp[n][:T, :3, :3])
rv_body = (Rt.inv() * Ra).as_rotvec(); rv_world = (Ra * Rt.inv()).as_rotvec()
d = mujoco.MjData(env.m)
for s0 in range(0, T, 10):
    sl = slice(s0, s0 + 10)
    pens = []
    for k in range(s0, min(T, s0 + 10), 3):
        d.qpos[:] = Q[k]; pt = tp[n][k]; d.qpos[a:a + 3] = pt[:3, 3]; d.qpos[a + 3:a + 7] = R.from_matrix(pt[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(env.m, d); dp = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = env.m.geom_bodyid[c.geom1], env.m.geom_bodyid[c.geom2]; nm = {mujoco.mj_id2name(env.m, 1, b1), mujoco.mj_id2name(env.m, 1, b2)}
            if any(x and 'spoon' in x for x in nm) and any(x and 'rack' in x for x in nm): dp = min(dp, c.dist)
        pens.append(dp * 100)
    print(f'{s0/20:4.1f}s deg {e[n][1][sl].mean():5.1f} pos {e[n][0][sl].mean():4.2f} | body rotvec(deg) {np.degrees(rv_body[sl].mean(0)).round(1)} world {np.degrees(rv_world[sl].mean(0)).round(1)} | dpos_world(cm) {(100*(P[n][sl,:3,3]-tp[n][sl,:3,3])).mean(0).round(2)} | tgt-pose spoon-rack pen(cm) {min(pens):.2f}')
