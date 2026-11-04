"""U8: nearest rack-feasible spoon pose to the target, per frame (physics world, achieved rack/hands from a rollout).
Samples rotations (rotvec ball around the target) + translations, measures the spoon-rack penetration (mj_forward contact dist),
returns min error ( pos(anchor) cm, deg ) among poses with penetration >= PEN_OK (achieved-rollout level). Reports the mean floor."""
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
ep, actor, meshes, npz = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; f0, f1, stride = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
env = mres.Env(dict(ep=ep, actor=actor, meshes=meshes)); Q = np.load(npz)['Q']; m = env.m; d = mujoco.MjData(m)
P = env.poses(Q); e, A = mres.L.per_frame(ep, P, env.reg, env.tgt, want_A=True); tp = mres.L.target_phys(ep, env.reg, A, env.tgt)
n = 'wooden_spoon'; a = env.oa[n]; ca = env.reg[n][0]
sp = [b for b in range(m.nbody) if 'spoon' in (mujoco.mj_id2name(m, 1, b) or '')]; rk = [b for b in range(m.nbody) if b == 0 or any(s in (mujoco.mj_id2name(m, 1, b) or '') for s in ('rack', 'table'))]
def pen(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d); dp = 0.0
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in sp and b2 in rk) or (b2 in sp and b1 in rk): dp = min(dp, c.dist)
    return dp * 100
rng = np.random.default_rng(0); out = []
for k in range(f0, min(f1, len(Q)), stride):
    ach_pen = pen(Q[k]); thr = min(ach_pen, -0.2)
    Rt = R.from_matrix(tp[n][k, :3, :3]); at = tp[n][k, :3, 3] + Rt.apply(ca)
    Ra = R.from_matrix(P[n][k, :3, :3]); aa = P[n][k, :3, 3] + Ra.apply(ca)
    best = (1e9, None)
    cands = [(Ra, aa)]
    for _ in range(1500):
        s = rng.uniform(0, 1); Rc = R.from_rotvec(rng.normal(size=3) * np.radians(rng.uniform(1, 25)) / np.sqrt(3)) * Rt if rng.uniform() < 0.7 else R.from_rotvec(s * (Rt * Ra.inv()).as_rotvec()) * Ra
        ac = at + rng.normal(size=3) * rng.uniform(0, 0.015)
        cands.append((Rc, ac))
    for Rc, ac in cands:
        dg = np.degrees((Rc * Rt.inv()).magnitude()); pc = 100 * np.linalg.norm(ac - at); c = pc / 1.3 + dg / 15
        if c >= best[0]: continue
        q = Q[k].copy(); q[a:a + 3] = ac - Rc.apply(ca); q[a + 3:a + 7] = Rc.as_quat(scalar_first=True)
        if pen(q) >= thr: best = (c, (pc, dg))
    out.append((k, e[n][0][k], e[n][1][k], ach_pen, best[1]))
    print(f'f{k:3d} ach {e[n][0][k]:4.2f}/{e[n][1][k]:5.1f} achpen {ach_pen:5.2f}  feasible-floor {best[1][0]:4.2f}/{best[1][1]:5.1f}', flush=True)
o = np.array([(x[1], x[2], x[4][0], x[4][1]) for x in out]); print('MEAN ach', o[:, :2].mean(0).round(2), 'floor', o[:, 2:].mean(0).round(2))
