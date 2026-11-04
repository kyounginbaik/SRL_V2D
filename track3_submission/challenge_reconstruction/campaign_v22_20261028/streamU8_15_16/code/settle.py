"""U8 floor test for a released object: re-roll the actor to frame K (exact evaluator schedule), replace the object's pose by
candidate poses near its target (non-penetrating, sampled as in feas.py), zero its velocity, continue the closed-loop actor to
the end, and score the object over frames K..end vs target_final_v5 (scorer alignment from the base rollout's frame 0).
If no candidate settles closer than the base rest, the released rest pose error is a physical floor of our rack/spoon geometry.
usage: settle.py EP ACTOR MESHES K NCAND [TABLE]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, numpy as np, mujoco
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
ep, actor, meshes, K, NC = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); TB = sys.argv[6] if len(sys.argv) > 6 else None
n = 'wooden_spoon'
def setup():
    env = mres.Env(dict(ep=ep, actor=actor, meshes=meshes, table=TB)); env.d['phase_weights'] = env.weights(env.table0)
    m = env.m; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d); Q = np.empty((env.T, m.nq)); Q[0] = d.qpos
    for k in range(1, K + 1):
        nn = round(k / (env.fps * env.dt)) - round((k - 1) / (env.fps * env.dt))
        for _ in range(nn): d.ctrl[:] = env.pol(d.time + env.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
        Q[k] = d.qpos
    return env, d, Q
E = {}
def init():
    E['env'], E['d'], E['Q'] = setup()
def cont(q_obj):
    env, d0, Q0 = E['env'], E['d'], E['Q']; m = env.m; d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0); a = env.oa[n]
    if q_obj is not None:
        d.qpos[a:a + 7] = q_obj; dv = int(m.joint(f'{n}_free').dofadr[0]); d.qvel[dv:dv + 6] = 0; mujoco.mj_forward(m, d)
    Q = Q0.copy()
    for k in range(K + 1, env.T):
        nn = round(k / (env.fps * env.dt)) - round((k - 1) / (env.fps * env.dt))
        for _ in range(nn): d.ctrl[:] = env.pol(d.time + env.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
        Q[k] = d.qpos
    if q_obj is not None: Q[K, a:a + 7] = q_obj
    e = mres.L.per_frame(ep, env.poses(Q), env.reg, env.tgt)
    return dict(win=[float(e[n][0][K:].mean()), float(e[n][1][K:].mean())], full=[float(e[n][0].mean()), float(e[n][1].mean())], start=[float(e[n][0][K]), float(e[n][1][K])])
def main():
    env, d, Q = setup(); m = env.m; a = env.oa[n]; ca = env.reg[n][0]
    P = env.poses(Q[:K + 1]); _, A = mres.L.per_frame(ep, {k: np.concatenate([v, np.repeat(v[-1:], env.T - K - 1, 0)]) for k, v in P.items()}, env.reg, env.tgt, want_A=True)
    tp = mres.L.target_phys(ep, env.reg, A, env.tgt)[n]
    sp = [b for b in range(m.nbody) if 'spoon' in (mujoco.mj_id2name(m, 1, b) or '')]; rk = [b for b in range(m.nbody) if b not in sp]
    dd = mujoco.MjData(m)
    def pen(qo):
        mujoco.mj_copyData(dd, m, d); dd.qpos[a:a + 7] = qo; mujoco.mj_forward(m, dd); p = 0.0
        for c in dd.contact[:dd.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in sp) != (b2 in sp): p = min(p, c.dist)
        return p * 100
    rng = np.random.default_rng(1); cands = []
    # target poses of frames K..end (the target rest jitters) as centres
    for _ in range(40 * NC):
        kk = rng.integers(K, env.T); Rt = R.from_matrix(tp[kk, :3, :3]); at = tp[kk, :3, 3] + Rt.apply(ca)
        Rc = R.from_rotvec(rng.normal(size=3) * np.radians(rng.uniform(0, 12)) / np.sqrt(3)) * Rt; ac = at + rng.normal(size=3) * rng.uniform(0, 0.012) + np.array([0, 0, rng.uniform(0, 0.015)])
        qo = np.r_[ac - Rc.apply(ca), Rc.as_quat(scalar_first=True)]
        if pen(qo) >= -0.15: cands.append(qo)
        if len(cands) >= NC: break
    print('candidates', len(cands), flush=True)
    with ProcessPoolExecutor(max_workers=30, mp_context=mp.get_context('spawn'), initializer=init) as pool:
        reps = list(pool.map(cont, [None] + cands, chunksize=1))
    base = reps[0]; print('BASE', json.dumps(base), flush=True)
    rs = sorted(reps[1:], key=lambda r: r['win'][0] / 1.3 + r['win'][1] / 15)
    for r in rs[:12]: print('TOP', json.dumps(r), flush=True)
    w = np.array([r['win'] for r in reps[1:]]); print('PCT win pos', np.percentile(w[:, 0], [0, 5, 25, 50]).round(2).tolist(), 'deg', np.percentile(w[:, 1], [0, 5, 25, 50]).round(1).tolist())
main() if __name__ == '__main__' else None
