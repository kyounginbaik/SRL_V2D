"""Stream L8: non-holding-hand clearance test (after P8's ep35 finding: the other hand pressing the spoon drives the in-hand bifurcation).
Candidates = per-frame LEFT-hand actuator bias (wrist xy away from the spoon, or left thumb flex opening) in a window, baked into the
phase weights exactly like l8res.py residuals. Each candidate is scored (closed-loop rollout, scorer-faithful) and its robustness is
measured as the median spoon error over N random small right-wrist knot residuals in the early window (the P8 test).
usage: l8clear.py --ep 5 --actor DIR --meshdir DIR --table T.npz --out DIR --win 2.6,8.2 [--rob-t0 3.0 --rob-t1 9.6]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8res as L
from tool_library.common import hand as Hh

E = None
def init(cfg):
    global E; E = L.Env(cfg)
def ev(job):
    bias, table = job; E.bias = bias; r = E.evaluate(table); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True)
    ap.add_argument('--table'); ap.add_argument('--out', required=True); ap.add_argument('--win', default='2.6,8.2'); ap.add_argument('--base')
    ap.add_argument('--rob-t0', type=float, default=3.0); ap.add_argument('--rob-t1', type=float, default=9.6); ap.add_argument('--rob-n', type=int, default=16)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, cache_t=0.0)
    env = L.Env(cfg); r0 = env.evaluate(env.table0, keep=True); Q = r0['Q']
    base = json.loads(Path(a.base).read_text()) if a.base else r0['objects']; cfg['base'] = base; env.base = base
    m = env.m; names = list(env.d['actuator_names']); nu = len(names)
    lw = [int(m.joint(f'robot_left_left_pos_{c}').qposadr[0]) for c in 'xy']; sp = env.adr['wooden_spoon']
    rig = Hh.HandRig.build(m, 'robot_left_', 'left'); thumb = []
    for jn in rig.flexion_joints('thumb'):
        jid = m.joint('robot_left_left_' + jn).id if isinstance(jn, str) else int(jn)
        hits = np.flatnonzero(m.actuator_trnid[:, 0] == jid)
        if len(hits) == 1: thumb.append(int(hits[0]))
    axy = [names.index(f'robot_left_left_pos_{c}_position') for c in 'xy']
    rng = np.random.default_rng(7); K = 6
    rob = [L.knot_table(env, rng.normal(size=K * L.NCH) * np.tile([0.006] * 3 + [0.04] * 3 + [0.04] * 2 + [0.0] * 6, K) * 0.05, a.rob_t0, a.rob_t1, K) for _ in range(a.rob_n)]
    cands = [('none', None)]
    for w in [tuple(map(float, x.split(','))) for x in a.win.split(';')]:
        t0, t1 = w; ramp = 0.3; wt = L.sm((env.t - t0) / ramp) * (1 - L.sm((env.t - (t1 - ramp)) / ramp))
        sel = (env.t >= t0) & (env.t <= t1); dvec = (Q[sel][:, lw] - Q[sel][:, [sp, sp + 1]]).mean(0); dvec /= np.linalg.norm(dvec)
        for mag in (0.005, 0.01, 0.02, 0.03, 0.04):
            b = np.zeros((env.T, nu)); b[:, axy[0]] = wt * mag * dvec[0]; b[:, axy[1]] = wt * mag * dvec[1]; cands.append((f'wrist_xy_{mag*100:.1f}cm_{t0}-{t1}', b))
        for th in (-0.3, -0.15, 0.15, 0.3):
            b = np.zeros((env.T, nu)); b[:, thumb] = (wt * th)[:, None]; cands.append((f'thumb_{th:+.2f}rad_{t0}-{t1}', b))
    print('thumb actuators', [names[i] for i in thumb], 'n cands', len(cands), flush=True)
    jobs = []; idx = []
    for ci, (nm, b) in enumerate(cands):
        jobs.append((b, env.table0)); idx.append((ci, -1))
        for ri, rt in enumerate(rob): jobs.append((b, env.table0 + rt)); idx.append((ci, ri))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        reps = list(pool.map(ev, jobs, chunksize=1))
    res = []
    for ci, (nm, b) in enumerate(cands):
        own = [r for (c, ri), r in zip(idx, reps) if c == ci and ri < 0][0]
        rr = [r for (c, ri), r in zip(idx, reps) if c == ci and ri >= 0]
        sp_ = np.array([r['objects']['wooden_spoon'] if 'objects' in r else (99, 999) for r in rr])
        row = dict(name=nm, cost=own['cost'], objects=own.get('objects'), rob_median_spoon=np.median(sp_, 0).tolist(), rob_frac_spoon_lt3cm=float((sp_[:, 0] < 3).mean()))
        res.append(row); print('CAND', json.dumps(row), flush=True)
        if b is not None: np.savez_compressed(out / f'bias_{ci:02d}.npz', bias=b)
    (out / 'results.json').write_text(json.dumps(dict(cands=[r['name'] for r in res], results=res), indent=1))


if __name__ == '__main__': main()
