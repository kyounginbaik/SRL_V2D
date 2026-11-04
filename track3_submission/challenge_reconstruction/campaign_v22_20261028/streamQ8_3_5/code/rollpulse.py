"""Stream Q8 ep3: pickup ROLL-PULSE lottery on the actor's phase weights.
The ep3 pickup (2.3-3.4 s) rolls the spoon about +35 deg while the target rolls about -70 deg and back (90 deg error spike, ~2 deg of the
episode mean). L8's pickup lottery used small isotropic wrist noise (6 mm / 0.04 rad), which cannot reach a different roll mode.
Here each sample is a structured wrist edit: rotation of the right wrist about the ACHIEVED spoon long axis through the spoon centre
(rigid about the spoon, so the grasp is carried, not broken) by theta * bump(t; tc, width), plus optional small translation and finger
squeeze; converted to joint-space deltas (pos m, XYZ-euler rad) and baked into the phase weights (l8res.Env, closed-loop rollout,
scorer-faithful cost, strictly-better penalty vs --base).
usage: rollpulse.py --ep 3 --actor A --meshdir M --base B --out O [--n 400] [--theta-lo -80 --theta-hi 40] [--tc-lo 2.0 --tc-hi 3.2]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import l8res as L

W = None
def init(cfg):
    global W; W = L.Env(cfg); W.prefix()
def ev(table): return W.evaluate(table)


def spoon_axis(m):
    b = m.jnt_bodyid[m.joint('wooden_spoon_free').id]; V = []
    for g in np.flatnonzero(m.geom_bodyid == b):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        Rg = R.from_quat(m.geom_quat[g], scalar_first=True); V.append(Rg.apply(v) + m.geom_pos[g])
    V = np.vstack(V); c = V.mean(0); w, U = np.linalg.eigh(np.cov((V - c).T)); return U[:, -1], c


def pulse_table(env, Q, ax_b, c_b, theta, tc, width, dp, sq, sqt):
    sc = env.sc; wq = env.wq; d = np.zeros((env.T, L.NCH))
    u = (env.t - tc) / width; bump = np.where(np.abs(u) < 1, 0.5 * (1 + np.cos(np.pi * u)), 0.0)
    Rs = R.from_quat(Q[:, sc + 3:sc + 7], scalar_first=True); ps = Q[:, sc:sc + 3]; Rw = R.from_euler('XYZ', Q[:, wq[3:]]); pw = Q[:, wq[:3]]
    for k in np.flatnonzero(bump > 0):
        axw = Rs[k].apply(ax_b); cw = ps[k] + Rs[k].apply(c_b); Rp = R.from_rotvec(axw * np.radians(theta) * bump[k])
        pn = cw + Rp.apply(pw[k] - cw); Rn = Rp * Rw[k]; eu = Rn.as_euler('XYZ'); eu = Q[k, wq[3:]] + (eu - Q[k, wq[3:]] + np.pi) % (2 * np.pi) - np.pi
        d[k, :3] = pn - pw[k] + dp * bump[k]; d[k, 3:6] = eu - Q[k, wq[3:]]; d[k, 6] = sq * bump[k]; d[k, 7] = sqt * bump[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--bias'); ap.add_argument('--base'); ap.add_argument('--n', type=int, default=400); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--theta-lo', type=float, default=-80); ap.add_argument('--theta-hi', type=float, default=40); ap.add_argument('--tc-lo', type=float, default=2.0); ap.add_argument('--tc-hi', type=float, default=3.2)
    ap.add_argument('--w-lo', type=float, default=0.3); ap.add_argument('--w-hi', type=float, default=0.9); ap.add_argument('--dp', type=float, default=0.005); ap.add_argument('--sq', type=float, default=0.08)
    ap.add_argument('--workers', type=int, default=40); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t_first = a.tc_lo - a.w_hi
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias, cache_t=max(0.0, t_first - 1.6))
    env = L.Env(dict(cfg, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True); Q = rep['Q']
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; cfg['base'] = base; env.base = base; rep = dict(rep, cost=env.score(Q)[0])
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep['objects'], base=base)), flush=True)
    ax_b, c_b = spoon_axis(env.m); print('spoon axis (body)', ax_b.round(3), 'centre', c_b.round(4), flush=True)
    envc = L.Env(cfg); d0, Q0 = envc.prefix(); probe = table + pulse_table(env, Q, ax_b, c_b, -60, a.tc_lo, a.w_hi, np.full(3, 0.005), 0.05, 0.05)
    lk = envc.leak(probe); same = bool(np.array_equal(Q0[:envc.k_pre + 1], Q[:envc.k_pre + 1])); print('CACHE k_pre', envc.k_pre, 'prefix==full', same, 'leak', lk, flush=True)
    if lk != 0.0 or not same: raise SystemExit('cache check failed')
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    rng = np.random.default_rng(a.seed); P = []
    for i in range(a.n):
        P.append(dict(theta=float(rng.uniform(a.theta_lo, a.theta_hi)), tc=float(rng.uniform(a.tc_lo, a.tc_hi)), width=float(rng.uniform(a.w_lo, a.w_hi)),
                      dp=(rng.normal(size=3) * a.dp).tolist(), sq=float(rng.normal() * a.sq), sqt=float(rng.normal() * a.sq)))
    cands = [table + pulse_table(env, Q, ax_b, c_b, p['theta'], p['tc'], p['width'], np.array(p['dp']), p['sq'], p['sqt']) for p in P]
    best = dict(cost=rep['cost'], objects=rep['objects'], params=None); bt = table; t0 = time.monotonic(); f = open(out / 'samples.jsonl', 'w')
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.map(ev, cands, chunksize=1)):
            f.write(json.dumps(dict(P[i], cost=r['cost'], objects=r.get('objects'))) + '\n'); f.flush()
            if r['cost'] < best['cost']:
                best = dict(cost=r['cost'], objects=r['objects'], params=P[i]); bt = cands[i]; np.savez_compressed(out / 'best_table.npz', table=bt)
                print('BEST', i, json.dumps(best), int(time.monotonic() - t0), flush=True)
            if i % 40 == 39: print('PROG', i + 1, int(time.monotonic() - t0), flush=True)
    (out / 'best.json').write_text(json.dumps(best))
    if not a.no_save and best['params'] is not None:
        L.save_actor(env, bt, out / 'actor', dict(mode='q8 pickup roll-pulse lottery', args=vars(a), search_best=best, script=str(Path(__file__).resolve())))


if __name__ == '__main__': main()
