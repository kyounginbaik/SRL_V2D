"""U8: place-offset scan. Constant right(or left)-wrist world-frame translation (+ optional rotation) residual on the phase weights,
smoothly ramped in over [t0, t0+ramp_in] and out over [t1-ramp, t1] (mres.Env rollout + scorer; same residual mechanism as mres).
Grid over dx,dy,dz (cm) and rx,ry,rz (deg); writes samples.jsonl and the best table (best_table.npz) for mres --table chaining.
usage: gridoff.py --ep 15 --actor A --meshes M --out OUT --t0 2.4 --t1 5.6 --dx=-1,0,1 --dy=... [--table T] [--cap ..] [--P --D --goal]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, itertools, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
def win(env, t0, t1, rin, rout):
    return mres.sm((env.t - t0) / rin) * (1 - mres.sm((env.t - (t1 - rout)) / rout))
def table_for(env, base, x, a):
    w = win(env, a['t0'], a['t1'], a['ramp_in'], a['ramp_out']); tb = base.copy(); o = mres.HOFF[a['side']]
    tb[:, o:o + 3] += np.outer(w, np.asarray(x[:3]) / 100.0); tb[:, o + 3:o + 6] += np.outer(w, np.radians(x[3:6])); return tb
G = {}
def init(cfg, a, base):
    G['env'] = mres.Env(cfg); G['a'] = a; G['base'] = base
def ev(x):
    env = G['env']; r = env.evaluate(table_for(env, G['base'], x, G['a'])); r['x'] = list(x); return r
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshes', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--table'); ap.add_argument('--cap', default='{}'); ap.add_argument('--side', default='right')
    ap.add_argument('--t0', type=float, required=True); ap.add_argument('--t1', type=float, required=True); ap.add_argument('--ramp-in', type=float, default=0.5); ap.add_argument('--ramp-out', type=float, default=0.4)
    for c in ('dx', 'dy', 'dz', 'rx', 'ry', 'rz'): ap.add_argument('--' + c, default='0')
    ap.add_argument('--P', type=float, default=1.3); ap.add_argument('--D', type=float, default=15.0); ap.add_argument('--goal'); ap.add_argument('--workers', type=int, default=30)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshes=a.meshes, table=a.table, cap=json.loads(a.cap), P=a.P, D=a.D, goal=[float(v) for v in a.goal.split(',')] if a.goal else None)
    env = mres.Env(cfg); base = env.table0.copy(); r0 = env.evaluate(base); print('BASE', json.dumps(r0), flush=True)
    grid = list(itertools.product(*[[float(v) for v in getattr(a, c).split(',')] for c in ('dx', 'dy', 'dz', 'rx', 'ry', 'rz')]))
    aa = dict(t0=a.t0, t1=a.t1, ramp_in=a.ramp_in, ramp_out=a.ramp_out, side=a.side)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__, n=len(grid)), indent=1)); t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg, aa, base)) as pool:
        reps = list(pool.map(ev, grid, chunksize=1))
    with open(out / 'samples.jsonl', 'w') as f:
        for r in reps: f.write(json.dumps(r) + '\n')
    reps.sort(key=lambda r: r['cost'])
    for r in reps[:15]: print('TOP', json.dumps(dict(x=r['x'], cost=round(r['cost'], 3), objects={k: [round(v[0], 3), round(v[1], 2)] for k, v in r.get('objects', {}).items()})), flush=True)
    best = reps[0]; np.savez_compressed(out / 'best_table.npz', table=table_for(env, base, best['x'], aa))
    (out / 'best.json').write_text(json.dumps(dict(best=best, base=r0, elapsed_s=time.monotonic() - t0), indent=1))
if __name__ == '__main__': main()
