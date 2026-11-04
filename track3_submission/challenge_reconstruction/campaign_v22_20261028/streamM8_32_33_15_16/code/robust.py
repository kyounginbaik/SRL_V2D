"""Stream M8: P8-style decoupling test. Shift the NON-holding (left) wrist by a constant world offset in [t0,t1] (smooth ramps),
then measure (a) the shifted rollout score and (b) robustness of the spoon hold: N random tiny right-wrist knot perturbations in
[p0,p1] (lottery); report median / quantiles of the spoon error. Writes tables to OUT/shift_<i>.npz.
usage: robust.py --ep 32 --actor A --meshes M --table T --out OUT --shifts 'dx,dy,dz;...' --t0 3.0 --t1 9.4 [--n 30]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mres
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int); ap.add_argument('--actor'); ap.add_argument('--meshes'); ap.add_argument('--table'); ap.add_argument('--out')
    ap.add_argument('--shifts'); ap.add_argument('--side', default='left'); ap.add_argument('--t0', type=float); ap.add_argument('--t1', type=float); ap.add_argument('--ramp', type=float, default=0.4)
    ap.add_argument('--p0', type=float, default=3.8); ap.add_argument('--p1', type=float, default=9.2); ap.add_argument('--n', type=int, default=30); ap.add_argument('--std', type=float, default=0.001)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=a.actor, meshes=a.meshes, table=a.table); env = mres.Env(cfg); base = env.table0.copy(); t = env.t; o = mres.HOFF[a.side]
    w = mres.sm((t - a.t0) / a.ramp) * (1 - mres.sm((t - a.t1) / a.ramp)) if a.t1 < t[-1] else mres.sm((t - a.t0) / a.ramp)
    shifts = [np.array([float(v) for v in s.split(',')]) for s in a.shifts.split(';')]; rng = np.random.default_rng(0); K = 8
    std = np.tile([a.std] * 3 + [a.std * 8] * 3 + [a.std * 8] * 2, K)
    pert = [rng.normal(size=K * 8) * std for _ in range(a.n)]
    with ProcessPoolExecutor(max_workers=30, mp_context=mp.get_context('spawn'), initializer=mres.init, initargs=(cfg,)) as pool:
        for i, s in enumerate(shifts):
            tb = base.copy(); tb[:, o:o + 3] += w[:, None] * s; np.savez_compressed(out / f'shift_{i}.npz', table=tb)
            cands = [tb] + [tb + mres.knot_table(env, x, a.p0, a.p1, K, 'right') for x in pert]
            reps = list(pool.map(mres.ev, cands, chunksize=1)); sp = [r['objects']['wooden_spoon'][0] if 'objects' in r else 99 for r in reps[1:]]
            row = dict(shift=s.tolist(), nominal=reps[0].get('objects'), nominal_cost=reps[0]['cost'], spoon_pert_median=float(np.median(sp)),
                       spoon_pert_q25=float(np.percentile(sp, 25)), spoon_pert_q75=float(np.percentile(sp, 75)), spoon_pert_min=float(min(sp)), frac_lt_3=float(np.mean(np.array(sp) < 3)))
            print('SHIFT', json.dumps(row), flush=True)
if __name__ == '__main__': main()
