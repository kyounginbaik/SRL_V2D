"""V8: robustness test of ep32 hold modifications (P8-style). For each MOD (smooth additive edits to the actor's phase-table channels,
mres layout: right 0-2 wrist pos m, 3-5 wrist rot rad, 6 finger squeeze, 7 thumb squeeze; left +8) evaluate the nominal rollout and
N random tiny right-wrist knot perturbations in [p0,p1] (1 mm / 8 mrad / 8 mrad squeeze, 8 knots, same draws for every mod).
Reports per mod: nominal scores, median / q25 / min spoon error over the perturbations, fraction < 3 cm, window 4.5-9.2 s errors.
usage: rob.py --actor A --meshes M [--table T] --mods MODS.json --out O [--n 16]
MODS.json: [{"name": "z2", "edits": [[ch, t0, t1, value, ramp], ...]}, ...]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code')
import mres
def edit_table(env, edits):
    tb = np.zeros((env.T, mres.NCH)); t = env.t
    for ch, t0, t1, v, rp in edits:
        w = mres.sm((t - t0) / rp) * (1 - mres.sm((t - (t1 - rp)) / rp)) if t1 < t[-1] else mres.sm((t - t0) / rp); tb[:, int(ch)] += v * w
    return tb
def ev(args):
    tb, = args; r = mres.W.evaluate(tb, keep=True)
    if 'err' not in r: return dict(cost=r['cost'])
    e = r['err']; k0, k1 = 90, 184
    return dict(cost=r['cost'], objects=r['objects'], win=[float(e['wooden_spoon'][0][k0:k1].mean()), float(e['wooden_spoon'][1][k0:k1].mean())],
                pre=[float(e['wooden_spoon'][0][:k1].mean()), float(e['wooden_spoon'][1][:k1].mean())])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor'); ap.add_argument('--meshes'); ap.add_argument('--table'); ap.add_argument('--mods'); ap.add_argument('--out')
    ap.add_argument('--n', type=int, default=16); ap.add_argument('--p0', type=float, default=3.8); ap.add_argument('--p1', type=float, default=9.2); ap.add_argument('--std', type=float, default=0.001)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=32, actor=a.actor, meshes=a.meshes, table=a.table); env = mres.Env(cfg); base = env.table0.copy()
    mods = json.loads(Path(a.mods).read_text()); rng = np.random.default_rng(a.seed); K = 8
    std = np.tile([a.std] * 3 + [a.std * 8] * 3 + [a.std * 8] * 2, K); pert = [mres.knot_table(env, rng.normal(size=K * 8) * std, a.p0, a.p1, K, 'right') for _ in range(a.n)]
    jobs = []
    for m in mods:
        tb = base + edit_table(env, m['edits']); np.savez_compressed(out / f"mod_{m['name']}.npz", table=tb); jobs += [(tb,)] + [(tb + p,) for p in pert]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=mres.init, initargs=(cfg,)) as pool:
        reps = list(pool.map(ev, jobs, chunksize=1))
    rows = []
    for i, m in enumerate(mods):
        rr = reps[i * (a.n + 1):(i + 1) * (a.n + 1)]; nom = rr[0]; ps = rr[1:]
        sp = np.array([r['objects']['wooden_spoon'][0] if 'objects' in r else 99 for r in ps]); sd = np.array([r['objects']['wooden_spoon'][1] if 'objects' in r else 999 for r in ps])
        wn = np.array([r['win'][0] if 'win' in r else 99 for r in ps])
        row = dict(mod=m['name'], nominal=nom.get('objects'), nominal_win=nom.get('win'), spoon_med=[float(np.median(sp)), float(np.median(sd))], spoon_q25=float(np.percentile(sp, 25)),
                   spoon_min=float(sp.min()), frac_lt3=float((sp < 3).mean()), win_med=float(np.median(wn)), win_min=float(wn.min()),
                   cup_med=float(np.median([r['objects']['blue_cup'][0] if 'objects' in r else 99 for r in ps])), all=[(r.get('objects'), r.get('win')) for r in ps])
        rows.append(row); print('MOD', json.dumps({k: v for k, v in row.items() if k != 'all'}), flush=True)
    (out / 'rob.json').write_text(json.dumps(rows))
if __name__ == '__main__': main()
