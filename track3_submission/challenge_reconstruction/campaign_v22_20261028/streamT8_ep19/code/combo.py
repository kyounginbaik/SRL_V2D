"""T8 ep19: evaluate superpositions (all non-empty subsets) of residual tables found on the same actor/sides (phase weights are linear in the table).
usage: combo.py --actor A --sides right name=table.npz [...]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code')
import actor_res19 as AR
W = None
def init(cfg):
    global W; W = AR.Env(cfg)
def ev(t):
    r = W.evaluate(t, keep=True)
    if 'err' in r: r['p90'] = {n: float(np.percentile(r['err'][n]['cm_t'], 90)) for n in AR.CF.NAMES}; r.pop('err'); r.pop('Q')
    return r
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor', required=True); ap.add_argument('--sides', default='right'); ap.add_argument('--out'); ap.add_argument('tables', nargs='+')
    a = ap.parse_args(); cfg = dict(actor=str(Path(a.actor).resolve()), table=None, sides=a.sides.split(','), P=1.3, D=15., peak_w=0.1)
    T = {k: np.load(v)['table'] for k, v in (x.split('=') for x in a.tables)}
    subs = [c for r in range(1, len(T) + 1) for c in itertools.combinations(T, r)]
    tabs = [sum(T[k] for k in c) for c in subs]
    with ProcessPoolExecutor(max_workers=min(16, len(subs)), mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool: reps = list(pool.map(ev, tabs))
    rows = []
    for c, r, t in zip(subs, reps, tabs):
        rows.append(dict(combo='+'.join(c), cost=r['cost'], objects=r.get('objects'), p90=r.get('p90'))); print(json.dumps(rows[-1]), flush=True)
        if a.out: Path(a.out).mkdir(parents=True, exist_ok=True); np.savez_compressed(Path(a.out) / ('+'.join(c) + '.npz'), table=t)


if __name__ == '__main__': main()
