"""Stream Q8: slow piecewise-constant wrist OFFSET search (the slow spoon error in ep5 is a +z 0.4-0.7 cm bias while stirring plus a
+-1 cm y drift; the per-frame ILC is too noisy for the chaotic hold, so try a handful of smooth DOF).
delta(t) = sum_j off_j * bump_j(t) on right-wrist pos (xyz, m) [and optional rot], segments = smooth plateaus with 0.3 s ramps.
Random sampling (n) over per-segment offsets; closed-loop full rollouts, l8res cost (strictly-better vs --base).
usage: offgrid.py --ep 5 --actor A --meshdir M --base B --out O --segs 2:5,5:8,8:11.2 [--n 200] [--xy 0.006] [--zlo -0.008 --zhi 0.003] [--rot 0]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import l8res as L

W = None
def init(cfg):
    global W; W = L.Env(cfg); W.prefix()
def ev(table): return W.evaluate(table)


def seg_table(env, segs, offs, ramp=0.3):
    d = np.zeros((env.T, L.NCH))
    for (s0, s1), o in zip(segs, offs):
        w = L.sm((env.t - (s0 - ramp)) / ramp) * (1 - L.sm((env.t - s1) / ramp)); d[:, :len(o)] += np.outer(w, o)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--bias'); ap.add_argument('--base'); ap.add_argument('--segs', required=True); ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--xy', type=float, default=0.006); ap.add_argument('--zlo', type=float, default=-0.008); ap.add_argument('--zhi', type=float, default=0.003); ap.add_argument('--rot', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--workers', type=int, default=40); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    segs = [tuple(float(v) for v in s.split(':')) for s in a.segs.split(',')]
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias, cache_t=max(0.0, segs[0][0] - 0.6 - 1.6))
    env = L.Env(dict(cfg, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; cfg['base'] = base; env.base = base; rep = dict(rep, cost=env.score(rep['Q'])[0])
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep['objects'], base=base)), flush=True)
    envc = L.Env(cfg); d0, Q0 = envc.prefix(); rng = np.random.default_rng(a.seed); nd = 6 if a.rot > 0 else 3
    P = []
    for i in range(a.n):
        offs = []
        for _ in segs:
            o = [rng.uniform(-a.xy, a.xy), rng.uniform(-a.xy, a.xy), rng.uniform(a.zlo, a.zhi)] + ([rng.normal() * a.rot for _ in range(3)] if nd == 6 else [])
            offs.append(o)
        P.append(offs)
    cands = [table + seg_table(env, segs, o) for o in P]
    lk = max(envc.leak(c) for c in cands[:5]); same = bool(np.array_equal(Q0[:envc.k_pre + 1], rep['Q'][:envc.k_pre + 1]))
    print('CACHE k_pre', envc.k_pre, 'prefix==full', same, 'leak', lk, flush=True)
    if lk != 0.0 or not same: raise SystemExit('cache check failed')
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    best = dict(cost=rep['cost'], objects=rep['objects'], offs=None); bt = table; f = open(out / 'samples.jsonl', 'w'); t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.map(ev, cands, chunksize=1)):
            f.write(json.dumps(dict(offs=P[i], cost=r['cost'], objects=r.get('objects'))) + '\n'); f.flush()
            if r['cost'] < best['cost']:
                best = dict(cost=r['cost'], objects=r['objects'], offs=P[i]); bt = cands[i]; np.savez_compressed(out / 'best_table.npz', table=bt); print('BEST', i, json.dumps(best), int(time.monotonic() - t0), flush=True)
    (out / 'best.json').write_text(json.dumps(best))
    if not a.no_save and best['offs'] is not None:
        L.save_actor(env, bt, out / 'actor', dict(mode='q8 piecewise wrist offset search', args=vars(a), search_best=best, script=str(Path(__file__).resolve())))


if __name__ == '__main__': main()
