"""T8 ep19: rigid-hold ILC on a distilled actor's wrist PHASE WEIGHTS, scored on the actor's own closed-loop rollout.
Reuses J8 actor_res19.Env (residual table baked into phase weights; rollouts identical to evaluate_policy_component; scorer-faithful errors).
Each term (side, object, t0, t1) adds gain * (target - achieved) object-anchor position error, expressed in the physics world through the
scorer's frame-0 sweeper alignment, to that side's wrist position channels (lead `lag` frames, cosine-tapered window). Every iteration
evaluates a gain x lag grid in parallel and keeps the best (cost = actor_res19 cost). Writes best_table.npz (full table incl. --table base).
usage: ilc19t.py --actor A --out O --term left:mini_dust_pan:6.8:8.0 [--term right:mini_sweeper:10.8:11.7] [--table T]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep19/code')
import actor_res19 as AR
CF = AR.CF

def phys_err(env, e_unused, Q):
    """object-anchor position error target - achieved, in the physics world (T, 3) per object."""
    ach = {n: CF.to_T(Q[:, a:a + 3], Q[:, a + 3:a + 7]) for n, a in env.adr.items()}
    reg = env.reg; A, Rf = {}, {}
    for n in CF.NAMES:
        r = reg[n]; a = ach[n].copy(); a[:, :3, 3] += np.einsum('tij,j->ti', a[:, :3, :3], r['candidate_anchor'])
        a[:, :3, :3] = a[:, :3, :3] @ r['rot'].T
        b = env.comp[n].copy(); b[:, :3, 3] += np.einsum('tij,j->ti', b[:, :3, :3], r['anchor']); A[n], Rf[n] = a, b
    Ra = Rf[CF.NAMES[0]][0, :3, :3] @ A[CF.NAMES[0]][0, :3, :3].T; off = Rf[CF.NAMES[0]][0, :3, 3] - Ra @ A[CF.NAMES[0]][0, :3, 3]
    out = {}
    for n in CF.NAMES:
        T = min(len(A[n]), len(Rf[n])); tgt = (Rf[n][:T, :3, 3] - off) @ Ra   # Ra^T (q - off)
        out[n] = tgt - A[n][:T, :3, 3]
    return out

def taper(t, t0, t1, r=0.25):
    w = np.zeros_like(t); m = (t >= t0) & (t <= t1); w[m] = 1
    up = (t >= t0 - r) & (t < t0); w[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - (t0 - r)) / r)
    dn = (t > t1) & (t <= t1 + r); w[dn] = 0.5 + 0.5 * np.cos(np.pi * (t[dn] - t1) / r)
    return w

def smooth(x, s=1.0):
    if s <= 0: return x
    k = np.arange(-3 * int(np.ceil(s)), 3 * int(np.ceil(s)) + 1); g = np.exp(-0.5 * (k / s) ** 2); g /= g.sum()
    return np.stack([np.convolve(np.pad(x[:, c], len(k) // 2, mode='edge'), g, mode='valid') for c in range(x.shape[1])], 1)

W = None
def init(cfg):
    global W; W = AR.Env(cfg)
def ev(table):
    r = W.evaluate(table, keep=True)
    if 'Q' in r:
        pe = phys_err(W, None, r['Q']); r['pe'] = pe
        r['cm_t'] = {n: r['err'][n]['cm_t'] for n in CF.NAMES}; r.pop('Q'); r.pop('err')
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor', required=True); ap.add_argument('--out', required=True); ap.add_argument('--table')
    ap.add_argument('--term', action='append', required=True); ap.add_argument('--iterations', type=int, default=6)
    ap.add_argument('--gains', default='0.3,0.6,1.0'); ap.add_argument('--lags', default='0,1,2'); ap.add_argument('--smooth', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--p', type=float, default=1.3); ap.add_argument('--d', type=float, default=15.)
    ap.add_argument('--peak-w', type=float, default=0.1); ap.add_argument('--clip', type=float, default=0.06)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    terms = [(x[0], x[1], float(x[2]), float(x[3]), float(x[4]) if len(x) > 4 else 1.0) for x in (y.split(':') for y in a.term)]
    sides = sorted({t[0] for t in terms}, key=['left', 'right'].index)
    cfg = dict(actor=str(Path(a.actor).resolve()), table=a.table, sides=sides, P=a.p, D=a.d, peak_w=a.peak_w)
    gains = [float(g) for g in a.gains.split(',')]; lags = [int(l) for l in a.lags.split(',')]
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, terms=terms, script=__file__), indent=1))
    t00 = time.monotonic(); hist = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        env = AR.Env(cfg); table = env.table0.copy(); cur = pool.submit(ev, table).result()
        print('INIT', json.dumps(dict(cost=cur['cost'], objects=cur['objects'])), flush=True)
        best = dict(cost=cur['cost'], objects=cur['objects']); best_table = table.copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
        for it in range(a.iterations):
            cands, meta = [], []
            for g in gains:
                for lag in lags:
                    tb = table.copy()
                    for s, o, t0, t1, gs in terms:
                        si = sides.index(s); e = smooth(cur['pe'][o], a.smooth); e = np.roll(e, -lag, 0)
                        if lag: e[-lag:] = e[-lag - 1]
                        w = taper(env.t, t0, t1)[:, None]; tb[:, si * 6:si * 6 + 3] += g * gs * w * e
                    tb = np.clip(tb, -a.clip, a.clip) if a.clip else tb
                    # rot channels unaffected; clip only pos channels
                    for si in range(len(sides)): tb[:, si * 6 + 3:si * 6 + 6] = table[:, si * 6 + 3:si * 6 + 6]
                    cands.append(tb); meta.append((g, lag))
            reps = list(pool.map(ev, cands, chunksize=1)); costs = [r['cost'] for r in reps]; i = int(np.argmin(costs))
            row = dict(iteration=it, chosen=meta[i], cost=costs[i], objects=reps[i].get('objects'), grid=[(m, round(c, 4), r.get('objects')) for m, c, r in zip(meta, costs, reps)], elapsed_s=time.monotonic() - t00)
            hist.append(row); print('IT', json.dumps(row), flush=True)
            if costs[i] < cur['cost'] or it == 0 or True:
                if costs[i] < best['cost']:
                    best = dict(cost=costs[i], objects=reps[i]['objects']); best_table = cands[i].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                if costs[i] < cur['cost'] * 1.05: table, cur = cands[i], reps[i]
                else: print('no improvement; stop', flush=True); break
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    print('BEST', json.dumps(best), flush=True)

if __name__ == '__main__': main()
