"""Stream Q8 (v23): filtered-residual ILC with lead on the actor's phase weights (eps 3/5 pot + spoon).
Uses streamL8 l8res.Env unchanged (closed-loop rollouts, prefix cache with leak check, scorer-faithful cost vs target_final_v5).
New vs l8res ilc:
  - the rigid-hold wrist correction (target spoon * inv(achieved spoon) * achieved wrist) is LOW-PASS filtered with a set of sigmas
    (Gaussian, seconds) and optionally split into band (G_s1 - G_s2) so only the slow component of the error is fed back;
  - LEAD: the correction is shifted earlier by `lead` frames (u_{k+1}(t) = u_k(t) + a * L e_k(t + lead)), the classic fix for a
    lagging plant (the ep5 spoon lags the target by ~1 frame at the 2.1 Hz stirring and has gain ~0.5);
  - GAIN per channel group (pos / rot / full) and alpha grid; accepted only if the closed-loop cost drops (strictly-better penalty
    vs --base as in l8res).
modes: ilc (iterated), eval.
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
import l8res as L
import plib as B

W = None
def init(cfg):
    global W; W = L.Env(cfg); W.prefix()
def ev(table): return W.evaluate(table)


def shift(x, lead):
    if lead == 0: return x
    y = np.empty_like(x)
    if lead > 0: y[:-lead] = x[lead:]; y[-lead:] = x[-1]
    else: y[-lead:] = x[:lead]; y[:-lead] = x[0]
    return y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['ilc', 'eval'])
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--bias'); ap.add_argument('--base')
    ap.add_argument('--t0', type=float, required=True); ap.add_argument('--t1', type=float, required=True); ap.add_argument('--ramp', type=float, default=0.3)
    ap.add_argument('--iterations', type=int, default=10); ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--alphas', default='0.2,0.4,0.7,1.0'); ap.add_argument('--sigmas', default='0.1,0.2,0.4'); ap.add_argument('--hp', type=float, default=0.0, help='if >0 subtract G_hp (band-pass)')
    ap.add_argument('--leads', default='0,1,2'); ap.add_argument('--modes', default='pos,full'); ap.add_argument('--hand', default='right'); ap.add_argument('--obj', default='wooden_spoon')
    ap.add_argument('--shrink', type=float, default=0.5); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias, cache_t=max(0.0, a.t0 - 1.6))
    T00 = time.monotonic(); env = L.Env(dict(cfg, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; cfg['base'] = base; env.base = base
    rep = dict(rep, cost=env.score(rep['Q'])[0])
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep['objects'], base=base, rollout_s=time.monotonic() - T00)), flush=True)
    if a.mode == 'eval':
        e = rep['err']
        for s0 in range(0, env.T, 10): print(f'   {s0/env.fps:5.1f}s', ' '.join(f"{e[n][0][s0:s0+10].mean():5.2f}/{e[n][1][s0:s0+10].mean():5.1f}" for n in B.NAMES))
        return
    envc = L.Env(cfg); d0, Q0 = envc.prefix()
    probe = envc.table0.copy(); probe[envc.t >= a.t0 - 0.2, :6] += 0.01; probe[envc.t >= a.t0 - 0.2, 8:] += 0.01
    lk = envc.leak(probe); same = bool(np.array_equal(Q0[:envc.k_pre + 1], rep['Q'][:envc.k_pre + 1]))
    print('CACHE k_pre', envc.k_pre, 'prefix==full', same, 'leak', lk, flush=True)
    if lk != 0.0 or not same: raise SystemExit('cache check failed')
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    alphas = [float(x) for x in a.alphas.split(',')]; sigmas = [float(x) for x in a.sigmas.split(',')]; leads = [int(x) for x in a.leads.split(',')]; modes = a.modes.split(',')
    off = 0 if a.hand == 'right' else 8; Q = rep['Q']; cur = table.copy(); curcost = rep['cost']; best = dict(cost=curcost, objects=rep['objects']); hist = []; t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(a.iterations):
            raw = env.ilc_correction(Q, a.t0, a.t1, ramp=a.ramp, hand=a.hand, obj=a.obj)
            w = L.sm((env.t - a.t0) / a.ramp) * (1 - L.sm((env.t - (a.t1 - a.ramp)) / a.ramp)) if a.t1 < env.t[-1] else L.sm((env.t - a.t0) / a.ramp)
            cands, meta = [], []
            for s, ld in itertools.product(sigmas, leads):
                # filter the UNWINDOWED-by-ramp correction is not available; filter the windowed one, then re-window so nothing leaks outside
                c = gf(raw, s * env.fps, axis=0, mode='nearest') if s > 0 else raw.copy()
                if a.hp > 0: c = c - gf(raw, a.hp * env.fps, axis=0, mode='nearest')
                c = shift(c, ld) * w[:, None]
                for al, md in itertools.product(alphas, modes):
                    sl = {'full': slice(0, 6), 'pos': slice(0, 3), 'rot': slice(3, 6)}[md]
                    dl = np.zeros((env.T, L.NCH)); dl[:, off + sl.start:off + sl.stop] = al * c[:, sl]; dl[env.t < a.t0 - 0.2] = 0
                    cands.append(cur + dl); meta.append(dict(sigma=s, lead=ld, alpha=al, mode=md))
            reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); i = int(order[0])
            row = dict(iteration=it, n=len(cands), best_candidate=meta[i], cost=reps[i]['cost'], objects=reps[i].get('objects'), prev_cost=curcost,
                       top=[dict(meta[j], cost=reps[j]['cost'], objects=reps[j].get('objects')) for j in order[:8]], elapsed_s=time.monotonic() - t0)
            hist.append(row); print('IT', json.dumps(row), flush=True)
            if reps[i]['cost'] < curcost:
                cur = cands[i]; curcost = reps[i]['cost']; Q = env.evaluate(cur, keep=True)['Q']
                best = dict(cost=curcost, objects=reps[i]['objects']); np.savez_compressed(out / 'best_table.npz', table=cur)
            else:
                alphas = [x * a.shrink for x in alphas]
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best)))
    np.savez_compressed(out / 'best_table.npz', table=cur)
    if not a.no_save and curcost < rep['cost']:
        L.save_actor(env, cur, out / 'actor', dict(mode='q8 filtered-lead ilc', args=vars(a), search_best=best))


if __name__ == '__main__': main()
