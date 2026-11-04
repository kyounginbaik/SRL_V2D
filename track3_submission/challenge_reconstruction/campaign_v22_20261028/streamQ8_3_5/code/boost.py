"""Stream Q8: stirring-band feed-forward boost on the actor's phase weights.
Finding (band.py): in ep5 the wrist PD passes only ~50 % of the commanded 2.1 Hz stirring amplitude (2-frame lag), so the spoon stirs at
~0.48 of the target amplitude. Residual (actuator units, baked into the phase weights as in l8res):
    delta_c(t) = w(t) * (g_c - 1) * HP_s[phase_cmd_c](t + lead)        c in right-wrist channels (pos xyz / rot xyz)
HP_s = x - Gauss_s(x) (s frames), phase_cmd = scale * phi(t) @ W (the open-loop part of the actor's command), w = smooth window.
Grid search (+ optional local refinement) over (g_pos, g_rot, lead, s); every candidate = closed-loop rollout scored scorer-faithfully (l8res.Env).
usage: boost.py --ep 5 --actor A --meshdir M --base B --out O --t0 2 --t1 11.2 [--gpos 1,1.3,1.6,2] [--grot ...] [--leads 0,1,2,3] [--hps 3,6]"""
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
from learned_feedback_policy import features

W = None
def init(cfg):
    global W; W = L.Env(cfg); W.prefix()
def ev(table): return W.evaluate(table)


def phase_cmd(env, table):
    """open-loop phase command of the right-wrist actuators at the frame times (T x 6), including the current table."""
    Wt = env.weights(table); ph = features(env.t / env.dur, env.d['centers'], float(env.d['width']))
    return np.stack([env.scale[a] * (ph @ Wt[:, a]) for a in env.wa], 1)


def shift(x, lead):
    if lead == 0: return x
    y = np.empty_like(x); y[:-lead] = x[lead:]; y[-lead:] = x[-1]; return y


ADV = False   # --advance: delta = g * HP(t + lead) - HP(t) (time-advance of the stirring band, g = 1 is a pure lead)
def boost_table(env, table, g_pos, g_rot, lead, s, t0, t1, ramp=0.3, pc=None):
    pc = phase_cmd(env, table) if pc is None else pc
    hp0 = pc - gf(pc, s, axis=0, mode='nearest'); hp = shift(hp0, lead)
    if ADV:
        w = L.sm((env.t - t0) / ramp) * (1 - L.sm((env.t - (t1 - ramp)) / ramp))
        d = np.zeros((env.T, L.NCH)); d[:, 0:3] = (np.asarray(g_pos) * hp[:, :3] - hp0[:, :3]) * w[:, None]; d[:, 3:6] = (np.asarray(g_rot) * hp[:, 3:] - hp0[:, 3:]) * w[:, None]
        d[env.t < t0 - 0.2] = 0; return table + d
    w = L.sm((env.t - t0) / ramp) * (1 - L.sm((env.t - (t1 - ramp)) / ramp))
    d = np.zeros((env.T, L.NCH)); d[:, 0:3] = (g_pos - 1) * hp[:, :3] * w[:, None]; d[:, 3:6] = (g_rot - 1) * hp[:, 3:] * w[:, None]
    d[env.t < t0 - 0.2] = 0; return table + d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--bias'); ap.add_argument('--base'); ap.add_argument('--t0', type=float, required=True); ap.add_argument('--t1', type=float, required=True)
    ap.add_argument('--gpos', default='1.0,1.3,1.6,2.0,2.5'); ap.add_argument('--grot', default='1.0,1.3,1.6,2.0'); ap.add_argument('--leads', default='0,1,2,3'); ap.add_argument('--hps', default='3,6')
    ap.add_argument('--workers', type=int, default=40); ap.add_argument('--refine', type=int, default=3); ap.add_argument('--advance', action='store_true'); ap.add_argument('--random', type=int, default=0, help='N random per-axis samples around --center (advance mode)'); ap.add_argument('--center', default='1.12,1.12,1.12,1.05,1.05,1.05,1,2.1'); ap.add_argument('--gstd', type=float, default=0.2); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); global ADV; ADV = a.advance
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias, cache_t=max(0.0, a.t0 - 1.6))
    env = L.Env(dict(cfg, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; cfg['base'] = base; env.base = base; rep = dict(rep, cost=env.score(rep['Q'])[0])
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep['objects'], base=base)), flush=True)
    envc = L.Env(cfg); d0, Q0 = envc.prefix(); pc = phase_cmd(env, table)
    probe = boost_table(envc, envc.table0, 2.0, 2.0, 0, 3, a.t0, a.t1, pc=pc); lk = envc.leak(probe); same = bool(np.array_equal(Q0[:envc.k_pre + 1], rep['Q'][:envc.k_pre + 1]))
    print('CACHE k_pre', envc.k_pre, 'prefix==full', same, 'leak', lk, flush=True)
    if lk != 0.0 or not same: raise SystemExit('cache check failed')
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    grid = [dict(g_pos=gp, g_rot=gr, lead=ld, s=s) for gp, gr, ld, s in itertools.product([float(x) for x in a.gpos.split(',')], [float(x) for x in a.grot.split(',')],
            [int(x) for x in a.leads.split(',')], [float(x) for x in a.hps.split(',')]) if ADV or not (gp == 1.0 and gr == 1.0)]
    if a.random:
        c_ = [float(v) for v in a.center.split(',')]; rg = np.random.default_rng(a.seed)
        grid = [dict(g_pos=c_[0:3], g_rot=c_[3:6], lead=int(c_[6]), s=c_[7])] + [dict(g_pos=(np.array(c_[0:3]) + rg.normal(size=3) * a.gstd).tolist(), g_rot=(np.array(c_[3:6]) + rg.normal(size=3) * a.gstd).tolist(),
                lead=int(rg.choice([0, 1, 1, 1, 2])), s=float(np.clip(c_[7] * np.exp(rg.normal() * 0.3), 1.2, 8))) for _ in range(a.random - 1)]
        a.refine = 0
    best = dict(cost=rep['cost'], objects=rep['objects'], params=None); best_table = table; rows = []; t00 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for rnd in range(1 + a.refine):
            cands = [boost_table(env, table, p['g_pos'], p['g_rot'], p['lead'], p['s'], a.t0, a.t1, pc=pc) for p in grid]
            reps = list(pool.map(ev, cands, chunksize=1))
            for p, r, c in zip(grid, reps, cands):
                rows.append(dict(p, round=rnd, cost=r['cost'], objects=r.get('objects')))
                if r['cost'] < best['cost']: best = dict(cost=r['cost'], objects=r['objects'], params=p); best_table = c
            order = np.argsort([r['cost'] for r in reps])
            print('ROUND', rnd, 'n', len(grid), 'best', json.dumps(best), 'top', json.dumps([rows[-len(grid) + int(i)] for i in order[:6]]), 'elapsed', int(time.monotonic() - t00), flush=True)
            (out / 'grid.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            if best['params'] is None or rnd == a.refine: break
            bp = best['params']; rng = np.random.default_rng(rnd)
            grid = [dict(g_pos=max(0.5, bp['g_pos'] + rng.normal() * 0.25), g_rot=max(0.5, bp['g_rot'] + rng.normal() * 0.25), lead=int(np.clip(bp['lead'] + rng.integers(-1, 2), 0, 5)), s=float(np.clip(bp['s'] * np.exp(rng.normal() * 0.3), 1.5, 12))) for _ in range(a.workers)]
    np.savez_compressed(out / 'best_table.npz', table=best_table); (out / 'best.json').write_text(json.dumps(best))
    if not a.no_save and best['params'] is not None:
        L.save_actor(env, best_table, out / 'actor', dict(mode='q8 stirring-band feed-forward boost', args=vars(a), search_best=best, script=str(Path(__file__).resolve())))


if __name__ == '__main__': main()
