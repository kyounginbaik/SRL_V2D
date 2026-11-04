"""Stream Q8: sequential WINDOWED lottery with a locked prefix (window-checkpoint refinement) on the actor's phase weights.
Why: in the chaotic spoon holds (ep3 roll, ep5 stirring) any edit changes the whole downstream rollout, so a full-episode cost turns
every search into a lottery whose tails are dominated by downstream chaos. Here the episode is cut into short windows; for window
[ws, we] we sample N random right-wrist/finger residuals (knots over [ws - pre, we + post]) and score ONLY the scorer-faithful per-frame
error on frames [ws, we + look] (truncated rollout from the prefix cache). The best (incl. the zero edit) is baked into the residual
table, the prefix is locked, and the next window starts from that state. The final table is re-evaluated over the full episode and,
if requested, saved as an actor (official evaluation).
Prefix cache: l8res.Env with cache_t = ws - pre - 1.6 (RBF leak checked = 0, prefix == full rollout checked).
usage: wlot.py --ep 5 --actor A --meshdir M --out O --windows 2:3,3:4,... [--n 120] [--look 0.5] [--pos-std 0.004 --rot-std 0.03 --sq-std 0.04]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8res as L
import plib as B

TOL = (1.3, 15.0)
W = None; KMAX = None; WS = None; WOBJ = None
def init(cfg, kmax, ws, wobj):
    global W, KMAX, WS, WOBJ; W = L.Env(cfg); W.prefix(); KMAX = kmax; WS = ws; WOBJ = wobj


def rollout_to(env, table, kmax):
    d0, Q0 = env.prefix(); env.d['phase_weights'] = env.weights(table); m = env.m; pol = env.pol; dt = env.dt
    d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0); Q = Q0[:kmax + 1].copy()
    for k in range(env.k_pre + 1, kmax + 1):
        n = round(k / (env.fps * dt)) - round((k - 1) / (env.fps * dt))
        for _ in range(n):
            d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
        Q[k] = d.qpos
    return Q


def wcost(env, Q, ws, wobj):
    e = B.per_frame(env.ep, env.poses(Q), env.reg, env.tgt); k = slice(ws, len(Q))
    c = sum(wobj[n] * float((e[n][0][k] / TOL[0] + e[n][1][k] / TOL[1]).mean()) for n in B.NAMES)
    return c, {n: (float(e[n][0][k].mean()), float(e[n][1][k].mean())) for n in B.NAMES}


def ev(table):
    try:
        if W.leak(table) != 0.0: return dict(cost=1e6, error='leak')
        Q = rollout_to(W, table, KMAX)
        if Q is None: return dict(cost=1e6, unstable=True)
        c, o = wcost(W, Q, WS, WOBJ); return dict(cost=c, objects=o)
    except Exception as ex: return dict(cost=1e6, error=repr(ex)[:200])


def knots(env, x, a0, a1, K):
    kn = np.linspace(a0, a1, K + 2); P = np.vstack([np.zeros(L.NCH), x.reshape(K, L.NCH), np.zeros(L.NCH)])
    return np.stack([np.interp(env.t, kn, P[:, c]) for c in range(L.NCH)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--bias'); ap.add_argument('--base'); ap.add_argument('--windows', required=True)
    ap.add_argument('--n', type=int, default=120); ap.add_argument('--look', type=float, default=0.5); ap.add_argument('--pre', type=float, default=0.3); ap.add_argument('--post', type=float, default=0.2)
    ap.add_argument('--knots', type=int, default=3); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.04)
    ap.add_argument('--scales', default='0.25,0.5,1.0,1.5'); ap.add_argument('--pot-w', type=float, default=0.5); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workers', type=int, default=40); ap.add_argument('--no-save', action='store_true'); ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--ilc', default='', help='alphas for windowed low-pass rigid-hold ILC candidates, e.g. 0.1,0.25,0.5,1.0'); ap.add_argument('--ilc-sigmas', default='0.1,0.2,0.4'); ap.add_argument('--ilc-leads', default='0,1'); ap.add_argument('--topk', type=int, default=0, help='>0: re-run the top-K window candidates over the FULL episode and accept only a full-cost improvement')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg0 = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias)
    env = L.Env(dict(cfg0, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; env.base = base; init_cost = env.score(rep['Q'])[0]; Qfull = rep['Q']
    print('INIT', json.dumps(dict(cost=init_cost, objects=rep['objects'], base=base)), flush=True)
    wins = [tuple(float(v) for v in w.split(':')) for w in a.windows.split(',')]; wobj = {'white_pot': a.pot_w, 'wooden_spoon': 1.0}
    scales = [float(s) for s in a.scales.split(',')]; std1 = np.array([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2 + [0.0] * 6)
    rng = np.random.default_rng(a.seed); log = []; t00 = time.monotonic()
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg0, script=__file__), indent=1))
    for rnd in range(a.rounds):
        for (w0, w1) in wins:
            a0, a1 = w0 - a.pre, w1 + a.post; ws = int(round(w0 * env.fps)); kmax = min(env.T - 1, int(round((w1 + a.look) * env.fps)))
            tpath = out / 'cur_table.npz'; np.savez_compressed(tpath, table=table)
            cfg = dict(cfg0, table=str(tpath), cache_t=max(0.0, a0 - 1.6))
            envc = L.Env(cfg); d0, Q0 = envc.prefix(); same = bool(np.array_equal(Q0[:envc.k_pre + 1], Qfull[:envc.k_pre + 1]))
            xs = [np.zeros(a.knots * L.NCH)] + [rng.normal(size=a.knots * L.NCH) * np.tile(std1, a.knots) * scales[i % len(scales)] for i in range(a.n - 1)]
            dels = [knots(envc, x, a0, a1, a.knots) for x in xs]; meta = [dict(kind='rand', scale=float(np.abs(x).max())) for x in xs]
            if a.ilc:
                from scipy.ndimage import gaussian_filter1d as gf
                raw = env.ilc_correction(Qfull, a0, a1 + 1e-6, ramp=0.25)
                wv = L.sm((env.t - a0) / 0.25) * (1 - L.sm((env.t - (a1 - 0.25)) / 0.25))
                for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                    cs_ = gf(raw, sg * env.fps, axis=0, mode='nearest')
                    for ld in [int(v) for v in a.ilc_leads.split(',')]:
                        c2 = np.r_[cs_[ld:], np.repeat(cs_[-1:], ld, 0)] * wv[:, None] if ld else cs_ * wv[:, None]
                        for al in [float(v) for v in a.ilc.split(',')]:
                            for md, sl in (('pos', slice(0, 3)), ('full', slice(0, 6))):
                                dl = np.zeros((env.T, L.NCH)); dl[:, sl] = al * c2[:, sl]; dl[env.t < a0] = 0; dels.append(dl); meta.append(dict(kind='ilc', sigma=sg, lead=ld, alpha=al, mode=md))
            lk = max(envc.leak(envc.table0 + dl) for dl in dels[1:4]);
            if lk != 0.0 or not same: raise SystemExit(f'cache check failed window {w0}-{w1}: same {same} leak {lk}')
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg, kmax, ws, wobj)) as pool:
                reps = list(pool.map(ev, [envc.table0 + dl for dl in dels], chunksize=1))
            cs = np.array([r['cost'] for r in reps]); i = int(np.argmin(cs))
            if a.topk:
                cur_full = env.score(Qfull)[0]; top = [int(j) for j in np.argsort(cs)[:a.topk] if j != 0 and cs[j] < 1e6]
                with ProcessPoolExecutor(max_workers=min(a.workers, max(1, len(top))), mp_context=mp.get_context('spawn'), initializer=L.init, initargs=(dict(cfg, base=base),)) as pool2:
                    freps = list(pool2.map(L.ev, [envc.table0 + dels[j] for j in top], chunksize=1))
                fc = [r['cost'] for r in freps]; i = 0
                if top and min(fc) < cur_full: i = top[int(np.argmin(fc))]
                print('TOPK', json.dumps(dict(window=[w0, w1], cur_full=cur_full, full=[dict(j=j, cost=r['cost'], objects=r.get('objects')) for j, r in zip(top, freps)])), flush=True)
            row = dict(round=rnd, window=[w0, w1], k_pre=envc.k_pre, n=len(dels), best_meta=meta[i], zero=reps[0], best_i=i, best=reps[i], median=float(np.median(cs[cs < 1e6])) if (cs < 1e6).any() else None,
                       n_better=int((cs < cs[0]).sum()), elapsed_s=int(time.monotonic() - t00))
            if i != 0: table = envc.table0 + dels[i]
            Qfull = env.evaluate(table, keep=True)['Q']; c, res, _ = env.score(Qfull); row['full_after'] = dict(cost=c, objects=res)
            log.append(row); print('WIN', json.dumps(row), flush=True)
            np.savez_compressed(out / 'best_table.npz', table=table); (out / 'progress.json').write_text(json.dumps(log))
    fin = env.evaluate(table, keep=True); c, res, _ = env.score(fin['Q'])
    print('FULL', json.dumps(dict(cost=c, objects=res, init_cost=init_cost)), flush=True)
    if not a.no_save and c < init_cost:
        L.save_actor(env, table, out / 'actor', dict(mode='q8 windowed lottery (locked prefix)', args=vars(a), windows=log, script=str(Path(__file__).resolve())))


if __name__ == '__main__': main()
