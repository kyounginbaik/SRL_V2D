"""U8: greedy window-by-window search with a locked prefix (port of streamQ8 wlot.py --ilc onto mres.Env, any episode, optional hold
primitive via --freeze). For window [w0,w1]: proposals = zero edit + N random knot residuals on SIDE (knots over [w0-pre, w1+post]) +
rigid-hold ILC steps toward the target-implied wrist pose (mres ilc_correction / rotation-about-anchor, Gaussian low-pass sigma, lead,
alpha, modes pos/full/rotanc). Each proposal = closed-loop rollout from the keyframe truncated at w1+look, scored ONLY on frames
[w0, w1+look] (scorer-faithful mlib errors, frame-0 alignment unchanged because the prefix is identical). Best is baked, the prefix is
locked (later windows' edits start at w_next-pre >= ... edits never touch frames before w0-pre), and the next window follows. The final
table is evaluated over the full episode and saved as an actor (official evaluation) if --save.
usage: wwin.py --ep 15 --actor A --meshes M --out O --windows 1.4:2.2,2.2:3.0,... [--table T] [--freeze side,t,ramp,tw] [--ilc 0.3,0.6,1.0]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres
TOL = [1.3, 15.0]; G = {}
def init(cfg, tol=None):
    G['env'] = mres.Env(cfg)
    if tol: TOL[:] = tol
def roll(env, table, kmax):
    env.d['phase_weights'] = env.weights(table); m = env.m; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    Q = np.empty((kmax + 1, m.nq)); Q[0] = d.qpos
    for k in range(1, kmax + 1):
        n = round(k / (env.fps * env.dt)) - round((k - 1) / (env.fps * env.dt))
        for _ in range(n):
            d.ctrl[:] = env.pol(d.time + env.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
        Q[k] = d.qpos
    return Q
def ev(args):
    table, ws, kmax, wobj = args; env = G['env']
    try:
        Q = roll(env, table, kmax)
        if Q is None: return dict(cost=1e6)
        e = mres.L.per_frame(env.ep, env.poses(Q), env.reg, env.tgt); k = slice(ws, kmax + 1)
        c = sum(wobj.get(n, 1.0) * float((e[n][0][k] / TOL[0] + e[n][1][k] / TOL[1]).mean()) for n in env.names)
        return dict(cost=c, objects={n: (float(e[n][0][k].mean()), float(e[n][1][k].mean())) for n in env.names})
    except Exception as ex: return dict(cost=1e6, error=repr(ex)[:200])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshes', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--table'); ap.add_argument('--freeze'); ap.add_argument('--windows', required=True); ap.add_argument('--side', default='right')
    ap.add_argument('--obj', default='wooden_spoon'); ap.add_argument('--wobj', default='{}'); ap.add_argument('--n', type=int, default=60); ap.add_argument('--look', type=float, default=0.5)
    ap.add_argument('--pre', type=float, default=0.3); ap.add_argument('--post', type=float, default=0.2); ap.add_argument('--knots', type=int, default=3)
    ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.03); ap.add_argument('--scales', default='0.25,0.5,1.0,1.5')
    ap.add_argument('--ilc', default='0.15,0.3,0.6,1.0'); ap.add_argument('--ilc-sigmas', default='0.1,0.2,0.4'); ap.add_argument('--ilc-leads', default='0,1'); ap.add_argument('--ilc-modes', default='pos,full,rotanc')
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--workers', type=int, default=30); ap.add_argument('--save', action='store_true'); ap.add_argument('--P', type=float, default=1.3); ap.add_argument('--D', type=float, default=15.0)
    ap.add_argument('--cap', default='{}'); ap.add_argument('--goal'); ap.add_argument('--tol', default='1.3,15', help='window cost scales cm,deg')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); wobj = json.loads(a.wobj)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshes=a.meshes, table=a.table, freeze=a.freeze, P=a.P, D=a.D, cap=json.loads(a.cap), goal=[float(v) for v in a.goal.split(',')] if a.goal else None)
    env = mres.Env(cfg); table = env.table0.copy(); r0 = env.evaluate(table, keep=True); Qf, Af = r0['Q'], r0['A']
    print('INIT', json.dumps(dict(cost=r0['cost'], objects=r0['objects'])), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    o = mres.HOFF[a.side]; std1 = np.array([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2); sc = [float(s) for s in a.scales.split(',')]
    rng = np.random.default_rng(a.seed); log = []; t00 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg, [float(v) for v in a.tol.split(',')])) as pool:
        for (w0, w1) in [tuple(float(v) for v in w.split(':')) for w in a.windows.split(',')]:
            a0, a1 = max(0.0, w0 - a.pre), min(env.t[-1], w1 + a.post); ws = int(round(w0 * env.fps)); kmax = min(env.T - 1, int(round((w1 + a.look) * env.fps)))
            dels, meta = [np.zeros((env.T, mres.NCH))], [dict(kind='zero')]
            for i in range(a.n):
                x = rng.normal(size=(a.knots, 8)) * std1 * sc[i % len(sc)]; kn = np.linspace(a0, a1, a.knots + 2); P_ = np.vstack([np.zeros(8), x, np.zeros(8)])
                dl = np.zeros((env.T, mres.NCH)); dl[:, o:o + 8] = np.stack([np.interp(env.t, kn, P_[:, c]) for c in range(8)], 1); dels.append(dl); meta.append(dict(kind='rand', scale=sc[i % len(sc)]))
            if a.ilc:
                raw = env.ilc_correction(Qf, Af, a.obj, a.side, a0, a1 + 1e-6, 0.25); rawa = env.last_corr_ra
                wv = mres.sm((env.t - a0) / 0.25) * (1 - mres.sm((env.t - (a1 - 0.25)) / 0.25))
                for md in a.ilc_modes.split(','):
                    src = rawa if md == 'rotanc' else raw
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs_ = gf(src, sg * env.fps, axis=0, mode='nearest')
                        for ld in [int(v) for v in a.ilc_leads.split(',')]:
                            c2 = (np.r_[cs_[ld:], np.repeat(cs_[-1:], ld, 0)] if ld else cs_) * wv[:, None]
                            for al in [float(v) for v in a.ilc.split(',')]:
                                dl = np.zeros((env.T, mres.NCH)); sl = slice(0, 3) if md == 'pos' else slice(0, 6); dl[:, o:o + 6][:, sl] = al * c2[:, sl]; dl[env.t < a0] = 0
                                dels.append(dl); meta.append(dict(kind='ilc', mode=md, sigma=sg, lead=ld, alpha=al))
            reps = list(pool.map(ev, [(table + dl, ws, kmax, wobj) for dl in dels], chunksize=1))
            cs = np.array([r['cost'] for r in reps]); i = int(np.argmin(cs))
            if i != 0: table = table + dels[i]
            rf = env.evaluate(table, keep=True); Qf, Af = rf['Q'], rf['A']
            row = dict(window=[w0, w1], n=len(dels), best=meta[i], win_zero=reps[0], win_best=reps[i], n_better=int((cs < cs[0]).sum()), full_after=dict(cost=rf['cost'], objects=rf['objects']), elapsed_s=int(time.monotonic() - t00))
            log.append(row); print('WIN', json.dumps(row), flush=True)
            np.savez_compressed(out / f'table_after_{w1:.2f}.npz', table=table); np.savez_compressed(out / 'best_table.npz', table=table); (out / 'progress.json').write_text(json.dumps(log))
    print('FULL', json.dumps(dict(cost=rf['cost'], objects=rf['objects'], init=r0['objects'])), flush=True)
    if a.save: mres.save_actor(env, table, out / 'actor', dict(mode='U8 greedy windowed ILC/lottery (locked prefix; port of Q8 wlot --ilc)', args=vars(a), windows=log, freeze=getattr(env, 'freeze_info', None)))
if __name__ == '__main__': main()
