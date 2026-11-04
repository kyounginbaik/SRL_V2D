"""H9 copy (target_final_v7_g9rot). B9 wbeam3 = B9 wbeam2 + time-ADVANCE candidates: right-wrist phase command shifted earlier by s frames (delta = g * (c(t+s) - c(t)), c = the parent's own
phase-part wrist command scale*phi(t)@W, smooth-windowed to [a0, a1]); fixes lift/stir lag (the v6 ep32 spoon lifts ~0.2 s later than the target at 2.4-2.8 s).
B9 copy of streamV8 wbeam2.py (scorer target = target_final_v6 via the B9 mlib copy; kc clamped at 0 so windows may start before the gap; --init resumes from saved beam tables).
V8 (ep32): BEAM search over short windows with locked prefixes ("window checkpoint refinement") on the actor's phase weights.
Why: the ep32 spoon hold is chaotic (any early edit re-rolls the grasp; 0/2880 whole-window draws improved it, rob.py: median of
1-mm perturbations 7.7 cm). A whole-hold search needs one draw to be lucky for 5 s at once. Here the hold is cut into short windows; for
each window and each of B beam parents (actor tables), N child edits (random right-wrist/finger knots + windowed rigid-hold ILC
candidates toward the target) are rolled out from the parent's exact physics state at the cache frame (mj_getState/mj_setState,
mjSTATE_INTEGRATION; RBF leak before the cache frame checked == 0) to window end + look-ahead, and ranked by the CUMULATIVE per-frame
scorer-faithful cost over [0, we + look] (spoon pos/1.3 + deg/15 + cup_w * cup). The top B (incl. unedited parents) form the next beam.
Final: every beam member is re-rolled from t=0 over the full horizon and scored (mlib.per_frame == score_any).
v2 (wbeam2): per-frame cost weights --tol P,D (spoon) and --cup-w; LEFT-wrist cup ILC candidates (axis-only rigid-hold correction toward
the cup target, channels 8-13) next to the right-hand ones; windows may run to the end of the episode (look clipped at T).
usage: wbeam2.py --actor A --meshes M --out O --windows 2.4:2.9,... [--beam 6 --n 64 --look 1.0 --tol 1.3,15 --cup-w 1.0 --lilc 0.3,0.6,1.0]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7/code')
import mres
from learned_feedback_policy import features
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
TOL = [1.3, 15.0]
E = None; CW = 0.5
def init(cfg, cw, tol):
    global E, CW; E = mres.Env(cfg); CW = cw; TOL[:] = tol

def frame_cost(env, Q):
    e = mres.L.per_frame(32, env.poses(Q), env.reg, env.tgt)
    s, c = e['wooden_spoon'], e['blue_cup']
    return s[0] / TOL[0] + s[1] / TOL[1] + CW * (c[0] / TOL[0] + c[1] / TOL[1]), e

def run(env, table, kc, state, Qpre, kmax, want_state_at=None):
    """roll from frame kc (state) to kmax; returns Q[:kmax+1] (and the state at frame want_state_at)."""
    m = env.m; dt = env.dt; env.d['phase_weights'] = env.weights(table); pol = env.pol
    d = mujoco.MjData(m)
    if state is None: mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    else: mujoco.mj_setState(m, d, state, SPEC)
    Q = np.zeros((kmax + 1, m.nq)); Q[:kc + 1] = Qpre[:kc + 1] if Qpre is not None else 0
    if state is None: Q[0] = d.qpos
    st = None
    for k in range(kc + 1, kmax + 1):
        n = round(k / (env.fps * dt)) - round((k - 1) / (env.fps * dt))
        for _ in range(n):
            d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None, None
        Q[k] = d.qpos
        if want_state_at == k:
            st = np.empty(mujoco.mj_stateSize(m, SPEC)); mujoco.mj_getState(m, d, st, SPEC)
    return Q, st

def job_prefix(args):
    table, kc = args; Q, st = run(E, table, 0, None, None, kc, want_state_at=kc); return Q, st

def job_child(args):
    table, kc, state, Qpre, kmax = args
    try:
        Q, _ = run(E, table, kc, state, Qpre, kmax)
        if Q is None: return dict(cum=1e9)
        c, e = frame_cost(E, Q)
        if not np.isfinite(c).all(): return dict(cum=1e9)
        return dict(cum=float(c.sum()), Q=Q, sp=[float(e['wooden_spoon'][0].mean()), float(e['wooden_spoon'][1].mean())], cup=[float(e['blue_cup'][0].mean()), float(e['blue_cup'][1].mean())])
    except Exception as ex: return dict(cum=1e9, error=repr(ex)[:200])

def job_full(table):
    r = E.evaluate(table, keep=True)
    if 'err' not in r: return dict(cost=1e6)
    c, _ = frame_cost(E, r['Q']); return dict(objects=r['objects'], cum=float(c.sum()), Q=r['Q'])

def knots(env, x, a0, a1, K):
    kn = np.linspace(a0, a1, K + 2); P = np.vstack([np.zeros(8), x.reshape(K, 8), np.zeros(8)]); tb = np.zeros((env.T, mres.NCH))
    tb[:, :8] = np.stack([np.interp(env.t, kn, P[:, c]) for c in range(8)], 1); return tb

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor', required=True); ap.add_argument('--meshes', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--windows', required=True); ap.add_argument('--beam', type=int, default=6); ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--look', type=float, default=1.0); ap.add_argument('--pre', type=float, default=0.3); ap.add_argument('--post', type=float, default=0.2); ap.add_argument('--gap', type=float, default=1.6)
    ap.add_argument('--knots', type=int, default=3); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.04)
    ap.add_argument('--scales', default='0.1,0.25,0.5,1.0'); ap.add_argument('--ilc', default='0.15,0.3,0.6,1.0'); ap.add_argument('--ilc-sigmas', default='0.1,0.25')
    ap.add_argument('--cup-w', type=float, default=1.0); ap.add_argument('--tol', default='1.3,15'); ap.add_argument('--lilc', default='0.3,0.6,1.0'); ap.add_argument('--workers', type=int, default=60); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--per-parent', type=int, default=0, help='max survivors per parent (0 = no limit)'); ap.add_argument('--adv', default='1,2,3,4,6'); ap.add_argument('--adv-g', default='0.5,1.0,1.3'); ap.add_argument('--sq', default='0.1,0.2'); ap.add_argument('--roll-alphas', default='0.5,0.8,1.0,1.3'); ap.add_argument('--init', help='npz with tables (beam_wXX.npz) to start the beam from')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=32, actor=str(Path(a.actor).resolve()), meshes=a.meshes, table=a.table); global CW; CW = a.cup_w; TOL[:] = [float(v) for v in a.tol.split(',')]
    env = mres.Env(cfg); t00 = time.monotonic(); rng = np.random.default_rng(a.seed)
    import trimesh; global AX; vv = np.asarray(trimesh.load(Path(a.meshes) / 'wooden_spoon.obj', force='mesh', process=False).vertices); vv = vv - vv.mean(0)
    AX = np.linalg.svd(vv, full_matrices=False)[2][0]; print('SPOON_AXIS_BODY', AX.round(4).tolist(), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    rep = env.evaluate(env.table0, keep=True); Qb = rep['Q']; _, A = mres.L.per_frame(32, env.poses(Qb), env.reg, env.tgt, want_A=True)
    cb, _ = frame_cost(env, Qb); print('INIT', json.dumps(dict(objects=rep['objects'], cum=float(cb.sum()))), flush=True)
    beam = [dict(table=env.table0.copy(), Q=Qb, hist=['base'])]
    if a.init:
        tbs = np.load(a.init)['tables']; beam = []
        for i, tb in enumerate(tbs):
            r = env.evaluate(tb, keep=True); beam.append(dict(table=tb.copy(), Q=r['Q'], hist=[f'init{i}'])); print('INITBEAM', i, json.dumps(r['objects']), flush=True)
    wins = [tuple(float(v) for v in w.split(':')) for w in a.windows.split(',')]
    std1 = np.array([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2); scales = [float(s) for s in a.scales.split(',')]
    ts = np.arange(1, int(round(env.T / (env.fps * env.dt))) + 1) * env.dt; PH = features(ts / env.dur, env.d['centers'], float(env.d['width'])); log = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg, a.cup_w, TOL)) as pool:
        for wi, (w0, w1) in enumerate(wins):
            a0, a1 = w0 - a.pre, w1 + a.post; kc = max(0, int(np.floor((a0 - a.gap) * env.fps))); kmax = min(env.T - 1, int(round((w1 + a.look) * env.fps)))
            nsub = int(round(kc / (env.fps * env.dt))); ph = PH[:nsub]
            pre = list(pool.map(job_prefix, [(b['table'], kc) for b in beam], chunksize=1))
            for b, (Qp, st) in zip(beam, pre):
                if not np.array_equal(Qp[:kc + 1], b['Q'][:kc + 1]): raise SystemExit('prefix mismatch')
                b['state'] = st; b['Qpre'] = Qp
            jobs, meta = [], []
            for pi, b in enumerate(beam):
                dels = [np.zeros((env.T, mres.NCH))]; mm = [dict(kind='zero')]
                for i in range(a.n):
                    x = rng.normal(size=a.knots * 8) * np.tile(std1, a.knots) * scales[i % len(scales)]; dels.append(knots(env, x, a0, a1, a.knots)); mm.append(dict(kind='rand', scale=scales[i % len(scales)]))
                Qf = np.vstack([b['Q'], np.repeat(b['Q'][-1:], env.T - len(b['Q']), 0)]) if len(b['Q']) < env.T else b['Q']
                if a.ilc:
                    Qf = np.vstack([b['Q'], np.repeat(b['Q'][-1:], env.T - len(b['Q']), 0)]) if len(b['Q']) < env.T else b['Q']
                    raw = env.ilc_correction(Qf, A, 'wooden_spoon', 'right', a0, a1, 0.2)
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs = gf(raw, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                        for al in [float(v) for v in a.ilc.split(',')]:
                            for md, sl in (('full', slice(0, 6)), ('pos', slice(0, 3)), ('rot', slice(3, 6))):
                                dl = np.zeros((env.T, mres.NCH)); dl[:, sl] = al * cs[:, sl]; dels.append(dl); mm.append(dict(kind='ilc', sigma=sg, alpha=al, mode=md))
                if a.roll_alphas:
                    rawr = env.ilc_correction(Qf, A, 'wooden_spoon', 'right', a0, a1, 0.2, roll_axis=AX)
                    wsq = mres.sm((env.t - a0) / 0.2) * (1 - mres.sm((env.t - (a1 - 0.2)) / 0.2))
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs = gf(rawr, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                        for al in [float(v) for v in a.roll_alphas.split(',')]:
                            for sq in [0.0] + [float(v) for v in a.sq.split(',') if v]:
                                dl = np.zeros((env.T, mres.NCH)); dl[:, 0:6] = al * cs; dl[:, 6] = sq * wsq; dl[:, 7] = sq * wsq; dels.append(dl); mm.append(dict(kind='roll', sigma=sg, alpha=al, sq=sq))
                    for sq in [float(v) for v in a.sq.split(',') if v]:
                        for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                            cs = gf(raw, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                            dl = np.zeros((env.T, mres.NCH)); dl[:, 0:6] = cs; dl[:, 6] = sq * wsq; dl[:, 7] = sq * wsq; dels.append(dl); mm.append(dict(kind='ilcsq', sigma=sg, alpha=1.0, sq=sq))
                if a.adv:
                    Wp = env.weights(b['table']); wa = env.wa['right']; PHf = features(env.t / env.dur, env.d['centers'], float(env.d['width']))
                    cmd = (PHf @ Wp[:, wa]) * env.scale[wa]; wwin = mres.sm((env.t - a0) / 0.2) * (1 - mres.sm((env.t - (a1 - 0.2)) / 0.2))
                    for sh in [int(v) for v in a.adv.split(',')]:
                        sc = np.vstack([cmd[sh:], np.repeat(cmd[-1:], sh, 0)]) - cmd
                        for g in [float(v) for v in a.adv_g.split(',')]:
                            for md, sl in (('full', slice(0, 6)), ('pos', slice(0, 3))):
                                dl = np.zeros((env.T, mres.NCH)); dl[:, sl] = g * sc[:, sl] * wwin[:, None]; dels.append(dl); mm.append(dict(kind='adv', shift=sh, g=g, mode=md))
                if a.lilc:
                    rawl = env.ilc_correction(Qf, A, 'blue_cup', 'left', a0, a1, 0.2, axis_only=True)
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs = gf(rawl, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                        for al in [float(v) for v in a.lilc.split(',')]:
                            for md, sl in (('full', slice(8, 14)), ('pos', slice(8, 11))):
                                dl = np.zeros((env.T, mres.NCH)); dl[:, sl] = al * cs[:, sl.start - 8:sl.stop - 8]; dels.append(dl); mm.append(dict(kind='lilc', sigma=sg, alpha=al, mode=md))
                for dl, m_ in zip(dels, mm):
                    tb = b['table'] + dl; dW = env.weights(tb) - env.weights(b['table']); rows = np.flatnonzero(np.abs(dW).sum(1) > 0)
                    if len(rows) and nsub > 0 and np.abs(ph[:, rows] @ dW[rows]).max() != 0.0: continue   # leaks before the cache frame -> skip
                    jobs.append((tb, kc, b['state'], b['Qpre'], kmax)); meta.append(dict(m_, parent=pi))
            reps = list(pool.map(job_child, jobs, chunksize=1)); cum = np.array([r['cum'] for r in reps]); order = np.argsort(cum)
            for j in range(len(jobs)):   # exactness: an unedited child continued from the cached state must equal its parent's own rollout
                if meta[j]['kind'] == 'zero' and 'Q' in reps[j]:
                    pq = beam[meta[j]['parent']]['Q']; kk = min(len(pq), kmax + 1)
                    if not np.array_equal(reps[j]['Q'][:kk], pq[:kk]): raise SystemExit(f'state-cache continuation differs from parent rollout (window {w0})')
            nb = []; seen = set(); cnt = {}
            for j in order:
                if cum[j] >= 1e9 or len(nb) >= a.beam: break
                key = round(float(cum[j]), 6)
                if key in seen: continue
                p = meta[j]['parent']
                if a.per_parent and cnt.get(p, 0) >= a.per_parent: continue
                seen.add(key); cnt[p] = cnt.get(p, 0) + 1
                nb.append(dict(table=jobs[j][0], Q=reps[j]['Q'], hist=beam[p]['hist'] + [f"{w0}:{json.dumps(meta[j])}"], cum=float(cum[j]), sp=reps[j]['sp'], cup=reps[j]['cup']))
            zero_cums = [float(cum[j]) for j in range(len(jobs)) if meta[j]['kind'] == 'zero']
            row = dict(window=[w0, w1], kc=kc, kmax=kmax, n=len(jobs), zero=zero_cums, best=[dict(cum=x['cum'], sp=x['sp'], cup=x['cup'], last=x['hist'][-1]) for x in nb],
                       n_better_than_best_zero=int((cum < min(zero_cums)).sum()), elapsed_s=int(time.monotonic() - t00))
            log.append(row); print('WIN', json.dumps(row), flush=True); beam = nb
            np.savez_compressed(out / f'beam_w{wi:02d}.npz', tables=np.stack([b['table'] for b in beam])); (out / 'progress.json').write_text(json.dumps(log))
        fulls = list(pool.map(job_full, [b['table'] for b in beam], chunksize=1))
    res = []
    for i, (b, f) in enumerate(zip(beam, fulls)):
        np.savez_compressed(out / f'final_{i}.npz', table=b['table']); res.append(dict(i=i, objects=f.get('objects'), cum=f.get('cum'), hist=b['hist']))
        print('FINAL', json.dumps(dict(i=i, objects=f.get('objects'), cum=f.get('cum'))), flush=True)
    (out / 'final.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__': main()
