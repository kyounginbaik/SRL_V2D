"""[R33 wbeamHGs (11/03): + --shinge w,acc_deg,inpalm_deg,f0,f1 = smoothness hinges: w*max(0,|d omega_spoon|-acc)/tol_deg per frame and
w*max(0,spoon-in-right-wrist rotation change-inpalm)/tol_deg per frame on held frames f0..f1] [E31 copy: scoring target = $E31_TGT (target_E14_HGA / target_E14_HGB)] [H13b copy of H10 code3/wbeam9.py: mresH (target_final_v12b_g15); + --adv fractional time-advance candidates on one/both wrists (H12 wbeam4 generalised)] [H10 copy of D9 wbeam9: scorer target = target_final_v8_g10b via the H10 mlib9 copy; + H9 roll-only twist ILC candidates about the mesh PCA long axis
(--roll obj:side, --roll-alphas, with optional finger squeeze --sq) and ILC+squeeze candidates; --init starts the beam from saved tables] Stream D9 (phase 9): window BEAM search with exact cached physics states on a shipped actor's phase weights, for any episode / both hands.
Generalisation of streamV8 wbeam2.py / streamU8 wbeam_u8.py:
- every object has its own hand (--hands obj:side,...) and gets rigid-hold ILC candidates (full/pos/rot/rotanc) on THAT hand's wrist channels;
- random knot children are drawn on each hand listed in --rand-sides (split evenly);
- per-frame cost (scorer-faithful vs target_final_v6, mlib9.per_frame) = sum_o w_o (pos_o/P + deg_o/D) + mppe_w * MPPE_frame/P,
  ranked CUMULATIVELY over [0, window end + look]; top B (max --per-parent per parent) form the next beam;
- children are rolled out from the parent's exact cached MuJoCo state (mjSTATE_INTEGRATION); edits that leak before the cache frame are
  skipped; an unedited child must equal its parent bit for bit (checked).
Final: every beam member re-rolled from t=0 over the full horizon (mres9.Env.evaluate == official evaluator rollout) and scored.
usage: wbeam9.py --ep 3 --actor A --meshes M --out O --windows 2.0:2.5,... --hands wooden_spoon:right,white_pot:left [--table T.npz]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as mres
from learned_feedback_policy import features
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
E = None; C = dict(tol=[1.3, 15.0], ow={}, mw=0.0, hw=0.0, hp=1.5)


def init(cfg, c):
    global E; E = mres.Env(cfg); C.update(c)


def frame_cost(env, Q):
    e = mres.L.per_frame(env.ep, env.poses(Q), env.reg, env.tgt, want_mppe=C['mw'] > 0)
    c = sum(C['ow'].get(n, 1.0) * (e[n][0] / C['tol'][0] + e[n][1] / C['tol'][1]) for n in env.names)
    if C.get('hw', 0) > 0: c = c + sum(C['hw'] * np.maximum(0, e[n][0] - C['hp']) / C['tol'][0] for n in env.names)   # H10: grasp-loss hinge on per-frame position
    if C.get('dw', 0) > 0: c = c + sum(C['dw'] * np.maximum(0, e[n][1] - C['dp']) / C['tol'][1] for n in env.names)   # K33-HG: per-frame rotation hinge (punishes a spin, not the mean)
    if C['mw'] > 0: c = c + C['mw'] * e['_mppe'] / C['tol'][0]
    if C.get('sw', 0) > 0:   # R33: smoothness (spoon angular-velocity change; spoon turning in the right palm while held)
        from scipy.spatial.transform import Rotation as _R
        a = env.oa['wooden_spoon']; Rq = _R.from_quat(Q[:, a + 3:a + 7], scalar_first=True); n = len(Q)
        w = (Rq[1:] * Rq[:-1].inv()).as_rotvec(); dw = np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=1))
        pen = np.zeros(n); pen[2:] = C['sw'] * np.maximum(0, dw - C['sa']) / C['tol'][1]
        wq = env.wq['right']; Rw = _R.from_euler('XYZ', Q[:, wq[3:]]); rel = Rw.inv() * Rq; dr = np.degrees((rel[1:] * rel[:-1].inv()).magnitude())
        held = np.zeros(n, bool); held[int(C['sf0']):int(C['sf1']) + 1] = True
        pen[1:] += np.where(held[1:], C['sw'] * np.maximum(0, dr - C['sr']) / C['tol'][1], 0.0)
        c = c + pen
    return c, e


def summ(env, e):
    o = {n: [float(e[n][0].mean()), float(e[n][1].mean())] for n in env.names}
    if '_mppe' in e: o['mppe'] = float(e['_mppe'].mean())
    o['_peak'] = [float(e['wooden_spoon'][1].max()), float(e['wooden_spoon'][1].argmax())] if 'wooden_spoon' in e else [0.0, 0.0]
    return o


def run(env, table, kc, state, Qpre, kmax, want_state_at=None):
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
    table, kc = args; return run(E, table, 0, None, None, kc, want_state_at=kc)


def job_child(args):
    table, kc, state, Qpre, kmax = args
    try:
        Q, _ = run(E, table, kc, state, Qpre, kmax)
        if Q is None: return dict(cum=1e9)
        c, e = frame_cost(E, Q)
        if not np.isfinite(c).all(): return dict(cum=1e9)
        return dict(cum=float(c.sum()), Q=Q, s=summ(E, e))
    except Exception as ex: return dict(cum=1e9, error=repr(ex)[:200])


def job_full(table):
    r = E.evaluate(table, keep=True)
    if 'err' not in r: return dict(cum=1e9)
    c, e = frame_cost(E, r["Q"])
    e2 = mres.L.per_frame(E.ep, E.poses(r['Q']), E.reg, E.tgt, want_mppe=True)
    return dict(objects=r['objects'], cum=float(c.sum()), mppe=float(e2['_mppe'].mean()), Q=r['Q'])


def knots(env, x, a0, a1, K, side):
    kn = np.linspace(a0, a1, K + 2); P = np.vstack([np.zeros(8), x.reshape(K, 8), np.zeros(8)]); tb = np.zeros((env.T, mres.NCH)); o = mres.HOFF[side]
    tb[:, o:o + 8] = np.stack([np.interp(env.t, kn, P[:, c]) for c in range(8)], 1); return tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshes', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--freeze'); ap.add_argument('--windows', required=True)
    ap.add_argument('--beam', type=int, default=4); ap.add_argument('--n', type=int, default=40); ap.add_argument('--per-parent', type=int, default=2)
    ap.add_argument('--look', type=float, default=1.0); ap.add_argument('--pre', type=float, default=0.3); ap.add_argument('--post', type=float, default=0.2); ap.add_argument('--gap', type=float, default=1.6)
    ap.add_argument('--knots', type=int, default=3); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.04)
    ap.add_argument('--scales', default='0.1,0.25,0.5,1.0'); ap.add_argument('--rand-sides', default='right')
    ap.add_argument('--hands', required=True, help='obj:side,... (ILC candidates per object on that hand; side "none" = no ILC)')
    ap.add_argument('--ilc', default='0.15,0.3,0.6,1.0'); ap.add_argument('--ilc-sigmas', default='0.1,0.25'); ap.add_argument('--ilc-modes', default='full,pos,rot,rotanc')
    ap.add_argument('--tol', default='1.3,15'); ap.add_argument('--ow', default='{}'); ap.add_argument('--mppe-w', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=48); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--hinge', default='0,1.5', help='w,cm: extra w*max(0,pos-cm)/P per frame and object'); ap.add_argument('--roll', default='', help='obj:side,... roll-only (twist about the mesh PCA long axis) ILC candidates'); ap.add_argument('--roll-alphas', default='0.5,0.8,1.0,1.3'); ap.add_argument('--sq', default='0.1,0.2')
    ap.add_argument('--adv', default='', help='H13b: fractional time-ADVANCE candidates (frames), wrist phase command of --adv-sides shifted earlier inside the window'); ap.add_argument('--adv-g', default='0.5,1.0'); ap.add_argument('--adv-sides', default='right,left,both')
    ap.add_argument('--shinge', default='0,3,4,55,300', help='R33 w,acc_deg,inpalm_deg,f0,f1 smoothness hinges (0 = off)')
    ap.add_argument('--dhinge', default='0,30', help='K33-HG w,deg: extra w*max(0,deg-D)/tol_deg per frame and object')
    ap.add_argument('--init', help='npz with tables (beam_wXX.npz) to start the beam from (added to --table)')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshes=a.meshes, table=a.table, freeze=a.freeze)
    c = dict(tol=[float(v) for v in a.tol.split(',')], ow=json.loads(a.ow), mw=a.mppe_w, hw=float(a.hinge.split(',')[0]), hp=float(a.hinge.split(',')[1]), dw=float(a.dhinge.split(',')[0]), dp=float(a.dhinge.split(',')[1]), **dict(zip(('sw', 'sa', 'sr', 'sf0', 'sf1'), map(float, a.shinge.split(','))))); C.update(c)
    hands = [tuple(h.split(':')) for h in a.hands.split(',')]; rsides = [s for s in a.rand_sides.split(',') if s]
    env = mres.Env(cfg); t00 = time.monotonic(); rng = np.random.default_rng(a.seed)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1))
    rep = env.evaluate(env.table0, keep=True); Qb = rep['Q']; _, A = mres.L.per_frame(a.ep, env.poses(Qb), env.reg, env.tgt, want_A=True)
    cb, eb = frame_cost(env, Qb); print('INIT', json.dumps(dict(objects=rep['objects'], s=summ(env, eb), cum=float(cb.sum()))), flush=True)
    beam = [dict(table=env.table0.copy(), Q=Qb, hist=['base'])]
    if a.init:
        tbs = np.load(a.init)['tables']; beam = []
        for i, tb in enumerate(tbs):
            r = env.evaluate(tb, keep=True); beam.append(dict(table=tb.copy(), Q=r['Q'], hist=[f'init{i}'])); print('INITBEAM', i, json.dumps(r['objects']), flush=True)
    import trimesh; AXS = {}
    rolls = [tuple(h.split(':')) for h in a.roll.split(',') if h]
    for obj, _ in rolls:
        vv = np.asarray(trimesh.load(Path(a.meshes) / f'{obj}.obj', force='mesh', process=False).vertices); vv = vv - vv.mean(0); AXS[obj] = np.linalg.svd(vv, full_matrices=False)[2][0]
        print('ROLL_AXIS_BODY', obj, AXS[obj].round(4).tolist(), flush=True)
    wins = [tuple(float(v) for v in w.split(':')) for w in a.windows.split(',')]
    std1 = np.array([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2); scales = [float(s) for s in a.scales.split(',')]
    ts = np.arange(1, int(round(env.T / (env.fps * env.dt))) + 1) * env.dt; PH = features(ts / env.dur, env.d['centers'], float(env.d['width'])); log = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg, c)) as pool:
        for wi, (w0, w1) in enumerate(wins):
            a0, a1 = w0 - a.pre, w1 + a.post; kc = max(1, int(np.floor((a0 - a.gap) * env.fps))); kmax = min(env.T - 1, int(round((w1 + a.look) * env.fps)))
            nsub = int(round(kc / (env.fps * env.dt))); ph = PH[:nsub]
            pre = list(pool.map(job_prefix, [(b['table'], kc) for b in beam], chunksize=1))
            for b, (Qp, st) in zip(beam, pre):
                if not np.array_equal(Qp[:kc + 1], b['Q'][:kc + 1]): raise SystemExit('prefix mismatch')
                b['state'] = st; b['Qpre'] = Qp
            jobs, meta = [], []
            for pi, b in enumerate(beam):
                dels = [np.zeros((env.T, mres.NCH))]; mm = [dict(kind='zero')]
                for i in range(a.n):
                    sd = rsides[i % len(rsides)]; sc = scales[(i // len(rsides)) % len(scales)]
                    x = rng.normal(size=a.knots * 8) * np.tile(std1, a.knots) * sc; dels.append(knots(env, x, a0, a1, a.knots, sd)); mm.append(dict(kind='rand', side=sd, scale=sc))
                Qf = np.vstack([b['Q'], np.repeat(b['Q'][-1:], env.T - len(b['Q']), 0)]) if len(b['Q']) < env.T else b['Q']
                for obj, side in hands:
                    if side == 'none' or not a.ilc: continue
                    o = mres.HOFF[side]
                    raw = env.ilc_correction(Qf, A, obj, side, a0, a1, 0.2); rawa = env.last_corr_ra
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs = gf(raw, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                        ca = gf(rawa, sg * env.fps, axis=0, mode='nearest'); ca[env.t < a0] = 0
                        for al in [float(v) for v in a.ilc.split(',')]:
                            for md in a.ilc_modes.split(','):
                                src = ca if md == 'rotanc' else cs; sl = {'full': slice(0, 6), 'pos': slice(0, 3), 'rot': slice(3, 6), 'rotanc': slice(0, 6)}[md]
                                dl = np.zeros((env.T, mres.NCH)); dl[:, o:o + 6][:, sl] = al * src[:, sl]; dels.append(dl); mm.append(dict(kind='ilc', obj=obj, side=side, sigma=sg, alpha=al, mode=md))
                for obj, side in rolls:
                    o = mres.HOFF[side]; wsq = mres.sm((env.t - a0) / 0.2) * (1 - mres.sm((env.t - (a1 - 0.2)) / 0.2))
                    rawr = env.ilc_correction(Qf, A, obj, side, a0, a1, 0.2, roll_axis=AXS[obj])
                    for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                        cs = gf(rawr, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                        for al in [float(v) for v in a.roll_alphas.split(',') if v]:
                            for sq in [0.0] + [float(v) for v in a.sq.split(',') if v]:
                                dl = np.zeros((env.T, mres.NCH)); dl[:, o:o + 6] = al * cs; dl[:, o + 6] = sq * wsq; dl[:, o + 7] = sq * wsq; dels.append(dl); mm.append(dict(kind='roll', obj=obj, sigma=sg, alpha=al, sq=sq))
                    raw = env.ilc_correction(Qf, A, obj, side, a0, a1, 0.2)
                    for sq in [float(v) for v in a.sq.split(',') if v]:
                        for sg in [float(v) for v in a.ilc_sigmas.split(',')]:
                            cs = gf(raw, sg * env.fps, axis=0, mode='nearest'); cs[env.t < a0] = 0
                            dl = np.zeros((env.T, mres.NCH)); dl[:, o:o + 6] = cs; dl[:, o + 6] = sq * wsq; dl[:, o + 7] = sq * wsq; dels.append(dl); mm.append(dict(kind='ilcsq', obj=obj, sigma=sg, alpha=1.0, sq=sq))
                if a.adv:   # H13b (H12 wbeam4 generalised): delta = g * (c(t+s) - c(t)) on the wrist channels of one or both hands, windowed to [a0, a1]
                    Wp = env.weights(b['table']); PHf = features(env.t / env.dur, env.d['centers'], float(env.d['width'])); wwin = mres.sm((env.t - a0) / 0.2) * (1 - mres.sm((env.t - (a1 - 0.2)) / 0.2))
                    SC = {}
                    for sd in ('right', 'left'):
                        wa = env.wa[sd]; cmd = (PHf @ Wp[:, wa]) * env.scale[wa]
                        for sh in [float(v) for v in a.adv.split(',')]:
                            tt = np.clip(env.t + sh / env.fps, 0, env.t[-1]); SC[sd, sh] = np.stack([np.interp(tt, env.t, cmd[:, c]) for c in range(6)], 1) - cmd
                    for sds in a.adv_sides.split(','):
                        sides = ('right', 'left') if sds == 'both' else (sds,)
                        for sh in [float(v) for v in a.adv.split(',')]:
                            for g in [float(v) for v in a.adv_g.split(',')]:
                                for md, sl in (('full', slice(0, 6)), ('pos', slice(0, 3))):
                                    dl = np.zeros((env.T, mres.NCH))
                                    for sd in sides: o = mres.HOFF[sd]; dl[:, o:o + 6][:, sl] = g * SC[sd, sh][:, sl] * wwin[:, None]
                                    dels.append(dl); mm.append(dict(kind='adv', sides=sds, shift=sh, g=g, mode=md))
                for dl, m_ in zip(dels, mm):
                    tb = b['table'] + dl; dW = env.weights(tb) - env.weights(b['table']); rows = np.flatnonzero(np.abs(dW).sum(1) > 0)
                    if len(rows) and np.abs(ph[:, rows] @ dW[rows]).max() != 0.0: continue
                    jobs.append((tb, kc, b['state'], b['Qpre'], kmax)); meta.append(dict(m_, parent=pi))
            reps = list(pool.map(job_child, jobs, chunksize=1)); cum = np.array([r['cum'] for r in reps]); order = np.argsort(cum)
            for j in range(len(jobs)):
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
                nb.append(dict(table=jobs[j][0], Q=reps[j]['Q'], hist=beam[p]['hist'] + [f"{w0}:{json.dumps(meta[j])}"], cum=float(cum[j]), s=reps[j]['s']))
            zero_cums = [float(cum[j]) for j in range(len(jobs)) if meta[j]['kind'] == 'zero']
            row = dict(window=[w0, w1], kc=kc, kmax=kmax, n=len(jobs), zero=zero_cums, best=[dict(cum=x['cum'], s=x['s'], last=x['hist'][-1]) for x in nb],
                       n_better_than_best_zero=int((cum < min(zero_cums)).sum()), elapsed_s=int(time.monotonic() - t00))
            log.append(row); print('WIN', json.dumps(row), flush=True); beam = nb
            np.savez_compressed(out / f'beam_w{wi:02d}.npz', tables=np.stack([b['table'] for b in beam])); (out / 'progress.json').write_text(json.dumps(log))
        fulls = list(pool.map(job_full, [b['table'] for b in beam], chunksize=1))
    res = []
    for i, (b, f) in enumerate(zip(beam, fulls)):
        np.savez_compressed(out / f'final_{i}.npz', table=b['table']); res.append(dict(i=i, objects=f.get('objects'), mppe=f.get('mppe'), cum=f.get('cum'), hist=b['hist']))
        print('FINAL', json.dumps(dict(i=i, objects=f.get('objects'), mppe=f.get('mppe'), cum=f.get('cum'))), flush=True)
    (out / 'final.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__': main()
