"""[C9 copy of U8 mres: c9lib (target_final_v6, eps 5/33/35/36), chart-aware wrist ILC (R_palm = Ac R_XYZ(q), K8/C7 fix)] Stream M8 (v23) [U8 copy: + ILC mode rotanc = wrist rotation about the achieved object anchor]: physical fine-tuning of a shipped actor's PHASE WEIGHTS (generalization of streamH actor_res.py to any episode,
both hands, and any registered wrapper: LearnedPolicy / PinkyMarginPolicy / ReleasePolicy / MultiReleasePolicy share one data dict).

Residual table (T x 16): per hand [wrist pos xyz (m), wrist rot xyz (rad, joint space), finger flex squeeze, thumb squeeze]; right = 0..7,
left = 8..15. Baked into W'[j,a] = W[j,a] + delta_a(t_j)/scale_a. Feedback weights, margins, wrappers, scene, start state untouched.
Every evaluation = full closed-loop rollout with the evaluator's controller object (ComponentPolicy) and substep schedule, scored
scorer-faithfully vs target_final_v5 (mlib.per_frame, == score_any.sh).
Cost (lower better): per object r = pos/P + deg/D (P,D = 1.2 cm / 12 deg goal); cost = max r + 0.5 sum r + wpk * sum peak (p95 pos/P)
+ 50 * violation of the baseline (any object pos or deg above the --cap values, i.e. the v22b component) .
modes: eval | ilc | cem | save   (ilc: rigid-hold correction of HAND to make OBJ follow its target in [t0,t1])"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
import c9lib as L
from tool_library.common import hand as Hh
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
NCH = 16
HOFF = {'right': 0, 'left': 8}


class Env:
    def __init__(self, cfg):
        self.cfg = cfg; actor = Path(cfg['actor']); ev = json.loads((actor / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.epd = Path(ev['input_dir']); self.ep = ep = cfg['ep']; self.names = L.NAMES[ep]
        self.m = mujoco.MjModel.from_xml_path(self.scene); self.dt = float(self.m.opt.timestep)
        from component_policy import ComponentPolicy
        self.ckpt = cfg.get('ckpt') or str(actor / 'training/policy.npz'); self.pol = ComponentPolicy(self.ckpt); self.d = self.pol.data
        self.W0 = self.d['phase_weights'].copy(); self._freeze(cfg.get('freeze')); self.fps = float(self.d['fps']); self.T = int(self.d['frames']); self.dur = float(self.d['duration_s'])
        self.t = np.arange(self.T) / self.fps; self.tc = self.d['centers'] * self.dur; self.scale = self.d['action_scale']
        names = list(self.d['actuator_names']); self.wa = {}; self.flex = {}; self.wq = {}
        for side in ('right', 'left'):
            self.wa[side] = [names.index(f'robot_{side}_{side}_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
            self.wq[side] = [int(self.m.joint(f'robot_{side}_{side}_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
            rig = Hh.HandRig.build(self.m, f'robot_{side}_', side); fl = []
            for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                for jn in rig.flexion_joints(fi):
                    jid = self.m.joint(f'robot_{side}_{side}_' + jn).id if isinstance(jn, str) else int(jn)
                    hits = np.flatnonzero(self.m.actuator_trnid[:, 0] == jid)
                    if len(hits) == 1: fl.append((fi == 'thumb', int(hits[0])))
            self.flex[side] = fl
            q = self.m.body_quat[self.m.body(f'robot_{side}_{side}_base_roll').id]; self.Ac = getattr(self, 'Ac', {}); self.Ac[side] = R.from_quat([q[1], q[2], q[3], q[0]])
        self.reg = L.register(ep, {n: Path(cfg['meshes']) / f'{n}.obj' for n in self.names}); self.tgt = L.target(ep)
        self.oa = {n: int(self.m.joint(f'{n}_free').qposadr[0]) for n in self.names}
        self.cap = cfg.get('cap') or {}; self.P = cfg.get('P', 1.2); self.D = cfg.get('D', 12.0); self.wpk = cfg.get('wpk', 0.1)
        self.ow = cfg.get('ow') or {}
        self.table0 = np.zeros((self.T, NCH))
        if cfg.get('table'):
            tb = np.load(cfg['table'])['table']; self.table0[:, :tb.shape[1]] = tb
        self.extra = cfg.get('extra') or {}
        for k, v in self.extra.items(): self.d[k] = np.asarray(v, dtype=self.d[k].dtype).reshape(self.d[k].shape)

    def _freeze(self, fr):
        """U8 hold primitive: for SIDE, every phase weight of that hand's actuators whose centre lies after t_h is set to the hand's own
        commanded (phase part) value at t_h, so the hand keeps the grasp/pose it had at t_h (no release); feedback weights unchanged.
        fr = 'side,t_h[,ramp_s]' ; with ramp the weights blend from the original to the frozen value over [t_h, t_h+ramp]."""
        if not fr: return
        parts = fr.split(','); side, th = parts[0], float(parts[1]); rp = float(parts[2]) if len(parts) > 2 else 0.0
        tw = float(parts[3]) if len(parts) > 3 else th   # optional separate freeze time for the 6 wrist actuators (fingers at t_h)
        from learned_feedback_policy import features
        names = [str(x) for x in self.d['actuator_names']]; allc = [i for i, n in enumerate(names) if n.startswith(f'robot_{side}_{side}_')]
        wr = [names.index(f'robot_{side}_{side}_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        dur = float(self.d['duration_s']); tc = self.d['centers'] * dur
        for cols, t_ in (([c for c in allc if c not in wr], th), (wr, tw)):
            phi = features([t_ / dur], self.d['centers'], float(self.d['width']))[0]; v = phi @ self.W0[:, cols]
            w = np.clip((tc - t_) / rp, 0, 1) if rp > 0 else (tc > t_).astype(float)
            self.W0[:, cols] = self.W0[:, cols] * (1 - w[:, None]) + v[None] * w[:, None]
        self.freeze_info = dict(side=side, t_h=th, t_wrist=tw, ramp=rp, n_actuators=len(allc))

    def weights(self, table):
        W = self.W0.copy(); tv = np.stack([np.interp(self.tc, self.t, table[:, c]) for c in range(NCH)], 1)
        for side, o in HOFF.items():
            if not np.any(tv[:, o:o + 8]): continue
            for i, a in enumerate(self.wa[side]): W[:, a] += tv[:, o + i] / self.scale[a]
            for th, a in self.flex[side]: W[:, a] += tv[:, o + (7 if th else 6)] / self.scale[a]
        return W

    def rollout(self, table):
        self.d['phase_weights'] = self.weights(table); m = self.m; pol = self.pol; dt = self.dt
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        Q = np.empty((self.T, m.nq)); Q[0] = d.qpos
        for k in range(1, self.T):
            n = round(k / (self.fps * dt)) - round((k - 1) / (self.fps * dt))
            for _ in range(n):
                d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
            Q[k] = d.qpos
        return Q

    def poses(self, Q):
        P = {}
        for nm, a in self.oa.items():
            T = np.tile(np.eye(4), (len(Q), 1, 1)); T[:, :3, :3] = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = Q[:, a:a + 3]; P[nm] = T
        return P

    def score(self, Q):
        e, A = L.per_frame(self.ep, self.poses(Q), self.reg, self.tgt, want_A=True)
        res = {n: (float(e[n][0].mean()), float(e[n][1].mean())) for n in self.names}
        pk = {n: float(np.percentile(e[n][0], 95)) for n in self.names}
        r = [self.ow.get(n, 1.0) * (p / self.P + d / self.D) for n, (p, d) in res.items()]
        if not np.isfinite(r).all(): return 1e6, res, e, A
        viol = sum(max(0, p - self.cap[n][0]) / 0.1 + max(0, d - self.cap[n][1]) / 1.0 for n, (p, d) in res.items() if n in self.cap)
        goal = self.cfg.get('goal')
        gp = sum(max(0, p - goal[0]) / 0.05 + max(0, d - goal[1]) / 0.5 for p, d in res.values()) if goal else 0.0
        c = max(r) + 0.5 * sum(r) + self.wpk * sum(pk[n] / self.P for n in self.names) + 50 * viol + gp
        return float(c), res, e, A

    def evaluate(self, table, keep=False):
        try:
            Q = self.rollout(table)
            if Q is None: return dict(cost=1e6, unstable=True)
            c, res, e, A = self.score(Q); out = dict(cost=c, objects=res)
            if keep: out['Q'] = Q; out['err'] = e; out['A'] = A
            return out
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])

    def ilc_correction(self, Q, A, obj, side, t0, t1, ramp=0.3, axis_only=False):
        """rigid-hold wrist correction (T x 6) of SIDE so that OBJ moves to its target (physics world), windowed."""
        corr = np.zeros((self.T, 6)); corr_ra = np.zeros((self.T, 6)); a = self.oa[obj]; wq = self.wq[side]
        w = sm((self.t - t0) / ramp) * (1 - sm((self.t - (t1 - ramp)) / ramp)) if t1 < self.t[-1] else sm((self.t - t0) / ramp)
        tp = L.target_phys(self.ep, self.reg, A, self.tgt)[obj]
        Ra = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True); pa = Q[:, a:a + 3]
        ca = self.reg[obj][0]
        Rt = R.from_matrix(tp[:self.T, :3, :3]); pt = tp[:self.T, :3, 3]
        ax = L.geometry(self.ep)[obj]['axis']; rot = self.reg[obj][1]
        Rw = R.from_euler('XYZ', Q[:, wq[3:]]); pw = Q[:, wq[:3]]
        for k in range(self.T):
            if w[k] <= 0: continue
            if axis_only:   # symmetric object: minimal rotation taking the achieved symmetry axis to the target one (axis in OUR body = rot^T-converted)
                axb = rot.T @ ax if np.any(ax) else np.zeros(3)
                u = Ra[k].apply(axb); v = Rt[k].apply(axb)
                cr = np.cross(u, v); s = np.linalg.norm(cr)
                D = R.from_rotvec(cr / s * np.arctan2(s, np.dot(u, v))) if s > 1e-9 else R.identity()
            else:
                D = Rt[k] * Ra[k].inv()
            # move the registered anchor point (not the body origin) onto its target
            qa = pa[k] + Ra[k].apply(ca); qt = pt[k] + Rt[k].apply(ca)
            pn = qt + D.apply(pw[k] - qa); Ac = self.Ac[side]; Rn = Ac.inv() * D * Ac * Rw[k]
            eu = Rn.as_euler('XYZ'); eu = Q[k, wq[3:]] + (eu - Q[k, wq[3:]] + np.pi) % (2 * np.pi) - np.pi
            corr[k, :3] = (pn - pw[k]) * w[k]; corr[k, 3:] = (eu - Q[k, wq[3:]]) * w[k]
            # U8: rotation about the achieved anchor (anchor position kept): rotate wrist by D about qa
            pn2 = qa + D.apply(pw[k] - qa); corr_ra[k, :3] = (pn2 - pw[k]) * w[k]; corr_ra[k, 3:] = corr[k, 3:]
        self.last_corr_ra = corr_ra
        return corr


W = None
def init(cfg):
    global W; W = Env(cfg)
def ev(table): return W.evaluate(table)
def ev_win(args):
    table, tw = args; r = W.evaluate(table, keep=True)
    if 'err' not in r: return dict(cost=r['cost'])
    k = int(round(tw * W.fps)); e = r['err']
    return dict(cost=r['cost'], objects=r['objects'], win={n: (float(e[n][0][:k].mean()), float(e[n][1][:k].mean())) for n in W.names},
                end={n: float(e[n][0][k - 5:k].mean()) for n in W.names})


def knot_table(env, x, t0, t1, K, side):
    kn = np.linspace(t0, t1, K + 2); P = np.vstack([np.zeros(8), x.reshape(K, 8), np.zeros(8)]); tb = np.zeros((env.T, NCH)); o = HOFF[side]
    tb[:, o:o + 8] = np.stack([np.interp(env.t, kn, P[:, c]) for c in range(8)], 1); return tb


def save_actor(env, table, out, info):
    out = Path(out); (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'
    with np.load(env.ckpt, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    for k in env.extra: d[k] = env.d[k]
    d['phase_weights'] = env.weights(table); np.savez_compressed(ck, **d); np.savez_compressed(out / 'residual_table.npz', table=table)
    from learned_feedback_policy import sha
    src = Path(env.ckpt).parent / 'training_report.json'
    tr = json.loads(src.read_text()) if src.exists() else {}
    tr.update(phase_residual=dict(method='stream M8: derivative-free physical fine-tuning (ILC / CEM over full-horizon closed-loop rollouts of this actor) of a time-local residual on the wrist and finger phase weights; feedback weights, margins, wrappers, scene unchanged',
              source_checkpoint=env.ckpt, source_checkpoint_sha256=sha(env.ckpt), trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), info=info), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from evaluate_policy_component import evaluate as ev_final
    ev_final(env.ep, env.scene, str(ck), str(out / 'evaluation'), str(env.epd))
    q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']; c, res, _, _ = env.score(q)
    r = json.loads((out / 'evaluation/result.json').read_text())
    print('FINAL', json.dumps(dict(official_eval_scorer=res, cost=c, proxy_mp_sr=r['metrics'].get('reference_proxy_mp_sr'), physics_verified=r['execution'].get('physics_verified'))), flush=True)
    return res


def windows(env, e):
    for s0 in range(0, env.T, 10): print(f'   {s0/env.fps:5.1f}s', ' '.join(f"{e[n][0][s0:s0+10].mean():5.2f}/{e[n][1][s0:s0+10].mean():5.1f}" for n in env.names), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['ilc', 'cem', 'eval', 'save', 'lot'])
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--ckpt'); ap.add_argument('--meshes', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--table'); ap.add_argument('--cap', default='{}', help='json n->[cm,deg]'); ap.add_argument('--extra', default='{}')
    ap.add_argument('--obj', default='wooden_spoon'); ap.add_argument('--side', default='right'); ap.add_argument('--axis-only', action='store_true')
    ap.add_argument('--t0', type=float, default=4.0); ap.add_argument('--t1', type=float, default=9.0); ap.add_argument('--ramp', type=float, default=0.3); ap.add_argument('--iterations', type=int, default=6)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--samples', type=int, default=64); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.03)
    ap.add_argument('--alphas', default='0.15,0.3,0.5,0.7,1.0'); ap.add_argument('--sigmas', default='0.1,0.3'); ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--wpk', type=float, default=0.1); ap.add_argument('--ow', default='{}'); ap.add_argument('--P', type=float, default=1.2); ap.add_argument('--D', type=float, default=12.0); ap.add_argument('--goal', help='cm,deg hinge per object'); ap.add_argument('--ilc-modes', default='full,pos,rot,rotanc'); ap.add_argument('--freeze', help="side,t_h[,ramp]: hold primitive (no release) for that hand after t_h")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), ckpt=a.ckpt, meshes=a.meshes, table=a.table, cap=json.loads(a.cap), extra=json.loads(a.extra),
               wpk=a.wpk, ow=json.loads(a.ow), P=a.P, D=a.D, goal=[float(v) for v in a.goal.split(',')] if a.goal else None, freeze=a.freeze)
    T00 = time.monotonic(); env = Env(cfg); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep.get('objects'), rollout_s=time.monotonic() - T00)), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1)); hist = []; t0 = time.monotonic()
    if a.mode == 'save':
        save_actor(env, table, out / 'actor', dict(mode='save', table=a.table, extra=cfg['extra'], search_eval=dict(cost=rep['cost'], objects=rep.get('objects')))); return
    if a.mode == 'eval':
        windows(env, rep['err']); np.savez_compressed(out / 'eval.npz', Q=rep['Q'], **{f'{n}_pos': rep['err'][n][0] for n in env.names}, **{f'{n}_deg': rep['err'][n][1] for n in env.names}); return
    best_table = table.copy(); best = dict(cost=rep['cost'], objects=rep['objects'])
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        if a.mode == 'lot':
            # lottery: independent random knot residuals in [t0,t1]; logs per-sample prefix-window errors (frames < t1) to samples.jsonl
            K = a.knots; std = np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2, K); rng = np.random.default_rng(a.seed)
            k = int(round(a.t1 * env.fps)); e = rep['err']; bw = {n: (float(e[n][0][:k].mean()), float(e[n][1][:k].mean())) for n in env.names}
            print('BASEWIN', json.dumps(bw), flush=True); f = open(out / 'samples.jsonl', 'a')
            for it in range(a.iterations):
                xs = rng.normal(size=(a.samples, K * 8)) * std
                reps = list(pool.map(ev_win, [(table + knot_table(env, x, a.t0, a.t1, K, a.side), a.t1) for x in xs], chunksize=1))
                for x, r in zip(xs, reps): f.write(json.dumps(dict(x=x.tolist(), **r)) + '\n')
                f.flush(); ok = [r for r in reps if 'win' in r]
                bb = min(ok, key=lambda r: sum(p / a.P + d / a.D for p, d in r['win'].values())) if ok else None
                print('IT', json.dumps(dict(iteration=it, best_win=bb and bb['win'], best_end=bb and bb['end'])), flush=True)
            return
        if a.mode == 'ilc':
            Q = rep['Q']; A = rep['A']; cur = table.copy(); curcost = rep['cost']; o = HOFF[a.side]
            alphas = [float(x) for x in a.alphas.split(',')]; sigmas = [float(x) for x in a.sigmas.split(',')]
            for it in range(a.iterations):
                corr = env.ilc_correction(Q, A, a.obj, a.side, a.t0, a.t1, a.ramp, a.axis_only); cra = env.last_corr_ra; cands = []; meta = []
                for s in sigmas:
                    cs = gaussian_filter1d(corr, s * env.fps, axis=0, mode='nearest') if s > 0 else corr
                    cr = gaussian_filter1d(cra, s * env.fps, axis=0, mode='nearest') if s > 0 else cra
                    for al in alphas:
                        for mode_ in a.ilc_modes.split(','):
                            dlt = np.zeros((env.T, NCH))
                            if mode_ == 'rotanc': dlt[:, o:o + 6] = al * cr
                            else:
                                m_ = {'full': slice(0, 6), 'pos': slice(0, 3), 'rot': slice(3, 6)}[mode_]; dlt[:, o:o + 6][:, m_] = al * cs[:, m_]
                            cands.append(cur + dlt); meta.append(dict(alpha=al, sigma=s, mode=mode_))
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); i = int(order[0])
                row = dict(iteration=it, best_candidate=meta[i], cost=reps[i]['cost'], objects=reps[i].get('objects'), prev_cost=curcost,
                           top=[dict(meta[j], cost=reps[j]['cost'], objects=reps[j].get('objects')) for j in order[:4]], elapsed_s=time.monotonic() - t0)
                hist.append(row); print('IT', json.dumps(row), flush=True)
                if reps[i]['cost'] < curcost:
                    cur = cands[i]; curcost = reps[i]['cost']; r2 = env.evaluate(cur, keep=True); Q = r2['Q']; A = r2['A']
                    best = dict(cost=curcost, objects=reps[i]['objects']); best_table = cur.copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                else:
                    alphas = [x * 0.5 for x in alphas]
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best)))
        else:
            K = a.knots; base = np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2, K); std = base.copy(); mean = np.zeros(K * 8); bx = mean.copy()
            rng = np.random.default_rng(a.seed); bound = 4 * base
            for it in range(a.iterations):
                xs = np.clip(np.vstack([bx, mean, mean + rng.normal(size=(a.samples - 2, K * 8)) * std]), -bound, bound)
                cands = [table + knot_table(env, x, a.t0, a.t1, K, a.side) for x in xs]
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = xs[order[:max(6, a.samples // 8)]]
                if reps[order[0]]['cost'] < best['cost']:
                    bx = xs[order[0]].copy(); best = dict(cost=reps[order[0]]['cost'], objects=reps[order[0]]['objects']); best_table = cands[order[0]].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * base)
                row = dict(iteration=it, best=best, median=float(np.median([r['cost'] for r in reps])), elapsed_s=time.monotonic() - t0); hist.append(row); print('IT', json.dumps(row), flush=True)
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best, best_x=bx.tolist())))
    np.savez_compressed(out / 'best_table.npz', table=best_table)
    if not a.no_save and best['cost'] < rep['cost']:
        save_actor(env, best_table, out / 'actor', dict(mode=a.mode, args=vars(a), search_best=best))


if __name__ == '__main__': main()
