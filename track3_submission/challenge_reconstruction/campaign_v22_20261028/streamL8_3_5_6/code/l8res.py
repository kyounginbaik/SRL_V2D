"""Stream L8 (v23, eps 3/5/6 pot + spoon): physical fine-tuning of a distilled actor's PHASE WEIGHTS on its own closed-loop rollout.
Generalisation of streamH_ep32/code/actor_res.py (same method) to the pot+spoon legacy scenes:
  - per-frame residual table (T x 8) on the right wrist (pos xyz m, rot xyz rad in joint space) + right finger flex / thumb squeeze,
    baked into the LearnedPolicy phase weights W'[j,a] = W[j,a] + delta_a(t_j)/scale_a. Feedback weights, scene, start state untouched.
  - every evaluation = closed-loop rollout with the evaluator substep schedule (causal inference before mj_step), scored scorer-
    faithfully vs target_final_v5 (plib.per_frame = streamB blib with NAMES = white_pot, wooden_spoon; MP-SR registration of the
    shipped meshes + anchor frame-0 alignment).
  - prefix cache: the rollout up to (window start - 1 s) is computed once and continued with mj_copyData; the RBF features of the
    modified centres underflow to exactly 0 there, so commands before the cache frame are bit-identical (checked at start-up:
    max |phi @ (W'-W0)| before the cache time == 0 for a probe residual, and the cached prefix == the base rollout).
  - cost: per object r = max(pos/1.2, deg/12); cost = max(r) + 0.25 sum(r) + peak term + penalty for any object worse than the base
    (pos or deg) -> strictly-better search.
modes: eval / ilc / cem / rand / save (as actor_res.py)."""
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
import plib as B
from tool_library.common import tracking as TR, hand as Hh
from learned_feedback_policy import features
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
NCH = 14   # 0-5 right wrist, 6 right finger squeeze, 7 right thumb squeeze, 8-13 left wrist (pos xyz m, rot xyz rad)
LEFT = slice(8, 14)
TOL = (1.3, 15.0)


class Env:
    def __init__(self, cfg):
        self.cfg = cfg; actor = Path(cfg['actor']); ev = json.loads((actor / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.epd = Path(ev['input_dir']); self.ep = cfg['ep']
        self.m = mujoco.MjModel.from_xml_path(self.scene); self.dt = float(self.m.opt.timestep)
        from component_policy import ComponentPolicy
        self.ckpt = cfg.get('ckpt') or str(actor / 'training/policy.npz'); self.pol = ComponentPolicy(self.ckpt); self.d = self.pol.data
        self.W0 = self.d['phase_weights'].copy(); self.fps = float(self.d['fps']); self.T = int(self.d['frames']); self.dur = float(self.d['duration_s'])
        self.t = np.arange(self.T) / self.fps; self.tc = self.d['centers'] * self.dur
        names = list(self.d['actuator_names']); self.wa = [names.index(f'robot_right_right_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.wal = [names.index(f'robot_left_left_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.scale = self.d['action_scale']
        rig = Hh.HandRig.build(self.m, 'robot_right_', 'right'); self.flex = []
        for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            for jn in rig.flexion_joints(fi):
                jid = self.m.joint('robot_right_right_' + jn).id if isinstance(jn, str) else int(jn)
                hits = np.flatnonzero(self.m.actuator_trnid[:, 0] == jid)
                if len(hits) == 1: self.flex.append((fi == 'thumb', int(hits[0])))
        self.reg = B.register(self.ep, {nm: Path(cfg['meshdir']) / f'{nm}.obj' for nm in B.NAMES}); self.tgt = B.target(self.ep)
        self.adr = {n: int(self.m.joint(n + '_free').qposadr[0]) for n in B.NAMES}; self.sc = self.adr['wooden_spoon']
        self.wq = [int(self.m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.wql = [int(self.m.joint(f'robot_left_left_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.qt = TR.load_reference(self.epd / 'tracking_reference.npz', self.m)['qpos']
        self.table0 = np.zeros((self.T, NCH))
        if cfg.get('table'):
            t_ = np.load(cfg['table'])['table']; self.table0[:, :t_.shape[1]] = t_
        self.bias = np.load(cfg['bias'])['bias'] if cfg.get('bias') else None   # optional per-frame actuator bias (T x nu, actuator units), e.g. left-hand clearance
        self.base = cfg.get('base')            # dict n -> (pos, deg) of the v22b component (strictly-better constraint)
        self.k_pre = max(0, int(np.floor((cfg.get('cache_t', 0.0)) * self.fps + 1e-9))); self.pre = None

    def weights(self, table):
        W = self.W0.copy(); tv = np.stack([np.interp(self.tc, self.t, table[:, c]) for c in range(NCH)], 1)
        for i, a in enumerate(self.wa): W[:, a] += tv[:, i] / self.scale[a]
        for th, a in self.flex: W[:, a] += tv[:, 7 if th else 6] / self.scale[a]
        for i, a in enumerate(self.wal): W[:, a] += tv[:, 8 + i] / self.scale[a]
        if self.bias is not None:
            for a in np.flatnonzero(np.abs(self.bias).sum(0) > 0): W[:, a] += np.interp(self.tc, self.t, self.bias[:, a]) / self.scale[a]
        return W

    def leak(self, table):
        """max |phi(t) @ (W' - W0)| over the substep times before the cache frame (must be exactly 0)."""
        if self.k_pre == 0: return 0.0
        dW = self.weights(table) - self.weights(self.table0)
        if getattr(self, '_ph', None) is None:
            ts = np.arange(1, int(round(self.k_pre / (self.fps * self.dt))) + 1) * self.dt
            self._ph = features(ts / self.dur, self.d['centers'], float(self.d['width']))
        rows = np.flatnonzero(np.abs(dW).sum(1) > 0)
        return float(np.abs(self._ph[:, rows] @ dW[rows]).max()) if len(rows) else 0.0

    def prefix(self):
        if self.pre is not None: return self.pre
        self.d['phase_weights'] = self.weights(self.table0); m = self.m; pol = self.pol; dt = self.dt
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        Q = np.empty((self.T, m.nq)); Q[0] = d.qpos
        for k in range(1, self.k_pre + 1):
            n = round(k / (self.fps * dt)) - round((k - 1) / (self.fps * dt))
            for _ in range(n): d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            Q[k] = d.qpos
        self.pre = (d, Q); return self.pre

    def rollout(self, table):
        d0, Q0 = self.prefix(); self.d['phase_weights'] = self.weights(table); m = self.m; pol = self.pol; dt = self.dt
        d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0); Q = Q0.copy()
        for k in range(self.k_pre + 1, self.T):
            n = round(k / (self.fps * dt)) - round((k - 1) / (self.fps * dt))
            for _ in range(n):
                d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
            Q[k] = d.qpos
        return Q

    def poses(self, Q):
        P = {}
        for nm, a in self.adr.items():
            T = np.tile(np.eye(4), (len(Q), 1, 1)); T[:, :3, :3] = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = Q[:, a:a + 3]; P[nm] = T
        return P

    def score(self, Q):
        e = B.per_frame(self.ep, self.poses(Q), self.reg, self.tgt); res = {nm: (float(e[nm][0].mean()), float(e[nm][1].mean())) for nm in B.NAMES}
        r = [max(p / TOL[0], dg / TOL[1]) for p, dg in res.values()]
        if not np.isfinite(r).all(): return 1e6, res, e
        peak = 0.1 * sum(float(np.percentile(e[n][0], 95)) / 5.0 + float(np.percentile(e[n][1], 95)) / 50.0 for n in B.NAMES)
        pen = 0.0
        if self.base:
            for n, (p, dg) in res.items():
                bp, bd = self.base[n]; pen += 100 * (max(0, p - bp) / bp + max(0, dg - bd) / bd) + (1.0 if (p > bp or dg > bd) else 0.0)
        return float(max(r) + 0.25 * sum(r) + peak + pen), res, e

    def evaluate(self, table, keep=False):
        try:
            if self.k_pre and self.leak(table) != 0.0: return dict(cost=1e6, error='residual leaks before the cache frame')
            Q = self.rollout(table)
            if Q is None: return dict(cost=1e6, unstable=True)
            c, res, e = self.score(Q); out = dict(cost=c, objects=res)
            if keep: out['Q'] = Q; out['err'] = e
            return out
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])

    def ilc_correction(self, Q, t0, t1, ramp=0.3, hand='right', obj='wooden_spoon'):
        """rigid-hold wrist correction (target spoon * inv(achieved spoon) * achieved wrist) in joint space (T x 6), windowed."""
        sc = self.adr[obj]; wq = self.wq if hand == 'right' else self.wql; corr = np.zeros((self.T, 6))
        w = sm((self.t - t0) / ramp) * (1 - sm((self.t - (t1 - ramp)) / ramp)) if t1 < self.t[-1] else sm((self.t - t0) / ramp)
        Ra = R.from_quat(Q[:, sc + 3:sc + 7], scalar_first=True); pa = Q[:, sc:sc + 3]
        Rt = R.from_quat(self.qt[:, sc + 3:sc + 7], scalar_first=True); pt = self.qt[:, sc:sc + 3]
        Rw = R.from_euler('XYZ', Q[:, wq[3:]]); pw = Q[:, wq[:3]]
        for k in range(self.T):
            if w[k] <= 0: continue
            D = Rt[k] * Ra[k].inv()
            pn = pt[k] + D.apply(pw[k] - pa[k]); Rn = D * Rw[k]
            eu = Rn.as_euler('XYZ'); eu = Q[k, wq[3:]] + (eu - Q[k, wq[3:]] + np.pi) % (2 * np.pi) - np.pi
            corr[k, :3] = (pn - pw[k]) * w[k]; corr[k, 3:] = (eu - Q[k, wq[3:]]) * w[k]
        return corr


W = None
def init(cfg):
    global W; W = Env(cfg); W.prefix()
def ev(table): return W.evaluate(table)


def knot_table(env, x, t0, t1, K):
    kn = np.linspace(t0, t1, K + 2); P = np.vstack([np.zeros(NCH), x.reshape(K, NCH), np.zeros(NCH)])
    return np.stack([np.interp(env.t, kn, P[:, c]) for c in range(NCH)], 1)


def save_actor(env, table, out, info):
    out = Path(out); (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'
    with np.load(env.ckpt, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    d['phase_weights'] = env.weights(table); np.savez_compressed(ck, **d); np.savez_compressed(out / 'residual_table.npz', table=table)
    from learned_feedback_policy import sha
    tr = json.loads((Path(env.cfg['actor']) / 'training/training_report.json').read_text())
    tr.update(phase_residual=dict(method='derivative-free physical fine-tuning (ILC / CEM over closed-loop rollouts of this actor) of a time-local residual on the right-wrist and right-finger phase weights; feedback weights, scene unchanged (stream L8, after streamH actor_res.py)',
              source_checkpoint=env.ckpt, source_checkpoint_sha256=sha(env.ckpt), trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), info=info), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from evaluate_policy_component import evaluate as ev_final
    ev_final(env.ep, env.scene, str(ck), str(out / 'evaluation'), str(env.epd))
    q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']; c, res, _ = env.score(q)
    r = json.loads((out / 'evaluation/result.json').read_text())
    print('FINAL', json.dumps(dict(official_eval_scorer=res, cost=c, proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'], physics_verified=r['execution']['physics_verified'])), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['ilc', 'cem', 'eval', 'save', 'rand'])
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--meshdir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--table'); ap.add_argument('--base', help='json {obj: [pos, deg]} strictly-better reference (default: the initial rollout)')
    ap.add_argument('--t0', type=float, default=4.0); ap.add_argument('--t1', type=float, default=9.0); ap.add_argument('--iterations', type=int, default=6)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--samples', type=int, default=64); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--pos-std', type=float, default=0.006); ap.add_argument('--rot-std', type=float, default=0.04); ap.add_argument('--sq-std', type=float, default=0.04)
    ap.add_argument('--alphas', default='0.15,0.3,0.5,0.7,1.0'); ap.add_argument('--sigmas', default='0.1,0.3'); ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--modes', default='full,pos,rot'); ap.add_argument('--hand', default='right', choices=['right', 'left']); ap.add_argument('--obj', default='wooden_spoon'); ap.add_argument('--bias'); ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), meshdir=a.meshdir, table=a.table, bias=a.bias, cache_t=0.0 if a.no_cache else max(0.0, a.t0 - 1.6))
    T00 = time.monotonic(); env = Env(dict(cfg, cache_t=0.0)); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    base = json.loads(Path(a.base).read_text()) if a.base else rep['objects']; cfg['base'] = base; env.base = base
    rep = dict(rep, cost=env.score(rep['Q'])[0])
    ref = np.load(Path(a.actor) / 'evaluation/execution/trajectory.npz')['qpos']
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep.get('objects'), base=base, rollout_s=time.monotonic() - T00, same_as_actor_eval=bool(a.table) or bool(np.array_equal(rep['Q'], ref)))), flush=True)
    envc = Env(cfg); d0, Q0 = envc.prefix(); probe = envc.table0 + knot_table(envc, np.ones(a.knots * NCH) * 0.01, a.t0, a.t1, a.knots)
    print('CACHE k_pre', envc.k_pre, 'prefix==full rollout', bool(np.array_equal(Q0[:envc.k_pre + 1], rep['Q'][:envc.k_pre + 1])), 'leak', envc.leak(probe), flush=True)
    if envc.leak(probe) != 0.0: raise SystemExit('residual leaks before the cache frame; use --no-cache')
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1)); hist = []; t0 = time.monotonic()
    if a.mode == 'save':
        save_actor(env, table, out / 'actor', dict(mode='save', table=a.table, search_eval=dict(cost=rep['cost'], objects=rep.get('objects')))); return
    if a.mode == 'eval':
        e = rep['err']
        for s0 in range(0, env.T, 10): print(f'   {s0/env.fps:5.1f}s', ' '.join(f"{e[n][0][s0:s0+10].mean():5.2f}/{e[n][1][s0:s0+10].mean():5.1f}" for n in B.NAMES))
        return
    best_table = table.copy(); best = dict(cost=rep['cost'], objects=rep['objects'])
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        if a.mode == 'rand':
            K = a.knots; base_std = np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2 + [0.0] * 6, K); rng = np.random.default_rng(a.seed)
            scales = [float(x) for x in a.alphas.split(',')]; f = open(out / 'samples.jsonl', 'a'); n = 0
            for it in range(a.iterations):
                xs = [rng.normal(size=K * NCH) * base_std * scales[(n + i) % len(scales)] for i in range(a.samples)]
                sc_ = [scales[(n + i) % len(scales)] for i in range(a.samples)]; n += a.samples
                reps = list(pool.map(ev, [table + knot_table(env, x, a.t0, a.t1, K) for x in xs], chunksize=1))
                for x, s_, r in zip(xs, sc_, reps):
                    f.write(json.dumps(dict(scale=s_, x=x.tolist(), cost=r['cost'], objects=r.get('objects'))) + '\n')
                    if r['cost'] < best['cost']: best = dict(cost=r['cost'], objects=r['objects'], scale=s_); best_table = table + knot_table(env, x, a.t0, a.t1, K); np.savez_compressed(out / 'best_table.npz', table=best_table)
                f.flush(); print('IT', it, json.dumps(best), flush=True)
        elif a.mode == 'ilc':
            Q = rep['Q']; cur = table.copy(); curcost = rep['cost']
            alphas = [float(x) for x in a.alphas.split(',')]; sigmas = [float(x) for x in a.sigmas.split(',')]; modes = a.modes.split(',')
            for it in range(a.iterations):
                corr = env.ilc_correction(Q, a.t0, a.t1, hand=a.hand, obj=a.obj); cands = []; meta = []; off = 0 if a.hand == 'right' else 8
                for s in sigmas:
                    cs = gaussian_filter1d(corr, s * env.fps, axis=0, mode='nearest') if s > 0 else corr
                    for al in alphas:
                        for mode_ in modes:
                            dlt = np.zeros((env.T, NCH)); m_ = {'full': slice(0, 6), 'pos': slice(0, 3), 'rot': slice(3, 6)}[mode_]
                            dlt[:, off + m_.start:off + m_.stop] = al * cs[:, m_]; dlt[env.t < a.t0 - 0.2] = 0; cands.append(cur + dlt); meta.append(dict(alpha=al, sigma=s, mode=mode_))
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); i = int(order[0])
                row = dict(iteration=it, best_candidate=meta[i], cost=reps[i]['cost'], objects=reps[i].get('objects'), prev_cost=curcost,
                           top=[dict(meta[j], cost=reps[j]['cost'], objects=reps[j].get('objects')) for j in order[:6]], elapsed_s=time.monotonic() - t0)
                hist.append(row); print('IT', json.dumps(row), flush=True)
                if reps[i]['cost'] < curcost:
                    cur = cands[i]; curcost = reps[i]['cost']; Q = env.evaluate(cur, keep=True)['Q']
                    if curcost < best['cost']: best = dict(cost=curcost, objects=reps[i]['objects']); best_table = cur.copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                else:
                    alphas = [x * 0.5 for x in alphas]
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best)))
        else:
            K = a.knots; std = np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2 + [0.0] * 6, K); mean = np.zeros(K * NCH); bx = mean.copy()
            rng = np.random.default_rng(a.seed); bound = 4 * std
            for it in range(a.iterations):
                xs = np.clip(np.vstack([bx, mean, mean + rng.normal(size=(a.samples - 2, K * NCH)) * std]), -bound, bound)
                cands = [table + knot_table(env, x, a.t0, a.t1, K) for x in xs]
                reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = xs[order[:max(6, a.samples // 8)]]
                if reps[order[0]]['cost'] < best['cost']:
                    bx = xs[order[0]].copy(); best = dict(cost=reps[order[0]]['cost'], objects=reps[order[0]]['objects']); best_table = cands[order[0]].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2 + [0.0] * 6, K))
                nb = int(sum(1 for r in reps if 'objects' in r and all(r['objects'][n][0] < base[n][0] and r['objects'][n][1] < base[n][1] for n in B.NAMES)))
                row = dict(iteration=it, best=best, n_better=nb, elapsed_s=time.monotonic() - t0); hist.append(row); print('IT', json.dumps(row), flush=True)
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best, best_x=bx.tolist(), mean=mean.tolist(), std=std.tolist())))
    np.savez_compressed(out / 'best_table.npz', table=best_table)
    if not a.no_save and best['cost'] < rep['cost']:
        save_actor(env, best_table, out / 'actor', dict(mode=a.mode, args=vars(a), search_best=best))


if __name__ == '__main__': main()
