"""Stream K8 (v23) eps 35/36/37 (single wooden_spoon): physical fine-tuning of an actor's PHASE WEIGHTS on its own closed-loop rollout.
Adapted from streamH_ep32/code/actor_res.py (H ep32 late-window ILC). Differences:
- single object: score = tlibK.errors (scorer-identical: MP-SR registration of our mesh onto target_final_v5, frame-0 alignment of
  the spoon itself), cost aims at the phase-8 goal (1.2 cm / 12 deg) + per-frame peak term + never-worse penalty (--failp/--faild);
- the ILC target in the physics world = frame-0 aligned target (A0 @ T_target X, A0 = achieved f0 * target f0^-1 = scorer alignment);
- wrist chart: the rot hinges sit under a fixed chart rotation Ac (body_quat of robot_right_right_base_roll; ep35 virtual chart),
  R_palm = Ac R_XYZ(q); a world correction D maps to R_XYZ(q') = Ac^-1 D Ac R_XYZ(q) (stream C7 chart fix);
- separate position / rotation gains (ap, ar) in the candidate grid; optional finger squeeze channel unchanged from actor_res;
- prefix cache: the current table's rollout stores the MuJoCo integration state at every frame; a candidate restarts from the last
  frame before its phase-command output first differs (bitwise) from the current table's (exactness checked at start).
The shipped architecture is unchanged (ComponentPolicy dispatch of the same checkpoint fields): W'[j, a] = W[j, a] + delta_a(t_j)/scale_a.
Every saved actor is re-run with the unchanged evaluate_policy_component (official actor evaluation).
usage: ilcK.py MODE --ep EP --actor DIR --out DIR --windows 2.0:4.5,6.0:9.0 [--table npz] ..."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
import tlibK as TL
from tool_library.common import hand as Hh
from learned_feedback_policy import features
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
NCH = 8
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class Env:
    def __init__(self, cfg):
        self.cfg = cfg; actor = Path(cfg['actor']); ev = json.loads((actor / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.epd = Path(cfg.get('epd') or ev['input_dir']); self.ep = cfg['ep']
        self.m = mujoco.MjModel.from_xml_path(self.scene); self.dt = float(self.m.opt.timestep)
        from component_policy import ComponentPolicy
        self.ckpt = str(actor / 'training/policy.npz'); self.pol = ComponentPolicy(self.ckpt); self.d = self.pol.data
        self.W0 = self.d['phase_weights'].copy(); self.fps = float(self.d['fps']); self.T = int(self.d['frames']); self.dur = float(self.d['duration_s'])
        self.t = np.arange(self.T) / self.fps; self.tc = self.d['centers'] * self.dur
        names = [str(x) for x in self.d['actuator_names']]
        self.wa = [names.index(f'robot_right_right_{j}_position') for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.scale = self.d['action_scale']
        rig = Hh.HandRig.build(self.m, 'robot_right_', 'right'); self.flex = []
        for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            for jn in rig.flexion_joints(fi):
                jid = self.m.joint('robot_right_right_' + jn).id if isinstance(jn, str) else int(jn)
                hits = np.flatnonzero(self.m.actuator_trnid[:, 0] == jid)
                if len(hits) == 1: self.flex.append((fi == 'thumb', int(hits[0])))
        self.sc = int(self.m.joint('wooden_spoon_free').qposadr[0])
        self.wq = [int(self.m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        q = self.m.body_quat[self.m.body('robot_right_right_base_roll').id]; self.Ac = R.from_quat([q[1], q[2], q[3], q[0]])
        X, anchor, ca, rot = TL.registration(self.ep); self.Tb = TL.target(self.ep) @ X
        self.table0 = np.zeros((self.T, NCH))
        if cfg.get('table'): self.table0 = np.load(cfg['table'])['table']
        # substep schedule (evaluate_policy_component): frame k gets n_k substeps
        self.nsub = [0] + [round(k / (self.fps * self.dt)) - round((k - 1) / (self.fps * self.dt)) for k in range(1, self.T)]
        self.fail = (cfg.get('failp', 99.0), cfg.get('faild', 999.0))
        self.cache = None

    def weights(self, table):
        W = self.W0.copy(); tv = np.stack([np.interp(self.tc, self.t, table[:, c]) for c in range(NCH)], 1)
        for i, a in enumerate(self.wa): W[:, a] += tv[:, i] / self.scale[a]
        for th, a in self.flex: W[:, a] += tv[:, 7 if th else 6] / self.scale[a]
        return W

    def _run(self, d, k_start, Q, S=None):
        m = self.m; pol = self.pol; dt = self.dt
        for k in range(k_start, self.T):
            for _ in range(self.nsub[k]):
                d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
            Q[k] = d.qpos
            if S is not None: mujoco.mj_getState(m, d, S[k], SPEC)
        return Q

    def rollout(self, table, store=False):
        """full rollout from the keyframe; store=True keeps per-frame integration states (prefix cache for this table)."""
        self.d['phase_weights'] = self.weights(table); m = self.m
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        Q = np.empty((self.T, m.nq)); Q[0] = d.qpos; S = None
        if store:
            S = np.empty((self.T, mujoco.mj_stateSize(m, SPEC))); mujoco.mj_getState(m, d, S[0], SPEC)
        Q = self._run(d, 1, Q, S)
        if store and Q is not None: self.cache = dict(W=self.d['phase_weights'].copy(), Q=Q.copy(), S=S)
        return Q

    def first_diff_frame(self, Wn):
        """first frame k whose substeps see a phase output different (bitwise) from the cached weights."""
        Wc = self.cache['W']; cols = np.flatnonzero((Wn != Wc).any(0))
        if len(cols) == 0: return self.T
        tt = np.arange(1, sum(self.nsub) + 1) * self.dt; ph = features(tt / self.dur, self.d['centers'], float(self.d['width']))
        diff = ((ph @ Wn)[:, cols] != (ph @ Wc)[:, cols]).any(1); i = int(np.argmax(diff)) if diff.any() else len(tt)
        cum = np.cumsum(self.nsub); return int(np.searchsorted(cum, i + 1))  # frame containing substep i (0-based)

    def rollout_cached(self, table):
        if self.cache is None: return self.rollout(table)
        Wn = self.weights(table); kf = self.first_diff_frame(Wn) - 2  # 1-frame safety margin (substep time rounding)
        if kf < 1: return self.rollout(table)
        self.d['phase_weights'] = Wn; m = self.m; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_setState(m, d, self.cache['S'][kf], SPEC); mujoco.mj_forward(m, d)
        Q = self.cache['Q'].copy(); return self._run(d, kf + 1, Q)

    def poses(self, Q):
        a = self.sc; T = np.tile(np.eye(4), (len(Q), 1, 1)); T[:, :3, :3] = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = Q[:, a:a + 3]; return T

    def score(self, Q):
        pe, re = TL.errors(self.ep, self.poses(Q)); pm, dm = float(pe.mean()), float(re.mean())
        if not (np.isfinite(pm) and np.isfinite(dm)): return 1e6, (pm, dm), (pe, re)
        g = max(pm / 1.2, dm / 12); bad = max(0., pm - self.fail[0]) + max(0., dm - self.fail[1]) / 10
        peak = np.percentile(pe, 95) / 3 + np.percentile(re, 95) / 30
        return float(g + .25 * (pm / 1.2 + dm / 12) + .05 * peak + 10 * bad + 100 * max(0., max(pm / 3, dm / 30) - 1)), (pm, dm), (pe, re)

    def evaluate(self, table, keep=False, cached=True):
        try:
            Q = self.rollout_cached(table) if cached else self.rollout(table)
            if Q is None: return dict(cost=1e6, unstable=True)
            c, res, e = self.score(Q); out = dict(cost=c, spoon=res, p95=[float(np.percentile(e[0], 95)), float(np.percentile(e[1], 95))])
            if keep: out['Q'] = Q; out['err'] = e
            return out
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])

    def target_world(self, Q):
        Ta = self.poses(Q[:1]); A0 = Ta[0] @ np.linalg.inv(self.Tb[0]); return A0 @ self.Tb[:self.T]

    def ilc_correction(self, Q, t0, t1, ramp=0.3):
        """rigid-hold wrist correction (T x 6: slide m, hinge rad), windowed; chart-aware."""
        sc = self.sc; corr = np.zeros((self.T, 6)); Tt = self.target_world(Q)
        w = sm((self.t - t0) / ramp) * (1 - sm((self.t - (t1 - ramp)) / ramp)) if t1 < self.t[-1] else sm((self.t - t0) / ramp)
        Ra = R.from_quat(Q[:, sc + 3:sc + 7], scalar_first=True); pa = Q[:, sc:sc + 3]
        Rt = R.from_matrix(Tt[:, :3, :3]); pt = Tt[:, :3, 3]
        qw = Q[:, self.wq[3:]]; pw = Q[:, self.wq[:3]]
        for k in range(self.T):
            if w[k] <= 0: continue
            D = Rt[k] * Ra[k].inv()
            pn = pt[k] + D.apply(pw[k] - pa[k])
            Rn = self.Ac.inv() * D * self.Ac * R.from_euler('XYZ', qw[k])
            eu = Rn.as_euler('XYZ'); eu = qw[k] + (eu - qw[k] + np.pi) % (2 * np.pi) - np.pi
            corr[k, :3] = (pn - pw[k]) * w[k]; corr[k, 3:] = (eu - qw[k]) * w[k]
        return corr


W = None
def init(cfg):
    global W; W = Env(cfg)
def ev(args):
    base, cand = args; key = hashlib.sha256(base.tobytes()).hexdigest()
    if getattr(W, 'cache_key', None) != key: W.rollout(base, store=True); W.cache_key = key
    return W.evaluate(cand)


def knot_table(env, x, t0, t1, K):
    kn = np.linspace(t0, t1, K + 2); P = np.vstack([np.zeros(NCH), x.reshape(K, NCH), np.zeros(NCH)])
    return np.stack([np.interp(env.t, kn, P[:, c]) for c in range(NCH)], 1)


def save_actor(env, table, out, info):
    out = Path(out); (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'
    with np.load(env.ckpt, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    d['phase_weights'] = env.weights(table); np.savez_compressed(ck, **d); np.savez_compressed(out / 'residual_table.npz', table=table)
    tr = json.loads((Path(env.cfg['actor']) / 'training/training_report.json').read_text())
    tr.update(phase_residual=dict(method='derivative-free physical fine-tuning (windowed ILC / knot CEM over full-horizon closed-loop rollouts of this actor) of a time-local residual on the right-wrist (and right-finger) phase weights; feedback weights, margins, scene unchanged',
              source_checkpoint=env.ckpt, source_checkpoint_sha256=sha(env.ckpt), trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), info=info), checkpoint=str(ck), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from evaluate_policy_component import evaluate as ev_final
    ev_final(env.ep, env.scene, str(ck), str(out / 'evaluation'), str(env.epd))
    q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']; c, res, _ = env.score(q)
    r = json.loads((out / 'evaluation/result.json').read_text())
    print('FINAL', json.dumps(dict(official_eval_scorer=res, cost=c, proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'], physics_verified=r['execution']['physics_verified'])), flush=True)
    return res


def winprint(e, fps=20):
    for s0 in range(0, len(e[0]), 20): print(f'   {s0/fps:5.1f}s {e[0][s0:s0+20].mean():5.2f}/{e[1][s0:s0+20].mean():5.1f}  max {e[0][s0:s0+20].max():5.2f}/{e[1][s0:s0+20].max():5.1f}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['ilc', 'cem', 'eval', 'save'])
    ap.add_argument('--ep', type=int, required=True); ap.add_argument('--actor', required=True); ap.add_argument('--out', required=True); ap.add_argument('--table'); ap.add_argument('--epd')
    ap.add_argument('--windows', default='4.0:9.0'); ap.add_argument('--iterations', type=int, default=6)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--samples', type=int, default=48); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--pos-std', type=float, default=0.004); ap.add_argument('--rot-std', type=float, default=0.03); ap.add_argument('--sq-std', type=float, default=0.0)
    ap.add_argument('--aps', default='0,0.3,0.6,1.0'); ap.add_argument('--ars', default='0,0.3,0.6,1.0'); ap.add_argument('--sigmas', default='0.1,0.3'); ap.add_argument('--ramp', type=float, default=0.3)
    ap.add_argument('--failp', type=float, default=99.0); ap.add_argument('--faild', type=float, default=999.0); ap.add_argument('--no-save', action='store_true'); ap.add_argument('--time-budget', type=float, default=2400)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, actor=str(Path(a.actor).resolve()), table=a.table, epd=a.epd, failp=a.failp, faild=a.faild)
    T00 = time.monotonic(); env = Env(cfg); table = env.table0.copy(); rep = env.evaluate(table, keep=True, cached=False)
    print('INIT', json.dumps(dict(cost=rep['cost'], spoon=rep.get('spoon'), p95=rep.get('p95'), rollout_s=time.monotonic() - T00)), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__, script_sha256=sha(__file__)), indent=1))
    if a.mode == 'save':
        save_actor(env, table, out / 'actor', dict(mode='save', table=a.table, search_eval=dict(cost=rep['cost'], spoon=rep.get('spoon')))); return
    if a.mode == 'eval':
        winprint(rep['err']); return
    windows = [tuple(map(float, w.split(':'))) for w in a.windows.split(',')]
    best_table = table.copy(); best = dict(cost=rep['cost'], spoon=rep['spoon'], p95=rep['p95']); hist = []; t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        # exactness check of the cache path
        env.rollout(table, store=True); tw = sm((env.t - windows[0][0]) / 0.3) * (1 - sm((env.t - windows[0][1] + 0.3) / 0.3)); tchk = table.copy(); tchk[:, 2] += 0.002 * tw
        chk = env.rollout_cached(tchk); full = env.rollout(tchk); print('CACHE exact', bool(chk is not None and full is not None and np.array_equal(chk, full)), 'restart frame', env.first_diff_frame(env.weights(tchk)) if env.cache else None, flush=True)
        env.cache = None
        if a.mode == 'ilc':
            Q = rep['Q']; cur = table.copy(); curcost = rep['cost']; aps = [float(x) for x in a.aps.split(',')]; ars = [float(x) for x in a.ars.split(',')]
            sigmas = [float(x) for x in a.sigmas.split(',')]
            for (w0, w1) in windows:
                scale = 1.0
                for it in range(a.iterations):
                    if time.monotonic() - t0 > a.time_budget: break
                    corr = env.ilc_correction(Q, w0, w1, a.ramp); cands = []; meta = []
                    for s in sigmas:
                        cs = gaussian_filter1d(corr, s * env.fps, axis=0, mode='nearest') if s > 0 else corr
                        for gp_ in aps:
                            for gr_ in ars:
                                if gp_ == 0 and gr_ == 0: continue
                                dlt = np.zeros((env.T, NCH)); dlt[:, :3] = gp_ * scale * cs[:, :3]; dlt[:, 3:6] = gr_ * scale * cs[:, 3:6]
                                cands.append(cur + dlt); meta.append(dict(ap=gp_ * scale, ar=gr_ * scale, sigma=s))
                    reps = list(pool.map(ev, [(cur, c_) for c_ in cands], chunksize=1)); order = np.argsort([r['cost'] for r in reps]); i = int(order[0])
                    row = dict(window=[w0, w1], iteration=it, best_candidate=meta[i], cost=reps[i]['cost'], spoon=reps[i].get('spoon'), p95=reps[i].get('p95'), prev_cost=curcost,
                               top=[dict(meta[j], cost=round(reps[j]['cost'], 4), spoon=reps[j].get('spoon')) for j in order[:5]], n_better=int(sum(r['cost'] < curcost for r in reps)), elapsed_s=time.monotonic() - t0)
                    hist.append(row); print('IT', json.dumps(row), flush=True)
                    if reps[i]['cost'] < curcost:
                        cur = cands[i]; curcost = reps[i]['cost']; r2 = env.evaluate(cur, keep=True, cached=False); Q = r2['Q']
                        if abs(r2['cost'] - curcost) > 1e-9: print('WARN cached vs full cost', curcost, r2['cost'], flush=True); curcost = r2['cost']
                        if curcost < best['cost']: best = dict(cost=curcost, spoon=r2['spoon'], p95=r2['p95']); best_table = cur.copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                    else:
                        scale *= 0.5
                    (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best)))
        else:
            (w0, w1) = windows[0]; K = a.knots; s0 = np.tile([a.pos_std] * 3 + [a.rot_std] * 3 + [a.sq_std] * 2, K); std = s0.copy(); mean = np.zeros(K * NCH); bx = mean.copy()
            rng = np.random.default_rng(a.seed); bound = 4 * s0 + 1e-12
            for it in range(a.iterations):
                if time.monotonic() - t0 > a.time_budget: break
                xs = np.clip(np.vstack([bx, mean, mean + rng.normal(size=(a.samples - 2, K * NCH)) * std]), -bound, bound)
                cands = [table + knot_table(env, x, w0, w1, K) for x in xs]
                reps = list(pool.map(ev, [(table, c_) for c_ in cands], chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = xs[order[:max(6, a.samples // 8)]]
                if reps[order[0]]['cost'] < best['cost']:
                    bx = xs[order[0]].copy(); best = dict(cost=reps[order[0]]['cost'], spoon=reps[order[0]]['spoon'], p95=reps[order[0]]['p95']); best_table = cands[order[0]].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
                mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * s0)
                row = dict(iteration=it, best=best, it_best=reps[order[0]].get('spoon'), n_better=int(sum(r['cost'] < rep['cost'] for r in reps)), elapsed_s=time.monotonic() - t0); hist.append(row); print('IT', json.dumps(row), flush=True)
                (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best, best_x=bx.tolist(), mean=mean.tolist(), std=std.tolist())))
    np.savez_compressed(out / 'best_table.npz', table=best_table)
    if not a.no_save and best['cost'] < rep['cost']:
        res = save_actor(env, best_table, out / 'actor', dict(mode=a.mode, args=vars(a), search_best=best))
        q = np.load(out / 'actor/evaluation/execution/trajectory.npz')['qpos']; _, _, e = env.score(q); winprint(e)


if __name__ == '__main__': main()
