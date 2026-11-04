"""J8 ep19: physical fine-tuning of a distilled actor's PHASE WEIGHTS (adapted from streamH_ep32 actor_res.py; modes cem / eval / save).

Unchanged architecture: plain LearnedPolicy dispatched by ComponentPolicy, output = phase RBF(t) @ W + state @ F. A time-local residual
delta(t) on the wrist actuators of the chosen sides (6 per side: pos xyz m, rot xyz rad in joint space) is baked into W:
W'[j, a] = W[j, a] + delta_a(t_j) / action_scale_a. Feedback weights, scene, start state untouched. Every evaluation is a full closed-loop
rollout with the same ComponentPolicy object and substep schedule as evaluate_policy_component, scored scorer-faithfully vs
target_final_v5 (common19F.scorer_errors = score_any registration and frame-0 sweeper alignment).
Cost = max_o r_o + 0.25 sum_o r_o + peak_w * mean_o p90(pos_o)/P, r_o = max(pos/P, deg/D) with P = 1.2 cm, D = 12 deg (--p/--d), plus a
hinge that keeps every object strictly under the v22b numbers (--keep).
usage: actor_res19.py cem --actor DIR --out DIR --sides left,right --t0 5 --t1 9 --knots 6 [...]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code']
import common19F as CF
V22B = {'mini_sweeper': (1.95, 14.8), 'mini_dust_pan': (2.38, 9.9)}


class Env:
    def __init__(self, cfg):
        self.cfg = cfg; actor = Path(cfg['actor']); ev = json.loads((actor / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.epd = Path(ev['input_dir'])
        self.m = mujoco.MjModel.from_xml_path(self.scene); self.dt = float(self.m.opt.timestep)
        from component_policy import ComponentPolicy
        self.ckpt = str(actor / 'training/policy.npz'); self.pol = ComponentPolicy(self.ckpt); self.d = self.pol.data
        self.W0 = self.d['phase_weights'].copy(); self.fps = float(self.d['fps']); self.T = int(self.d['frames']); self.dur = float(self.d['duration_s'])
        self.t = np.arange(self.T) / self.fps; self.tc = self.d['centers'] * self.dur
        names = list(self.d['actuator_names']); self.sides = cfg['sides']
        self.wa = [names.index(f'robot_{s}_{s}_{j}_position') for s in self.sides for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        self.nch = len(self.wa); self.scale = self.d['action_scale']
        self.reg = CF.registration(CF.packed_meshes()); self.comp = CF.compose_tracks()
        self.adr = {n: int(self.m.joint(n + '_free').qposadr[0]) for n in CF.NAMES}
        self.table0 = np.zeros((self.T, self.nch))
        if cfg.get('table'): self.table0 = np.load(cfg['table'])['table']
        self.P = cfg.get('P', 1.2); self.D = cfg.get('D', 12.); self.pw = cfg.get('peak_w', 0.1); self.keep = cfg.get('keep', 1)

    def weights(self, table):
        W = self.W0.copy(); tv = np.stack([np.interp(self.tc, self.t, table[:, c]) for c in range(self.nch)], 1)
        for i, a in enumerate(self.wa): W[:, a] += tv[:, i] / self.scale[a]
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

    def score(self, Q):
        ach = {n: CF.to_T(Q[:, a:a + 3], Q[:, a + 3:a + 7]) for n, a in self.adr.items()}
        e = CF.scorer_errors(ach, self.comp, self.reg); res = {n: (e[n]['cm'], e[n]['deg']) for n in CF.NAMES}
        r = [max(p / self.P, dg / self.D) for p, dg in res.values()]
        if not np.isfinite(r).all(): return 1e6, res, e
        pk = np.mean([np.percentile(e[n]['cm_t'], 90) / self.P for n in CF.NAMES])
        hinge = sum(max(0., res[n][0] - (V22B[n][0] - .05)) / .05 + max(0., res[n][1] - (V22B[n][1] - .3)) / .3 for n in CF.NAMES) if self.keep else 0.
        return float(max(r) + .25 * sum(r) + self.pw * pk + .2 * hinge), res, e

    def evaluate(self, table, keep=False):
        try:
            Q = self.rollout(table)
            if Q is None: return dict(cost=1e6, unstable=True)
            c, res, e = self.score(Q); out = dict(cost=c, objects=res)
            if keep: out['Q'] = Q; out['err'] = e
            return out
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])


W = None
def init(cfg):
    global W; W = Env(cfg)
def ev(table): return W.evaluate(table)


def knot_table(env, x, t0, t1, K):
    kn = np.linspace(t0, t1, K + 2); P = np.vstack([np.zeros(env.nch), x.reshape(K, env.nch), np.zeros(env.nch)])
    return np.stack([np.interp(env.t, kn, P[:, c]) for c in range(env.nch)], 1)


def save_actor(env, table, out, info):
    out = Path(out); (out / 'training').mkdir(parents=True, exist_ok=False); ck = out / 'training/policy.npz'
    with np.load(env.ckpt, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    d['phase_weights'] = env.weights(table); np.savez_compressed(ck, **d); np.savez_compressed(out / 'residual_table.npz', table=table)
    from learned_feedback_policy import sha
    tr = json.loads((Path(env.cfg['actor']) / 'training/training_report.json').read_text())
    tr.update(phase_residual=dict(method='derivative-free physical fine-tuning (CEM over full-horizon closed-loop rollouts of this actor) of a time-local residual on the wrist phase weights (sides %s); feedback weights, scene unchanged' % env.sides,
              source_checkpoint=env.ckpt, source_checkpoint_sha256=sha(env.ckpt), trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), info=info), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    for f in ('attempts.json',):
        pass
    from evaluate_policy_component import evaluate as ev_final
    ev_final(19, env.scene, str(ck), str(out / 'evaluation'), str(env.epd))
    q = np.load(out / 'evaluation/execution/trajectory.npz')['qpos']; c, res, _ = env.score(q)
    r = json.loads((out / 'evaluation/result.json').read_text())
    print('FINAL', json.dumps(dict(official_eval_scorer=res, cost=c, proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'], physics_verified=r['execution']['physics_verified'])), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['cem', 'eval', 'save'])
    ap.add_argument('--actor', required=True); ap.add_argument('--out', required=True); ap.add_argument('--table'); ap.add_argument('--sides', default='right')
    ap.add_argument('--t0', type=float, default=4.0); ap.add_argument('--t1', type=float, default=9.0); ap.add_argument('--iterations', type=int, default=8)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--samples', type=int, default=48); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--knots', type=int, default=6); ap.add_argument('--pos-std', type=float, default=0.006); ap.add_argument('--rot-std', type=float, default=0.04)
    ap.add_argument('--p', type=float, default=1.2); ap.add_argument('--d', type=float, default=12.); ap.add_argument('--peak-w', type=float, default=0.1); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    cfg = dict(actor=str(Path(a.actor).resolve()), table=a.table, sides=a.sides.split(','), P=a.p, D=a.d, peak_w=a.peak_w)
    T00 = time.monotonic(); env = Env(cfg); table = env.table0.copy(); rep = env.evaluate(table, keep=True)
    print('INIT', json.dumps(dict(cost=rep['cost'], objects=rep.get('objects'), rollout_s=time.monotonic() - T00)), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(args=vars(a), cfg=cfg, script=__file__), indent=1)); hist = []; t0 = time.monotonic()
    if a.mode == 'save': save_actor(env, table, out / 'actor', dict(mode='save', table=a.table)); return
    if a.mode == 'eval':
        e = rep['err']
        for s0 in range(0, env.T, 10): print(f'   {s0/20:5.1f}s', ' '.join(f"{e[n]['cm_t'][s0:s0+10].mean():5.2f}/{e[n]['deg_t'][s0:s0+10].mean():5.1f}" for n in CF.NAMES))
        return
    best_table = table.copy(); best = dict(cost=rep['cost'], objects=rep['objects'])
    K = a.knots; base = np.tile([a.pos_std] * 3 + [a.rot_std] * 3, len(env.sides)); std = np.tile(base, K); mean = np.zeros(K * env.nch); bx = mean.copy()
    rng = np.random.default_rng(a.seed); bound = 4 * std
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(a.iterations):
            xs = np.clip(np.vstack([bx, mean, mean + rng.normal(size=(a.samples - 2, K * env.nch)) * std]), -bound, bound)
            cands = [table + knot_table(env, x, a.t0, a.t1, K) for x in xs]
            reps = list(pool.map(ev, cands, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = xs[order[:max(6, a.samples // 8)]]
            if reps[order[0]]['cost'] < best['cost']:
                bx = xs[order[0]].copy(); best = dict(cost=reps[order[0]]['cost'], objects=reps[order[0]]['objects']); best_table = cands[order[0]].copy(); np.savez_compressed(out / 'best_table.npz', table=best_table)
            mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * np.tile(base, K))
            row = dict(iteration=it, best=best, elapsed_s=time.monotonic() - t0); hist.append(row); print('IT', json.dumps(row), flush=True)
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best=best, best_x=bx.tolist())))
    np.savez_compressed(out / 'best_table.npz', table=best_table)
    if not a.no_save and best['cost'] < rep['cost']: save_actor(env, best_table, out / 'actor', dict(mode=a.mode, args=vars(a), search_best=best))


if __name__ == '__main__': main()
