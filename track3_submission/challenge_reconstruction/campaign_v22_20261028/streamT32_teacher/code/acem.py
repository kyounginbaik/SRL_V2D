"""Stream B: ACTOR-level CEM. Same plan parameterization as fcem.py (target-following right wrist, knots, squeeze), but every sample is
  plan -> teacher execute_reference (unchanged physics, frozen fixed-point scene) -> learned_feedback_policy.train (the registered RBF
  actor, fixed hyperparameters, seed 44, exactly as actor_grid_c.py) -> closed-loop actor rollout with ComponentPolicy (evaluator
  substep schedule, as streamK search_margins_k.py) -> scorer-faithful cost (blib.per_frame).
cost = actor cost + 0.3 * teacher cost (cost = worst ratio + 0.25 * sum of ratios, ratio = max(pos/3 cm, deg/30)).
The best sample is exported as a teacher dir (reference.npz, result.json, execution/) plus actor/ = train + the unchanged
evaluate_policy_component (staging-ready).
usage: acem.py EP BASE_DIR EPD OUT --t-lock --t-off [--ramp] [--knots] [--centers 512 --gain 0.003 --vgain 0.0001] [--samples 48] [--iterations 10] [--workers 16] [--seed] [--init PROGRESS] [--scene FIXED_POINT] [--resume]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time, shutil, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcem
from fcem import TR, execute_reference, evaluate_files
TMP = Path(os.environ.get('ACEM_TMP', '/tmp'))
class A(fcem.Plan):
    def actor_rollout(self, ckpt, scene):
        from component_policy import ComponentPolicy
        m = mujoco.MjModel.from_xml_path(scene); pol = ComponentPolicy(ckpt); dt = float(m.opt.timestep)
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        Q = np.empty((self.n, m.nq)); Q[0] = d.qpos
        for k in range(1, self.n):
            nsub = round(k / (self.fps * dt)) - round((k - 1) / (self.fps * dt))
            for _ in range(nsub):
                d.ctrl[:] = pol(d.time + dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all(): return None
            Q[k] = d.qpos
        return Q
    def run(self, x, out=None):
        from learned_feedback_policy import train
        Q = self.commands(x); keep = out is not None
        work = Path(out) if keep else Path(tempfile.mkdtemp(prefix='acem_', dir=TMP))
        try:
            ref = dict(qpos=Q, fps=self.fps, labels=np.asarray(self.ref['labels']))
            ex = execute_reference(self.scene, ref, work / 'execution')
            tq = np.load(ex['trajectory'])['qpos']; tc, tres = self.score(tq)
            train(ex['scene'], ex['trajectory'], work / 'training', centers=self.cfg['centers'], feedback_gain=self.cfg['gain'], velocity_gain=self.cfg['vgain'], seed=44)
            aq = self.actor_rollout(str(work / 'training/policy.npz'), ex['scene'])
            if aq is None: return dict(cost=1e6, teacher=tres)
            ac, ares = self.score(aq)
            return dict(cost=float(min(ac, 1e6) + 0.3 * min(tc, 1e6)), actor_cost=ac, teacher_cost=tc, objects=ares, teacher=tres)
        except Exception as e: return dict(cost=1e6, error=repr(e)[:300])
        finally:
            if not keep: shutil.rmtree(work, ignore_errors=True)
W = None
def init(cfg):
    global W; W = A(cfg)
def ev(x): return W.run(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('base'); ap.add_argument('epd'); ap.add_argument('out')
    ap.add_argument('--t-lock', type=float, required=True); ap.add_argument('--t-off', type=float, required=True); ap.add_argument('--ramp', type=float, default=0.3); ap.add_argument('--knots', type=int, default=8)
    ap.add_argument('--centers', type=int, default=512); ap.add_argument('--gain', type=float, default=0.003); ap.add_argument('--vgain', type=float, default=0.0001)
    ap.add_argument('--samples', type=int, default=48); ap.add_argument('--iterations', type=int, default=10); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--init'); ap.add_argument('--scene'); ap.add_argument('--resume', action='store_true'); ap.add_argument('--std-scale', type=float, default=0.5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, base=str(Path(a.base).resolve()), epd=str(Path(a.epd).resolve()), t_lock=a.t_lock, t_off=a.t_off, ramp=a.ramp, knots=a.knots, centers=a.centers, gain=a.gain, vgain=a.vgain, scene=str(Path(a.scene).resolve()) if a.scene else None)
    plan = A(cfg); rng = np.random.default_rng(a.seed); mean = np.zeros(plan.dim); std = plan.scale * a.std_scale; hist = []; it0 = 0
    if a.init: mean = np.asarray(json.loads(Path(a.init).read_text())['best_parameters'])
    best = mean.copy(); best_rep = plan.run(best)
    if a.resume and (out / 'progress.json').exists():
        p = json.loads((out / 'progress.json').read_text()); best = np.asarray(p['best_parameters']); mean = np.asarray(p['mean']); std = np.asarray(p['std']); hist = p['history']; it0 = len(hist); best_rep = hist[-1]['best']; rng = np.random.default_rng(a.seed + 1000 * it0)
    (out / 'settings.json').write_text(json.dumps(dict(cfg=cfg, args=vars(a), script=__file__), indent=1)); print('INIT', json.dumps(best_rep), flush=True); t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(it0, a.iterations):
            props = np.clip(np.vstack([best, mean, mean + rng.normal(size=(a.samples - 2, plan.dim)) * std]), -plan.bound, plan.bound)
            reps = list(pool.map(ev, props, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = props[order[:max(6, a.samples // 8)]]
            if reps[order[0]]['cost'] < best_rep['cost']: best = props[order[0]].copy(); best_rep = reps[order[0]]
            mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.08 * plan.scale)
            npass = int(sum(1 for r in reps if 'objects' in r and all(p <= 2.4 and d <= 24 for p, d in r['objects'].values())))
            hist.append(dict(iteration=it, best=best_rep, n_pass_target=npass, elapsed_s=time.monotonic() - t0)); print('IT', json.dumps(hist[-1]), flush=True)
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best_parameters=best.tolist(), mean=mean.tolist(), std=std.tolist())))
    # export: teacher dir + official actor evaluation
    Q = plan.commands(best); refp = TR.save_reference(out / 'reference.npz', plan.m, Q, plan.ref['labels'], plan.fps, plan.ref['segment'], plan.ref['segments'], source=f'acem.py best plan: {json.dumps(cfg)}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(plan.scene, refp, out / 'execution'); mf = Path(cfg['epd']) / 'input_manifest.json'; mm = json.loads(mf.read_text())
    mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], Path(cfg['epd']) / 'metric_geometry.json'); tc, tres = plan.score(np.load(ex['trajectory'])['qpos'])
    (out / 'result.json').write_text(json.dumps(dict(episode=mm['episode'], input_dir=cfg['epd'], reference=str(refp), execution=ex, metrics=mt, best=best_rep, best_parameters=best.tolist(), scorer_faithful=dict(cost=tc, objects=tres), cfg=cfg, script=__file__), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from learned_feedback_policy import train
    from evaluate_policy_component import evaluate
    dest = out / f"actor_c{cfg['centers']}_p{cfg['gain']:g}_v{cfg['vgain']:g}"
    train(ex['scene'], ex['trajectory'], dest / 'training', centers=cfg['centers'], feedback_gain=cfg['gain'], velocity_gain=cfg['vgain'], seed=44)
    evaluate(a.ep, ex['scene'], str(dest / 'training/policy.npz'), str(dest / 'evaluation'), cfg['epd'])
    aq = np.load(dest / 'evaluation/execution/trajectory.npz')['qpos']; ac, ares = plan.score(aq)
    print('RESULT teacher', {k: (round(p, 2), round(d, 1)) for k, (p, d) in tres.items()}, 'actor(official eval)', {k: (round(p, 2), round(d, 1)) for k, (p, d) in ares.items()}, 'search-best actor', best_rep.get('objects'), flush=True)
if __name__ == '__main__': main()
