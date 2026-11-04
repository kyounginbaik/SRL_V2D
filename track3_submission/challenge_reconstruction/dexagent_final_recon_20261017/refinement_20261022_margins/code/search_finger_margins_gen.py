"""Generic (episodes 15/36): fit bounded constant target margins on all fingers of one hand (default right, the spoon hand),
warm-started from any margins already in the checkpoint (union of actuator sets), on top of a distilled actor
(PinkyMarginPolicy over LearnedPolicy), by full-horizon physical rollouts (CEM). Objective = search_pinky_margins_v2.py
(worst-object MP-SR ratio + 0.25 mean ratio + finger soft-limit penalty), no task-specific terms.
usage: search_finger_margins_gen.py --episode N --actor DIR (training/policy.npz + evaluation/result.json) --output DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, multiprocessing as mp, sys, time
import numpy as np, mujoco
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from refine_grasp_cem import Experiment, angular_errors, dump
from learned_feedback_policy import sha
DONOR = B / 'refinement_20261018/expanded_batch/bimanual_release/episode_000033/case_0/policy.npz'
RELEASE_KEYS = None


class Search:
    def __init__(self, spec):
        self.spec = spec; inp = Path(spec['episode_dir']); man = json.loads((inp / 'input_manifest.json').read_text())
        self.metric = Experiment(dict(scene=spec['scene'], reference=str(inp / 'tracking_reference.npz'), original_reference=str(inp / 'tracking_reference.npz'), source=man['source'], geometry=str(inp / 'metric_geometry.json')))
        self.m = self.metric.model
        if spec['release']:
            from multi_release_feedback_policy import MultiReleasePolicy as P
        else:
            from pinky_margin_policy_v2 import PinkyMarginPolicy as P
        self.policy = P(spec['augmented_checkpoint']); self.dt = float(self.m.opt.timestep); self.steps = round(1 / self.metric.fps / self.dt)
        self.fj = np.array([j for j in range(self.m.njnt) if (self.m.joint(j).name or '').startswith('robot_') and any(f'_{f}_' in self.m.joint(j).name for f in ['thumb', 'index', 'middle', 'ring', 'pinky'])])
        self.fq = self.m.jnt_qposadr[self.fj]; self.frange = self.m.jnt_range[self.fj]

    def body(self, p, q):
        from scipy.spatial.transform import Rotation as Rot
        Rm = lambda j: Rot.from_quat(q[:, j][:, [1, 2, 3, 0]]).as_matrix(); Rs, Rc = Rm(self.si), Rm(self.ci)
        os_ = p[:, self.si] - np.einsum('tij,j->ti', Rs, self.anch[self.si]); oc = p[:, self.ci] - np.einsum('tij,j->ti', Rc, self.anch[self.ci])
        return np.einsum('tij,j->ti', Rs, self.tip) + os_, (oc, Rc)

    def local(self, tw, cup): oc, Rc = cup; return np.einsum('tji,tj->ti', Rc, tw - oc)

    def evaluate(self, x):
        self.policy.data['margin_bias_rad'] = np.asarray(x)
        d = mujoco.MjData(self.m); mujoco.mj_resetDataKeyframe(self.m, d, 0); mujoco.mj_forward(self.m, d)
        Q = np.empty_like(self.metric.base); Q[0] = d.qpos; limit = 0.
        for k in range(1, len(Q)):
            for _ in range(self.steps):
                d.ctrl[:] = self.policy(d.time + self.dt, d.qpos, d.qvel); mujoco.mj_step(self.m, d)
                limit = max(limit, float(np.max(np.maximum(self.frange[:, 0] - d.qpos[self.fq], d.qpos[self.fq] - self.frange[:, 1]))))
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return dict(cost=1e6, unstable=True)
            Q[k] = d.qpos
        p, q = self.metric.object_poses(Q); ep = np.linalg.norm(p - self.metric.tp, axis=-1)
        er = angular_errors(q[:, None], self.metric.tq[:, None], self.metric.axes, self.metric.kinds)[:, 0]
        pos = ep.mean(0); ang = er.mean(0); ratio = np.maximum(pos / .03, ang / np.deg2rad(30)); limit_deg = float(np.degrees(limit))
        cost = float(ratio.max() + .25 * ratio.mean() + 100 * max(0., limit_deg / self.spec['tolerance_deg'] - 1., float(ratio.max()) - 1.))
        return dict(cost=cost, mean_position_cm=(100 * pos).tolist(), mean_orientation_deg=np.degrees(ang).tolist(), max_ratio=float(ratio.max()),
                    mp_sr=float(np.all(ratio < 1)), all_finger_limit_max_degrees=limit_deg, quality_pass=limit_deg <= self.spec['tolerance_deg'], unstable=False)


WORK = None
def init(spec):
    global WORK; WORK = Search(spec)
def evaluate(x): return WORK.evaluate(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--actor', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--episode', type=int, required=True); ap.add_argument('--hand', default='right'); ap.add_argument('--release', action='store_true'); ap.add_argument('--left-pinky', action='store_true'); ap.add_argument('--bound', type=float, default=.25)
    ap.add_argument('--samples', type=int, default=96); ap.add_argument('--iterations', type=int, default=12); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=3333); ap.add_argument('--tolerance-deg', type=float, default=1.15); ap.add_argument('--std', type=float, default=.08)
    a = ap.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'settings.json').exists(): raise FileExistsError(out)
    ev = json.loads((a.actor / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']; inp = Path(ev['input_dir']); m = mujoco.MjModel.from_xml_path(scene)
    with np.load(a.actor / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
    base_keys = set(d)
    if 'margin_base_module' not in d:   # plain LearnedPolicy -> wrap in PinkyMarginPolicy
        d.update(margin_base_module=np.array('learned_feedback_policy'), margin_base_class=np.array('LearnedPolicy'))
    old_ids = [int(i) for i in d['margin_actuator_ids']] if 'margin_actuator_ids' in d else []
    old_bias = dict(zip(old_ids, np.asarray(d['margin_bias_rad'], float))) if old_ids else {}
    old_early = dict(zip(old_ids, np.asarray(d.get('margin_startup_only', np.zeros(len(old_ids), bool)), bool))) if old_ids else {}
    ids = list(old_ids); pre = f'robot_{a.hand}_{a.hand}_'
    for j in range(m.njnt):
        n = m.joint(j).name or ''
        if n.startswith(pre) and any(f'_{f}_' in n for f in ['thumb', 'index', 'middle', 'ring', 'pinky']) or (a.left_pinky and n.startswith('robot_left_left_pinky')):
            hits = np.flatnonzero(m.actuator_trnid[:, 0] == j)
            if len(hits) == 1 and int(hits[0]) not in ids: ids.append(int(hits[0]))
    init_x = np.array([old_bias.get(i, 0.) for i in ids]); early = np.array([old_early.get(i, False) for i in ids], bool)
    d.update(margin_actuator_ids=np.asarray(ids, int), margin_bias_rad=init_x.copy(), margin_startup_only=early, margin_startup_end_s=np.asarray(d.get('margin_startup_end_s', .8)))
    if a.release:
        with np.load(DONOR, allow_pickle=False) as z: donor = {k: z[k] for k in z.files}
        for k, v in donor.items():
            if k.startswith('release') or k.startswith('multi_release'): d[k] = v
    aug = out / 'initial_policy.npz'; np.savez_compressed(aug, **d)
    spec = dict(scene=str(Path(scene).resolve()), episode_dir=str(inp), augmented_checkpoint=str(aug), tolerance_deg=a.tolerance_deg, release=a.release)
    s = Search(spec); best = init_x.copy(); rep = s.evaluate(best)
    print(json.dumps(dict(initial_margins=rep, dims=len(ids), names=[m.actuator(i).name for i in ids])), flush=True)
    rng = np.random.default_rng(a.seed); mean = init_x.copy(); std = np.full(len(ids), a.std); lo, hi = np.minimum(-a.bound, init_x), np.maximum(a.bound, init_x); hist = []; t0 = time.monotonic()
    best_mp = (best.copy(), dict(rep)) if rep.get('mp_sr') == 1. else None
    dump(out / 'settings.json', dict(spec=spec, source_actor=str(a.actor), source_checkpoint_sha256=sha(a.actor / 'training/policy.npz'), actuator_names=[m.actuator(i).name for i in ids], bound=a.bound, samples=a.samples, iterations=a.iterations, seed=a.seed))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(spec,)) as pool:
        for it in range(a.iterations):
            xs = np.vstack([best, mean, np.clip(mean + rng.normal(size=(a.samples - 2, len(ids))) * std, lo, hi)]); rs = list(pool.map(evaluate, xs, chunksize=1))
            for x, r in zip(xs, rs):
                if r.get('mp_sr') == 1. and r.get('quality_pass') and (best_mp is None or r['cost'] < best_mp[1]['cost']): best_mp = (x.copy(), dict(r)); dump(out / 'best_mp.json', dict(parameters=x, report=r))
            order = np.argsort([r['cost'] for r in rs]); elite = xs[order[:max(6, a.samples // 8)]]
            if rs[order[0]]['cost'] < rep['cost']: best = xs[order[0]].copy(); rep = rs[order[0]]
            mean = .25 * mean + .75 * elite.mean(0); std = np.maximum(.3 * std + .7 * elite.std(0), .01)
            row = dict(iteration=it, best=rep, n_pass=int(sum(r.get('mp_sr') == 1. for r in rs)), elapsed_s=time.monotonic() - t0); hist.append(row)
            dump(out / 'progress.json', dict(history=hist, best_parameters=best)); print(json.dumps(row, default=lambda x: x.tolist()), flush=True)
    if best_mp is not None: best, rep = best_mp
    d['margin_bias_rad'] = best; (out / 'training').mkdir(exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
    tr = json.loads((a.actor / 'training/training_report.json').read_text())
    tr.update(finger_margins=dict(method='derivative-free physical fine-tuning of bounded constant finger target margins (CEM, full-horizon rollouts), warm-started from existing margins', parameters=best.tolist(),
              actuator_names=[m.actuator(i).name for i in ids], source_checkpoint_sha256=sha(a.actor / 'training/policy.npz'), release_keys_grafted_from=str(DONOR) if a.release else None, warm_start_margins=init_x.tolist(),
              trainer=str(Path(__file__).resolve()), trainer_sha256=sha(__file__), best=rep), checkpoint_sha256=sha(ck))
    (out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    from evaluate_policy_component import evaluate as ev_final
    ev_final(a.episode, scene, str(ck), str(out / 'evaluation'), str(inp))
    r = json.loads((out / 'evaluation/result.json').read_text()); mm = r['metrics']['official_style_reference_proxy']['objects']
    print('FINAL', json.dumps(dict(passed=r['metrics']['reference_proxy_mp_sr'] == 1 and r['execution']['physics_verified'], arch=r.get('policy_architecture'), search_best=rep,
          objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in mm.items()})), flush=True)


if __name__ == '__main__': main()
