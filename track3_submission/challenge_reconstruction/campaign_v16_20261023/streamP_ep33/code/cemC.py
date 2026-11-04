"""Stream P ep33 phase 6: physical CEM over the compose-target right-hand re-plan parameters (code/planC.py) on top of a
compose-world base actor (cup already re-targeted). Copied from roll_cem.py (phase 5b) with the planner and reference swapped.
Each sample: phase weights rebuilt by rollplan.Plan.build, full-horizon rollout of the SAME controller the evaluator runs
(ComponentPolicy incl. command rounding, evaluator substep count, keyframe reset), objects scored against the roll-CORRECTED
reference (setup frame; spoon full orientation, cup axial). Cost = worst-object MP-SR ratio + 0.25 mean ratio (+100 x excess if
any object fails). Resumable state.json (<= 50 min jobs). The best passing sample is written as a checkpoint and evaluated with
the unchanged evaluate_policy_component against the corrected episode dir.
usage: roll_cem.py --base DIR --output DIR [--samples 64 --iterations 10 --workers 16 --seed S --resume] [--init JSON]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, time, argparse, hashlib, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import planC as rollplan
OLD = os.environ['CEMC_OLD_ROLLOUT']
from scipy.spatial.transform import Rotation as R
NAMES = ['t_on', 't_off', 'g_roll', 'g_tilt', 'g_pos', 't_w0', 't_w1', 'ox', 'oy', 'oz', 'to0', 'to1', 'k_td']
LO = np.array([3.1, 15.0, 0.0, 0.0, 0.0, 4.5, 4.8, -3., -3., -3., 4.0, 5.0, 0.0]); HI = np.array([4.5, 17.4, 1.2, 1.0, 1.0, 7.0, 7.6, 3., 3., 3., 7.5, 9.0, 1.0])
INIT = np.array([3.4, 17.0, 0.6, 0.3, 0.4, 5.4, 5.9, 0.0, 0.0, 0.0, 5.5, 7.5, 0.0]); STD0 = np.array([0.3, 0.5, 0.3, 0.25, 0.25, 0.4, 0.4, 1.2, 1.2, 1.2, 0.6, 0.6, 0.3])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W = None
COST = os.environ.get('ROLLCEM_COST', 'worst')
FIXED = json.loads(os.environ.get('ROLLCEM_FIXED', '{}'))
class Worker:
    def __init__(self, base):
        from component_policy import ComponentPolicy
        self.plan = rollplan.Plan(base, OLD); self.m = mujoco.MjModel.from_xml_path(self.plan.scene); self.pol = ComponentPolicy(str(Path(base) / 'training/policy.npz'))
        self.W0 = self.pol.data['phase_weights'].copy(); self.dt = float(self.m.opt.timestep)
        man = json.loads((rollplan.OEP / 'input_manifest.json').read_text()); S = np.asarray(man['setup_from_source'])
        z = np.load(Path(man['source']) / 'object_poses.npz'); names = [str(x) for x in z['names']]; T = S @ z['world_from_object']
        self.tp = T[:, :, :3, 3]; self.tR = T[:, :, :3, :3]; self.oi = names
        self.adr = [int(self.m.jnt_qposadr[self.m.body(o).jntadr[0]]) for o in names]
        ax = json.loads((rollplan.OEP / 'metric_geometry.json').read_text())['objects']; self.cupax = np.array(ax['blue_cup']['axis_local'])
        self.anch = [np.array(ax[o]['anchor_local_m']) for o in names]
    def run(self, x):
        p = dict(zip(NAMES, x)); p.update(FIXED); Wn, info = self.plan.build(p); self.pol.data['phase_weights'][:] = Wn
        m = self.m; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d); n = 348; Q = np.empty((n, m.nq)); Q[0] = d.qpos
        for k in range(1, n):
            for _ in range(round(k / (20 * self.dt)) - round((k - 1) / (20 * self.dt))):
                d.ctrl[:] = self.pol(d.time + self.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return dict(cost=1e6, unstable=True, params=p)
            Q[k] = d.qpos
        self.pol.data['phase_weights'][:] = self.W0
        res = {}
        for i, o in enumerate(self.oi):
            a = self.adr[i]; Ra = R.from_quat(Q[:, a + 3:a + 7][:, [1, 2, 3, 0]]).as_matrix(); pa = Q[:, a:a + 3] + np.einsum('tij,j->ti', Ra, self.anch[i])
            pr = self.tp[:, i] + np.einsum('tij,j->ti', self.tR[:, i], self.anch[i]); pe = np.linalg.norm(pa - pr, axis=1).mean() * 100
            if o == 'blue_cup':
                u = np.einsum('tij,j->ti', Ra, self.cupax); v = np.einsum('tij,j->ti', self.tR[:, i], self.cupax); re = np.degrees(np.arccos(np.clip((u * v).sum(1), -1, 1))).mean()
            else:
                re = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', self.tR[:, i], Ra)).magnitude()).mean()
            res[o] = (float(pe), float(re))
        ratio = np.array([max(v[0] / 3, v[1] / 30) for v in res.values()])
        if COST == 'sum': cost = float(sum(v[0] / 3 + v[1] / 30 for v in res.values()) + 100 * max(0., float(ratio.max()) - 1.))
        else: cost = float(ratio.max() + .25 * ratio.mean() + 100 * max(0., float(ratio.max()) - 1.))
        return dict(cost=cost, objects=res, mp_sr=float(ratio.max() < 1), params=p, **info)
def init(base):
    global W; W = Worker(base)
def ev(x): return W.run(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--output', required=True); ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--iterations', type=int, default=10); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init'); ap.add_argument('--std-scale', type=float, default=1.0); ap.add_argument('--time-budget', type=float, default=2700)
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True); t0 = time.monotonic(); stp = out / 'state.json'
    mean = INIT.copy() if not a.init else np.array([json.loads(a.init).get(k, 0.0) for k in NAMES]); std = STD0.copy() * a.std_scale; best = None; hist = []; it0 = 0
    if stp.exists():
        s = json.loads(stp.read_text()); mean, std, best, hist, it0 = np.array(s['mean']), np.array(s['std']), s['best'], s['hist'], s['next']
    else: (out / 'settings.json').write_text(json.dumps(dict(base=a.base, names=NAMES, lo=LO.tolist(), hi=HI.tolist(), init=mean.tolist(), std0=std.tolist(), samples=a.samples, seed=a.seed, cost=COST, fixed=FIXED, script=__file__, script_sha256=sha(__file__), rollplan_sha256=sha(rollplan.__file__)), indent=1))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(a.base,)) as pool:
        for it in range(it0, a.iterations):
            if time.monotonic() - t0 > a.time_budget: break
            rng = np.random.default_rng([a.seed, it]); xs = np.clip(mean + rng.normal(size=(a.samples, len(NAMES))) * std, LO, HI); xs[0] = mean
            if best: xs[1] = np.array([best['params'][k] for k in NAMES])
            rs = list(pool.map(ev, xs, chunksize=1))
            for r in rs:
                if best is None or r['cost'] < best['cost']: best = r
            order = np.argsort([r['cost'] for r in rs]); el = xs[order[:max(6, a.samples // 8)]]
            mean = .3 * mean + .7 * el.mean(0); std = np.maximum(.3 * std + .7 * el.std(0), STD0 * 0.08)
            row = dict(iteration=it, best_cost=best['cost'], best=best.get('objects'), best_params=best['params'], n_pass=int(sum(r.get('mp_sr', 0) for r in rs)), it_best=rs[order[0]].get('objects'), elapsed=time.monotonic() - t0)
            hist.append(row); print(json.dumps(row), flush=True)
            stp.write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist(), best=best, hist=hist, next=it + 1), default=float))
    if best and best.get('mp_sr') == 1 and len(hist) >= a.iterations and not (out / 'evaluation/result.json').exists():
        plan = rollplan.Plan(a.base, OLD); Wn, info = plan.build(best['params']); d = dict(plan.d); d['phase_weights'] = Wn
        (out / 'training').mkdir(exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
        rep = json.loads((Path(a.base) / 'training/training_report.json').read_text())
        rep.update(checkpoint_sha256=sha(ck), compose_right_cem=dict(old_rollout=OLD, base_actor=str(Path(a.base) / 'training/policy.npz'), base_sha256=sha(Path(a.base) / 'training/policy.npz'), params=best['params'], search_objects=best['objects'],
            reference_episode_dir=str(rollplan.OEP), planner=str(rollplan.__file__), planner_sha256=sha(rollplan.__file__), search=str(Path(__file__).resolve()), search_sha256=sha(__file__), info=info))
        (out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
        from evaluate_policy_component import evaluate
        evaluate(33, plan.scene, str(ck), str(out / 'evaluation'), str(rollplan.OEP))
        r = json.loads((out / 'evaluation/result.json').read_text()); mm = r['metrics']['official_style_reference_proxy']['objects']
        print('FINAL', json.dumps({k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in mm.items()}), flush=True)
if __name__ == '__main__': main()
