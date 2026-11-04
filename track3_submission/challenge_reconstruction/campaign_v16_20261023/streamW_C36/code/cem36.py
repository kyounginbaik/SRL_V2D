"""Stream W ep36 (v21C): physical CEM over the compose-following plan parameters (code/plan36.py) on top of a base actor.
Each sample: phase weights rebuilt by plan36.Plan.build, full-horizon rollout of the SAME controller the evaluator runs
(ComponentPolicy incl. command rounding, evaluator substep count, keyframe reset), spoon scored against the compose-retargeted
reference (episode_compose; setup frame; anchor point; full orientation) = the scorer-identical numbers vs compose_target.
Cost = max(pos/3, rot/30) + 0.25 * mean ratio + 100 * excess. Resumable state.json (<= 50 min jobs). The best passing sample is
written as a checkpoint and evaluated with the unchanged evaluate_policy_component against episode_compose.
usage: cem36.py --base DIR --output DIR [--samples 48 --iterations 8 --workers 16 --seed S] [--init JSON] [--names a,b,..]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, time, argparse, hashlib, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent)); import plan36
from scipy.spatial.transform import Rotation as R
SPACE = dict(gr=(0.0, 1.2, 0.25), gp=(0.0, 1.2, 0.25), sig=(0.0, 0.6, 0.1), clip=(0.0, 180.0, 25.0), cmax=(10.0, 180.0, 25.0), t_on=(1.6, 3.5, 0.3), t_off=(16.5, 19.5, 0.4),
             lead=(-0.3, 0.3, 0.08), rc0=(11.8, 12.4, 0.12), rc1=(12.5, 13.1, 0.12), rc2=(14.2, 14.8, 0.12), rc3=(14.8, 15.4, 0.12), squeeze=(-0.05, 0.3, 0.08))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
Wk = None; FIXED = json.loads(os.environ.get('CEM36_FIXED', '{}'))
class Worker:
    def __init__(self, base):
        from component_policy import ComponentPolicy
        self.plan = plan36.Plan(base); self.m = mujoco.MjModel.from_xml_path(self.plan.scene); self.pol = ComponentPolicy(str(Path(base) / 'training/policy.npz'))
        self.W0 = self.pol.data['phase_weights'].copy(); self.dt = float(self.m.opt.timestep)
        self.Tc = self.plan.Tc; self.adr = int(self.m.jnt_qposadr[self.m.body('wooden_spoon').jntadr[0]])
        self.anch = np.array(json.loads((plan36.OEP / 'metric_geometry.json').read_text())['objects']['wooden_spoon']['anchor_local_m'])
    def rollout(self, p):
        Wn, info = self.plan.build(p); self.pol.data['phase_weights'][:] = Wn
        m = self.m; d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d); n = 407; Q = np.empty((n, m.nq)); Q[0] = d.qpos
        try:
            for k in range(1, n):
                for _ in range(round(k / (20 * self.dt)) - round((k - 1) / (20 * self.dt))):
                    d.ctrl[:] = self.pol(d.time + self.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None, info
                Q[k] = d.qpos
        finally: self.pol.data['phase_weights'][:] = self.W0
        return Q, info
    def score(self, Q):
        a = self.adr; Ra = R.from_quat(Q[:, a + 3:a + 7][:, [1, 2, 3, 0]]).as_matrix(); pa = Q[:, a:a + 3] + np.einsum('tij,j->ti', Ra, self.anch)
        pr = self.Tc[:, :3, 3] + np.einsum('tij,j->ti', self.Tc[:, :3, :3], self.anch); pe = np.linalg.norm(pa - pr, axis=1) * 100
        re = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', self.Tc[:, :3, :3], Ra)).magnitude()); return pe, re
    def run(self, p):
        p = dict(p); p.update(FIXED); Q, info = self.rollout(p)
        if Q is None: return dict(cost=1e6, unstable=True, params=p)
        pe, re = self.score(Q); ratio = max(pe.mean() / 3, re.mean() / 30)
        return dict(cost=float(ratio + .25 * ratio + 100 * max(0., ratio - 1.)), objects={'wooden_spoon': (float(pe.mean()), float(re.mean()))}, mp_sr=float(ratio < 1),
                    win=[round(float(re[i:i + 40].mean()), 1) for i in range(0, 407, 40)], params=p, **info)
def init(base):
    global Wk; Wk = Worker(base)
def ev(p): return Wk.run(p)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--output', required=True); ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--iterations', type=int, default=8); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--init', default='{}'); ap.add_argument('--names', default='gr,gp,sig,clip,cmax,t_on,t_off,lead'); ap.add_argument('--std-scale', type=float, default=1.0)
    ap.add_argument('--time-budget', type=float, default=2500); ap.add_argument('--grid', help='JSON list of param dicts: evaluate only (no CEM)')
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    base_p = dict(plan36.DEFAULT); base_p.update(json.loads(a.init)); names = a.names.split(',')
    lo = np.array([SPACE[k][0] for k in names]); hi = np.array([SPACE[k][1] for k in names]); std0 = np.array([SPACE[k][2] for k in names]) * a.std_scale
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(a.base,)) as pool:
        if a.grid:
            ps = []
            for g in json.loads(a.grid): q = dict(base_p); q.update(g); ps.append(q)
            rs = list(pool.map(ev, ps, chunksize=1))
            for r in rs: print(json.dumps({k: r.get(k) for k in ('objects', 'win', 'mp_sr', 'cost')}), json.dumps({k: r['params'][k] for k in r['params'] if k in json.loads(a.grid)[0] or True}), flush=True)
            (out / 'grid.json').write_text(json.dumps(rs, indent=1, default=float)); return
        stp = out / 'state.json'; mean = np.array([base_p[k] for k in names]); std = std0.copy(); best = None; hist = []; it0 = 0
        if stp.exists():
            s = json.loads(stp.read_text()); mean, std, best, hist, it0 = np.array(s['mean']), np.array(s['std']), s['best'], s['hist'], s['next']
        else: (out / 'settings.json').write_text(json.dumps(dict(base=a.base, names=names, lo=lo.tolist(), hi=hi.tolist(), init=base_p, std0=std0.tolist(), samples=a.samples, seed=a.seed, fixed=FIXED, script=__file__, script_sha256=sha(__file__), plan_sha256=sha(plan36.__file__)), indent=1))
        for it in range(it0, a.iterations):
            if time.monotonic() - t0 > a.time_budget: break
            rng = np.random.default_rng([a.seed, it]); xs = np.clip(mean + rng.normal(size=(a.samples, len(names))) * std, lo, hi); xs[0] = mean
            if best: xs[1] = np.array([best['params'][k] for k in names])
            ps = []
            for x in xs: q = dict(base_p); q.update(dict(zip(names, map(float, x)))); ps.append(q)
            rs = list(pool.map(ev, ps, chunksize=1))
            for r in rs:
                if best is None or r['cost'] < best['cost']: best = r
            order = np.argsort([r['cost'] for r in rs]); el = xs[order[:max(6, a.samples // 8)]]
            mean = .3 * mean + .7 * el.mean(0); std = np.maximum(.3 * std + .7 * el.std(0), std0 * 0.08)
            row = dict(iteration=it, best_cost=best['cost'], best=best.get('objects'), best_win=best.get('win'), best_params={k: round(best['params'][k], 3) for k in names}, n_pass=int(sum(r.get('mp_sr', 0) for r in rs)), it_best=rs[order[0]].get('objects'), elapsed=time.monotonic() - t0)
            hist.append(row); print(json.dumps(row), flush=True)
            stp.write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist(), best=best, hist=hist, next=it + 1), default=float))
    if best and best.get('mp_sr') == 1 and len(hist) >= a.iterations and not (out / 'evaluation/result.json').exists():
        write_checkpoint(a.base, best['params'], out, dict(search_objects=best['objects'], search=str(Path(__file__).resolve()), search_sha256=sha(__file__)))
def write_checkpoint(base, params, out, extra):
    out = Path(out); plan = plan36.Plan(base); Wn, info = plan.build(params); d = dict(plan.d); d['phase_weights'] = Wn
    (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
    rep = json.loads((Path(base) / 'training/training_report.json').read_text())
    rep.update(checkpoint=str(ck), checkpoint_sha256=sha(ck), compose_follow_replan=dict(base_actor=str(Path(base) / 'training/policy.npz'), base_sha256=sha(Path(base) / 'training/policy.npz'), params=params,
        compose_retarget=str(plan36.REC), compose_retarget_sha256=sha(plan36.REC), planner=str(plan36.__file__), planner_sha256=sha(plan36.__file__), info=info, **extra))
    (out / 'training/training_report.json').write_text(json.dumps(rep, indent=2, default=str))
    from evaluate_policy_component import evaluate
    evaluate(36, plan.scene, str(ck), str(out / 'evaluation'), str(plan36.OEP))
    r = json.loads((out / 'evaluation/result.json').read_text()); mm = r['metrics']['official_style_reference_proxy']['objects']
    print('FINAL', json.dumps({k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in mm.items()}), flush=True)
if __name__ == '__main__': main()
