"""T33: CEM over the LEFT-hand cup grasp on the cup's image-left side (fcem.py pattern, left hand, cup frame).
Plan = lside33 side transform (base variant V, e.g. follow -100) + phi offset x[0] (deg, added to the follow goal / D) + left squeeze x[1]
(all left flexion joints, from LT0) + K knots over [T0, T1] of a cup-frame correction D(t) = (rotvec 3, translation 3):
  W'(t) = C(t) D(t) C(t)^-1 W_side(t)  (C = executed cup frame of the BASE teacher rollout; frozen after t_rel).
Cost = worst ratio max(cm/3, deg/30) + 0.25 * sum of ratios (cup and spoon, mlibH vs $E31_TGT)
     + wa * mean over [1.5, t_rel] of max(0, |phi_left - phi_goal| - 30) / 60   (keep the grasp on the left side)
     + wl * fraction of frames in [1.5, t_rel] with < 2 left fingers on the cup.
Every sample = one full-horizon servo_rollout (keyframe start). The best plan is written + executed (execute_reference) like lside33 make.
usage: lcem33.py BASE OUT --v JSON [--knots 8] [--samples 64] [--iterations 15] [--workers 16] [--seed 1] [--scene ..]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import t33lib as T, lside33 as LS
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from wrist_reference import continuous_euler_xyz
sm = LS.sm

class Plan(LS.Base):
    def setup(self, cfg):
        self.cfg = cfg; self.K = cfg['knots']; self.dim = 2 + 6 * self.K
        self.scale = np.r_[10.0, 0.05, np.tile([0.08, 0.08, 0.08, 0.006, 0.006, 0.006], self.K)]
        self.lo = np.r_[-40.0, -0.1, np.tile([-0.4] * 3 + [-0.03] * 3, self.K)]; self.hi = np.r_[40.0, 0.25, np.tile([0.4] * 3 + [0.03] * 3, self.K)]
        n = self.n; ca = int(self.m.joint('blue_cup_free').qposadr[0]); kr = min(n - 1, int(round(cfg['v'].get('t_rel', LS.DEF['t_rel']) * self.fps)))
        self.Rc = R.from_quat(self.Qx[:n, ca + 3:ca + 7], scalar_first=True).as_matrix(); self.Rc[kr:] = self.Rc[kr]; self.cen_f = self.cen.copy(); self.cen_f[kr:] = self.cen_f[kr]
        self.kr = kr
    def goal(self, x):
        v = self.cfg['v']; return (v['follow'] if v.get('follow') is not None else -100.0) + x[0]
    def plan_x(self, x):
        v = dict(self.cfg['v']); c = self.cfg
        if v.get('follow') is not None: v['follow'] = v['follow'] + x[0]
        else: v['D'] = v.get('D', LS.DEF['D']) + x[0]
        v['lsq'] = float(x[1]); v.setdefault('lt0', c['lt0']); Q = self.commands(v); t = self.t; a = self.adl
        kn = np.linspace(c['t0'], c['t1'], self.K); P = x[2:].reshape(self.K, 6); D = np.stack([np.interp(t, kn, P[:, j]) for j in range(6)], 1)
        w = sm((t - (c['t0'] - 0.3)) / 0.3) * (1 - sm((t - c['t1']) / 0.3)); D = D * w[:, None]
        for k in range(1, self.n):
            if not D[k].any(): continue
            Rc = self.Rc[k]; cen = self.cen_f[k]; Dr = R.from_rotvec(D[k, :3]).as_matrix(); Rw = Rc @ Dr @ Rc.T
            Q[k, a[:3]] = cen + Rw @ (Q[k, a[:3]] - cen) + Rc @ D[k, 3:]
            Q[k, a[3:]] = continuous_euler_xyz(R.from_matrix((Rw @ R.from_euler('XYZ', Q[k, a[3:]]).as_matrix())[None]), Q[k - 1, a[3:]])[0]
        Q[0] = self.Q0[0]; return Q
    def cost(self, Qx, x):
        e = T.errors(self.m, Qx); res = T.summary(e); ratios = [max(p / 3, d / 30) for p, d in res.values()]
        if not np.isfinite(ratios).all(): return 1e6, res, {}
        k0, k1 = int(1.5 * self.fps), self.kr; az = T.azimuth(self.m, Qx[k0:k1])[:, 0]; dphi = np.abs((az - self.goal(x) + 180) % 360 - 180)
        _, C, _ = T.contacts(self.m, Qx[k0:k1]); nl = np.array([len(s) for s in C])
        ex = dict(phi_med=float(np.median(az)), az_pen=float(np.mean(np.maximum(0, dphi - 30) / 60)), l_lt2=float((nl < 2).mean()), l_mean=float(nl.mean()))
        c = max(ratios) + 0.25 * sum(ratios) + self.cfg['wa'] * ex['az_pen'] + self.cfg['wl'] * ex['l_lt2']
        return float(c), res, ex
    def evaluate(self, x):
        try:
            Qx = self.rollout(self.plan_x(x)); c, res, ex = self.cost(Qx, x); return dict(cost=min(c, 1e6), objects=res, left=ex)
        except Exception as e: return dict(cost=1e6, error=repr(e)[:200])
W = None
def init(cfg):
    global W; W = Plan(cfg['base'], cfg['scene']); W.setup(cfg)
def ev(x): return W.evaluate(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--v', required=True); ap.add_argument('--scene')
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--t0', type=float, default=1.2); ap.add_argument('--t1', type=float, default=16.2); ap.add_argument('--lt0', type=float, default=1.6)
    ap.add_argument('--wa', type=float, default=1.0); ap.add_argument('--wl', type=float, default=1.5)
    ap.add_argument('--samples', type=int, default=64); ap.add_argument('--iterations', type=int, default=15); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(base=str(Path(a.base).resolve()), scene=a.scene, v=json.loads(a.v), knots=a.knots, t0=a.t0, t1=a.t1, lt0=a.lt0, wa=a.wa, wl=a.wl)
    plan = Plan(cfg['base'], cfg['scene']); plan.setup(cfg); rng = np.random.default_rng(a.seed); mean = np.zeros(plan.dim); std = plan.scale.copy(); hist = []
    best = mean.copy(); best_rep = plan.evaluate(best); print('INIT', json.dumps(best_rep), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(cfg=cfg, args=vars(a), script=__file__), indent=1)); t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(a.iterations):
            props = np.clip(np.vstack([best, mean, mean + rng.normal(size=(a.samples - 2, plan.dim)) * std]), plan.lo, plan.hi)
            reps = list(pool.map(ev, props, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = props[order[:max(6, a.samples // 8)]]
            if reps[order[0]]['cost'] < best_rep['cost']: best = props[order[0]].copy(); best_rep = reps[order[0]]
            mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.12 * plan.scale)
            hist.append(dict(iteration=it, best=best_rep, elapsed_s=time.monotonic() - t0)); print('IT', json.dumps(hist[-1]), flush=True)
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best_parameters=best.tolist(), mean=mean.tolist(), std=std.tolist())))
    Q = plan.plan_x(best); refp = TR.save_reference(out / 'reference.npz', plan.m, Q, plan.ref['labels'], plan.fps, plan.ref['segment'], plan.ref['segments'], source=f'lcem33 best plan: {json.dumps(cfg)}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(plan.scene, refp, out / 'execution'); Qx = np.load(ex['trajectory'])['qpos']; res, txt = T.table(plan.m, Qx, 'lcem33 best'); print(txt, flush=True)
    r0 = json.loads((plan.base / 'result.json').read_text()); r0.update(execution=ex, lcem33=dict(cfg=cfg, best=best_rep, best_parameters=best.tolist(), script=__file__), t33=dict(summary=res['summary'], short=LS.short(res)))
    (out / 'result.json').write_text(json.dumps(r0, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))); (out / 't33table.json').write_text(json.dumps(res, indent=1))
    print('RESULT', json.dumps(LS.short(res)), flush=True)
if __name__ == '__main__': main()
