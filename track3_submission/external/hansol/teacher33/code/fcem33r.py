"""[T33 fcem33r = fcem33 + team fcem_r.py robustness: every sample also rolled out under --robust N smooth noise draws (--noise rad|m on all robot command columns, sigma 5 frames, x3.5, seeds 777+j, as fcem_r / lside33.noise_draws); cost = mean of the N+1 costs + 0.5 * max; reported objects = nominal] [T33 teacher33 copy of retarget32 fcem32.py for ep33 (MD 33, NAMES by ep; separate index+middle / thumb / ring+pinky squeeze; cost + wg*max(0,3.5-mean right fingers on spoon)/3.5 + wd*frac(frames <2 fingers) over the hold); R32 retarget32 copy of the team streamB_ep3233/code/fcem.py: scorer = mlibH (target $E31_TGT, our R32_HGA) instead of blib (target_final_v5); meshes = G18_ep32_H12e1f0; everything else unchanged] Stream B: CEM over a target-following right-wrist plan (follow_plan_v2 generalized) with a SCORER-FAITHFUL cost.
Plan: base teacher commands (port); after t_on the right wrist command is slerped (ramp) to F(t) = T_target(t) * D(t) * offset, offset =
commanded wrist vs ACHIEVED spoon of the base rollout at t_lock; D(t) = piecewise-linear knots (K over [t_on, t_off]) of a local
rotation (rotvec, spoon frame) + translation (spoon frame); plus a right-finger squeeze (non-thumb, thumb) over the window; ramp back to the
base commands before t_off. Every sample = one full-horizon servo_rollout (the unchanged floating_runtime physics, keyframe start).
Cost = worst ratio max(mean pos/3 cm, mean deg/30) + 0.25 * sum of ratios, errors computed exactly as the scorer (blib.per_frame:
MP-SR registration of the shipped meshes onto target_final_v5, spoon-anchored frame-0 alignment). The best plan is re-executed with
execute_reference and scored with evaluate_files (teacher dir: reference.npz + result.json + execution/).
NOTE: _freeze is not idempotent for every scene (ep33 W1: quaternion print rounding flips once); pass --scene = a fixed point.
usage: fcem.py EP BASE_DIR EPD OUT --t-lock 3.0 --t-off 10.8 [--ramp 0.3] [--knots 8] [--samples 64] [--iterations 12] [--workers 16] [--seed 1] [--resume]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
from tool_library.common import tracking as TR, hand as Hh
from floating_runtime import execute_reference, servo_rollout
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
import mlibH as B   # R32: E31 scorer copy (target = $E31_TGT)
C16 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
MD = {33: "/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033"}   # T33: shipped G18 meshes of the ep33 package
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)

class Plan:
    def __init__(self, cfg):
        self.cfg = cfg; base = Path(cfg['base']); r = json.loads((base / 'result.json').read_text()); self.scene = cfg.get('scene') or r['execution']['scene']
        self.m = mujoco.MjModel.from_xml_path(self.scene); ref = TR.load_reference(base / 'reference.npz', self.m); self.ref = ref
        self.Q0 = ref['qpos'].copy(); self.n = len(self.Q0); self.fps = ref['fps']; self.t = np.arange(self.n) / self.fps
        tr = np.load(r['execution']['trajectory']); si = [str(x) for x in tr['object_names']].index('wooden_spoon'); self.tr = tr; self.si = si
        self.qt = TR.load_reference(Path(cfg['epd']) / 'tracking_reference.npz', self.m)['qpos']; self.sc = int(self.m.joint('wooden_spoon_free').qposadr[0]); self.cc = int(self.m.joint('blue_cup_free').qposadr[0])
        self.adr = [int(self.m.joint(f'robot_right_right_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        rig = Hh.HandRig.build(self.m, 'robot_right_', 'right'); self.flex = []
        for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            for jn in rig.flexion_joints(fi): self.flex.append((fi, rig.finger_qpos[jn], *rig.finger_range[jn]))
        self.K = cfg['knots']; self.dim = 5 + 6 * self.K   # T33: x[2] index+middle, x[3] thumb, x[4] ring+pinky squeeze
        self.scale = np.r_[0.15, 0.08, 0.1, 0.1, 0.1, np.tile([0.12, 0.08, 0.08, 0.008, 0.008, 0.008], self.K)]
        self.bound = np.r_[0.4, 0.2, 0.4, 0.4, 0.4, np.tile([0.6, 0.4, 0.4, 0.03, 0.03, 0.03], self.K)]
        self.reg = B.register(cfg['ep'], {nm: Path(MD[cfg['ep']]) / f'{nm}.obj' for nm in B.NAMES[cfg["ep"]]}); self.tgt = B.target(cfg['ep'])
        self.qstart = None
        import lside33; self.noise = lside33.noise_draws(self.m, self.n, cfg['noise'], cfg['robust']) if cfg['robust'] else []
    def commands(self, x):
        c = self.cfg; Q = self.Q0.copy(); t = self.t; tl = c['t_lock'] + x[0]; ramp = max(0.08, c['ramp'] + x[1])
        k0 = int(round(tl * self.fps)); tr = self.tr; ps0 = tr['object_positions'][k0, self.si]; Rs0 = R.from_quat(tr['object_quaternions_wxyz'][k0, self.si][[1, 2, 3, 0]])
        adr = self.adr; Rw0 = R.from_euler('XYZ', Q[k0, adr[3:]]); off_p = Rs0.inv().apply(Q[k0, adr[:3]] - ps0); off_R = Rs0.inv() * Rw0
        w = sm((t - tl) / ramp) * (1 - sm((t - (c['t_off'] - ramp)) / ramp))
        kn = np.linspace(tl, c['t_off'], self.K); P = x[5:].reshape(self.K, 6); D = np.stack([np.interp(t, kn, P[:, j]) for j in range(6)], 1)
        P_ = self.qt[:, self.sc:self.sc + 3]; RT = R.from_quat(self.qt[:, self.sc + 3:self.sc + 7], scalar_first=True)
        for k in range(1, self.n):
            if w[k] <= 0: continue
            FR = RT[k] * R.from_rotvec(D[k, :3]) * off_R; Fp = P_[k] + RT[k].apply(D[k, 3:] + R.from_rotvec(D[k, :3]).apply(off_p))
            Bp = Q[k, adr[:3]]; BR = R.from_euler('XYZ', Q[k, adr[3:]])
            Q[k, adr[:3]] = (1 - w[k]) * Bp + w[k] * Fp; Rn = BR * R.from_rotvec((BR.inv() * FR).as_rotvec() * w[k])
            Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[k - 1, adr[3:]])[0]
        for fi, col, lo, hi in self.flex: Q[:, col] = np.clip(Q[:, col] + (x[3] if fi == 'thumb' else x[2] if fi in ('index', 'middle') else x[4]) * w, lo, hi)
        Q[0] = self.Q0[0]; return Q
    def score(self, qpos):
        P = {}
        for nm, a in (('blue_cup', self.cc), ('wooden_spoon', self.sc)):
            T = np.tile(np.eye(4), (len(qpos), 1, 1)); T[:, :3, :3] = R.from_quat(qpos[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = qpos[:, a:a + 3]; P[nm] = T
        e = B.per_frame(self.cfg['ep'], P, self.reg, self.tgt); res = {nm: (float(e[nm][0].mean()), float(e[nm][1].mean())) for nm in B.NAMES[self.cfg["ep"]]}
        ratios = [max(p / 3, d / 30) for p, d in res.values()]
        if not np.isfinite(ratios).all(): return 1e6, res
        return float(max(ratios) + 0.25 * sum(ratios)), res
    def evaluate(self, x):
        try:
            Q = self.commands(x)
            if self.qstart is None:
                d = mujoco.MjData(self.m); mujoco.mj_resetDataKeyframe(self.m, d, 0); self.qstart = d.qpos.copy()
            cs = []; out0 = None
            for Z in [None] + self.noise:
                out = servo_rollout(self.m, Q if Z is None else Q + Z, self.fps, qpos_start=self.qstart, record_substeps=False)
                qp = np.asarray(out['qpos']); c, res = self.score(qp); g = self.grasp(qp); c = min(1e3, c + self.cfg['wg'] * max(0.0, 3.5 - g['mean']) / 3.5 + self.cfg['wd'] * g['frac_lt2'])
                cs.append(c)
                if out0 is None: out0 = (c, res, g)
            c = float(np.mean(cs) + 0.5 * np.max(cs)) if len(cs) > 1 else cs[0]
            return dict(cost=min(c, 1e6), objects=out0[1], grasp=out0[2], nominal_cost=out0[0], costs=[round(x, 3) for x in cs])
        except Exception as ex: return dict(cost=1e6, error=str(ex)[:200])
    def grasp(self, qp):
        """T33: right-hand fingers on the spoon over the hold [t_lock - 1, t_off - 0.5] (t33lib.contacts): mean count, fraction of frames < 2."""
        import t33lib
        k0, k1 = int((self.cfg['t_lock'] - 1.0) * self.fps), int((self.cfg['t_off'] - 0.5) * self.fps); S, _, _ = t33lib.contacts(self.m, qp[k0:k1])
        n = np.array([len(x) for x in S]); return dict(mean=float(n.mean()), frac_lt2=float((n < 2).mean()))
W = None
def init(cfg):
    global W; W = Plan(cfg)
def ev(x): return W.evaluate(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('base'); ap.add_argument('epd'); ap.add_argument('out')
    ap.add_argument('--t-lock', type=float, required=True); ap.add_argument('--t-off', type=float, required=True); ap.add_argument('--ramp', type=float, default=0.3)
    ap.add_argument('--robust', type=int, default=3); ap.add_argument('--noise', type=float, default=0.0005); ap.add_argument('--wg', type=float, default=0.3); ap.add_argument('--wd', type=float, default=1.0); ap.add_argument('--knots', type=int, default=8); ap.add_argument('--samples', type=int, default=64); ap.add_argument('--iterations', type=int, default=12)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--resume', action='store_true'); ap.add_argument('--init'); ap.add_argument('--scene', help='physics scene for every rollout (must be a _freeze fixed point so execute_reference reproduces it); default = base execution scene')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(robust=a.robust, noise=a.noise, wg=a.wg, wd=a.wd, ep=a.ep, base=str(Path(a.base).resolve()), epd=str(Path(a.epd).resolve()), t_lock=a.t_lock, t_off=a.t_off, ramp=a.ramp, knots=a.knots, scene=str(Path(a.scene).resolve()) if a.scene else None)
    plan = Plan(cfg); rng = np.random.default_rng(a.seed); mean = np.zeros(plan.dim); std = plan.scale.copy(); hist = []; it0 = 0
    if a.init: mean = np.asarray(json.loads(Path(a.init).read_text())['best_parameters'])
    best = mean.copy(); best_rep = plan.evaluate(best)
    if a.resume and (out / 'progress.json').exists():
        p = json.loads((out / 'progress.json').read_text()); best = np.asarray(p['best_parameters']); mean = np.asarray(p['mean']); std = np.asarray(p['std']); hist = p['history']; it0 = len(hist); best_rep = hist[-1]['best']
        rng = np.random.default_rng(a.seed + 1000 * it0)
    (out / 'settings.json').write_text(json.dumps(dict(cfg=cfg, args=vars(a), script=__file__), indent=1)); print('INIT', json.dumps(best_rep), flush=True)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(it0, a.iterations):
            props = np.clip(np.vstack([best, mean, mean + rng.normal(size=(a.samples - 2, plan.dim)) * std]), -plan.bound, plan.bound)
            reps = list(pool.map(ev, props, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = props[order[:max(6, a.samples // 8)]]
            if reps[order[0]]['cost'] < best_rep['cost']: best = props[order[0]].copy(); best_rep = reps[order[0]]
            mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.12 * plan.scale)
            npass = int(sum(1 for r in reps if 'objects' in r and all(p <= 3 and d <= 30 for p, d in r['objects'].values())))
            hist.append(dict(iteration=it, best=best_rep, n_pass=npass, elapsed_s=time.monotonic() - t0)); print('IT', json.dumps(hist[-1]), flush=True)
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best_parameters=best.tolist(), mean=mean.tolist(), std=std.tolist())))
    Q = plan.commands(best); refp = TR.save_reference(out / 'reference.npz', plan.m, Q, plan.ref['labels'], plan.fps, plan.ref['segment'], plan.ref['segments'], source=f'fcem.py best plan: {json.dumps(cfg)}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(plan.scene, refp, out / 'execution'); mf = Path(cfg['epd']) / 'input_manifest.json'; mm = json.loads(mf.read_text())
    mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], Path(cfg['epd']) / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
    c, res = plan.score(np.load(ex['trajectory'])['qpos'])
    (out / 'result.json').write_text(json.dumps(dict(episode=mm['episode'], input_dir=cfg['epd'], reference=str(refp), execution=ex, metrics=mt, best=best_rep, best_parameters=best.tolist(), scorer_faithful=dict(cost=c, objects=res), cfg=cfg, script=__file__), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    print('RESULT', o['mp_sr'], 'proxy', {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, 'scorer', {k: (round(p, 2), round(d, 1)) for k, (p, d) in res.items()}, flush=True)
if __name__ == '__main__': main()
