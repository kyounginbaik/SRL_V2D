"""Stream B: CEM over a target-following right-wrist plan (follow_plan_v2 generalized) with a SCORER-FAITHFUL cost.
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
import blib as B
C16 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
MD = {32: f'{C16}/R_ep32_regrasp_power06/episode_000032/meshes/episode_000032', 33: f'{C16}/P_ep33_regrasp_cem3/episode_000033/meshes/episode_000033'}
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)

HANDS = [('right', 'wooden_spoon', 'robot_right_'), ('left', 'blue_cup', 'robot_left_')]
class Plan:
    """fcem2: BOTH wrists follow their object's target (right: spoon, left: cup) from t_lock with per-hand knots and per-hand finger squeeze."""
    def __init__(self, cfg):
        self.cfg = cfg; base = Path(cfg['base']); r = json.loads((base / 'result.json').read_text()); self.scene = cfg.get('scene') or r['execution']['scene']
        self.m = mujoco.MjModel.from_xml_path(self.scene); ref = TR.load_reference(base / 'reference.npz', self.m); self.ref = ref
        self.Q0 = ref['qpos'].copy(); self.n = len(self.Q0); self.fps = ref['fps']; self.t = np.arange(self.n) / self.fps
        tr = np.load(r['execution']['trajectory']); self.tr = tr; names = [str(x) for x in tr['object_names']]
        self.qt = TR.load_reference(Path(cfg['epd']) / 'tracking_reference.npz', self.m)['qpos']; self.sc = int(self.m.joint('wooden_spoon_free').qposadr[0]); self.cc = int(self.m.joint('blue_cup_free').qposadr[0])
        self.H = []
        for side, obj, pre in HANDS:
            if side == 'left' and not cfg.get('left', True): continue
            h = dict(side=side, obj=obj, oi=names.index(obj), oc=int(self.m.joint(f'{obj}_free').qposadr[0]), adr=[int(self.m.joint(f'{pre}{side}_{j}').qposadr[0]) for j in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')], flex=[])
            rig = Hh.HandRig.build(self.m, pre, side)
            for fi in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                for jn in rig.flexion_joints(fi): h['flex'].append((0 if fi == 'thumb' else (1 if fi in ('index', 'middle') else 2), rig.finger_qpos[jn], *rig.finger_range[jn]))
            self.H.append(h)
        self.K = cfg['knots']; self.nh = len(self.H); self.dim = 2 + 3 * self.nh + 6 * self.K * self.nh
        self.scale = np.r_[0.15, 0.08, np.tile([0.12, 0.12, 0.12], self.nh), np.tile([0.12, 0.08, 0.08, 0.008, 0.008, 0.008], self.K * self.nh)]
        self.bound = np.r_[0.4, 0.2, np.tile([0.8, 0.8, 0.8], self.nh), np.tile([0.6, 0.4, 0.4, 0.03, 0.03, 0.03], self.K * self.nh)]
        self.reg = B.register(cfg['ep'], {nm: Path(MD[cfg['ep']]) / f'{nm}.obj' for nm in B.NAMES}); self.tgt = B.target(cfg['ep'])
        self.qstart = None
    def commands(self, x):
        c = self.cfg; Q = self.Q0.copy(); t = self.t; tl = c['t_lock'] + x[0]; ramp = max(0.08, c['ramp'] + x[1])
        k0 = int(round(tl * self.fps)); tr = self.tr; ton = min(c.get('t_on') or tl, tl); w = sm((t - ton) / ramp) * (1 - sm((t - (c['t_off'] - ramp)) / ramp)); kn = np.linspace(ton, c['t_off'], self.K); o = 2 + 3 * self.nh
        for hi, h in enumerate(self.H):
            adr = h['adr']; oi = h['oi']; ps0 = tr['object_positions'][k0, oi]; Rs0 = R.from_quat(tr['object_quaternions_wxyz'][k0, oi][[1, 2, 3, 0]])
            Rw0 = R.from_euler('XYZ', Q[k0, adr[3:]]); off_p = Rs0.inv().apply(Q[k0, adr[:3]] - ps0); off_R = Rs0.inv() * Rw0
            P = x[o + 6 * self.K * hi:o + 6 * self.K * (hi + 1)].reshape(self.K, 6); D = np.stack([np.interp(t, kn, P[:, j]) for j in range(6)], 1)
            oc = h['oc']; P_ = self.qt[:, oc:oc + 3]; RT = R.from_quat(self.qt[:, oc + 3:oc + 7], scalar_first=True)
            for k in range(1, self.n):
                if w[k] <= 0: continue
                FR = RT[k] * R.from_rotvec(D[k, :3]) * off_R; Fp = P_[k] + RT[k].apply(D[k, 3:] + R.from_rotvec(D[k, :3]).apply(off_p))
                Bp = Q[k, adr[:3]]; BR = R.from_euler('XYZ', Q[k, adr[3:]])
                Q[k, adr[:3]] = (1 - w[k]) * Bp + w[k] * Fp; Rn = BR * R.from_rotvec((BR.inv() * FR).as_rotvec() * w[k])
                Q[k, adr[3:]] = continuous_euler_xyz(R.from_matrix(Rn.as_matrix()[None]), Q[k - 1, adr[3:]])[0]
            sq = x[2 + 3 * hi:5 + 3 * hi]
            for gi, col, lo, hi_ in h['flex']: Q[:, col] = np.clip(Q[:, col] + sq[gi] * w, lo, hi_)
        Q[0] = self.Q0[0]; return Q
    def score(self, qpos):
        P = {}
        for nm, a in (('blue_cup', self.cc), ('wooden_spoon', self.sc)):
            T = np.tile(np.eye(4), (len(qpos), 1, 1)); T[:, :3, :3] = R.from_quat(qpos[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = qpos[:, a:a + 3]; P[nm] = T
        e = B.per_frame(self.cfg['ep'], P, self.reg, self.tgt); sw = float(self.cfg.get('scoop_w') or 1.0)
        if sw > 1.0:
            N_ = len(e[B.NAMES[0]][0]); wt = np.ones(N_); wt[65:101] = sw; wt[130:171] = sw; wt /= wt.mean()
            res = {nm: (float((e[nm][0] * wt).mean()), float((e[nm][1] * wt).mean())) for nm in B.NAMES}
        else: res = {nm: (float(e[nm][0].mean()), float(e[nm][1].mean())) for nm in B.NAMES}
        ratios = [max(p / 1.2, d / 15) for p, d in res.values()]
        if not np.isfinite(ratios).all(): return 1e6, res
        return float(max(ratios) + 0.25 * sum(ratios)), res
    def evaluate(self, x):
        try:
            Q = self.commands(x)
            if self.qstart is None:
                d = mujoco.MjData(self.m); mujoco.mj_resetDataKeyframe(self.m, d, 0); self.qstart = d.qpos.copy()
            out = servo_rollout(self.m, Q, self.fps, qpos_start=self.qstart, record_substeps=False)
            c, res = self.score(np.asarray(out['qpos'])); return dict(cost=min(c, 1e6), objects=res)
        except Exception as ex: return dict(cost=1e6, error=str(ex)[:200])
W = None
def init(cfg):
    global W; W = Plan(cfg)
def ev(x): return W.evaluate(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('base'); ap.add_argument('epd'); ap.add_argument('out')
    ap.add_argument('--t-lock', type=float, required=True); ap.add_argument('--t-off', type=float, required=True); ap.add_argument('--ramp', type=float, default=0.3)
    ap.add_argument('--knots', type=int, default=8); ap.add_argument('--samples', type=int, default=64); ap.add_argument('--iterations', type=int, default=12)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--resume', action='store_true'); ap.add_argument('--init'); ap.add_argument('--no-left', action='store_true'); ap.add_argument('--t-on', type=float, default=None); ap.add_argument('--scoop-w', type=float, default=1.0); ap.add_argument('--scene', help='physics scene for every rollout (must be a _freeze fixed point so execute_reference reproduces it); default = base execution scene')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(ep=a.ep, base=str(Path(a.base).resolve()), epd=str(Path(a.epd).resolve()), t_lock=a.t_lock, t_off=a.t_off, ramp=a.ramp, knots=a.knots, left=not a.no_left, t_on=a.t_on, scoop_w=a.scoop_w, scene=str(Path(a.scene).resolve()) if a.scene else None)
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
    Q = plan.commands(best); refp = TR.save_reference(out / 'reference.npz', plan.m, Q, plan.ref['labels'], plan.fps, plan.ref['segment'], plan.ref['segments'], source=f'fcem3.py best plan: {json.dumps(cfg)}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(plan.scene, refp, out / 'execution'); mf = Path(cfg['epd']) / 'input_manifest.json'; mm = json.loads(mf.read_text())
    mt = evaluate_files(mm['source'], ex['trajectory'], mf, ex['scene'], Path(cfg['epd']) / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
    c, res = plan.score(np.load(ex['trajectory'])['qpos'])
    (out / 'result.json').write_text(json.dumps(dict(episode=mm['episode'], input_dir=cfg['epd'], reference=str(refp), execution=ex, metrics=mt, best=best_rep, best_parameters=best.tolist(), scorer_faithful=dict(cost=c, objects=res), cfg=cfg, script=__file__), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    print('RESULT', o['mp_sr'], 'proxy', {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, 'scorer', {k: (round(p, 2), round(d, 1)) for k, (p, d) in res.items()}, flush=True)
if __name__ == '__main__': main()
