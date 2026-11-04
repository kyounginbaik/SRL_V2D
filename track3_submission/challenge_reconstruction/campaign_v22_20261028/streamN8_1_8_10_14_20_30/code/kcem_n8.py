"""Stream N8 copy of stream G kcem.py (per-object --P cm scales, --D deg scale; any episode). Original doc: Stream G (ep10): CEM over time-varying wrist offsets (piecewise-linear knots) on top of a ported teacher, evaluated by full
physics rollouts (unchanged floating runtime, servo_rollout from the scene keyframe) against the world's tracking reference
(= target_final_v5 in our body frames placed by G; the physics frame 0 equals the target layout, so these unaligned anchor-point
errors equal the scorer's aligned errors - checked 2.59 vs 2.59 on port/p3r_none).
Plan per side s in --sides: wrist_pos(t) += sum_k w_k(t) dp_k (world frame), wrist_rot(t) = R(sum_k w_k(t) dr_k) * wrist_rot(t),
hat-function knots evenly over [t0, t1] (zero at both ends), fingers += squeeze(t) * c over the window (flexion only via all22 offsets).
Cost = max over objects of max(pos/P_o, deg/24) + 0.25 * sum, P = blue 2.4 cm, beige --beige-p (2.1) cm; hand penetration penalty.
Best plan re-executed with execute_reference + evaluate_files (teacher dir layout: reference.npz, result.json, execution/).
usage: kcem.py BASE_DIR EPD OUT --sides right --t0 1.5 --t1 7.5 --knots 8 [--samples 64 --iterations 12 --workers 16 --seed 1]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from refine_grasp_cem import Experiment, dump
from tool_library.common import tracking as TR
from floating_runtime import servo_rollout, execute_reference
from evaluate_mp_sr import angular_errors, evaluate_files
from wrist_reference import continuous_euler_xyz


class KExp(Experiment):
    def __init__(self, spec):
        if spec.get('scene_home'):   # ep20: the port moved the robot home keyframe (hand holds the cup at frame 0) -> scene initial state
            import refine_grasp_cem as RG; ac = RG.np.allclose; RG.np.allclose = lambda *x, **k: True   # skip only the base-class frame-0 equality check
            try: super().__init__(spec)
            finally: RG.np.allclose = ac
        else: super().__init__(spec)
        self.home = bool(spec.get('scene_home')); self.sides = spec['sides']; self.K = spec['knots']
        t = np.arange(self.T) / self.fps; c = np.linspace(spec['t0'], spec['t1'], self.K + 2)
        self.W = np.stack([np.interp(t, c, np.eye(self.K + 2)[k]) for k in range(1, self.K + 1)], 1)   # (T, K) hat functions
        self.sq = np.interp(t, [spec['t0'], spec['t0'] + .3, spec['t1'] - .3, spec['t1']], [0, 1, 1, 0])
        per = 6 * self.K + 1; self.dim = per * len(self.sides)
        sc = np.r_[np.full(3 * self.K, spec['pos_scale']), np.full(3 * self.K, spec['rot_scale']), spec['sq_scale']]
        self.scale = np.tile(sc, len(self.sides)); self.bound = 4 * self.scale
        self.P = np.array([spec['P'].get(n, .012) for n in self.names]); self.Dm = np.deg2rad(np.array([spec.get('Dobj', {}).get(n, spec.get('D', 24.)) for n in self.names]))

    def commands(self, x):
        x = np.asarray(x); Q = self.base.copy(); per = 6 * self.K + 1
        for si, side in enumerate(self.sides):
            v = x[si * per:(si + 1) * per]; c = self.cols[side]
            dp = self.W @ v[:3 * self.K].reshape(self.K, 3); dr = self.W @ v[3 * self.K:6 * self.K].reshape(self.K, 3)
            Q[:, c[:3]] += dp
            if np.any(dr): Q[:, c[3:6]] = continuous_euler_xyz(R.from_rotvec(dr) * self.rot[side], Q[0, c[3:6]])
            if v[-1]: Q[:, c[6:]] = np.clip(Q[:, c[6:]] + self.sq[:, None] * v[-1], self.limits[side][6:, 0], self.limits[side][6:, 1])
        Q[0] = self.base[0] if self.home else self.target[0]; return Q

    def evaluate(self, x):
        Q = self.commands(x); roll = servo_rollout(self.model, Q, self.fps, qpos_start=self.initial, record_substeps=False)
        if roll['unstable'] or len(roll['qpos']) != self.T: return dict(cost=1e9, unstable=True)
        p, q = self.object_poses(roll['qpos']); ep = np.linalg.norm(p - self.tp, axis=-1)
        er = angular_errors(q[:, None], self.tq[:, None], self.axes, self.kinds)[:, 0]
        pos = ep.mean(0); ang = er.mean(0); ratio = np.maximum(pos / self.P, ang / self.Dm)
        hp = float(np.max(roll['hand_pen_m'])); cost = float(ratio.max() + .25 * ratio.sum()) + 100 * max(0., hp - .01) + .001 * float(np.mean((x / self.bound) ** 2))
        return dict(cost=cost, mean_position_cm=(100 * pos).tolist(), mean_orientation_deg=np.rad2deg(ang).tolist(), max_hand_pen_m=hp,
                    mp_sr=float(np.all((pos < .03) & (ang < np.deg2rad(30)))), unstable=False)


W = None
def init(spec):
    global W; W = KExp(spec)
def ev(x): return W.evaluate(x)


def main():
    p = argparse.ArgumentParser(); p.add_argument('base'); p.add_argument('epd'); p.add_argument('out')
    p.add_argument('--sides', default='right'); p.add_argument('--t0', type=float, default=1.5); p.add_argument('--t1', type=float, default=7.5)
    p.add_argument('--knots', type=int, default=8); p.add_argument('--samples', type=int, default=64); p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--seed', type=int, default=1); p.add_argument('--P', default='', help='obj=cm,... per-object position scale (default 1.2 cm)'); p.add_argument('--D', type=float, default=12.); p.add_argument('--scene-home', action='store_true', help='start from the scene keyframe (robot home moved by the port); frame-0 command = teacher reference'); p.add_argument('--Dobj', default='', help='obj=deg,... per-object orientation scale')
    p.add_argument('--pos-scale', type=float, default=.008); p.add_argument('--rot-scale', type=float, default=.05); p.add_argument('--sq-scale', type=float, default=.05)
    p.add_argument('--init', help='progress.json of a previous run to start from (same knots/sides)'); a = p.parse_args()
    base, epd, out = Path(a.base).resolve(), Path(a.epd).resolve(), Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    r = json.loads((base / 'result.json').read_text()); man = json.loads((epd / 'input_manifest.json').read_text())
    spec = dict(scene=r['execution']['scene'], reference=str(base / 'reference.npz'), original_reference=str(epd / 'tracking_reference.npz'), source=man['source'],
                geometry=str(epd / 'metric_geometry.json'), sides=a.sides.split(','), knots=a.knots, t0=a.t0, t1=a.t1, P={k: float(v) / 100 for k, v in (x.split('=') for x in filter(None, a.P.split(',')))}, D=a.D, scene_home=a.scene_home, Dobj={k: float(v) for k, v in (x.split('=') for x in filter(None, a.Dobj.split(',')))},
                pos_scale=a.pos_scale, rot_scale=a.rot_scale, sq_scale=a.sq_scale)
    exp = KExp(spec); rng = np.random.default_rng(a.seed); best = np.zeros(exp.dim)
    if a.init: best = np.asarray(json.loads(Path(a.init).read_text())['best_parameters'], float); assert len(best) == exp.dim
    mean = best.copy(); std = exp.scale.copy(); rep = exp.evaluate(best); hist = []; t0 = time.monotonic()
    dump(out / 'settings.json', dict(spec=spec, args=vars(a), base=str(base), objective='scorer-faithful full-episode errors vs target_final_v5 (frame 0 = target layout); no object actuation, no time editing'))
    dump(out / 'initial.json', rep); print('initial', json.dumps(rep), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(spec,)) as pool:
        for it in range(a.iterations):
            props = [best.copy(), mean.copy()]
            for j in range(a.samples - 2):
                c = best if j % 2 == 0 else mean; props.append(np.clip(c + rng.normal(size=exp.dim) * std, -exp.bound, exp.bound))
            props = np.asarray(props); reps = list(pool.map(ev, props, chunksize=1)); o = np.argsort([x['cost'] for x in reps]); el = props[o[:max(6, a.samples // 8)]]
            if reps[o[0]]['cost'] < rep['cost']: best = props[o[0]].copy(); rep = reps[o[0]]
            mean = .25 * mean + .75 * el.mean(0); std = np.maximum(.3 * std + .7 * el.std(0), .12 * exp.scale)
            row = dict(iteration=it, best=rep, elapsed_s=time.monotonic() - t0); hist.append(row)
            dump(out / 'progress.json', dict(history=hist, best_parameters=best)); print(json.dumps(row), flush=True)
    Q = exp.commands(best)
    path = TR.save_reference(out / 'reference.npz', exp.model, Q, exp.ref['labels'], exp.fps, exp.ref['segment'], exp.ref['segments'],
                             source=f'stream N8 knot CEM (copy of stream G kcem.py) on {base}', arms_coupled=False, floating_runtime=True)
    ex = execute_reference(spec['scene'], path, out / 'execution')
    met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
    dump(out / 'result.json', dict(episode=epd.name, input_dir=str(epd), parent=str(base), best=rep, parameters=best, history=hist, reference=str(path), execution=ex, metrics=met, spec=spec, learned_policy=False))
    print('RESULT', met['reference_proxy_mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}, flush=True)


if __name__ == '__main__': main()
