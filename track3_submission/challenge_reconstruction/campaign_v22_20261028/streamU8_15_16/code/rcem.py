"""[U8 copy: + --P/--D/--goal passthrough] Stream M8 (v23): generalization of stream I release_cem.py to any episode (scorer = mlib/mres cost vs target_final_v5 with the
v22b caps; --ep, --meshes, --side). Original doc:
Stream I (v22 phase 7) ep33: CEM over a RIGHT-hand release primitive wrapped around an existing distilled actor
(release_feedback_policy.ReleasePolicy over pinky_margin_policy_v2.PinkyMarginPolicy (no margins) over the unchanged LearnedPolicy weights).

Before release_start_s the wrapped actor's commands are bit-identical to the base actor, so the rollout prefix is the base actor's own
rollout. After the start the right-hand learned phase is held (proprioceptive feedback still active) and the fitted primitive places
(world wrist translation + local wrist rotation over place_dur, optional closing bias), opens the fingers (scale x (open pose - pose at
start), open pose = the base actor's own opened command) and retreats. Every sample = closed-loop actor rollout under the evaluator
substep schedule (evaluate_policy_component loop), continued bit-exactly from the cached base-actor state at the earliest admissible
start frame (mj_copyData). Cost = scorer-faithful (streamB blib.per_frame: MP-SR registration of the shipped meshes onto
target_final_v5, spoon-anchored frame-0 alignment) worst ratio max(pos/2.4, deg/24) + 0.25 * sum.
usage: release_cem.py BASE_ACTOR_DIR OUT [--tmin 13.6] [--tmax 15.0] [--samples 64] [--iterations 16] [--workers 16] [--seed 1] [--init JSON]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, argparse, time, tempfile, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
import mlib as L, mres
from component_policy import ComponentPolicy
TMP = Path(os.environ.get('TMPDIR', '/tmp'))
# x = [start, pdx, pdy, pdz, rx, ry, rz, place_dur, open_dur, open_scale, rtx, rty, rtz, retreat_dur, closed_bias]
NAMES = ['start', 'pdx', 'pdy', 'pdz', 'rx', 'ry', 'rz', 'place_dur', 'open_dur', 'open_scale', 'rtx', 'rty', 'rtz', 'retreat_dur', 'closed_bias']
X0 = np.array([14.0, 0, 0, 0, 0, 0, 0, 0.8, 0.4, 1.0, 0, 0, 0.08, 0.8, 0.0])
SD = np.array([0.3, 0.03, 0.03, 0.03, 0.25, 0.25, 0.25, 0.3, 0.2, 0.3, 0.03, 0.03, 0.03, 0.3, 0.1])
LO = np.array([13.6, -0.15, -0.15, -0.2, -1.2, -1.2, -1.2, 0.15, 0.1, 0.0, -0.1, -0.1, -0.02, 0.2, -0.3])
HI = np.array([15.2, 0.15, 0.15, 0.1, 1.2, 1.2, 1.2, 2.0, 1.5, 1.5, 0.1, 0.1, 0.15, 2.0, 0.5])
SIDE = ['right']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def wrapper_data(base, x, open_cmd, roll_ctrl, fps):
    """checkpoint dict: base LearnedPolicy arrays + empty PinkyMargin + right-hand ReleasePolicy fields."""
    d = dict(base); names = [str(s) for s in d['actuator_names']]
    S_ = SIDE[0]; Rr = [i for i, n in enumerate(names) if n.startswith(f'robot_{S_}_{S_}_')]
    tr = [names.index(f'robot_{S_}_{S_}_pos_{a}_position') for a in 'xyz']; ro = [names.index(f'robot_{S_}_{S_}_rot_{a}_position') for a in 'xyz']
    fing = [i for i in Rr if i not in tr + ro]
    p = dict(zip(NAMES, map(float, x)))
    k0 = min(int(round(p['start'] * fps)), len(roll_ctrl) - 1)
    d.update(margin_base_module=np.array('learned_feedback_policy'), margin_base_class=np.array('LearnedPolicy'),
             margin_actuator_ids=np.zeros(0, dtype=np.int64), margin_bias_rad=np.zeros(0),
             release_base_module=np.array('pinky_margin_policy_v2'), release_base_class=np.array('PinkyMarginPolicy'),
             release_start_s=np.array(p['start']), release_open_duration_s=np.array(p['open_dur']), release_retreat_duration_s=np.array(p['retreat_dur']),
             release_actuator_ids=np.array(Rr), release_finger_actuator_ids=np.array(fing), release_translation_actuator_ids=np.array(tr),
             release_rotation_actuator_ids=np.array(ro), release_open_delta_rad=(open_cmd[fing] - roll_ctrl[k0, fing]) * p['open_scale'],
             release_retreat_m=np.array([p['rtx'], p['rty'], p['rtz']]),
             release_place_duration_s=np.array(p['place_dur']), release_place_delta_m=np.array([p['pdx'], p['pdy'], p['pdz']]),
             release_place_rotation_vector=np.array([p['rx'], p['ry'], p['rz']]))
    if p['closed_bias'] != 0: d['release_closed_bias_rad'] = np.full(len(fing), p['closed_bias'])
    return d


class Searcher:
    def __init__(self, cfg):
        self.cfg = cfg; ad = Path(cfg['actor']); ev = json.loads((ad / 'evaluation/result.json').read_text())
        self.scene = ev['execution']['scene']; self.m = mujoco.MjModel.from_xml_path(self.scene)
        with np.load(ad / 'training/policy.npz', allow_pickle=False) as z: self.base = {k: z[k] for k in z.files}
        roll = np.load(ad / 'evaluation/execution/trajectory.npz'); self.roll_q = roll['qpos']; self.roll_ctrl = roll['ctrl']
        self.fps = float(self.base['fps']); self.T = int(self.base['frames']); self.dt = float(self.m.opt.timestep)
        names = [str(s) for s in self.base['actuator_names']]
        self.open_cmd = self.roll_ctrl[int(round(cfg['open_t'] * self.fps))].copy()
        self.env = mres.Env(dict(ep=cfg['ep'], actor=cfg['actor'], meshes=cfg['meshes'], cap=cfg['cap'], P=cfg.get('P', 1.2), D=cfg.get('D', 12.0), goal=cfg.get('goal')))
        self.k_pre = int(np.floor(cfg['tmin'] * self.fps + 1e-9)); self.pre = None
    def prefix(self):
        """base actor from the keyframe to frame k_pre (evaluate_policy_component schedule); cached."""
        if self.pre is not None: return self.pre
        m = self.m; pol = ComponentPolicy(str(Path(self.cfg['actor']) / 'training/policy.npz')); d = mujoco.MjData(m)
        if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d); Q = np.zeros((self.T, m.nq)); Q[0] = d.qpos
        for t in range(1, self.k_pre + 1):
            n = round(t / (self.fps * self.dt)) - round((t - 1) / (self.fps * self.dt))
            for _ in range(n): d.ctrl[:] = pol(d.time + self.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            Q[t] = d.qpos
        exact = bool(np.array_equal(Q[:self.k_pre + 1], self.roll_q[:self.k_pre + 1]))
        self.pre = (d, Q, exact); return self.pre
    def rollout(self, ckpt):
        d0, Q0, _ = self.prefix(); m = self.m; d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0); Q = Q0.copy(); pol = ComponentPolicy(ckpt)
        for t in range(self.k_pre + 1, self.T):
            n = round(t / (self.fps * self.dt)) - round((t - 1) / (self.fps * self.dt))
            for _ in range(n):
                d.ctrl[:] = pol(d.time + self.dt, d.qpos, d.qvel); mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number: return None
            Q[t] = d.qpos
        return Q
    def score(self, Q):
        c, res, _, _ = self.env.score(Q); return c, res
    def evaluate(self, x, keep=None):
        x = np.clip(x, LO, HI)
        try:
            d = wrapper_data(self.base, x, self.open_cmd, self.roll_ctrl, self.fps)
            f = Path(keep) if keep else Path(tempfile.mkstemp(suffix='.npz', dir=TMP)[1])
            np.savez_compressed(f, **d)
            try: Q = self.rollout(str(f))
            finally:
                if not keep: f.unlink(missing_ok=True)
            if Q is None: return dict(cost=1e6, error='unstable')
            c, res = self.score(Q); return dict(cost=c, objects=res)
        except Exception as ex: return dict(cost=1e6, error=repr(ex)[:300])
W = None
def init(cfg):
    global W; SIDE[0] = cfg['side']; W = Searcher(cfg); W.prefix()
def ev(x): return W.evaluate(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('actor'); ap.add_argument('out'); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--meshes', required=True); ap.add_argument('--cap', default='{}'); ap.add_argument('--side', default='right')
    ap.add_argument('--tmin', type=float, default=13.6); ap.add_argument('--tmax', type=float, default=15.0); ap.add_argument('--open-t', type=float, default=14.4)
    ap.add_argument('--samples', type=int, default=64); ap.add_argument('--iterations', type=int, default=16); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--init'); ap.add_argument('--x0', help='comma list overriding X0'); ap.add_argument('--sd-scale', type=float, default=1.0); ap.add_argument('--lo'); ap.add_argument('--hi'); ap.add_argument('--P', type=float, default=1.2); ap.add_argument('--D', type=float, default=12.0); ap.add_argument('--goal')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    LO[0] = a.tmin; HI[0] = a.tmax
    if a.lo: LO[:] = [float(v) for v in a.lo.split(',')]
    if a.hi: HI[:] = [float(v) for v in a.hi.split(',')]
    cfg = dict(actor=str(Path(a.actor).resolve()), tmin=a.tmin, open_t=a.open_t, ep=a.ep, meshes=a.meshes, cap=json.loads(a.cap), side=a.side, P=a.P, D=a.D, goal=[float(v) for v in a.goal.split(',')] if a.goal else None); SIDE[0] = a.side
    s = Searcher(cfg); _, _, exact = s.prefix(); print('PREFIX exact vs base rollout:', exact, 'k_pre', s.k_pre, flush=True)
    base_c, base_res = s.score(s.roll_q); print('BASE', base_c, base_res, flush=True)
    mean = X0.copy()
    if a.x0: mean = np.array([float(v) for v in a.x0.split(',')])
    if a.init: mean = np.asarray(json.loads(Path(a.init).read_text())['best_parameters'])
    mean = np.clip(mean, LO, HI); std = SD * a.sd_scale; rng = np.random.default_rng(a.seed)
    best = mean.copy(); best_rep = s.evaluate(best); hist = []; t0 = time.monotonic(); print('INIT', json.dumps(best_rep), flush=True)
    (out / 'settings.json').write_text(json.dumps(dict(cfg=cfg, args=vars(a), names=NAMES, script=__file__, script_sha256=sha(__file__)), indent=1))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(cfg,)) as pool:
        for it in range(a.iterations):
            props = np.clip(np.vstack([best, mean, mean + rng.normal(size=(a.samples - 2, len(mean))) * std]), LO, HI)
            reps = list(pool.map(ev, props, chunksize=1)); order = np.argsort([r['cost'] for r in reps]); elite = props[order[:max(6, a.samples // 8)]]
            if reps[order[0]]['cost'] < best_rep['cost']: best = props[order[0]].copy(); best_rep = reps[order[0]]
            mean = 0.2 * mean + 0.8 * elite.mean(0); std = np.maximum(0.3 * std + 0.7 * elite.std(0), 0.1 * SD * a.sd_scale)
            npass = int(np.median([r['cost'] for r in reps]))
            hist.append(dict(iteration=it, best=best_rep, n_margin=npass, elapsed_s=time.monotonic() - t0)); print('IT', json.dumps(hist[-1]), flush=True)
            (out / 'progress.json').write_text(json.dumps(dict(history=hist, best_parameters=best.tolist(), best_named=dict(zip(NAMES, best.tolist())), mean=mean.tolist(), std=std.tolist())))
    # export the best wrapper checkpoint + its official evaluation (unchanged evaluate_policy_component)
    from evaluate_policy_component import evaluate
    dest = out / 'actor'; (dest / 'training').mkdir(parents=True, exist_ok=True); ck = dest / 'training/policy.npz'
    np.savez_compressed(ck, **wrapper_data(s.base, best, s.open_cmd, s.roll_ctrl, s.fps))
    tr = json.loads((Path(cfg['actor']) / 'training/training_report.json').read_text())
    tr.update(right_release_wrapper=dict(base_actor=str(Path(cfg['actor']) / 'training/policy.npz'), base_sha256=sha(Path(cfg['actor']) / 'training/policy.npz'),
              parameters=dict(zip(NAMES, best.tolist())), open_pose_source=str(Path(cfg['actor']) / 'evaluation/execution/trajectory.npz'), open_t=a.open_t,
              search=str(out), script=str(Path(__file__).resolve()), script_sha256=sha(__file__)), checkpoint=str(ck), checkpoint_sha256=sha(ck))
    (dest / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
    ev_ = json.loads((Path(cfg['actor']) / 'evaluation/result.json').read_text())
    evaluate(a.ep, s.scene, str(ck), str(dest / 'evaluation'), ev_['input_dir'])
    q = np.load(dest / 'evaluation/execution/trajectory.npz')['qpos']; c, res = s.score(q)
    print('RESULT search', best_rep.get('objects'), 'official-eval', {k: (round(p, 3), round(dg, 2)) for k, (p, dg) in res.items()}, 'params', dict(zip(NAMES, np.round(best, 4).tolist())), flush=True)


if __name__ == '__main__': main()
