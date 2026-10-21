"""Evaluate a trained GCRL transition policy on every (start, goal) pair and render feedback videos.
Numbers: per pair, object kept (< 2 cm / 15 deg from where it started), goal reached (mean finger-joint error to the goal < 0.15 rad and
palm within 2 cm / 10 deg of the goal pose in the object frame), drop. Video: a grid of selected pairs; each panel = the physics rollout
with the trained policy + the GOAL hand pose as a transparent light-blue ghost; captions show phase, object error and finger error.
Usage (training image, GPU): eval_transition_video.py ROOT SEG POLICY.safetensors OUT.mp4 [PAIRS="start:goal,..."]"""
import json, os, re, sys, subprocess, hashlib
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, warp as wp, newton, cv2, imageio_ffmpeg
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.lifecycle.reset import build_reference_joint_q
from flash_chord.runtime.jax import to_jax
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.retargeting.parallel_rollout import pack_candidate_references, install_candidate_references
from flash_chord.visualization.viewer import ViewerConfig, configure_camera, configure_render_style, create_viewer
ROOT, SEG, POL, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]); SEL = sys.argv[5] if len(sys.argv) > 5 else ''
EP = [p.name for p in (ROOT / 'frozen').iterdir()][0]; PARQ = ROOT / 'frozen' / EP / 'reference.parquet'
TMP = OUT.parent / (OUT.stem + '_parts'); TMP.mkdir(parents=True, exist_ok=True); PANEL = os.environ.get('TV_PANEL')
meta = json.loads((POL.parent / 'pairs.json').read_text()); names = meta['names']; HOLD_A, TRANS = meta['hold_a'], meta['transition']
pairs = [(a, b) for a in meta['starts'] for b in names]; W = len(pairs); PW, PH, HEAD = 400, 300, 46
B = ROOT; a = type('A', (), dict(phase='pilot', seed=44, worlds=W))(); train = ROOT / 'runs/eval_tmp'; total = interval = search_interval = W * 1024
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
ov = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.', 'task.robot_reference', 'task.parquet', 'training.', 'scene.world_count'))]
PPO = POL.name.startswith('model_')
ov += [f'task.parquet={PARQ}', f'scene.world_count={W}'] + (['training=reference_ppo'] if PPO else [])
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose(config_name='train_flash_sac', overrides=ov)
resolved = OmegaConf.to_container(cfg, resolve=True); apply_evaluation_composition(resolved, world_count=W)
env_cfg = instantiate_typed(OmegaConf.create(resolved['env']), RLEnvConfig); emb = instantiate_typed(cfg.embodiment, Embodiment)
if PPO:
    from flash_chord.training.ppo.config import TrainingConfig as PPOConfig
    training = instantiate_typed(cfg.training, PPOConfig)
else:
    training = instantiate_typed(cfg.training, TrainingConfig)
fit = np.load(ROOT / 'priors/dexfit_types.npz'); fnames = [str(x).replace('dexonomy_', '') for x in fit['prototype_names']]
order = ['thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP', 'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP', 'middle_MCP_FE', 'middle_MCP_AA',
         'middle_PIP', 'middle_DIP', 'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP', 'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP']
flex = np.array([not (j.endswith('AA') or j in ('thumb_CMC_FE', 'pinky_CMC')) for j in order], float); side = SEG.split('_')[0]
def build_Q(setup, wpairs):
    scene = setup.scene; F = scene.robot_reference.num_frames
    builder = newton.ModelBuilder(); layout = emb.build(builder); mdl = builder.finalize(device='cpu', skip_all_validations=True)
    jn = list(layout.scalar_joints.names); qids = list(layout.scalar_joints.q_ids); dids = np.asarray(layout.scalar_joints.dof_ids)
    lo, hi = mdl.joint_limit_lower.numpy()[dids], mdl.joint_limit_upper.numpy()[dids]
    balls = {('left' if 'left' in str(builder.joint_label[j]) else 'right'): (builder.joint_q_start[j], builder.joint_qd_start[j]) for j, k in enumerate(builder.joint_type) if k == newton.JointType.BALL}
    pc = [qids[jn.index(f'{side}_wrist_{c}')] for c in 'xyz']; bq, bd = balls[side]; fc = [qids[jn.index(f'{side}_{j}')] for j in order]
    o = int(fit[SEG + '_meta'][3]); op = np.asarray(setup.reference.object_body_pos_w())[0, o]; Ro = R.from_quat(np.asarray(setup.reference.object_body_quat_w())[0, o][[1, 2, 3, 0]])
    def grasp(n):
        i = fnames.index(n); return op + Ro.apply(fit[SEG + '_palm_in_obj_pos'][i]), Ro * R.from_quat(fit[SEG + '_palm_in_obj_quat_xyzw'][i]), fit[SEG + '_fingers'][i] + .3 * flex
    sm = lambda x: .5 - .5 * np.cos(np.pi * np.clip(x, 0, 1)); wgt = sm((np.arange(F) - HOLD_A) / TRANS)
    Q = np.repeat(scene.robot_reference.joint_q.astype(np.float64)[None], len(wpairs), 0); TG = np.repeat(scene.robot_reference.joint_target.astype(np.float64)[None], len(wpairs), 0)
    for w, (sa, gb) in enumerate(wpairs):
        pa, ra, fa = grasp(sa); pb, rb, fb = grasp(gb); rot = R.from_rotvec(wgt[:, None] * (rb * ra.inv()).as_rotvec()[None]) * ra
        Q[w][:, pc] = (1 - wgt)[:, None] * pa + wgt[:, None] * pb; Q[w][:, bq:bq + 4] = rot.as_quat()
        Q[w][:, fc] = np.clip((1 - wgt)[:, None] * fa + wgt[:, None] * fb, lo[[qids.index(c) for c in fc]], hi[[qids.index(c) for c in fc]])
        TG[w][:, dids] = Q[w][:, qids]
        for sd, (qs, ds) in balls.items(): TG[w][:, ds:ds + 3] = R.from_quat(Q[w][:, qs:qs + 4]).as_rotvec()
    return Q, TG, fc, pc, bq, o
def mk(worlds):
    return setup_scene(parquet=str(PARQ), control_fps=env_cfg.sim.fps, motion_speed=1., embodiment=emb, collision=instantiate_typed(cfg.collision, CollisionPolicy),
                       world_count=worlds, include_support=resolved['scene'].get('support', True), decompose_objects=True)
with wp.ScopedDevice('cuda:0'):
    if PANEL is None and not (TMP / 'rollout.npz').exists():
        setup = mk(W); Q, TG, fc, pc, bq, o = build_Q(setup, pairs); env = RLEnv(setup.scene, setup.reference, config=env_cfg); env.reset(frame_id=0)
        install_candidate_references(env, pack_candidate_references(env, Q.astype(np.float32), TG.astype(np.float32)), allow_auto_reset=True)
        env._step_graph = None; env.reset(frame_id=0); env.capture_step(); env.reset(frame_id=0)
        ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=PPO); obs = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
        if PPO:   # arms B/C (PPO checkpoints model_<iteration>.safetensors)
            from flash_chord.training.ppo.learner import create_learner
            from flash_chord.training.ppo.evaluation import policy_action, EvaluationConfig
            from flash_chord.training.checkpoint import load_checkpoint_for_inference
            lrn = load_checkpoint_for_inference(str(POL), create_learner(ad, training)); ecfg = EvaluationConfig(checkpoint=str(POL)); key = None; st = None
            pol = lambda st_, o_, k_: (policy_action(lrn, o_, training, ecfg), k_)
        else:
            st, pol, key, _ = load_policy_for_inference(str(POL), training, obs, ad.action_dim, deterministic=True)
        ids = env.command.body_ids_w.numpy().reshape(W, -1)[:, o]; F = setup.scene.robot_reference.num_frames; BQ, JQ = [], []
        nq = setup.scene.model.joint_coord_count // W
        for t in range(F - 1):
            act, key = pol(st, obs, key); obs = jnp.array(ad.step(act).observation, copy=True)
            BQ.append(env.state_0.body_q.numpy().copy()); JQ.append(env.state_0.joint_q.numpy().reshape(W, nq).copy())
        BQ, JQ = np.asarray(BQ), np.asarray(JQ); nb = BQ.shape[1] // W; B4 = BQ.reshape(len(BQ), W, nb, 7)
        obj = np.stack([B4[:, w, ids[w] - w * nb] for w in range(W)], 1)                    # (T, W, 7)
        oerr = np.linalg.norm(obj[:, :, :3] - obj[:1, :, :3], axis=-1) * 100
        orot = np.degrees((R.from_quat(obj[:, :, 3:].reshape(-1, 4)) * R.from_quat(np.repeat(obj[:1, :, 3:], len(obj), 0).reshape(-1, 4)).inv()).magnitude().reshape(len(obj), W))
        ferr = np.abs(JQ[:, :, fc] - Q[:, -1, fc][None]).mean(-1)   # (T, W): mean finger-joint error to the goal posture
        kept = (oerr.max(0) < 2) & (orot.max(0) < 15); reached = ferr[-1] < .15; dropped = oerr.max(0) > 5
        res = [dict(start=a_, goal=b_, kept=bool(kept[w]), goal_reached=bool(reached[w]), dropped=bool(dropped[w]), max_object_cm=round(float(oerr[:, w].max()), 2),
                    max_object_deg=round(float(orot[:, w].max()), 1), final_finger_err_rad=round(float(ferr[-1, w]), 3)) for w, (a_, b_) in enumerate(pairs)]
        summary = dict(pairs=len(pairs), success=float(np.mean(kept & reached)), kept=float(kept.mean()), reached=float(reached.mean()), dropped=float(dropped.mean()), results=res)
        OUT.with_suffix('.json').write_text(json.dumps(summary, indent=1))
        print('[eval] success %.3f kept %.3f reached %.3f dropped %.3f' % (summary['success'], summary['kept'], summary['reached'], summary['dropped']), flush=True)
        np.savez(TMP / 'rollout.npz', BQ=BQ, Q=Q, oerr=oerr, ferr=ferr, nb=nb, obj0=obj[0, :, :3])
    d = np.load(TMP / 'rollout.npz'); BQ, Q, nb = d['BQ'], d['Q'], int(d['nb'])
    if SEL: show = [p for p in [tuple(x.split(':')) for x in SEL.split(',')] if p in pairs]
    else:
        res = json.loads(OUT.with_suffix('.json').read_text())['results']; vg = [r for r in res if r['goal'] == 'video']
        good = sorted([r for r in res if r['kept'] and r['goal_reached'] and r['goal'] != 'video'], key=lambda r: r['final_finger_err_rad'])[:2]
        bad = sorted([r for r in res if not r['kept']], key=lambda r: -r['max_object_cm'])[:1]
        show = [(r['start'], r['goal']) for r in vg[:3] + good + bad]
    if PANEL is not None:
        k, kind = PANEL.split(':'); k = int(k); w = pairs.index(show[k]); setup = mk(1); s = setup.scene
        tgt = d['obj0'][w].astype(np.float64); pos = tgt + np.array([.22, -.28, .18])
        flags = s.model.shape_flags.numpy(); saved = flags.copy()
        if kind == 'ghost':
            flags[:s.collision_layout.robot.start] = 0; flags[s.collision_layout.robot.stop:] = 0; s.model.shape_flags.assign(flags)
        vc = ViewerConfig(backend='mp4', output_path=str(TMP / f'{kind}{k}.mp4'), num_frames=len(BQ), video_fps=20, width=PW, height=PH, headless=True, realtime=False,
                          camera_position=tuple(pos), camera_target=tuple(tgt), background_color=(0., 0., 0.), support_color=(.96, .62, .33))
        v = create_viewer(vc); v.set_model(s.model); configure_camera(v, vc); configure_render_style(v, vc, scene=s); state = s.model.state()
        if kind == 'ghost':
            q = build_reference_joint_q(s, setup.reference)[-1].copy(); q[:s.robot_reference.num_joint_q] = Q[w][-1]
        try:
            for i in range(len(BQ)):
                if kind == 'ghost':
                    state.joint_q.assign(q.astype(np.float32)); state.joint_qd.zero_(); newton.eval_fk(s.model, state.joint_q, state.joint_qd, state)
                else:
                    state.body_q.assign(BQ[i].reshape(len(pairs), nb, 7)[w].astype(np.float32))
                v.begin_frame(i / 20); v.log_state(state); v.end_frame()
        finally:
            v.close(); s.model.shape_flags.assign(saved)
        sys.exit(0)
for k in range(len(show)):
    for kind in ('real', 'ghost'):
        if not (TMP / f'{kind}{k}.mp4').exists(): subprocess.run([sys.executable, __file__, *sys.argv[1:]], check=True, env={**os.environ, 'TV_PANEL': f'{k}:{kind}'})
def frames_of(p):
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, im = cap.read()
        if not ok: break
        out.append(im)
    return out
res = {(r['start'], r['goal']): r for r in json.loads(OUT.with_suffix('.json').read_text())['results']}
REAL = [frames_of(TMP / f'real{k}.mp4') for k in range(len(show))]; GH = [frames_of(TMP / f'ghost{k}.mp4') for k in range(len(show))]
n = min(len(x) for x in REAL + GH); C = 3; Rw = (len(show) + C - 1) // C; lb = np.array([250, 200, 120], np.float32)
raw = TMP / 'grid.mp4'; wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 10, (C * PW, Rw * (PH + HEAD) + 30))
for i in range(n):
    cv = np.full((Rw * (PH + HEAD) + 30, C * PW, 3), 24, np.uint8); ph = 'holding start grasp' if i < HOLD_A else ('transition' if i < HOLD_A + TRANS else 'holding goal grasp')
    cv2.putText(cv, f'GCRL in-hand transition (trained policy, physics). light blue = GOAL hand pose.  t={i / 20:.2f}s  [{ph}]  half speed', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
    for k, (sa, gb) in enumerate(show):
        r_, c_ = divmod(k, C); y = 30 + r_ * (PH + HEAD); x = c_ * PW; w = pairs.index((sa, gb))
        g = GH[k][i].astype(np.float32); m = (g.max(axis=2, keepdims=True) > 12).astype(np.float32) * .45
        im = (REAL[k][i].astype(np.float32) * (1 - m) + m * (.7 * lb + .3 * g)).clip(0, 255).astype(np.uint8)
        oe = float(d['oerr'][min(i, len(d['oerr']) - 1), w]); fe = float(d['ferr'][min(i, len(d['ferr']) - 1), w]); rr = res[(sa, gb)]
        col = (80, 200, 80) if rr['kept'] and rr['goal_reached'] else ((60, 180, 230) if rr['kept'] else (60, 60, 230))
        cv2.rectangle(im, (1, 1), (PW - 2, PH - 2), col, 3)
        cv2.putText(cv, f'{sa.replace("_", " ")} -> {gb.replace("_", " ")}', (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(cv, f'object moved {oe:.1f} cm   finger err to goal {fe:.2f} rad', (x + 8, y + 38), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
        cv[y + HEAD:y + HEAD + PH, x:x + PW] = im
    wr.write(cv)
wr.release()
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(OUT)], check=True)
print('[eval] VIDEO', OUT, show, flush=True)
