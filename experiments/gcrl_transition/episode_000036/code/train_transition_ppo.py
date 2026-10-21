"""ARM B (SimToolReal recipe, adapted): PPO + progress reward against a running best + sustained-success bonus, drop termination.
Same task/data as train_transition.py (FlashSAC arm A).
Goal-conditioned in-hand transition policy (GCRL) on top of the UniRetarget tracking stack.
Each world gets its own hand reference: hold START grasp A (frames 0-9) -> smooth transition (10-39) -> hold GOAL grasp B (40-T),
with the object target fixed in the air (make_transition_parquet.py). The goal enters the policy through the commanded hand / object
targets in its observation. Reward: hand tracking + OBJECT tracking (object must stay put while the hand regrasps); terminate on drop.
Usage (training image, GPU): train_transition.py ROOT SEG STEPS [WORLDS=2048]"""
import json, os, re, sys, time, hashlib
from pathlib import Path
import numpy as np, jax, warp as wp, newton
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import build_checkpoint_metadata, critic_schema, policy_schema, resolved_reference_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.runner import run_training
import jax.numpy as jnp
from flash_chord.training.progress import ConsoleTrainingProgress
from flash_chord.retargeting.parallel_rollout import pack_candidate_references, install_candidate_references
ROOT, SEG, STEPS = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]); W = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
EP = ROOT.name if (ROOT / 'frozen' / ROOT.name).exists() else [p.name for p in (ROOT / 'frozen').iterdir()][0]
PARQ = ROOT / 'frozen' / EP / 'reference.parquet'; OUT = ROOT / 'runs' / f'gcrl_ppo_{SEG}'; OUT.mkdir(parents=True, exist_ok=True)
START = os.environ.get('GCRL_START', 'large_diameter,power_sphere,fixed_hook,light_tool,extensior_type,small_diameter').split(',')
HOLD_A, TRANS = 10, 30
B = ROOT; a = type('A', (), dict(phase='pilot', seed=44, worlds=W))(); train = OUT; total = STEPS; interval = search_interval = 5 * W * 1024
cmd = re.search(r"cmd=\[(.*?)\]\n", (ROOT / 'code/run_mixed.py').read_text(), re.S).group(1)
ov = [x for x in eval('[' + cmd + ']') if not x.startswith(('scripts/', 'output_dir=', 'hydra.', '+bilevel', '+adaptive_voc', 'logging.', 'task.robot_reference', 'task.parquet',
      'training.', '+training.', 'scene.world_count', 'termination.object', 'objective.object_keypoints', 'env.voc_scale', 'reset.'))]
ov += ['training=reference_ppo', f'training.max_iterations={STEPS // (W * 24)}', 'training.save_interval=500', f'task.parquet={PARQ}', f'scene.world_count={W}',
       'objective.object_keypoints.weight=2.0', 'termination.object_position.enabled=true', 'termination.object_position.threshold=0.05',
       'termination.object_orientation.enabled=true', 'termination.object_orientation.threshold=0.5', 'env.voc_scale=0.0', 'reset.voc_decay_steps=0', 'logging.mode=disabled']
with initialize_config_dir(config_dir='/workspace/src/flash_chord/configs', version_base='1.3'):
    cfg = compose(config_name='train_flash_sac', overrides=ov)
env_cfg = instantiate_typed(cfg.env, RLEnvConfig); emb = instantiate_typed(cfg.embodiment, Embodiment)
training = instantiate_typed(cfg.training, TrainingConfig)
fit = np.load(ROOT / 'priors/dexfit_types.npz'); names = [str(x).replace('dexonomy_', '') for x in fit['prototype_names']]
order = ['thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP', 'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP', 'middle_MCP_FE', 'middle_MCP_AA',
         'middle_PIP', 'middle_DIP', 'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP', 'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP']
flex = np.array([not (j.endswith('AA') or j in ('thumb_CMC_FE', 'pinky_CMC')) for j in order], float)
side = SEG.split('_')[0]
with wp.ScopedDevice(training.device):
    setup = setup_scene(parquet=str(PARQ), control_fps=env_cfg.sim.fps, motion_speed=1., embodiment=emb, collision=instantiate_typed(cfg.collision, CollisionPolicy),
                        world_count=W, include_support=cfg.scene.support, decompose_objects=True)
    scene = setup.scene; F = scene.robot_reference.num_frames
    builder = newton.ModelBuilder(); layout = emb.build(builder); mdl = builder.finalize(device='cpu', skip_all_validations=True)
    jn = list(layout.scalar_joints.names); qids = list(layout.scalar_joints.q_ids); dids = np.asarray(layout.scalar_joints.dof_ids)
    lo, hi = mdl.joint_limit_lower.numpy()[dids], mdl.joint_limit_upper.numpy()[dids]
    balls = {('left' if 'left' in str(builder.joint_label[j]) else 'right'): (builder.joint_q_start[j], builder.joint_qd_start[j])
             for j, k in enumerate(builder.joint_type) if k == newton.JointType.BALL}
    pc = [qids[jn.index(f'{side}_wrist_{c}')] for c in 'xyz']; bq, bd = balls[side]; fc = [qids[jn.index(f'{side}_{j}')] for j in order]
    base_q = scene.robot_reference.joint_q.astype(np.float64); base_t = scene.robot_reference.joint_target.astype(np.float64)
    op = np.asarray(setup.reference.object_body_pos_w())[0, int(fit[SEG + '_meta'][3])]; oq = np.asarray(setup.reference.object_body_quat_w())[0, int(fit[SEG + '_meta'][3])][[1, 2, 3, 0]]
    Ro = R.from_quat(oq)
    def grasp(i):
        return op + Ro.apply(fit[SEG + '_palm_in_obj_pos'][i]), (Ro * R.from_quat(fit[SEG + '_palm_in_obj_quat_xyzw'][i])), fit[SEG + '_fingers'][i] + .3 * flex
    starts = [names.index(s) for s in START if s in names]; goals = list(range(len(names)))
    pairs = [(s, g) for s in starts for g in goals]; rng = np.random.default_rng(0); wpair = [pairs[w % len(pairs)] for w in range(W)]
    sm = lambda x: .5 - .5 * np.cos(np.pi * np.clip(x, 0, 1)); tt = np.arange(F); wgt = sm((tt - HOLD_A) / TRANS)
    Q = np.repeat(base_q[None], W, 0); TG = np.repeat(base_t[None], W, 0)
    for w, (s, g) in enumerate(wpair):
        pa, ra, fa = grasp(s); pb, rb, fb = grasp(g)
        rel = (rb * ra.inv()).as_rotvec()
        rot = R.from_rotvec(wgt[:, None] * rel[None]) * ra
        Q[w][:, pc] = (1 - wgt)[:, None] * pa + wgt[:, None] * pb; Q[w][:, bq:bq + 4] = rot.as_quat()
        Q[w][:, fc] = np.clip((1 - wgt)[:, None] * fa + wgt[:, None] * fb, lo[[qids.index(c) for c in fc]], hi[[qids.index(c) for c in fc]])
        TG[w][:, dids] = Q[w][:, qids]
        for sd, (qs, ds) in balls.items(): TG[w][:, ds:ds + 3] = R.from_quat(Q[w][:, qs:qs + 4]).as_rotvec()
    env = RLEnv(scene, setup.reference, config=env_cfg); env.reset(frame_id=0)
    install_candidate_references(env, pack_candidate_references(env, Q.astype(np.float32), TG.astype(np.float32)), allow_auto_reset=True)
    env._step_graph = None; env.reset(frame_id=0); env.capture_step(); env.reset(frame_id=0)
    jax_env = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=True)   # PPO reads reward from the step metrics
    class ProgressReward:
        # SimToolReal-style goal reward: reward only NEW progress toward the goal (running best of the hand-goal closeness, reset per
        # episode) plus a bonus while the goal grasp is held with the object still; base tracking reward kept.
        def __init__(s, inner):
            s.inner = inner; names = list(inner.objective_term_names); s.ih = names.index('hand_keypoints'); s.io = names.index('object_keypoints')
            s.best = jnp.full((inner.world_count,), -1.0); s.printed = False
        def __getattr__(s, k): return getattr(s.inner, k)
        def reset(s, *a, **k): s.best = jnp.full((s.inner.world_count,), -1.0); return s.inner.reset(*a, **k)
        def step(s, action):
            tr = s.inner.step(action); h = tr.objective_terms[:, s.ih]; o = tr.objective_terms[:, s.io]
            if not s.printed: print(f'[gcrl-ppo] closeness terms: hand {float(h.min()):.3f}..{float(h.max()):.3f}, object {float(o.min()):.3f}..{float(o.max()):.3f}', flush=True); s.printed = True
            best = jnp.where(s.best < 0, h, s.best); prog = jnp.maximum(h - best, 0.); s.best = jnp.maximum(best, h)
            bonus = ((h > .85) & (o > .85)).astype(jnp.float32)
            done = jnp.logical_or(tr.terminated.astype(bool), tr.truncated.astype(bool)); s.best = jnp.where(done, -1.0, s.best)
            return tr.replace(reward=tr.reward + 10. * prog + bonus)
    jax_env = ProgressReward(jax_env)
    ec = resolved_dict(cfg); ec['task']['parquet'] = str(PARQ); ec['reference'] = resolved_reference_config(setup.reference, str(PARQ))
    meta = build_checkpoint_metadata(ec, policy_schema(env.action, env.observation_strategy), critic_schema(env.action, env.observation_strategy, getattr(jax_env, 'critic_context_names', ())))
    (OUT / 'pairs.json').write_text(json.dumps(dict(names=names, starts=[names[s] for s in starts], pairs_per_world=[[names[s], names[g]] for s, g in wpair[:len(pairs)]],
                                                    frames=F, hold_a=HOLD_A, transition=TRANS), indent=1))
    class L:
        def __init__(s, p): s.f = open(p, 'a')
        def log(s, m, step): s.f.write(json.dumps({'step': step, **{k: (float(v) if hasattr(v, '__float__') else str(v)) for k, v in m.items()}}) + '\n'); s.f.flush()
        def close(s): s.f.close()
    print(f'[gcrl] {len(starts)} start grasps x {len(goals)} goals = {len(pairs)} pairs over {W} worlds; {F} frames; training {STEPS} steps', flush=True)
    learner = run_training(jax_env, training, logger=L(OUT / 'training_metrics.jsonl'), checkpoint_dir=OUT, log_interval=20, voc_scale_override=0.0,
                           checkpoint_metadata=meta, progress=ConsoleTrainingProgress(training.max_iterations * W * 24, OUT))
    print('[gcrl-ppo] done', flush=True)
