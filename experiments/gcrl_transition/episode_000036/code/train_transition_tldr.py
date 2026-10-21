"""ARM C (TLDR-style goal reward on the same PPO base as arm B): a temporal-distance representation phi is learned online
(||phi(x)-phi(x')|| <= 1 for consecutive states, distances between random states maximised); the goal reward is the per-step DECREASE of
||phi(x) - phi(x_goal)||, x = (finger joints, object pose in the wrist frame), x_goal = the commanded (goal) fingers and object-in-goal-wrist pose.
ARM B (SimToolReal recipe, adapted): PPO + progress reward against a running best + sustained-success bonus, drop termination.
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
PARQ = ROOT / 'frozen' / EP / 'reference.parquet'; OUT = ROOT / 'runs' / f'gcrl_tldr_{SEG}'; OUT.mkdir(parents=True, exist_ok=True)
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
    import flax.linen as nn, optax
    names_b = list(env.observation_strategy.block_names); ranges_b = list(env.observation_strategy.block_ranges)
    def blk(prefix):
        k = [i for i, n in enumerate(names_b) if n.startswith(prefix)]
        assert k, (prefix, names_b); return ranges_b[k[0]]
    RF, RO, RQ, RW, RWQ = blk('finger_joint_position'), blk('object_position_e'), blk('object_orientation_e'), blk('wrist_position_e'), blk('wrist_orientation_e')
    RCF, RCW, RCO, RCOQ = blk('command_finger'), blk('command_wrist'), blk('command_object_relative_position'), blk('command_object_relative_orientation')
    print('[gcrl-tldr] blocks', [names_b[i] for i in range(len(names_b))][:30], flush=True)
    def qmul(a, b):
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return jnp.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)
    qinv = lambda q: q * jnp.array([1., -1., -1., -1.])
    def rot(q, v): return qmul(qmul(q, jnp.concatenate([jnp.zeros(v.shape[:-1] + (1,)), v], -1)), qinv(q))[..., 1:]
    def state_and_goal(o):
        f = o[:, RF[0]:RF[1]]; pw, qw = o[:, RW[0]:RW[0] + 3], o[:, RWQ[0]:RWQ[0] + 4]; po, qo = o[:, RO[0]:RO[0] + 3], o[:, RQ[0]:RQ[0] + 4]
        x = jnp.concatenate([f, rot(qinv(qw), po - pw) * 10., qmul(qinv(qw), qo)], -1)           # object pose in the (right) wrist frame
        cw = o[:, RCW[0]:RCW[0] + 7]; cwp, cwq = cw[:, :3], cw[:, 3:7]                              # goal wrist in the current wrist frame
        cop, coq = o[:, RCO[0]:RCO[0] + 3], o[:, RCOQ[0]:RCOQ[0] + 4]                              # object target in the current wrist frame
        g = jnp.concatenate([o[:, RCF[0]:RCF[1]], rot(qinv(cwq), cop - cwp) * 10., qmul(qinv(cwq), coq)], -1)   # object in the GOAL wrist frame
        return x, g
    class Phi(nn.Module):
        @nn.compact
        def __call__(s, x):
            for h in (256, 256): x = nn.elu(nn.Dense(h)(x))
            return nn.Dense(32)(x)
    class TemporalDistanceReward:
        def __init__(s, inner):
            s.inner = inner; s.phi = Phi(); s.key = jax.random.PRNGKey(0); o0 = jnp.asarray(inner.reset())
            x0, _ = state_and_goal(o0); s.params = s.phi.init(s.key, x0[:2]); s.opt = optax.adam(3e-4); s.ost = s.opt.init(s.params)
            s.buf = jnp.zeros((65536, x0.shape[1])); s.nbuf = 0; s.pairs = []; s.prev_obs = o0; s.lam = 10.; s.t = 0
            names = list(inner.objective_term_names); s.ih = names.index('hand_keypoints'); s.io = names.index('object_keypoints')
            def loss(p, a, b, c, d):   # a,b consecutive; c,d random pairs
                ea, eb, ec, ed = (s.phi.apply(p, z) for z in (a, b, c, d))
                cons = jnp.linalg.norm(ea - eb, axis=-1); spread = jnp.linalg.norm(ec - ed, axis=-1)
                return -jnp.minimum(spread, 100.).mean() + s.lam * jnp.square(jnp.maximum(cons - 1., 0.)).mean(), (cons.mean(), spread.mean())
            s.grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
            s.dist = jax.jit(lambda p, x, g: jnp.linalg.norm(s.phi.apply(p, x) - s.phi.apply(p, g), axis=-1))
        def __getattr__(s, k): return getattr(s.inner, k)
        def reset(s, *a, **k): o = s.inner.reset(*a, **k); s.prev_obs = jnp.asarray(o); return o
        def step(s, action):
            tr = s.inner.step(action); o = jnp.asarray(tr.observation); x0, g0 = state_and_goal(s.prev_obs); x1, g1 = state_and_goal(o)
            done = jnp.logical_or(tr.terminated.astype(bool), tr.truncated.astype(bool))
            d0, d1 = s.dist(s.params, x0, g0), s.dist(s.params, x1, g1)
            prog = jnp.where(done, 0., jnp.clip(d0 - d1, -1., 1.))
            h = tr.objective_terms[:, s.ih]; ob = tr.objective_terms[:, s.io]; bonus = ((h > .85) & (ob > .85)).astype(jnp.float32)
            # online representation learning (TLDR objective with a fixed penalty multiplier) on consecutive, non-terminal pairs
            keep = ~done; i = jnp.flatnonzero(keep, size=keep.shape[0], fill_value=0)[:256]
            s.pairs.append((x0[i], x1[i])); n = x1.shape[0]; ins = x1[:min(n, 256)]
            s.buf = jax.lax.dynamic_update_slice(s.buf, ins, (s.nbuf % (s.buf.shape[0] - ins.shape[0]), 0)); s.nbuf += ins.shape[0]; s.t += 1
            if s.t % 4 == 0 and s.nbuf > 4096:
                a = jnp.concatenate([p[0] for p in s.pairs]); b = jnp.concatenate([p[1] for p in s.pairs]); s.pairs = []
                s.key, k1, k2 = jax.random.split(s.key, 3); hi = min(s.nbuf, s.buf.shape[0])
                c = s.buf[jax.random.randint(k1, (a.shape[0],), 0, hi)]; dd = s.buf[jax.random.randint(k2, (a.shape[0],), 0, hi)]
                (l, (cm, sm_)), gr = s.grad(s.params, a, b, c, dd); up, s.ost = s.opt.update(gr, s.ost); s.params = optax.apply_updates(s.params, up)
                if s.t % 400 == 0: print(f'[gcrl-tldr] step {s.t}: phi loss {float(l):.3f} consecutive {float(cm):.3f} spread {float(sm_):.2f}; mean goal distance {float(d1.mean()):.1f}', flush=True)
            elif s.t % 4 == 0: s.pairs = []
            s.prev_obs = o
            return tr.replace(reward=tr.reward + prog + bonus)
    jax_env = TemporalDistanceReward(jax_env)
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
    print('[gcrl-tldr] done', flush=True)
