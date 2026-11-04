"""Score frozen arm checkpoints with the upper level's own unassisted frame-0 test (YamReferenceEditor.rollout)."""
import os, sys, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np, jax, jax.numpy as jnp
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.envs.rl import RLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint
from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.retargeting.yam_bilevel import YamReferenceEditor

arm = sys.argv[1]; WORLDS = 2
base = Path(sys.argv[3] if len(sys.argv) > 3 else f"/uniretarget_yam/out/episode_000037/arm_{arm}-seed42")
ckpts = sorted(base.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
out = Path("/cmp/results"); work = out / f"work_clip_{arm}"; work.mkdir(parents=True, exist_ok=True)
config = checkpoint_config(read_checkpoint_metadata(ckpts[-1]))
config["task"].pop("robot_reference_path", None); config["task"].pop("robot_reference_sha256", None)
print("arm", arm, "parquet", config["task"]["parquet"], "object weight", config["objective"]["object_keypoints"]["weight"], flush=True)
cfg = OmegaConf.create(deepcopy(config))
env_cfg = instantiate_typed(cfg.env, RLEnvConfig); embodiment = instantiate_typed(cfg.embodiment, Embodiment)
collision = instantiate_typed(cfg.collision, CollisionPolicy); training = instantiate_typed(cfg.training, TrainingConfig)
setup = setup_scene(parquet=cfg.task.parquet, control_fps=env_cfg.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment,
    collision=collision, world_count=WORLDS, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
    contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
    source_frame_playback=bool(cfg.task.source_frame_playback), motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame))
options = dict(engine="yam_reference", interval=20000000, warmup=30000000, candidates=12, eval_worlds=WORLDS, side="right",
    episode="/uniretarget_yam/episodes/episode_000037", code_dir="/uniretarget_yam/code")
editor = YamReferenceEditor(options=options, setup=setup, embodiment=embodiment, collision=collision, training=training,
    environment_config=deepcopy(config), checkpoint_metadata={}, output_dir=work)
print("onset", editor.onset, flush=True)
probe = RLEnv(setup.scene, setup.reference, config=deepcopy(env_cfg))
_, template = create_actor_template(training, probe.observation_dim, WarpRLEnv(probe, publish_diagnostics=False, publish_step_metrics=False).action_dim)

from dataclasses import replace
from flash_chord.training.flash_sac.evaluation import compile_policy_action
from flash_chord.runtime.jax import to_jax
CKPT = os.environ.get("CKPT", ""); NAME = os.environ["CLIP"]
binding = editor.binding; pol = None
if CKPT:
    path = base / CKPT
    state, _ = load_actor_checkpoint(str(path), template)
    task = checkpoint_config(read_checkpoint_metadata(path))["task"]; ref_path = task.get("robot_reference_path")
    if ref_path:
        binding = setup_scene(parquet=cfg.task.parquet, control_fps=env_cfg.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment, collision=collision, world_count=1,
            include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects, contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback), motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame),
            robot_reference_path=ref_path, robot_reference_sha256=task.get("robot_reference_sha256")).scene.robot_reference
    print("policy", path.name, "reference", ref_path, flush=True)
env = RLEnv(replace(editor.eval_setup.scene, robot_reference=binding), editor.eval_setup.reference, config=deepcopy(editor.eval_config))
env.reset(frame_id=0)
if env.device.is_cuda:
    env.capture_step()
env.reset(frame_id=0)
ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
o = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
key = jax.random.PRNGKey(42); W = env.world_count
if CKPT:
    actor, _ = create_actor_template(training, o.shape[1], adim) if "adim" in globals() else create_actor_template(training, o.shape[1], ad.action_dim)
    pol = compile_policy_action(actor, state, o, key, deterministic=True, action_transform=training.network.action_transform, action_scale=training.network.action_scale)
Q, OBJ, GOAL, GQ = [], [], [], []
layout = editor.projector.layout.scalar_joints
for _ in range(binding.num_frames - 1):
    if pol is None:
        a = jnp.zeros((W, ad.action_dim), dtype=o.dtype)
    else:
        a, key = pol(state, o, key)
    o = jnp.array(ad.step(a).observation, copy=True); jax.block_until_ready(o)
    Q.append(env.state_0.joint_q.numpy().reshape(W, -1)[0, np.asarray(layout.q_ids)])
    x = env.state_0.body_q.numpy()[env.command.body_ids_w.numpy().reshape(-1)].reshape(W, -1, 7)[0, 0]
    OBJ.append(x); GOAL.append(env.command.body_target_pos_w.numpy().reshape(W, -1, 3)[0, 0]); GQ.append(env.command.body_target_quat_w.numpy().reshape(W, -1, 4)[0, 0])
np.savez_compressed(f"/vid/{NAME}.npz", q=np.asarray(Q), names=np.array(list(layout.names)), obj=np.asarray(OBJ), goal=np.asarray(GOAL), goal_quat=np.asarray(GQ), ref_q=np.asarray(binding.joint_q)[:, np.asarray(layout.q_ids)], human_q=np.asarray(editor.baseline.joint_q)[:, np.asarray(layout.q_ids)])
print("rise cm", float((np.asarray(OBJ)[:, 2].max() - OBJ[0][2]) * 100), "clip", NAME, flush=True)
