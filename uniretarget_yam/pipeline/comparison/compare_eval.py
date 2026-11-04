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

arm = sys.argv[1]; WORLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
base = Path(f"/uniretarget_yam/out/episode_000037/arm_{arm}-seed42")
ckpts = sorted(base.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
out = Path("/cmp/results"); work = out / f"work_{arm}"; work.mkdir(parents=True, exist_ok=True)
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
rows = []
for path in ckpts:
    state, _ = load_actor_checkpoint(str(path), template)
    learner = SimpleNamespace(actor=SimpleNamespace(params=state.params, batch_stats=state.batch_stats))
    for seed in (42, 1042):
        m, poses = editor.rollout(editor.binding, learner, seed)
        x = poses[:, :, 0]
        obj = np.linalg.norm(x[:, :, :3] - x[0, :, :3][None], axis=2).max(axis=0) * 100
        z = (x[:, :, 2] - x[0, :, 2][None]).max(axis=0) * 100
        row = dict(arm=arm, steps=int(path.stem.split("_")[1]), seed=seed, worlds=WORLDS, **m,
            object_max_move_cm=float(obj.mean()), object_max_rise_cm=float(z.mean()), object_max_rise_cm_worst=float(z.min()), object_max_rise_cm_best=float(z.max()))
        rows.append(row); print(json.dumps(row), flush=True)
(out / f"upper_test_{arm}.json").write_text(json.dumps(rows, indent=1))
print("DONE", arm, flush=True)
