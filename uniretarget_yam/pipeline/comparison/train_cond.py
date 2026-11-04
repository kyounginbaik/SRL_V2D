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
from dataclasses import replace
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.yam_bilevel import YamReferenceEditor, identity_edit, better
from flash_chord.training.flash_sac.evaluation import compile_policy_action
from flash_chord.runtime.jax import to_jax

arm = sys.argv[1]; WORLDS = 1024
base = Path(f"/uniretarget_yam/out/episode_000037/arm_{arm}-seed42")
ckpts = sorted(base.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
out = Path("/cmp/results"); work = out / f"work_cond_{arm}"; work.mkdir(parents=True, exist_ok=True)
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
adim = WarpRLEnv(probe, publish_diagnostics=False, publish_step_metrics=False).action_dim
_, template = create_actor_template(training, probe.observation_dim, adim)

results = {}
BINS = [(0, 1), (1, 20), (20, 40), (40, 60), (60, 80), (80, 150)]
def run(name, ckpt, prob, deterministic=True):
    env = RLEnv(setup.scene, setup.reference, config=deepcopy(env_cfg))
    env.set_voc_scale(0.0)
    env.set_reset_to_first_frame_probability(prob)
    env.reset()
    if env.device.is_cuda:
        env.capture_step()
    env.reset()
    ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    o = jnp.array(to_jax(env.reset(), (env.world_count, env.observation_dim)), copy=True)
    start = env.reset_policy.reset_frame.numpy().copy()
    key = jax.random.PRNGKey(42)
    if ckpt is None:
        pol = None
    else:
        state, _ = load_actor_checkpoint(str(base / f"policy_{ckpt}.safetensors"), template)
        actor, _ = create_actor_template(training, o.shape[1], adim)
        pol = compile_policy_action(actor, state, o, key, deterministic=deterministic, action_transform=training.network.action_transform, action_scale=training.network.action_scale)
    W = env.world_count; finished = np.zeros(W, bool); ended = np.zeros(W, bool); last = np.zeros(W, int)
    for _ in range(190):
        tprev = env.timestep.numpy().copy()
        if pol is None:
            a = jnp.zeros((W, adim), dtype=o.dtype)
        else:
            a, key = pol(state, o, key)
        vs = ad.step(a); o = jnp.array(vs.observation, copy=True)
        term = np.asarray(vs.terminated) != 0; tr = np.asarray(vs.truncated) != 0
        new = (term | tr) & ~finished
        ended[new & tr & ~term] = True; last[new] = tprev[new]; finished |= new
        if finished.all():
            break
    rows = {}
    for lo, hi in BINS:
        s = (start >= lo) & (start < hi)
        rows[f"{lo}-{hi-1}"] = dict(n=int(s.sum()), reached_end=float(ended[s].mean()) if s.any() else None, median_last=float(np.median(last[s])) if s.any() else None)
    tot = float(ended.mean())
    print(f"{name:34s} first_frame_p={prob} overall_reached_end={tot:.3f} " + " ".join(f"[{k}: n={v['n']} end={v['reached_end'] if v['reached_end'] is None else round(v['reached_end'],2)} last={v['median_last']}]" for k, v in rows.items()), flush=True)
    results[name] = dict(prob=prob, overall=tot, bins=rows)
for prob in (0.0, 1.0):
    tag = "random" if prob == 0 else "frame0"
    run(f"zero/{tag}", None, prob)
    run(f"250M_det/{tag}", 250009600, prob)
    run(f"250M_stoch/{tag}", 250009600, prob, deterministic=False)
    run(f"50M_det/{tag}", 50001920, prob)
(out / "train_conditions.json").write_text(json.dumps(results, indent=1))
print("COND_DONE", flush=True)
