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
out = Path("/cmp/results"); work = out / f"work_reset_{arm}"; work.mkdir(parents=True, exist_ok=True)
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
options = dict(engine="yam_reference", interval=20000000, warmup=30000000, candidates=12, eval_worlds=16, side="right",
    episode="/uniretarget_yam/episodes/episode_000037", code_dir="/uniretarget_yam/code")
editor = YamReferenceEditor(options=options, setup=setup, embodiment=embodiment, collision=collision, training=training,
    environment_config=deepcopy(config), checkpoint_metadata={}, output_dir=work)
print("onset", editor.onset, flush=True)
probe = RLEnv(setup.scene, setup.reference, config=deepcopy(env_cfg))
adim = WarpRLEnv(probe, publish_diagnostics=False, publish_step_metrics=False).action_dim
_, template = create_actor_template(training, probe.observation_dim, adim)


import warp as wp
results = {}
BINS = [(0, 1), (1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 150)]
def make_cfg(hold):
    c = deepcopy(config); c["env"]["reset"]["voc_decay_steps"] = hold
    return instantiate_typed(OmegaConf.create(c["env"]), RLEnvConfig)
def run(name, hold, opening, prob=0.0):
    env = RLEnv(setup.scene, setup.reference, config=make_cfg(hold))
    env.set_voc_scale(0.0); env.set_reset_to_first_frame_probability(prob)
    if not opening:
        orig = env.reset_policy.sample
        def sample(*a, **k):
            r = orig(*a, **k); env.reset_policy.finger_scale.fill_(1.0); return r
        env.reset_policy.sample = sample
    env.reset()
    if env.device.is_cuda:
        env.capture_step()
    env.reset()
    ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    env.reset()
    start = env.reset_policy.reset_frame.numpy().copy(); fs = env.reset_policy.finger_scale.numpy()
    W = env.world_count; finished = np.zeros(W, bool); ended = np.zeros(W, bool); last = np.zeros(W, int)
    a = jnp.zeros((W, adim), dtype=jnp.float32)
    for _ in range(230):
        tprev = env.timestep.numpy().copy()
        vs = ad.step(a)
        term = np.asarray(vs.terminated) != 0; tr = np.asarray(vs.truncated) != 0
        new = (term | tr) & ~finished
        ended[new & tr & ~term] = True; last[new] = tprev[new]; finished |= new
        if finished.all():
            break
    rows = {f"{lo}-{hi-1}": (float(ended[(start >= lo) & (start < hi)].mean()) if ((start >= lo) & (start < hi)).any() else None) for lo, hi in BINS}
    print(f"{name:26s} hold={hold:2d} open={opening!s:5s} finger_scale=[{fs.min():.2f},{fs.max():.2f}] all={ended.mean():.3f} " + " ".join(f"{k}:{'-' if v is None else format(v, '.2f')}" for k, v in rows.items()), flush=True)
    results[name] = dict(hold=hold, opening=opening, overall=float(ended.mean()), bins=rows)
run("default", 20, True)
run("no_open", 20, False)
run("no_hold", 0, True)
run("no_open_no_hold", 0, False)
run("no_open_hold40", 40, False)
run("frame0_default", 20, True, 1.0)
run("frame0_no_open", 20, False, 1.0)
run("frame0_no_hold", 0, True, 1.0)
# reset pose versus the pose playback actually reaches, spoon in palm frame
z = np.load(out / "diag_trace_zero.npz")
env = RLEnv(setup.scene, setup.reference, config=make_cfg(0)); env.set_voc_scale(0.0)
per = env.model.body_count // env.world_count
pose = {}
for f in (5, 15, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100):
    env.reset(frame_id=f)
    b = env.state_0.body_q.numpy().reshape(env.world_count, per, 7)[0]
    x = env.state_0.body_q.numpy()[env.command.body_ids_w.numpy().reshape(-1)].reshape(env.world_count, -1, 7)[0, 0]
    p = b[editor.palm_body]
    rp, rr = R.from_quat(p[3:7]).inv().apply(x[:3] - p[:3]), R.from_quat(p[3:7]).inv() * R.from_quat(x[3:7])
    i = f - 1; zp, zo = z["palm"][i], z["obj"][i]
    zrp = np.stack([R.from_quat(zp[w, 3:7]).inv().apply(zo[w, :3] - zp[w, :3]) for w in range(zp.shape[0])]).mean(0)
    zrr = R.from_quat(zp[0, 3:7]).inv() * R.from_quat(zo[0, 3:7])
    d = float(np.linalg.norm(rp - zrp) * 100); ang = float(np.degrees((rr * zrr.inv()).magnitude()))
    pose[f] = dict(cm=d, deg=ang)
    print(f"reset vs playback spoon-in-palm f={f:3d}: {d:6.2f} cm {ang:6.1f} deg  (reset palm height {p[2]:.3f}, playback {zp[:,2].mean():.3f})", flush=True)
results["pose_gap"] = pose
(out / "reset_check.json").write_text(json.dumps(results, indent=1))
print("RESET_DONE", flush=True)
