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
from flash_chord.retargeting.yam_bilevel import YamReferenceEditor, measure
from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
from flash_chord.training.flash_sac.evaluation import compile_policy_action
from flash_chord.runtime.jax import to_jax

arm = sys.argv[1]; WORLDS = 16
base = Path(f"/uniretarget_yam/out/episode_000037/arm_{arm}-seed42")
ckpts = sorted(base.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
out = Path("/cmp/results"); work = out / f"work_diag_{arm}"; work.mkdir(parents=True, exist_ok=True)
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

ranges = list(zip(probe.action.block_names, probe.action.block_ranges))
print("blocks", ranges, flush=True)
def idx(side, group):
    for n, (a, b) in ranges:
        if n.startswith(f"{side}_{group}"):
            return np.arange(a, b)
    raise KeyError((side, group, [n for n, _ in ranges]))
RA, RF, LA, LF = idx("right", "arm"), np.concatenate([np.arange(a, b) for n, (a, b) in ranges if n.startswith("right_") and "arm" not in n.split("[")[0]]), idx("left", "arm"), np.concatenate([np.arange(a, b) for n, (a, b) in ranges if n.startswith("left_") and "arm" not in n.split("[")[0]])
ALL = np.arange(adim)
print("right arm", RA, "right fingers", RF, "left arm", LA, "left fingers", LF, flush=True)
actor_compiled = {}
def rollout(state, select=None, active=lambda f: True, scale=1.0, seed=42):
    env = RLEnv(replace(editor.eval_setup.scene, robot_reference=editor.binding), editor.eval_setup.reference, config=deepcopy(editor.eval_config))
    env.reset(frame_id=0)
    if env.device.is_cuda:
        env.capture_step()
    env.reset(frame_id=0)
    ad = WarpRLEnv(env, publish_diagnostics=False, publish_step_metrics=False)
    obs = jnp.array(to_jax(env.reset(frame_id=0), (env.world_count, env.observation_dim)), copy=True)
    key = jax.random.PRNGKey(seed)
    if "p" not in actor_compiled and state is not None:
        actor, _ = create_actor_template(training, obs.shape[1], ad.action_dim)
        actor_compiled["p"] = compile_policy_action(actor, state, obs, key, deterministic=True,
            action_transform=training.network.action_transform, action_scale=training.network.action_scale)
    W = env.world_count; per = env.model.body_count // W
    mask = np.zeros(adim, dtype=np.float32); mask[ALL if select is None else select] = 1.0
    poses, palm, fq, aq, acts = [], [], [], [], []
    for f in range(editor.binding.num_frames - 1):
        if state is None:
            a = np.zeros((W, adim), dtype=np.float32)
        else:
            a, key = actor_compiled["p"](state, obs, key); a = np.asarray(a)
        acts.append(a.mean(axis=0))
        applied = a * mask * scale if active(f) else a * 0.0
        obs = jnp.array(ad.step(jnp.asarray(applied, dtype=obs.dtype)).observation, copy=True)
        jax.block_until_ready(obs)
        b = env.state_0.body_q.numpy()
        x = b[env.command.body_ids_w.numpy().reshape(-1)].reshape(W, -1, 7)
        gp = env.command.body_target_pos_w.numpy().reshape(W, -1, 3); gq = env.command.body_target_quat_w.numpy().reshape(W, -1, 4)
        poses.append(np.concatenate([x, gp, gq], axis=-1)); palm.append(b.reshape(W, per, 7)[:, editor.palm_body])
        q = env.state_0.joint_q.numpy().reshape(W, -1)
        fq.append(q[:, editor.finger_q]); aq.append(q[:, editor.arm_q])
    poses, palm, fq, aq, acts = map(np.asarray, (poses, palm, fq, aq, acts))
    pn = np.linalg.norm(palm[:, :, 3:7], axis=2); on = np.linalg.norm(poses[:, :, 0, 3:7], axis=2)
    bad = (~np.isfinite(pn)) | (pn < 1e-6) | (~np.isfinite(on)) | (on < 1e-6)
    badw = bad.any(axis=0); first_bad = int(np.flatnonzero(bad.any(axis=1))[0]) if badw.any() else -1
    if badw.any():
        print(f"  NONFINITE worlds={int(badw.sum())}/{W} first_frame={first_bad}", flush=True)
    if badw.all():
        m = dict(kind="sim_fail", kinds={"sim_fail": int(W)}, frontier=first_bad, frontier_min=first_bad, slip=-1, cm=float("nan"), deg=float("nan"), passed=0.0)
    else:
        good = ~badw
        m = measure(poses[:, good], palm[:, good], editor.onset)
        poses, palm, fq, aq = poses[:, good], palm[:, good], fq[:, good], aq[:, good]
    m["bad_worlds"] = int(badw.sum()); m["first_bad_frame"] = first_bad
    x = poses[:, :, 0]
    m["rise"] = float((x[:, :, 2] - x[0, :, 2][None]).max(axis=0).mean() * 100)
    return m, dict(palm=palm, fq=fq, aq=aq, acts=acts, obj=x[:, :, :7])
def line(name, m):
    print(f"{name:34s} held={m['kinds'].get('held',0):2d}/16 kind={m['kind']:10s} frontier={m['frontier']:3d} rise={m['rise']:6.2f} cm={m['cm']:5.2f} deg={m['deg']:5.2f} pass={m['passed']:.2f}", flush=True)
results = {}
m0, z = rollout(None); line("zero_action", m0); results["zero_action"] = m0
ref_f = editor.human_q[:, editor.finger_q]; ref_a = editor.human_q[:, editor.arm_q]
FR = [0, 10, 20, 30, 35, 40, 45, 50, 60, 80, 120]
print("sanity zero-run vs reference q (rad, mean abs): fingers", [round(float(np.abs(z["fq"][f].mean(0) - ref_f[f + 1]).mean()), 3) for f in FR], "arm", [round(float(np.abs(z["aq"][f].mean(0) - ref_a[f + 1]).mean()), 3) for f in FR], flush=True)
print("reference spoon z rise cm by frame", [round(float((editor.eval_setup.reference is not None) and 0), 1) for _ in []], flush=True)
print("zero-run rise cm", {f: round(float((z["obj"][f, :, 2] - z["obj"][0, :, 2]).mean() * 100), 2) for f in FR}, flush=True)
grasp = z["fq"][60].mean(0) - z["fq"][0].mean(0)
print("zero-run finger closure 0->60 rad", np.round(grasp, 3).tolist(), "names", editor.finger_names, flush=True)
def compare(tag, t):
    rows = []
    for f in FR:
        pp = np.linalg.norm(t["palm"][f, :, :3] - z["palm"][f, :, :3], axis=1).mean() * 100
        pr = np.degrees((R.from_quat(t["palm"][f, 0, 3:7]) * R.from_quat(z["palm"][f, 0, 3:7]).inv()).magnitude())
        closure = float(np.dot(t["fq"][f].mean(0) - z["fq"][0].mean(0), grasp) / np.dot(grasp, grasp))
        zc = float(np.dot(z["fq"][f].mean(0) - z["fq"][0].mean(0), grasp) / np.dot(grasp, grasp))
        d = t["palm"][f, :, :3] - t["obj"][f, :, :3]; dz = z["palm"][f, :, :3] - z["obj"][f, :, :3]
        rows.append(dict(frame=f, palm_cm=float(pp), palm_deg=float(pr), closure=closure, closure_zero=zc, palm_obj_cm=float(np.linalg.norm(d, axis=1).mean() * 100), palm_obj_zero=float(np.linalg.norm(dz, axis=1).mean() * 100),
            ra=float(np.abs(t["acts"][f][RA]).mean()), rf=float(np.abs(t["acts"][f][RF]).mean()), la=float(np.abs(t["acts"][f][LA]).mean()), lf=float(np.abs(t["acts"][f][LF]).mean()), rf_signed_along_close=float(np.dot(t["acts"][f][RF], np.sign(grasp[:len(RF)])) / len(RF)) if len(RF) == len(grasp) else None))
        r = rows[-1]; print(f"  {tag} f={f:3d} palm_diff={r['palm_cm']:5.2f}cm {r['palm_deg']:5.1f}deg closure={r['closure']:5.2f} (zero {r['closure_zero']:5.2f}) palm_obj={r['palm_obj_cm']:5.1f}cm (zero {r['palm_obj_zero']:5.1f}) |a| RA={r['ra']:.2f} RF={r['rf']:.2f} LA={r['la']:.2f} LF={r['lf']:.2f}", flush=True)
    return rows
for steps in (50001920, 250009600):
    path = base / f"policy_{steps}.safetensors"
    state, _ = load_actor_checkpoint(str(path), template)
    tag = f"{steps // 1000000}M"
    print("==", tag, flush=True)
    v = {
        "full": dict(),
        "right_arm_only": dict(select=RA), "right_fingers_only": dict(select=RF), "right_both": dict(select=np.concatenate([RA, RF])),
        "left_only": dict(select=np.concatenate([LA, LF])),
        "scale_0.1": dict(scale=0.1), "scale_0.25": dict(scale=0.25), "scale_0.5": dict(scale=0.5),
        "policy_until_f20": dict(active=lambda f: f < 20), "policy_until_f40": dict(active=lambda f: f < 40),
        "policy_from_f20": dict(active=lambda f: f >= 20), "policy_from_f30": dict(active=lambda f: f >= 30), "policy_from_f45": dict(active=lambda f: f >= 45), "policy_from_f60": dict(active=lambda f: f >= 60), "policy_from_f80": dict(active=lambda f: f >= 80),
    }
    for name, kw in v.items():
        m, t = rollout(state, **kw); line(f"{tag} {name}", m); results[f"{tag}/{name}"] = m
        if name == "full":
            results[f"{tag}/trace"] = compare(tag, t)
            np.savez(out / f"diag_trace_{tag}.npz", **t)
np.savez(out / "diag_trace_zero.npz", **z)
(out / f"diag_{arm}.json").write_text(json.dumps(results, indent=1))
print("DIAG_DONE", flush=True)
