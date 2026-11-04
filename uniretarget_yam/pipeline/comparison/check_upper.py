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

arm = sys.argv[1]; WORLDS = 4
base = Path(f"/uniretarget_yam/out/episode_000037/arm_{arm}-seed42")
ckpts = sorted(base.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
out = Path("/cmp/results"); work = out / f"work_check_{arm}"; work.mkdir(parents=True, exist_ok=True)
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

learner = SimpleNamespace(actor=SimpleNamespace(params=template.params, batch_stats=template.batch_stats))
editor.actor = object()
editor.policy = lambda s, o, k: (jnp.zeros((o.shape[0], adim), dtype=o.dtype), k)
P = editor.projector; base_ref = editor.baseline
def tq(ref, name):
    d = ref.joint_target[:, P.dids] - ref.joint_q[:, P.qids]
    print(f"{name}: |target-q| all max={np.abs(d).max():.4f} mean={np.abs(d).mean():.4f}", flush=True)
    ids = [editor.finger_q.index(i) if False else None for i in []]
    fmask = np.isin(P.qids, editor.finger_q); amask = np.isin(P.qids, editor.arm_q)
    print(f"  right fingers max={np.abs(d[:, fmask]).max():.4f} mean={np.abs(d[:, fmask]).mean():.4f}; right arm max={np.abs(d[:, amask]).max():.4f}", flush=True)
tq(base_ref, "fixed")
ident = editor.build(identity_edit())
print("identity vs baseline arm q rad: max", float(np.abs(ident.joint_q[:, editor.arm_q] - base_ref.joint_q[:, editor.arm_q]).max()), "by frame", {f: round(float(np.abs(ident.joint_q[f, editor.arm_q] - base_ref.joint_q[f, editor.arm_q]).max()), 3) for f in (0, 5, 10, 20, 30, 40, 45, 50, 60, 80, 120)}, flush=True)
ip, ir = editor._palm_path(ident.joint_q)
print("identity palm vs human cm by frame", {f: round(float(np.linalg.norm(ip[f] - editor.human_p[f]) * 100), 2) for f in (0, 5, 10, 20, 30, 40, 45, 50, 60, 80, 120)}, flush=True)
print("target changed by identity build: max", float(np.abs(ident.joint_target - base_ref.joint_target).max()), flush=True)
arm_v = np.abs(np.diff(base_ref.joint_q[:, editor.arm_q], axis=0)).max(axis=1) * 20
print("fixed arm speed rad/s max", float(arm_v.max()), "limit", editor.Y.ARM_SPEED, "frames over limit", int((arm_v > editor.Y.ARM_SPEED).sum()), flush=True)
print("object goal rise cm by frame", {f: round(float((editor.task.gp_raw[f, 2] - editor.task.gp_raw[0, 2]) * 100), 2) for f in (0, 10, 20, 25, 28, 30, 35, 38, 40, 45, 50, 60, 80, 120)}, "task.frames", editor.task.frames, "binding frames", base_ref.num_frames, flush=True)
# refined versus human
rs = setup_scene(parquet="/uniretarget_yam/episodes/episode_000037/yam/flash/refined/reference.parquet", control_fps=env_cfg.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=embodiment,
    collision=collision, world_count=1, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
    contact_friction=cfg.scene.get("contact_friction", 1.0), object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
    source_frame_playback=bool(cfg.task.source_frame_playback), motion_start_frame=int(cfg.task.motion_start_frame), motion_end_frame=int(cfg.task.motion_end_frame))
rr = rs.scene.robot_reference
tq(rr, "refined")
print("object commands equal fixed vs refined:", float(np.abs(np.asarray(rs.reference.object_body_pos_w()) - np.asarray(setup.reference.object_body_pos_w())).max()), flush=True)
rp, rrot = editor._palm_path(rr.joint_q)
gr = editor.task.gr_raw[:len(rp)]; gp = editor.task.gp_raw[:len(rp)]
for f in (0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 120):
    a = gr[f].inv().apply(editor.human_p[f] - gp[f]); b = gr[f].inv().apply(rp[f] - gp[f])
    dd = (gr[f].inv() * rrot[f]) * (gr[f].inv() * editor.human_r[f]).inv()
    print(f"  f={f:3d} refined-human palm in object frame {np.round((b-a)*100,1).tolist()} cm |{np.linalg.norm(b-a)*100:.1f}| rot {np.degrees(dd.magnitude()):.1f}deg  fingers mean|d|={np.abs(rr.joint_q[f, editor.finger_q]-base_ref.joint_q[f, editor.finger_q]).mean():.3f} rad", flush=True)
print("edit bounds: GRASP_P 4 cm, GRASP_R 30 deg, FINGER 0.3", flush=True)
# close direction
d = base_ref.joint_q[60, editor.finger_q] - base_ref.joint_q[0, editor.finger_q]
dr = rr.joint_q[60, editor.finger_q] - rr.joint_q[0, editor.finger_q]
print("finger motion 0->60: joints moving toward lower limit, human", int((d < -0.02).sum()), "refined", int((dr < -0.02).sum()), "of", len(d), flush=True)
for n, a, b, l in zip(editor.finger_names, d, dr, editor.limits):
    print(f"  {n:24s} human {a:+.3f} refined {b:+.3f} limits {l[0]:+.2f} {l[1]:+.2f} q60 {base_ref.joint_q[60, editor.finger_q[editor.finger_names.index(n)]]:+.3f}", flush=True)
# search landscape with a best-case zero-action policy, exactly the proposal rule the upper level uses
results = dict()
def test(name, onset, rounds):
    editor.onset = onset
    m0, poses = editor.rollout(editor.binding, learner, 42)
    print(f"[{name}] baseline {m0['kind']} frontier={m0['frontier']} cm={m0['cm']:.2f}", flush=True)
    im, ix = editor.rollout(editor.build(identity_edit()), learner, 42)
    print(f"[{name}] identity build {im['kind']} frontier={im['frontier']} cm={im['cm']:.2f} deg={im['deg']:.1f} rise={float((ix[:,:,0,2]-ix[0,:,0,2][None]).max(axis=0).mean()*100):.2f}", flush=True)
    hits = []
    for r in range(rounds):
        for e in editor.propose(m0, poses):
            m, x = editor.rollout(editor.build(e), learner, 42)
            rise = float((x[:, :, 0, 2] - x[0, :, 0, 2][None]).max(axis=0).mean() * 100)
            hits.append(dict(frontier=m["frontier"], kind=m["kind"], cm=m["cm"], deg=m["deg"], passed=m["passed"], rise=rise, accepted=bool(better(m, m0)), dp=np.round(np.asarray(e["dp"]) * 100, 1).tolist(), close=round(float(e["close"]), 3)))
            print(f"[{name}] {r} fr={m['frontier']:3d} {m['kind']:10s} rise={rise:6.2f} cm={m['cm']:5.2f} accept={hits[-1]['accepted']} dp={hits[-1]['dp']} close={hits[-1]['close']}", flush=True)
    print(f"[{name}] SUMMARY n={len(hits)} accepted={sum(o['accepted'] for o in hits)} rise>2cm={sum(o['rise']>2 for o in hits)} rise>10cm={sum(o['rise']>10 for o in hits)} passed={sum(o['passed']>0 for o in hits)}", flush=True)
    results[name] = hits
test("onset40", 40, 4)
test("onset28", 28, 4)
(out / "check_upper.json").write_text(json.dumps(results, indent=1))
print("CHECK_DONE", flush=True)
