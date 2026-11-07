"""Probe: Newton scene layout, hand FK, headless ViewerGL. Usage (container): nprobe.py CKPT CLIP_PARQUET EVAL_NPZ"""
import sys, numpy as np, warp as wp, newton
sys.path.insert(0, "/workspace/src")
from pathlib import Path
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
ckpt, pqp, ev = sys.argv[1:4]
cfg = OmegaConf.create(checkpoint_config(read_checkpoint_metadata(Path(ckpt))))
env_config = instantiate_typed(cfg.env, RLEnvConfig)
S = setup_scene(parquet=pqp, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=1, include_support=cfg.scene.support,
                decompose_objects=cfg.scene.decompose_objects, contact_friction=cfg.scene.get("contact_friction", 1.0),
                object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0))
sc = S.scene; print("scene fields", [k for k in vars(sc)] if hasattr(sc, "__dict__") else type(sc))
m = sc.model if hasattr(sc, "model") else None
print("model", type(m), m.body_count, m.joint_coord_count, m.shape_count)
lab = getattr(m, "body_label", None) or getattr(m, "body_key", None); print("bodies", list(lab)[:60])
print("joints", list(getattr(m, "joint_label", None) or getattr(m, "joint_key", None)))
d = np.load(ev); print({k: d[k].shape for k in d.files}); print("eval body labels[:60]", list(d["body_labels"][:60])); print("eval joint labels[:40]", list(d["joint_labels"][:40]), d["joint_q_start"][:40])
st = m.state(); q = m.joint_q.numpy(); print("joint_q0", np.round(q, 3))
newton.eval_fk(m, m.joint_q, m.joint_qd, st)
bq = st.body_q.numpy()
lo, hi = m.joint_limit_lower.numpy(), m.joint_limit_upper.numpy(); jl = list(getattr(m, "joint_label", None) or getattr(m, "joint_key", None)); qs = m.joint_q_start.numpy(); ds = m.joint_qd_start.numpy()
for j, nm in enumerate(jl):
    if nm.startswith("right_sharpa"): print("lim", nm, qs[j], ds[j], round(float(lo[ds[j]]), 2), round(float(hi[ds[j]]), 2))
for c in (0.0, 0.5, 1.0):
    q2 = q.copy()
    for j, nm in enumerate(jl):
        if nm.startswith("right_sharpa") and any(x in nm for x in ("FE", "PIP", "DIP", "IP")): q2[qs[j]] = lo[ds[j]] + c * (hi[ds[j]] - lo[ds[j]])
    m.joint_q.assign(q2); newton.eval_fk(m, m.joint_q, m.joint_qd, st); bq = st.body_q.numpy()
    print("closure", c, [(n.split("/")[-1], np.round(b[:3] - [0, .25, .5], 3).tolist()) for n, b in zip(lab, bq) if n.startswith("right") and n.endswith("DP")])
m.joint_q.assign(q); newton.eval_fk(m, m.joint_q, m.joint_qd, st)
try:
    import newton.viewer as V
    v = V.ViewerGL(width=640, height=480, headless=True); v.set_model(m)
    v.begin_frame(0.0); v.log_state(st); v.end_frame()
    img = v.get_frame().numpy(); print("frame", img.shape, img.dtype, img.mean())
    import imageio; imageio.imwrite("/oak/write/nprobe.png", img)
except Exception as e:
    import traceback; traceback.print_exc()
print("PROBE_DONE")
