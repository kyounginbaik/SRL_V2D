"""Render a Newton eval rollout with Newton's own headless viewer (no MuJoCo). Joint state per frame comes straight from the
eval npz (full joint_q of the world). Green = reference ink (goal tip on the board), orange = what the pen drew (tip within 3 mm of the board).
Also reports finger-pen penetration from Newton's own collision pipeline (min signed contact distance per frame).
Usage (container): nrender.py CKPT CLIP_PARQUET EVAL_NPZ WORLD OUT_FRAMES_NPZ [REF_SNAPSHOT]"""
import json, sys
from pathlib import Path
import numpy as np, warp as wp, newton, newton.viewer as V
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/workspace/src")
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config

ckpt, pqp, ev, w, outp = sys.argv[1:6]; w = int(w)
cfg = OmegaConf.create(checkpoint_config(read_checkpoint_metadata(Path(ckpt))))
env_config = instantiate_typed(cfg.env, RLEnvConfig)
S = setup_scene(parquet=pqp, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed, embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=1, include_support=cfg.scene.support,
                decompose_objects=cfg.scene.decompose_objects, contact_friction=cfg.scene.get("contact_friction", 1.0),
                object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0))
m = S.scene.model; st = m.state()
d = np.load(ev); Q = d["q"][:, w].astype(np.float32); T = len(Q)
assert Q.shape[1] == m.joint_coord_count, (Q.shape, m.joint_coord_count)
board = json.loads(Path("/oak/write/board.json").read_text())
tip_local, N, Sp = np.array(board["tip_local"]), np.array(board["normal"]), np.array(board["surface_point"])
gp = np.asarray(S.reference.object_body_pos_w(), float)[:T + 1, 0]; gq = np.asarray(S.reference.object_body_quat_w(), float)[:T + 1, 0]
gtip = R.from_quat(gq[:, [1, 2, 3, 0]]).apply(tip_local) + gp
hg = (gtip - Sp) @ N; on = hg < np.percentile(hg, 20) + 0.002
ref_ink = gtip[on]

# Newton collision for penetration: hand shapes vs pen shapes
blab = list(getattr(m, "body_label", None) or getattr(m, "body_key"))
sb = m.shape_body.numpy()
pen_b = [i for i, n in enumerate(blab) if n.startswith("object")]
hand_b = [i for i, n in enumerate(blab) if n.startswith("right_sharpa")]
try:
    contacts = m.contacts()
except Exception as e:
    print("no contacts:", repr(e)[:300]); contacts = None

v = V.ViewerGL(width=960, height=720, headless=True); v.set_model(m)
cen = ref_ink.mean(0) + N * 0.05
cam = cen + N * 0.45 + np.array([0.0, 0.0, 0.10])
fwd = (cen - cam) / np.linalg.norm(cen - cam)
yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0]))); pitch = float(np.degrees(np.arcsin(fwd[2])))
v.set_camera(wp.vec3(*cam.astype(float)), pitch, yaw)
frames, pen_min, ink = [], [], []
for t in range(T):
    m.joint_q.assign(Q[t]); newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    pq_ = Q[t, -7:].astype(float); tip = R.from_quat(pq_[3:7]).apply(tip_local) + pq_[:3]
    if (tip - Sp) @ N < 0.003: ink.append(tip)
    if contacts is not None:
        try:
            m.collide(st, contacts)
            n_ = int(contacts.rigid_contact_count.numpy()[0]); s0 = contacts.rigid_contact_shape0.numpy()[:n_]; s1 = contacts.rigid_contact_shape1.numpy()[:n_]
            b0, b1 = sb[s0], sb[s1]; sel = (np.isin(b0, pen_b) & np.isin(b1, hand_b)) | (np.isin(b1, pen_b) & np.isin(b0, hand_b))
            if sel.any():
                p0 = contacts.rigid_contact_point0.numpy()[:n_][sel]; p1 = contacts.rigid_contact_point1.numpy()[:n_][sel]; nn = contacts.rigid_contact_normal.numpy()[:n_][sel]
                bq = st.body_q.numpy()
                def wpt(b, p):
                    return R.from_quat(bq[b, 3:7]).apply(p) + bq[b, :3]
                dist = [float(((wpt(sb[a], pa) - wpt(sb[c], pc)) @ -nv)) for a, c, pa, pc, nv in zip(s0[sel], s1[sel], p0, p1, nn)]
                pen_min.append(min(dist))
            else:
                pen_min.append(np.nan)
        except Exception as e:
            if t == 0: print("contact query failed:", repr(e)[:300], flush=True)
            contacts = None
    v.begin_frame(t / 40.0); v.log_state(st)
    v.log_points("ref_ink", wp.array(ref_ink.astype(np.float32), dtype=wp.vec3), wp.full(len(ref_ink), 0.0015, dtype=wp.float32), wp.array(np.tile([[0.2, 0.8, 0.3]], (len(ref_ink), 1)).astype(np.float32), dtype=wp.vec3))
    if ink: v.log_points("ink", wp.array(np.array(ink, np.float32), dtype=wp.vec3), wp.full(len(ink), 0.002, dtype=wp.float32), wp.array(np.tile([[1.0, 0.55, 0.1]], (len(ink), 1)).astype(np.float32), dtype=wp.vec3))
    v.end_frame(); frames.append(v.get_frame().numpy())
pm = np.array(pen_min, float)
print("NRENDER frames", len(frames), "finger-pen min dist cm: p5", np.round(np.nanpercentile(pm, 5) * 100, 3) if np.isfinite(pm).any() else None,
      "min", np.round(np.nanmin(pm) * 100, 3) if np.isfinite(pm).any() else None, flush=True)
np.savez_compressed(outp, frames=np.stack(frames), pen=pm)
print("NRENDER_DONE", flush=True)
