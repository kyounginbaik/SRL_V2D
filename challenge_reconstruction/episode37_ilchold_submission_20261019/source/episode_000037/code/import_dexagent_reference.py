"""Convert DexAgent's achieved (physically successful) robot trajectory into a UniRetarget robot reference.

DexAgent MJCF: per hand slides pos_x/y/z then hinges rot_x -> rot_y -> rot_z (nested bodies, no offsets),
then <side>_hand_C_MC (our palm frame / hand root). World = DexAgent "setup" frame, which is the source
frame shifted +z by setup_from_source[2,3]. We take achieved qpos at 20 Hz (not the commands), map joint
names, undo the frame shift, and save a snapshot loadable via task.robot_reference_path. Object/contact
commands are untouched (snapshot provenance digest is checked at load).

Usage (in training image, cwd=/workspace): python import_dexagent_reference.py ROOT DEXAGENT_ORIGINAL_DIR
"""
import json
import re
import sys
from pathlib import Path

import newton
import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.embodiments.binding import _frame_poses, build_robot_reference_binding
from flash_chord.embodiments.sharpa_hands import SharpaHands, SharpaHandsConfig
from flash_chord.retargeting.reference_snapshot import command_digest, save_snapshot
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene

ROOT, DX = Path(sys.argv[1]), Path(sys.argv[2])
EP = ROOT.name
OUT = ROOT / 'priors' / 'dexagent_reference.npz'
REPORT = ROOT / 'reproduction' / 'dexagent_import_report.json'

embodiment = SharpaHands(SharpaHandsConfig.stronger_tracking())
setup = setup_scene(parquet=str(ROOT / 'frozen' / EP / 'reference.parquet'), control_fps=20.0, motion_speed=1.0,
                    embodiment=embodiment, collision=CollisionPolicy(), world_count=1,
                    include_support=False, decompose_objects=True)
reference, baseline = setup.reference, setup.scene.robot_reference
builder = newton.ModelBuilder()
layout = embodiment.build(builder)
model = builder.finalize(device='cpu', skip_all_validations=True)
scalar = layout.scalar_joints
names = list(scalar.names)
qids, dids = np.asarray(scalar.q_ids), np.asarray(scalar.dof_ids)
balls = [(builder.joint_q_start[j], builder.joint_qd_start[j], builder.joint_label[j] if hasattr(builder, 'joint_label') else str(j))
         for j, kind in enumerate(builder.joint_type) if kind == newton.JointType.BALL]

# DexAgent qpos layout: named joints in declaration order (robot), then object free joints.
xml = (DX / 'scene_physics.xml').read_text()
dx_names = []
for tag, attr in re.findall(r'<(joint|freejoint)\b([^>]*)>', xml):
    m = re.search(r'name="([^"]*)"', attr)
    if m:
        free = tag == 'freejoint' or 'type="free"' in attr
        dx_names.append((m.group(1), 7 if free else 1))
offsets, at = {}, 0
for n, size in dx_names:
    offsets[n] = at
    at += size
traj = np.load(DX / 'evaluation_trajectory.npz')
qpos = traj['qpos']
assert qpos.shape[1] == at, (qpos.shape, at)
manifest = json.loads((DX / 'input_manifest.json').read_text())
shift = np.asarray(manifest['source_from_setup'])[:3, 3]
F = baseline.num_frames
assert len(qpos) == F, (len(qpos), F)

q = baseline.joint_q.astype(np.float64).copy()
mapped = []
for side in ('left', 'right'):
    pre = f'robot_{side}_{side}_'
    pos = np.stack([qpos[:, offsets[pre + f'pos_{a}']] for a in 'xyz'], 1) + shift
    ang = np.stack([qpos[:, offsets[pre + f'rot_{a}']] for a in 'xyz'], 1)
    rot = Rotation.from_euler('XYZ', ang)  # nested hinges x then y then z == intrinsic XYZ
    for a, axis in enumerate('xyz'):
        q[:, qids[names.index(f'{side}_wrist_{axis}')]] = pos[:, a]
        mapped.append(f'{side}_wrist_{axis}')
    ball = [b for b in balls if side in str(b[2])]
    assert len(ball) == 1, (side, balls)
    quat = rot.as_quat()  # xyzw, Newton ball convention
    quat *= np.sign(np.sum(quat * q[:, ball[0][0]:ball[0][0] + 4], axis=1, keepdims=True) + 1e-12)
    q[:, ball[0][0]:ball[0][0] + 4] = quat
    for n in names:
        if n.startswith(f'{side}_') and pre + n[len(side) + 1:] in offsets and 'wrist' not in n:
            q[:, qids[names.index(n)]] = qpos[:, offsets[pre + n[len(side) + 1:]]]
            mapped.append(n)
unmapped = [n for n in names if n not in mapped]
assert not unmapped, unmapped

lower = model.joint_limit_lower.numpy()[dids]
upper = model.joint_limit_upper.numpy()[dids]
clipped = int(((q[:, qids] < lower - 1e-6) | (q[:, qids] > upper + 1e-6)).sum())
q[:, qids] = np.clip(q[:, qids], lower, upper)
target = baseline.joint_target.copy()
target[:, dids] = q[:, qids]
for qs, ds, _ in balls:
    target[:, ds:ds + 3] = Rotation.from_quat(q[:, qs:qs + 4]).as_rotvec()
binding = build_robot_reference_binding(builder, layout, q.astype(np.float32), target.astype(np.float32), baseline.fps)
checksum = save_snapshot(OUT, binding, setup.scene.layout, provenance=dict(
    mode='dexagent_achieved_qpos_import', episode=EP, object_contact_sha256=command_digest(reference),
    source=str(DX / 'evaluation_trajectory.npz'), frame_shift_m=shift.tolist()))

# Compare with the kinematic (IK) reference: this is the physical correction DexAgent found.
active = np.zeros(F, bool)
for side in reference.sides:
    active |= np.asarray(reference.contact_active(side)).reshape(F, -1).any(axis=1)
report = dict(episode=EP, snapshot=str(OUT), sha256=checksum, frames=F, clipped_joint_samples=clipped,
              frame_shift_m=shift.tolist(), contact_frames=int(active.sum()), hands={})
for hb, hd in zip(baseline.hands, binding.hands):
    dp = np.linalg.norm(hd.wrist_pos_w - hb.wrist_pos_w, axis=1)
    dr = (Rotation.from_quat(hd.wrist_quat_w[:, [1, 2, 3, 0]]) * Rotation.from_quat(hb.wrist_quat_w[:, [1, 2, 3, 0]]).inv()).magnitude()
    df = np.abs(hd.finger_joint_pos - hb.finger_joint_pos)
    dt = np.linalg.norm(hd.fingertip_pos_w - hb.fingertip_pos_w, axis=-1)
    report['hands'][hb.side] = dict(
        frame0_wrist_offset_m=float(dp[0]),
        wrist_offset_m=dict(mean=float(dp.mean()), p95=float(np.percentile(dp, 95)), max=float(dp.max()),
                            contact_mean=float(dp[active].mean())),
        wrist_rotation_deg=dict(mean=float(np.degrees(dr.mean())), p95=float(np.degrees(np.percentile(dr, 95)))),
        finger_joint_rad=dict(mean=float(df.mean()), contact_mean=float(df[active].mean()), max=float(df.max())),
        fingertip_offset_m=dict(mean=float(dt.mean()), contact_mean=float(dt[active].mean())))
REPORT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=1))
