"""First-success rule (2026-10-28): after every parallel rollout (window, candidate and confirmation rounds, all full sequences
from frame 0, deterministic policy), test every world on its own and save the first world that passes as the converted data.

A world passes when, after the official Track 3 frame-0 alignment (first pose of object 0, as tools/eval_fixed_reference.py),
for EVERY object:  episode-mean position error < 3 cm and mean orientation error < 30 deg   (official per-world MP-SR test)
                   and max position error over all frames < 10 cm and max orientation error < 60 deg   (per-frame cap)
and every recorded pose is finite.

Saved next to the training output (training/first_success.npz + .json): the simulated full joint_q of that world (hands +
object root), its object pose actual|target, palm/fingertip poses, and the reference joint_q / joint_target it tracked.
Mode +bilevel.first_success=stop cuts the training budget (as the 90% stop rule does); =record only saves and keeps training.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

POS_MEAN, ORI_MEAN, POS_MAX, ORI_MAX = .03, np.radians(30.), .10, np.radians(60.)


def _unit(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 1e-8, q / np.maximum(n, 1e-8), np.array([0., 0., 0., 1.]))


def world_errors(object_pose):
    """(Fm, W, O, 14) actual pos|quat xyzw | target pos|quat xyzw -> aligned position / orientation errors (Fm, W, O), finite (W,)."""
    OP = np.asarray(object_pose, np.float64)
    finite = np.isfinite(OP).all(axis=(0, 2, 3))
    OP = np.nan_to_num(OP)
    Fm, W, O = OP.shape[:3]
    Ra = Rotation.from_quat(_unit(OP[0, :, 0, 10:14])) * Rotation.from_quat(_unit(OP[0, :, 0, 3:7])).inv()   # (W,)
    off = OP[0, :, 0, 7:10] - Ra.apply(OP[0, :, 0, 0:3])
    RaF = Rotation.from_quat(np.broadcast_to(Ra.as_quat()[None, :, None], (Fm, W, O, 4)).reshape(-1, 4))
    pa = RaF.apply(OP[..., 0:3].reshape(-1, 3)).reshape(Fm, W, O, 3) + off[None, :, None]
    pos = np.linalg.norm(pa - OP[..., 7:10], axis=-1)
    qa = RaF * Rotation.from_quat(_unit(OP[..., 3:7]).reshape(-1, 4))
    ori = (qa * Rotation.from_quat(_unit(OP[..., 10:14]).reshape(-1, 4)).inv()).magnitude().reshape(Fm, W, O)
    return pos, ori, finite


def passing_worlds(object_pose):
    pos, ori, finite = world_errors(object_pose)
    ok = finite & ((pos.mean(0) < POS_MEAN) & (ori.mean(0) < ORI_MEAN) & (pos.max(0) < POS_MAX) & (ori.max(0) < ORI_MAX)).all(axis=1)
    return ok, pos, ori


def check(editor, learner, q, targets, joint_q):
    """Called at the end of parallel_rollout.rollout_candidates with that batch's per-world references and recorded joint_q."""
    mode = str(editor.options.get('first_success', '') or '')
    if mode not in ('stop', 'record') or getattr(editor, '_first_success', None) is not None:
        return
    ok, pos, ori = passing_worlds(editor._object_pose)
    import jax
    step = int(jax.device_get(learner.environment_steps))
    score = pos.mean(axis=(0, 2)) / POS_MEAN + ori.mean(axis=(0, 2)) / ORI_MEAN
    b = int(np.argmin(np.where(np.isfinite(score), score, np.inf)))
    print(f'[first_success] step={step} checked {len(ok)} worlds: {int(ok.sum())} pass; closest world mean {pos[:, b].mean() * 100:.2f} cm / '
          f'{np.degrees(ori[:, b].mean()):.1f} deg, max {pos[:, b].max() * 100:.2f} cm / {np.degrees(ori[:, b].max()):.1f} deg', flush=True)
    if not ok.any():
        return
    w = int(np.flatnonzero(ok)[np.argmin(score[ok])])   # best of the passing worlds (all of them pass the same rule)
    q, targets = np.asarray(q), np.asarray(targets)
    current = bool(np.allclose(q[w], editor.binding.joint_q, atol=1e-6))
    out = editor.output.parent
    np.savez_compressed(out / 'first_success.npz', sim_joint_q=np.asarray(joint_q)[:, w], object_pose=editor._object_pose[:, w],
                        palm_pose=editor._palm_pose[:, w], tip_pos=editor._tip_pos[:, w] if getattr(editor, '_tip_pos', None) is not None else np.zeros(0),
                        reference_joint_q=q[w], reference_joint_target=targets[w], fps=float(editor.baseline.fps),
                        position_error_m=pos[:, w], orientation_error_rad=ori[:, w])
    info = dict(step=step, mode=mode, world=w, worlds=int(len(ok)), passing_worlds=int(ok.sum()), reference_version=int(editor.version),
                reference='current reference' if current else 'candidate reference (saved in first_success.npz)',
                mean_pos_cm=round(float(pos[:, w].mean() * 100), 2), max_pos_cm=round(float(pos[:, w].max() * 100), 2),
                mean_ori_deg=round(float(np.degrees(ori[:, w].mean())), 1), max_ori_deg=round(float(np.degrees(ori[:, w].max())), 1),
                rule='aligned per object: mean pos < 3 cm, mean ori < 30 deg, max pos < 10 cm, max ori < 60 deg, all finite')
    (out / 'first_success.json').write_text(json.dumps(info, indent=1))
    editor._first_success = info
    print(f'[first_success] step={step} world {w} ({int(ok.sum())}/{len(ok)} worlds pass; {info["reference"]}): mean {info["mean_pos_cm"]} cm / '
          f'{info["mean_ori_deg"]} deg, max {info["max_pos_cm"]} cm / {info["max_ori_deg"]} deg', flush=True)
    if mode == 'stop':
        total = step + int(editor.training.world_count)
        object.__setattr__(editor.training, 'total_environment_steps', total)
        print(f'[first_success] mode stop: budget cut to {total}', flush=True)
