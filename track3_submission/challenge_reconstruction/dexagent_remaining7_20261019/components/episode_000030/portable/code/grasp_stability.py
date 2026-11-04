#!/usr/bin/env python3
"""Read-only contact and hand-relative motion audit of measured MuJoCo states.

This diagnostic does not run a controller, integrate physics, grade task success,
or modify MP-SR. Contact geometry alone cannot certify force closure or grasping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(mask, times):
    """Left-held sampled intervals; the final sample contributes zero duration."""
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    stops = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def interval_record(a, b, times):
    stop = float(times[min(b, len(times) - 1)])
    return dict(start_s=float(times[a]), end_s=stop, duration_s=stop-float(times[a]),
                first_sample=a, last_sample=b-1)


def mask_summary(mask, times, include_intervals=False):
    mask = np.asarray(mask, bool)
    segments = [interval_record(a, b, times) for a, b in intervals(mask, times)]
    weights = np.r_[np.diff(times), 0.]
    duration = float(np.dot(weights, mask))
    total = float(times[-1] - times[0])
    result = dict(duration_s=duration, fraction_of_recording=duration/total if total else None,
                  sampled_count=int(mask.sum()), interval_count=len(segments),
                  longest_interval_s=max((r['duration_s'] for r in segments), default=0.),
                  first_time_s=float(times[np.flatnonzero(mask)[0]]) if mask.any() else None)
    if include_intervals:
        result['intervals'] = segments
    return result


def relative_drift(position, rotation, a, b):
    """Rigid object COM/body motion in the palm frame; not contact-point slip."""
    delta = np.linalg.norm(position[a:b] - position[a], axis=-1)
    relative_rotation = np.einsum('ij,tjk->tik', rotation[a].T, rotation[a:b])
    angles = np.degrees(np.arccos(np.clip((np.trace(relative_rotation, axis1=1, axis2=2)-1)/2, -1, 1)))
    return dict(max_translation_from_entry_m=float(delta.max()),
                final_translation_from_entry_m=float(delta[-1]),
                max_rotation_from_entry_deg=float(angles.max()),
                final_rotation_from_entry_deg=float(angles[-1]))


def _body_lookup(model, names):
    roots = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names}
    if any(b < 0 for b in roots.values()):
        raise ValueError('Recorded object body missing from scene')
    by_body = {}
    for b in range(model.nbody):
        p = b
        while p:
            for name, root in roots.items():
                if p == root:
                    by_body[b] = name
            p = int(model.body_parentid[p])
    hand_bodies = {}
    for b in range(model.nbody):
        name = model.body(b).name or ''
        for side in ('left', 'right'):
            if name.startswith(f'robot_{side}_'):
                category = next((v for v in ('thumb', 'index', 'middle', 'ring', 'pinky', 'little')
                                 if f'_{v}_' in name), 'palm_or_other')
                if category == 'little':
                    category = 'pinky'
                hand_bodies[b] = (side, category)
    return roots, by_body, hand_bodies


def audit_grasp_stability(scene, trajectory, *, lift_threshold_m=.02,
                         contact_distance_m=0., sample_stride=1, frames_only=False):
    scene, trajectory = Path(scene).resolve(), Path(trajectory).resolve()
    if lift_threshold_m <= 0 or contact_distance_m < 0 or sample_stride < 1:
        raise ValueError('Use positive lift threshold/stride and nonnegative contact tolerance')
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(trajectory, allow_pickle=False) as z:
        if 'scene_sha256' not in z or str(z['scene_sha256']) != sha256(scene):
            raise ValueError('Trajectory does not identify this exact frozen scene')
        names = z['object_names'].astype(str).tolist()
        if len(names) != len(set(names)) or not names:
            raise ValueError('Expected unique recorded object names')
        use_substeps = not frames_only and 'substep_qpos' in z and len(z['substep_qpos']) > 0
        if use_substeps:
            qpos = np.concatenate([z['qpos'][:1], z['substep_qpos']], axis=0)
            times = np.r_[z['time_s'][:1], z['substep_time_s']]
        else:
            qpos, times = z['qpos'].copy(), z['time_s'].copy()
    if qpos.ndim != 2 or qpos.shape != (len(times), model.nq) or len(times) < 2:
        raise ValueError('Recording must have at least two aligned qpos/time samples')
    if not np.isfinite(qpos).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('States must be finite and timestamps strictly increasing')
    indices = np.unique(np.r_[np.arange(0, len(times), sample_stride), len(times)-1])
    qpos, times = qpos[indices], times[indices]
    roots, by_body, hand_bodies = _body_lookup(model, names)
    palm = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'robot_{s}_{s}_hand_C_MC')
            for s in ('left', 'right')}
    if any(b < 0 for b in palm.values()):
        raise ValueError('Expected exact source Sharpa palm frames')
    gravity = np.asarray(model.opt.gravity)
    if np.linalg.norm(gravity) == 0:
        raise ValueError('Lift direction requires nonzero model gravity')
    up = -gravity / np.linalg.norm(gravity)
    n = len(times)
    categories = ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm_or_other')
    contact = {(o, s): np.zeros((n, len(categories)), bool) for o in names for s in palm}
    depth = {(o, s): np.zeros(n) for o in names for s in palm}
    touched_bodies = {(o, s): set() for o in names for s in palm}
    rel_position = {(o, s): np.zeros((n, 3)) for o in names for s in palm}
    rel_rotation = {(o, s): np.zeros((n, 3, 3)) for o in names for s in palm}
    height = {o: np.zeros(n) for o in names}
    other_contact = {o: np.zeros(n, bool) for o in names}
    other_bodies = {o: set() for o in names}
    data = mujoco.MjData(model)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        # Geometry only. No mj_step, force re-solve, or changes to saved states.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        for o, body in roots.items():
            center, obj_rotation = data.xipos[body], data.xmat[body].reshape(3, 3)
            height[o][i] = center @ up
            for s, hand in palm.items():
                hand_rotation = data.xmat[hand].reshape(3, 3)
                rel_position[o, s][i] = hand_rotation.T @ (center - data.xpos[hand])
                rel_rotation[o, s][i] = hand_rotation.T @ obj_rotation
        for k in range(data.ncon):
            c = data.contact[k]
            if float(c.dist) > contact_distance_m:
                continue
            bodies = [int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])]
            for own, other in (bodies, bodies[::-1]):
                o = by_body.get(own)
                if o is None or by_body.get(other) == o:
                    continue
                hand = hand_bodies.get(other)
                if hand:
                    s, category = hand
                    contact[o, s][i, categories.index(category)] = True
                    depth[o, s][i] = max(depth[o, s][i], -float(c.dist))
                    touched_bodies[o, s].add(model.body(other).name)
                else:
                    other_contact[o][i] = True
                    other_bodies[o].add(model.body(other).name or 'world')

    weights = np.r_[np.diff(times), 0.]
    objects = {}
    for o in names:
        rise = height[o] - height[o][0]
        elevated = rise >= lift_threshold_m
        # This may include free flight. It deliberately does not assert grasping.
        free_of_other_contact = elevated & ~other_contact[o]
        entry = dict(initial_com_height_m=float(height[o][0]), max_com_rise_m=float(rise.max()),
                     min_com_rise_m=float(rise.min()),
                     elevated=mask_summary(elevated, times, True),
                     elevated_without_nonhand_contact=mask_summary(free_of_other_contact, times, True),
                     nonhand_contact_bodies=sorted(other_bodies[o]), hands={})
        for s in palm:
            flags = contact[o, s]
            any_contact = flags.any(axis=1)
            fingers = flags[:, 1:5].any(axis=1)
            thumb_finger = flags[:, 0] & fingers
            first = np.flatnonzero(any_contact)
            # Also include trailing loss after the hand has ever touched the object.
            after_first = np.arange(n) >= first[0] if len(first) else np.zeros(n, bool)
            windows = []
            for a, b in intervals(elevated, times):
                duration = float(weights[a:b].sum())
                row = interval_record(a, b, times)
                row.update(contact_fraction=float(np.dot(weights[a:b], any_contact[a:b])/duration) if duration else None,
                           thumb_and_finger_fraction=float(np.dot(weights[a:b], thumb_finger[a:b])/duration) if duration else None,
                           hand_relative_drift=relative_drift(rel_position[o, s], rel_rotation[o, s], a, b))
                windows.append(row)
            continuous = []
            for a, b in intervals(elevated & any_contact, times):
                continuous.append(dict(**interval_record(a, b, times),
                                       **relative_drift(rel_position[o, s], rel_rotation[o, s], a, b)))
            unsupported = []
            for a, b in intervals(free_of_other_contact, times):
                duration = float(weights[a:b].sum())
                unsupported.append(dict(**interval_record(a, b, times),
                    contact_fraction=float(np.dot(weights[a:b], any_contact[a:b])/duration) if duration else None,
                    thumb_and_finger_fraction=float(np.dot(weights[a:b], thumb_finger[a:b])/duration) if duration else None,
                    hand_relative_drift=relative_drift(rel_position[o, s], rel_rotation[o, s], a, b)))
            entry['hands'][s] = dict(
                any_contact=mask_summary(any_contact, times, True),
                thumb_and_any_finger_contact=mask_summary(thumb_finger, times, True),
                by_part={c: mask_summary(flags[:, j], times) for j, c in enumerate(categories)},
                touched_hand_bodies=sorted(touched_bodies[o, s]),
                max_geometric_penetration_m=float(depth[o, s].max()),
                contact_loss_after_first_touch=mask_summary(after_first & ~any_contact, times, True),
                contact_loss_while_elevated=mask_summary(elevated & ~any_contact, times, True),
                observed_elevation_windows=windows,
                elevated_without_nonhand_contact_windows=unsupported,
                continuous_contact_during_elevation=continuous)
        objects[o] = entry
    return dict(schema='dexagent.grasp_stability.v1', scene=str(scene), trajectory=str(trajectory),
                hashes=dict(scene=sha256(scene), trajectory=sha256(trajectory), audit_code=sha256(__file__)),
                sample_source='recorded_physics_substeps' if use_substeps else 'recorded_control_frames',
                sample_count=n, sample_stride=sample_stride, start_s=float(times[0]), end_s=float(times[-1]),
                max_sample_gap_s=float(np.diff(times).max()),
                settings=dict(lift_threshold_m=lift_threshold_m, contact_distance_m=contact_distance_m,
                              up_direction=up.tolist(), hand_frames={s:model.body(b).name for s,b in palm.items()}),
                interpretation=[
                    'Read-only geometric contact reconstruction from measured states; no rollout or physics modification.',
                    'Durations left-hold each sample until the next; the last sample adds zero duration. Events below the sample gap may be missed.',
                    'Contact means a MuJoCo collision pair at or below the declared distance; force, force closure, and actual grasp success are not inferred.',
                    'Elevation is actual object COM rise from its initial physical state; it may be tilt or flight, not necessarily a successful lift.',
                    'Elevated without nonhand contact removes table/other-object support but can still include free flight.',
                    'Hand-relative drift is a slip proxy using the palm frame and object COM/body orientation; finger articulation, rolling and regrasping also change it.',
                    'Thumb plus finger contact does not prove opposing normals or a stable grasp.',
                    'This report does not change MP-SR, independently verify control replay, or certify official challenge success.'
                ], objects=objects)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lift-threshold-m', type=float, default=.02)
    p.add_argument('--contact-distance-m', type=float, default=0.)
    p.add_argument('--sample-stride', type=int, default=1)
    p.add_argument('--frames-only', action='store_true')
    args = vars(p.parse_args()); output = args.pop('output')
    if output.resolve() in {args['scene'].resolve(), args['trajectory'].resolve()}:
        raise ValueError('The audit output must not overwrite either input')
    report = audit_grasp_stability(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=str(output), samples=report['sample_count'],
                         duration_s=report['end_s']-report['start_s'],
                         object_max_rise_m={o:r['max_com_rise_m'] for o,r in report['objects'].items()}), indent=2))


if __name__ == '__main__':
    main()
