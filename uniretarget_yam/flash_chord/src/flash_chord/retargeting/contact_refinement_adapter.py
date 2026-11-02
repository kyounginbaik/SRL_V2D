"""Adapter for the floating-hand parallel reference editor.

The reusable geometry/timing implementation lives in contact_refinement.py.
Only this adapter knows the editor's rollout buffers and floating-wrist schema.
"""
from dataclasses import asdict
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import (
    HandChannels, RefinementConfig, extract_segments, find_giver,
    generate_phase_edits, phase_mask, precision_cost, project_edit, supported_anchor,
)
from flash_chord.retargeting.contact_segment_window import aligned_pose_errors
from flash_chord.retargeting.refinement_window import (
    WindowConfig, choose_refinement_window, robust_tracking_error, window_mask,
)
from flash_chord.retargeting.contact_feedback import (
    acquisition_window, choose_contact_failure_window, hand_object_contacts, palm_reach_correction, phase_contact_statistics,
    transfer_quality,
)


def configuration(editor):
    return RefinementConfig(
        position_tolerance_m=float(editor.options.get('precision_position_m', .03)),
        orientation_tolerance_rad=float(editor.options.get('precision_orientation_rad', np.pi / 6)),
        max_finger_delta_rad=float(editor.options.get('structured_finger_delta_rad', .4)),
        max_candidates=int(editor.options.get('structured_max_candidates', 18)),
    )


def reference_contacts(reference, frames):
    objects = len(reference.object_body_names())
    result = {}
    for hand in reference.sides:
        raw_active = np.asarray(reference.contact_active(hand)).reshape(frames, -1)
        active = np.isfinite(raw_active) & (raw_active > 0)
        parts = np.asarray(reference.contact_part_ids(hand)).reshape(frames, -1) - 1
        result[hand] = np.stack([(active & (parts == o)).any(1) for o in range(objects)], axis=1)
    return result


def _reference_key(q):
    return hashlib.sha256(np.ascontiguousarray(q, dtype=np.float32).tobytes()).hexdigest()


def _finger_limits(editor, hand):
    lookup = {int(qid): i for i, qid in enumerate(editor.projector.qids)}
    indices = [lookup[qid] for qid in hand.fingers]
    lower = np.asarray(editor.projector.lower, float)[indices]
    upper = np.asarray(editor.projector.upper, float)[indices]
    dids = np.asarray(editor.projector.dids)[indices]
    # Minimum-norm joint posture is embodiment-neutral and respects even joints
    # whose physical range excludes zero. All hand fingers participate, including
    # opposition/abduction channels omitted by scalar closure heuristics.
    target = np.clip(np.zeros(len(indices)), lower, upper)
    supplied = editor.options.get('contact_full_release_open_posture', {})
    if hand.name in supplied:
        target = np.asarray(supplied[hand.name], float)
    return target, lower, upper, dids


def _validate_full_release(editor, q, specification):
    """Recheck the exact finger exception before projection or outer blending."""
    base = np.asarray(editor.binding.joint_q)
    candidate = np.asarray(q)
    ids = tuple(specification['finger_qids'])
    diagnostic = specification['diagnostic']
    start, restore = diagnostic['edit_bounds']
    first, last = diagnostic['open_interval']
    a, b = specification['allowed_window']
    values = candidate[:, ids]
    lower, upper = np.asarray(specification['lower']), np.asarray(specification['upper'])
    target = np.asarray(diagnostic['open_target'], dtype=candidate.dtype)
    if candidate.shape != base.shape or not np.isfinite(candidate).all():
        raise ValueError('full release requires finite matching robot references')
    if not 0 <= a <= start <= first <= last <= restore <= b < len(base):
        raise ValueError('full release escaped its declared finger window')
    if specification['baseline_sha256'] != _reference_key(base):
        raise ValueError('full release was validated against a different current reference')
    outside = (np.arange(len(base)) <= start) | (np.arange(len(base)) >= restore)
    if not np.array_equal(values[outside], base[:, ids][outside]):
        raise ValueError('full release changes protected finger commands')
    if not np.array_equal(candidate[[0, -1], :], base[[0, -1], :]):
        raise ValueError('full release changes protected trajectory endpoints')
    next_contact = diagnostic['next_contact_frame']
    if next_contact is not None and (restore > next_contact
            or not np.array_equal(values[next_contact:], base[next_contact:, ids])):
        raise ValueError('full release changes a later intended regrasp')
    if np.any(values < lower - 1e-6) or np.any(values > upper + 1e-6):
        raise ValueError('full release violates physical finger limits')
    if not np.array_equal(values[first:last + 1], np.broadcast_to(target, (last - first + 1, len(ids)))):
        raise ValueError('full release does not reach and hold its complete open target')
    speed = np.abs(np.diff(values[start:restore + 1], axis=0)).max(initial=0.) * editor.baseline.fps
    if speed > diagnostic['max_speed_rad_s'] + 1e-5:
        raise ValueError('full release exceeds its absolute finger velocity bound')


def _full_release_proposals(editor, hands, selected, giver, bounds, raw):
    """Build explicit complete releases, maximizing open time before restoration."""
    if not editor.options.get('contact_full_release'):
        return [], []
    from flash_chord.retargeting.contact_release import plan_full_release
    if giver is None:
        return [], [dict(feasible=False, reason='no_handover_giver_in_selected_window')]
    current, fps = editor.binding.joint_q, editor.baseline.fps
    if bounds is None:
        _, bounds = phase_mask(len(current), selected.start, selected.end, fps)
    bounds = tuple(bounds)
    hand = hands[giver.hand]
    target, lower, upper, dids = _finger_limits(editor, hand)
    extension = float(editor.options.get('contact_full_release_window_extension_s', 0.))
    hold_s = float(editor.options.get('contact_full_release_hold_s', .5))
    delays = tuple(editor.options.get('contact_full_release_delays_s', (0., .2, .4, .6)))
    if not np.isfinite([extension, hold_s]).all() or extension < 0 or hold_s <= 0:
        raise ValueError('full release requires nonnegative window extension and positive hold duration')
    next_contact = min((max(segment.start, giver.end + 1) for segment in editor._phase_graph
                        if segment.hand == giver.hand and segment != giver and segment.end > giver.end),
                       default=None)
    limit = min(len(current) - 1, bounds[1] + round(extension * fps),
                next_contact if next_contact is not None else len(current) - 1)
    minimum_hold = max(1, int(np.ceil(hold_s * fps)))
    proposals, diagnostics = [], []
    receiver_alignment = None
    for operation in ('achieved_reach_correction', 'contact_direction_correction', 'receiver_anchor', 'close_earlier'):
        receiver_alignment = next(((info, q) for info, q in raw if info['op'] == operation), None)
        if receiver_alignment is not None:
            break
    for delay in delays:
        # The planner reports its own opening duration, including infeasibility,
        # so the adapter never substitutes a shorter, partially open transition.
        probe = plan_full_release(current, hand, target, lower, upper, fps=fps, bounds=bounds,
            event_frame=giver.end, hold_until=limit, next_contact_frame=next_contact, delay_s=float(delay),
            max_speed_rad_s=3., max_window_extension_s=extension)
        complete = probe.diagnostic.get('opening_complete')
        result = probe
        if complete is not None:
            earliest_hold_end = complete + minimum_hold - 1
            for hold_until in range(min(selected.end, limit), earliest_hold_end - 1, -1):
                result = plan_full_release(current, hand, target, lower, upper, fps=fps, bounds=bounds,
                    event_frame=giver.end, hold_until=hold_until, next_contact_frame=next_contact,
                    delay_s=float(delay), max_speed_rad_s=3., max_window_extension_s=extension)
                if result.feasible:
                    break
            if earliest_hold_end > min(selected.end, limit):
                diagnostics.append(dict(probe.diagnostic, feasible=False, reason='insufficient_full_open_hold_time',
                                        minimum_hold_frames=minimum_hold, earliest_hold_end=earliest_hold_end))
                continue
        diagnostics.append(result.diagnostic)
        if not result.feasible:
            continue
        specification = dict(hand=giver.hand, finger_qids=list(map(int, hand.fingers)), finger_dids=dids.tolist(),
            lower=lower.tolist(), upper=upper.tolist(), allowed_window=[bounds[0], limit],
            selected_window=list(bounds), baseline_sha256=_reference_key(current), diagnostic=result.diagnostic)
        info = dict(op='giver_full_release', segment=selected.key, edit_window=list(bounds),
                    receiver=selected.hand, giver=giver.hand, object_id=selected.object_id,
                    full_release=specification, release_delay_s=float(delay))
        proposals.append((info, result.joint_q))
        if receiver_alignment is not None:
            alignment_info, aligned = receiver_alignment
            combined = np.array(aligned, copy=True)
            combined[:, hand.fingers] = result.joint_q[:, hand.fingers]
            proposals.append((dict(info, op='receiver_alignment_giver_full_release',
                                   receiver_operation=alignment_info['op']), combined))
    return proposals, diagnostics


def channels(editor):
    names = list(editor.projector.layout.scalar_joints.names)
    qids = list(editor.projector.qids)
    builder = editor.projector.builder
    import newton
    result = {}
    for hand in editor.projector.layout.hands:
        side = hand.side
        positions = tuple(qids[names.index(f'{side}_wrist_{axis}')] for axis in 'xyz')
        balls = [builder.joint_q_start[j] for j, kind in enumerate(builder.joint_type)
                 if kind == newton.JointType.BALL and side in str(builder.joint_label[j])]
        if len(balls) != 1:
            raise ValueError(f'{side}: this adapter requires one floating wrist; supply an IK adapter for arm embodiments')
        fingers = tuple(qids[j] for j, name in enumerate(names)
                        if name.startswith(side + '_') and '_wrist_' not in name)
        closure = tuple(float(not (names[qids.index(j)].endswith('AA')
                                   or names[qids.index(j)].endswith(('thumb_CMC_FE', 'pinky_CMC')))) for j in fingers)
        result[side] = HandChannels(side, positions, tuple(range(balls[0], balls[0] + 4)), fingers, closure)
    return result


def _segment_window(editor, learner, step):
    from flash_chord.retargeting.parallel_rollout import rollout_candidates
    q, target = editor.binding.joint_q, editor.binding.joint_target
    rollout_candidates(editor, learner, np.repeat(q[None], editor.upper_worlds, axis=0),
                       np.repeat(target[None], editor.upper_worlds, axis=0))
    editor._offset_cached = True
    frames, fps = len(q), editor.baseline.fps
    cfg = configuration(editor)
    segments = extract_segments(reference_contacts(editor.setup.reference, frames), fps)
    p, r = aligned_pose_errors(editor._object_pose)
    # Discard nonfinite trajectories from the robust window detector; all
    # candidates still face the full-horizon finite/instability gates later.
    # A valid initial state is sufficient: later explosions are failure evidence,
    # not a reason to discard every world and disable the corrective search.
    finite = np.isfinite(p[0]).all(1) & np.isfinite(r[0]).all(1)
    if finite.sum() < 4:
        editor._phase_segment = None
        editor._win_info = dict(detect='structured_contacts', window=[1, 1], reason='insufficient_finite_worlds')
        editor._win_chunks = np.zeros(editor.basis.shape[1], bool)
        return np.zeros(frames)
    pm = np.median(np.nan_to_num(p[:, finite], nan=1., posinf=1.), axis=1)
    rm = np.median(np.nan_to_num(r[:, finite], nan=np.pi, posinf=np.pi), axis=1)
    choices = []
    focus = editor.options.get('window_segments') or []
    for segment in segments:
        simple_key = f'{segment.hand}_seg{segment.index}'
        if focus and segment.key not in focus and simple_key not in focus:
            continue
        mask, (a, b) = phase_mask(frames, segment.start, segment.end, fps)
        error = np.maximum(pm[a:b + 1, segment.object_id] / cfg.position_tolerance_m,
                           rm[a:b + 1, segment.object_id] / cfg.orientation_tolerance_rad) - 1
        run = max(1, round(.25 * fps))
        if len(error) >= run and (np.convolve((error > 0).astype(int), np.ones(run, int), 'valid') >= run).any():
            choices.append((float(np.maximum(error, 0).mean()), segment, mask, (a, b)))
    editor._phase_graph = segments
    if not choices:
        editor._phase_segment = None
        editor._win_info = dict(detect='structured_contacts', window=[1, 1], reason='no_persistent_failure')
        editor._win_chunks = np.zeros(editor.basis.shape[1], bool)
        return np.zeros(frames)
    score, segment, mask, bounds = max(choices, key=lambda x: x[0])
    editor._phase_segment = segment
    giver = find_giver(segment, segments, fps)
    editor._win_info = dict(detect='structured_contacts', key=segment.key, window=list(bounds),
                            start=segment.start, end=segment.end, object_id=segment.object_id,
                            receiver=segment.hand, giver=giver.hand if giver else None, score=score,
                            precision_position_m=cfg.position_tolerance_m,
                            precision_orientation_rad=cfg.orientation_tolerance_rad)
    editor._win_chunks = (editor.basis * mask[:, None]).max(axis=0) > .3
    print(f'[phase-window] step={step} {segment.key} giver={giver.hand if giver else None} window={bounds}', flush=True)
    return mask


def window_configuration(editor):
    return WindowConfig(**{name: editor.options['window_' + name]
                           for name in WindowConfig.__dataclass_fields__
                           if 'window_' + name in editor.options})


def observed_contacts(editor):
    if not hasattr(editor, '_contact_active'):
        return {}
    from flash_chord.retargeting.complete_contacts import hand_contact_evidence
    contact = hand_contact_evidence(editor)
    return hand_object_contacts(contact['active'], contact['force'], contact['slot_hand'],
                                contact['slot_object'], contact['sides'],
                                len(editor.setup.reference.object_body_names()))


def pose_hand_index(editor, side):
    # Link-contact slots may use a different ordering from the embodiment's FK
    # frames. Never use a contact-slot index for a pose/proximity tensor.
    sides = getattr(editor, '_palm_sides', None)
    if sides is None:
        sides = tuple(h.side for h in editor.projector.layout.hands)
    return list(sides).index(side)


def distance_hand_index(editor, side):
    sides = getattr(editor, '_contact_distance_sides', None)
    if sides is None:
        sides = editor.contact_proximity.sides
    return list(sides).index(side)


def validate_feedback_frames(editor):
    expected = np.arange(len(editor.binding.joint_q) - 1)
    recorded = getattr(editor, '_rollout_reference_frames', expected)
    if not np.array_equal(recorded, expected):
        raise ValueError('this adapter requires full-horizon feedback indexed by executed reference commands 0..F-2')


def select_window(editor, learner, step):
    """Local failures are editable intervals; contacts remain their context."""
    editor._phase_feedback = {}
    editor._window_tracking_error = None
    mode = editor.options.get('structured_window_mode', 'local_failures')
    if mode == 'contact_segments':
        editor._phase_edit_mask = None
        editor._phase_edit_bounds = None
        return _segment_window(editor, learner, step)
    if mode != 'local_failures':
        raise ValueError(f'unknown structured window mode: {mode}')
    from flash_chord.retargeting.parallel_rollout import rollout_candidates
    q, target = editor.binding.joint_q, editor.binding.joint_target
    rollout_candidates(editor, learner, np.repeat(q[None], editor.upper_worlds, axis=0),
                       np.repeat(target[None], editor.upper_worlds, axis=0))
    editor._offset_cached = True
    frames, fps = len(q), editor.baseline.fps
    validate_feedback_frames(editor)
    cfg, wc = configuration(editor), window_configuration(editor)
    segments = extract_segments(reference_contacts(editor.setup.reference, frames), fps)
    editor._phase_graph = segments
    p, r = aligned_pose_errors(editor._object_pose)
    initial_valid = np.isfinite(p[0]).all(1) & np.isfinite(r[0]).all(1)
    selection = None; details = dict(reason='insufficient_finite_worlds')
    if initial_valid.sum() >= 4:
        error = robust_tracking_error(p[:, initial_valid], r[:, initial_valid],
                                       position_m=cfg.position_tolerance_m,
                                       orientation_rad=cfg.orientation_tolerance_rad,
                                       quantile=wc.replica_quantile)
        editor._window_tracking_error = error
        selection, details = choose_refinement_window(error, segments, fps, config=wc,
                                                       focus=editor.options.get('window_segments') or ())
        previous = getattr(editor, '_local_window_previous', None)
        last_record = getattr(editor, '_last_record', {}) or {}
        if (selection is not None and previous is not None
                and 'accepted' in last_record and not last_record['accepted']
                and previous['key'] == selection.segment.key
                and abs(previous.get('failure_origin', previous['failure_start']) - selection.failure_origin)
                    <= round(wc.persistence_s * fps)):
            retry = min(previous.get('retry', 0) + 1, 8)
            selection, details = choose_refinement_window(error, segments, fps, config=wc,
                                                          focus=editor.options.get('window_segments') or (), retry=retry)
    contact_selected = False
    if selection is None and initial_valid.sum() >= 4 and editor.options.get('contact_phase_feedback'):
        from flash_chord.retargeting.complete_contacts import hand_contact_evidence
        pose_reason = details.get('reason')
        if not hasattr(editor, '_physical_contact_active'):
            details = dict(reason='complete_physical_contact_evidence_required', pose_reason=pose_reason)
        else:
            contact = hand_contact_evidence(editor)
            touch = hand_object_contacts(contact['active'], contact['force'], contact['slot_hand'],
                contact['slot_object'], contact['sides'], len(editor.setup.reference.object_body_names()))
            touch = {side: values[:, initial_valid] for side, values in touch.items()}
            expected = reference_contacts(editor.setup.reference, frames)
            options = dict(config=wc, focus=editor.options.get('window_segments') or ())
            selection, details = choose_contact_failure_window(touch, expected, segments, fps, **options)
            previous = getattr(editor, '_local_window_previous', None)
            last_record = getattr(editor, '_last_record', {}) or {}
            if (selection is not None and previous is not None and last_record.get('accepted') is False
                    and previous['key'] == selection.segment.key
                    and abs(previous.get('failure_origin', previous['failure_start']) - selection.failure_origin)
                        <= round(wc.persistence_s * fps)):
                selection, details = choose_contact_failure_window(touch, expected, segments, fps,
                    retry=min(previous.get('retry', 0) + 1, 8), **options)
            details['pose_reason'] = pose_reason
            contact_selected = selection is not None
    if selection is None:
        editor._phase_segment = None
        editor._phase_edit_bounds = None
        mask = np.zeros(frames)
        editor._win_info = dict(detect='local_failures', window=[1, 1], **details)
        editor._local_window_previous = None
    else:
        editor._phase_segment = selection.segment
        info = dict(selection.metadata(), **details, replica_quantile=wc.replica_quantile,
                    precision_position_m=cfg.position_tolerance_m,
                    precision_orientation_rad=cfg.orientation_tolerance_rad)
        bounds = selection.bounds
        if editor.options.get('contact_phase_feedback') and not contact_selected:
            expected = reference_contacts(editor.setup.reference, frames)
            planned_giver = find_giver(selection.segment, segments, fps)
            contact_window = acquisition_window(selection, observed_contacts(editor), segments, frames, fps,
                                                 max_window_s=wc.max_window_s,
                                                 expected=expected[selection.segment.hand][:-1, selection.segment.object_id],
                                                 giver_expected=expected[planned_giver.hand][:-1, selection.segment.object_id]
                                                 if planned_giver is not None else None)
            if contact_window is not None:
                bounds, diagnosis = contact_window
                info.update(window=list(bounds), contact_diagnosis=diagnosis)
        editor._phase_edit_bounds = bounds
        mask = window_mask(frames, bounds, fps, blend_s=wc.blend_s)
        giver = find_giver(selection.segment, segments, fps)
        a, b = bounds
        info['giver'] = (giver.hand if giver is not None
                         and (a < selection.segment.start < b or a < giver.end < b) else None)
        editor._win_info = info
        editor._local_window_previous = info.copy()
    editor._phase_edit_mask = mask
    editor._win_chunks = (editor.basis * mask[:, None]).max(axis=0) > .3
    print(f'[local-window] step={step} {editor._win_info}', flush=True)
    return mask


def directed_candidates(editor):
    editor._validated_finger_releases = {}
    selected = getattr(editor, '_phase_segment', None)
    if selected is None:
        return []
    cfg = configuration(editor)
    hands = channels(editor)
    poses = np.asarray(editor._object_pose, float)
    o = selected.object_id
    valid = np.isfinite(poses[:, :, o]).all(2)
    actual = np.full((len(poses), 7), np.nan); target = actual.copy()
    for frame in range(len(poses)):
        if valid[frame].sum() >= 4:
            actual[frame] = np.median(poses[frame, valid[frame], o, :7], axis=0)
            target[frame] = np.median(poses[frame, valid[frame], o, 7:14], axis=0)
    supported = np.zeros(len(actual), bool)
    support_by_hand = {}
    if hasattr(editor, '_contact_active'):
        for h, side in enumerate(editor._contact_sides):
            slots = (editor._contact_slot_hand == h) & (editor._contact_slot_object == o)
            if slots.any():
                force = np.linalg.norm(np.asarray(editor._contact_force[:, :, slots], float), axis=-1)
                touching = (np.asarray(editor._contact_active[:, :, slots]) & (force > 1e-4)).any(2) & valid
                fraction = touching.sum(1) / np.maximum(valid.sum(1), 1)
                fraction[valid.sum(1) < 4] = 0
                support_by_hand[side] = fraction
                supported |= fraction >= .5
    giver = find_giver(selected, editor._phase_graph, editor.baseline.fps)
    if giver is None:
        # A stationary table-supported object is a valid acquisition target.
        speed = np.linalg.norm(np.diff(actual[:, :3], axis=0, prepend=actual[:1, :3]), axis=1) * editor.baseline.fps
        supported |= speed < .02
    bounds = getattr(editor, '_phase_edit_bounds', None)
    if (bounds is not None and giver is not None
            and not (bounds[0] < selected.start < bounds[1] or bounds[0] < giver.end < bounds[1])):
        giver = None
    event = getattr(editor, '_win_info', {}).get('failure_start', selected.start)
    anchor_frame = selected.start if bounds is None or bounds[0] < selected.start < bounds[1] else event
    anchor_row = max(0, min(len(actual) - 1, anchor_frame))
    anchor = supported_anchor(actual, target, supported, anchor_row)
    if anchor is None and bounds is not None:
        # Use support before the loss; never chase the later falling object.
        earlier = np.flatnonzero(supported[bounds[0]:anchor_row + 1])
        if len(earlier):
            anchor = supported_anchor(actual, target, supported, bounds[0] + int(earlier[-1]))
    own_support = support_by_hand.get(selected.hand, np.zeros(len(actual)))
    lo, hi = bounds if bounds is not None else (selected.start, selected.end)
    middle = own_support[max(0, lo):min(len(actual), hi + 1)]
    holding = bool(len(middle) and (middle >= .5).mean() >= .6)
    raw = generate_phase_edits(editor.binding.joint_q, hands, editor._phase_graph, selected,
                               np.asarray(editor.setup.reference.object_body_pos_w(), float),
                               np.asarray(editor.setup.reference.object_body_quat_w(), float)[..., [1, 2, 3, 0]],
                               editor.baseline.fps, anchor=anchor, holding=holding, config=cfg,
                               edit_mask=getattr(editor, '_phase_edit_mask', None),
                               edit_bounds=bounds, event_frame=event)
    reach = None
    if editor.options.get('contact_phase_feedback') and hasattr(editor, '_palm_pose'):
        hand_index = pose_hand_index(editor, selected.hand)
        reference_hand = editor.binding.hand(selected.hand)
        reference_palm = np.concatenate([reference_hand.wrist_pos_w,
                                        reference_hand.wrist_quat_w[:, [1, 2, 3, 0]]], axis=1)[:-1]
        reference = editor.setup.reference
        reference_object = np.concatenate([np.asarray(reference.object_body_pos_w())[:, o],
            np.asarray(reference.object_body_quat_w())[:, o, [1, 2, 3, 0]]], axis=1)[:-1]
        touch = observed_contacts(editor)
        support = np.zeros(valid.shape, bool)
        for values in touch.values(): support |= values[:, :, o]
        if editor.options.get('achieved_reach_feedback', True):
            reach = palm_reach_correction(editor._palm_pose[:, :, hand_index], poses[:, :, o, :7],
                                          reference_palm, reference_object, support & valid, anchor_row)
        measured = []
        from flash_chord.retargeting.contact_refinement import anchor_hand, squeeze_hand
        mask = getattr(editor, '_phase_edit_mask', None)
        if mask is None: mask, _ = phase_mask(len(editor.binding.joint_q), selected.start, selected.end, editor.baseline.fps)
        pivot = reference_hand.wrist_pos_w
        if reach is not None:
            for gain in (.5, 1., 1.5):
                for rotate in (False, True):
                    q = anchor_hand(editor.binding.joint_q, hands[selected.hand], gain * reach[0],
                                    gain * reach[1] if rotate else np.zeros(3), mask, pivot=pivot)
                    for squeeze in (0., .25):
                        qq = squeeze_hand(q, hands[selected.hand], squeeze, mask) if squeeze else q
                        measured.append((dict(op='achieved_reach_correction', receiver=selected.hand,
                            object_id=o, gain=gain, rotate=rotate, squeeze=squeeze,
                            translation_m=reach[0].tolist(), rotation_rad=reach[1].tolist()), qq))
            measured = measured[::2]
        if hasattr(editor, '_contact_direction'):
            ids = np.arange(max(0, anchor_row - 3), min(len(poses), anchor_row + 4))
            direction = editor._contact_direction[ids, :, distance_hand_index(editor, selected.hand), o]
            good = support[ids] & valid[ids] & np.isfinite(direction).all(2)
            good &= np.linalg.norm(direction, axis=-1) > 1e-8
            if good.any():
                delta = np.median(direction[good], axis=0)
                for gain in (.5, 1., 1.5):
                    q = anchor_hand(editor.binding.joint_q, hands[selected.hand], gain * delta,
                                    np.zeros(3), mask)
                    for squeeze in (0., .25):
                        qq = squeeze_hand(q, hands[selected.hand], squeeze, mask) if squeeze else q
                        measured.append((dict(op='contact_direction_correction', receiver=selected.hand,
                            object_id=o, gain=gain, squeeze=squeeze, translation_m=delta.tolist()), qq))
        raw = measured + raw[:max(0, cfg.max_candidates - len(measured))]
    if editor.options.get('contact_full_release'):
        raw = [(info, q) for info, q in raw if info['op'] not in
               ('giver_open', 'receiver_anchor_giver_open', 'receiver_anchor_giver_withdraw')]
    release_raw, release_diagnostics = _full_release_proposals(editor, hands, selected, giver, bounds, raw)
    raw = (release_raw + raw)[:cfg.max_candidates]
    output = []; seen = set()
    for info, q in raw:
        specification = info.get('full_release')
        protected_fingers = None
        if specification is not None:
            _validate_full_release(editor, q, specification)
            ids = tuple(specification['finger_qids'])
            protected_fingers = np.array(q[:, ids], dtype=np.float32, copy=True)
            q = np.array(q, copy=True)
            # Only the independently validated giver fingers bypass generic
            # correction-delta contraction. Every wrist and receiver finger
            # still goes through its ordinary metric/speed trust region.
            q[:, ids] = editor.binding.joint_q[:, ids]
        q, scale = project_edit(editor.binding.joint_q, q, list(hands.values()), editor.projector.qids,
                                editor.projector.lower, editor.projector.upper, editor.baseline.fps, cfg)
        if specification is not None:
            q[:, ids] = protected_fingers
            _validate_full_release(editor, q, specification)
        # Retain the editor's cumulative wrist trust region as references evolve.
        feasible = True
        for hand in hands.values():
            distance = np.linalg.norm(q[:, hand.position] - editor.baseline.joint_q[:, hand.position], axis=1).max()
            angle = (Rotation.from_quat(q[:, hand.quaternion])
                     * Rotation.from_quat(editor.baseline.joint_q[:, hand.quaternion]).inv()).magnitude().max()
            feasible &= distance <= float(editor.options.get('wrist_translation_offset_m', .08)) + 1e-5
            feasible &= angle <= float(editor.options.get('wrist_rotation_offset_rad', .7)) + 1e-5
        key = hashlib.sha256(np.round(q, 6).tobytes()).hexdigest()
        if not feasible or key in seen or np.max(np.abs(q - editor.binding.joint_q)) < 1e-6:
            continue
        seen.add(key)
        tt = editor.binding.joint_target.copy()
        tt[:, editor.projector.dids] = q[:, editor.projector.qids]
        for qi, di in editor.projector.ball_joints:
            tt[:, di:di + 3] = Rotation.from_quat(q[:, qi:qi + 4]).as_rotvec()
        if specification is not None:
            editor._validated_finger_releases[_reference_key(q)] = specification
        output.append((dict(info, projection_scale=scale), q, tt))
    editor._phase_feedback = dict(segment=asdict(selected), giver=asdict(giver) if giver else None,
                                  edit_window=list(bounds) if bounds is not None else None,
                                  receiver_holding=holding,
                                  anchor_translation_m=anchor[0].tolist() if anchor is not None else None,
                                  achieved_reach_translation_m=reach[0].tolist() if reach is not None else None,
                                  achieved_reach_rotation_rad=reach[1].tolist() if reach is not None else None,
                                  support_fraction_by_hand={h: np.round(v, 3).tolist() for h, v in support_by_hand.items()},
                                  full_release_diagnostics=release_diagnostics,
                                  generated=len(output), operations=[x[0] for x in output])
    return output


def local_costs(editor, global_cost):
    if getattr(editor, '_phase_segment', None) is None:
        return global_cost
    p, r = aligned_pose_errors(editor._object_pose)
    a, b = editor._win_info['window']; o = editor._phase_segment.object_id
    local = precision_cost(p[a:b + 1, :, o:o + 1], r[a:b + 1, :, o:o + 1], configuration(editor))
    cost = local + .25 * global_cost
    stats = contact_statistics(editor)
    if stats is not None:
        cost += float(editor.options.get('contact_phase_weight', 2.)) * stats['cost']
    return cost


def contact_statistics(editor):
    if not editor.options.get('contact_phase_feedback') or getattr(editor, '_phase_segment', None) is None:
        return None
    if not hasattr(editor, '_contact_distance'):
        raise RuntimeError('contact-phase feedback requires per-frame contact-distance recording')
    segment = editor._phase_segment
    hand = distance_hand_index(editor, segment.hand)
    touch = observed_contacts(editor)[segment.hand][:, :, segment.object_id]
    distance = editor._contact_distance[:, :, hand, segment.object_id]
    expected = reference_contacts(editor.setup.reference, len(touch) + 1)
    stats = phase_contact_statistics(touch, distance, segment, editor.baseline.fps,
                                     bounds=getattr(editor, '_phase_edit_bounds', None),
                                     expected=expected[segment.hand][:-1, segment.object_id])
    if stats is not None:
        giver = find_giver(segment, editor._phase_graph, editor.baseline.fps)
        if giver is not None:
            all_touch = observed_contacts(editor)
            transfer = transfer_quality(touch, all_touch[giver.hand][:, :, segment.object_id], expected[giver.hand][:-1, segment.object_id],
                                         segment, giver, editor.baseline.fps,
                                         bounds=getattr(editor, '_phase_edit_bounds', None),
                                         receiver_expected=expected[segment.hand][:-1, segment.object_id])
            stats.update(transfer)
            stats['cost'] += transfer['release_penalty']
        editor._contact_phase_latest = {k: np.asarray(v).tolist() for k, v in stats.items()}
    return stats


def contact_progress_gate(editor, candidates, replicas):
    stats = contact_statistics(editor)
    if stats is None: return np.ones(candidates, bool), None
    n = candidates * replicas
    distance, coverage, sustained = (np.asarray(stats[k]) for k in ('distance_m', 'coverage', 'sustained'))
    b_distance, b_coverage, b_sustained = (float(x[n:].mean()) for x in (distance, coverage, sustained))
    cd, cc, cs = (x[:n].reshape(candidates, replicas).mean(1) for x in (distance, coverage, sustained))
    dse = np.sqrt(distance[:n].reshape(candidates, replicas).var(1, ddof=1) / replicas
                  + distance[n:].var(ddof=1) / (len(distance) - n))
    progress = ((b_distance - cd > np.maximum(.001, 2 * dse))
                | (cc >= b_coverage + .05) | (cs >= b_sustained + .1))
    if 'release_quality' in stats:
        rq = stats['release_quality']
        progress |= rq[:n].reshape(candidates, replicas).mean(1) >= rq[n:].mean() + .05
    preserve = (cc >= b_coverage - .02) & (cs >= b_sustained - .05)
    # A healthy grip can improve task pose without increasing contact further.
    passed = preserve & (progress | (b_coverage >= .8 and b_sustained >= .8))
    return passed, dict(baseline_distance_m=b_distance, baseline_coverage=b_coverage,
        baseline_sustained=b_sustained, candidate_distance_m=cd.tolist(),
        candidate_coverage=cc.tolist(), candidate_sustained=cs.tolist(), passed=passed.tolist())


def task_frontier_acceptance(editor, candidates, replicas):
    from flash_chord.retargeting.contact_segment_window import task_frontier_gate
    passed, record = task_frontier_gate(editor._object_pose, candidates, replicas,
        anchor_normalized=getattr(editor, '_contact_task_origin', None))
    if getattr(editor, '_contact_task_origin', None) is None:
        editor._contact_task_origin = record['baseline_normalized_error']
    return passed, record


def install(editor_class):
    """Install opt-in hooks; the existing editor path remains the control."""
    if getattr(editor_class, '_structured_contacts_installed', False):
        return
    original_window = editor_class._failure_window
    original_candidates = editor_class._ilc_candidates
    original_init = editor_class.__init__
    original_blend = getattr(editor_class, '_window_blend', None)

    def initialize(self, **kwargs):
        options = dict(kwargs['options'])
        if options.get('structured_refinement'):
            options.setdefault('window_strategy', 'contact_segments')
            options.setdefault('structured_local_score', True)
            options.setdefault('structured_window_mode', 'local_failures')
            kwargs['options'] = options
        original_init(self, **kwargs)
        if options.get('structured_refinement'):
            self.wrench_feedback = True  # enable the rollout's contact-recording hook

    def window(self, learner, step):
        if self.options.get('structured_refinement'):
            return select_window(self, learner, step)
        return original_window(self, learner, step)

    def candidates(self, learner):
        self._validated_finger_releases = {}
        if not self.options.get('structured_refinement'):
            return original_candidates(self, learner)
        directed = directed_candidates(self)
        old = original_candidates(self, learner)
        # Keep random proposals: directed feedback is a proposal, not a decision.
        total = (self.upper_worlds - int(self.options.get('baseline_replicas', 64))) // int(self.options.get('replicas', 8))
        budget = max(0, total - max(4, total // 4))
        combined = directed[:budget] + old[:max(0, budget - len(directed))]
        summary = {k: v for k, v in getattr(self, '_phase_feedback', {}).items()
                   if k not in ('support_fraction_by_hand', 'operations')}
        self._ilc_feedback = dict(self._ilc_feedback, structured=summary)
        return combined

    def blend(self, q, mask):
        out, target = original_blend(self, q, mask)
        if self.options.get('structured_refinement'):
            # Quaternion conversions can change untouched rows by rounding.
            # Preserve both robot commands exactly outside the editable region.
            outside = np.asarray(mask) == 0
            out[outside] = self.binding.joint_q[outside]
            target[outside] = self.binding.joint_target[outside]
            specification = (getattr(self, '_validated_finger_releases', {}).get(_reference_key(q))
                             if self.options.get('contact_full_release') else None)
            if specification is not None:
                _validate_full_release(self, q, specification)
                ids = tuple(specification['finger_qids'])
                dids = tuple(specification['finger_dids'])
                start, restore = specification['diagnostic']['edit_bounds']
                # The selected wrist mask remains authoritative for every
                # other channel. Only these validated giver-finger rows can
                # extend past it, when the user explicitly allowed extension.
                rows = np.arange(start + 1, restore)
                out[np.ix_(rows, ids)] = q[np.ix_(rows, ids)]
                target[np.ix_(rows, dids)] = out[np.ix_(rows, ids)]
                _validate_full_release(self, out, specification)
        return out, target

    editor_class._failure_window = window
    editor_class._ilc_candidates = candidates
    editor_class.__init__ = initialize
    if original_blend is not None:
        editor_class._window_blend = blend
    editor_class._structured_contacts_installed = True
