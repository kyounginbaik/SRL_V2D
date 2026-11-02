"""Object/hand-specific contact feedback for manipulation refinement.

Contact presence is reach evidence, not a force-closure certificate. Geometry,
contact persistence and the unchanged object-tracking guards are used together.
All arrays use stepped frames; row i executes reference command frame i.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import find_giver
from flash_chord.retargeting.refinement_window import WindowConfig, choose_refinement_window


@dataclass(frozen=True)
class ContactFeedbackConfig:
    acquisition_grace_s: float = .15
    persistence_s: float = .25
    distance_scale_m: float = .02
    minimum_force_n: float = 1e-4
    window_lookback_s: float = 2.
    acquisition_context_s: float = .5

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all() or min(vars(self).values()) <= 0:
            raise ValueError('positive finite contact feedback settings required')


def hand_object_contacts(active, force, slot_hand, slot_object, sides, objects, *, minimum_force=1e-4):
    """Reduce simulator link slots without mixing hands or object identities."""
    active, force = np.asarray(active), np.asarray(force, float)
    hand, obj = np.asarray(slot_hand), np.asarray(slot_object)
    if active.ndim != 3 or force.shape != (*active.shape, 3) or hand.shape != (active.shape[2],) or obj.shape != hand.shape:
        raise ValueError('expected [frame,world,slot] contact buffers and matching slot identities')
    touching = (np.isfinite(active) & (active > 0) & np.isfinite(force).all(-1)
                & (np.linalg.norm(force, axis=-1) > minimum_force))
    output = {}
    for h, side in enumerate(sides):
        output[side] = np.stack([touching[:, :, (hand == h) & (obj == o)].any(2)
                                for o in range(objects)], axis=-1)
    return output


def contact_phase_mask(frames, segment, fps, *, bounds=None, expected=None, config=ContactFeedbackConfig()):
    t = np.arange(frames - 1)
    start = segment.start + round(config.acquisition_grace_s * fps)
    mask = (t >= start) & (t <= segment.end)
    if bounds is not None:
        mask &= (t > bounds[0]) & (t < bounds[1])
    if expected is not None:
        expected = np.asarray(expected)
        if expected.shape != mask.shape: raise ValueError('expected contacts must have one row per stepped frame')
        mask &= np.isfinite(expected) & (expected > 0)
    return mask


def max_contact_run(touch):
    """Longest continuous contact duration in frames for each rollout world."""
    touch = np.asarray(touch, bool)
    if touch.ndim != 2:
        raise ValueError('expected [frame,world] contact flags')
    run = np.zeros(touch.shape[1], int); longest = run.copy()
    for row in touch:
        run = np.where(row, run + 1, 0)
        longest = np.maximum(longest, run)
    return longest


def phase_contact_statistics(touch, distance, segment, fps, *, bounds=None, expected=None, config=ContactFeedbackConfig()):
    touch, distance = np.asarray(touch, bool), np.asarray(distance, float)
    if touch.shape != distance.shape or touch.ndim != 2:
        raise ValueError('matching [stepped frame,world] touch and distance required')
    mask = contact_phase_mask(len(touch) + 1, segment, fps, bounds=bounds, expected=expected, config=config)
    if not mask.any():
        # A preceding approach window has no mature acquisition interval yet.
        # It can still use continuous reach feedback wherever labels are active.
        t = np.arange(len(touch))
        mask = (t >= segment.start) & (t <= segment.end)
        if bounds is not None: mask &= (t > bounds[0]) & (t < bounds[1])
        if expected is not None: mask &= np.isfinite(expected) & (np.asarray(expected) > 0)
    if not mask.any():
        return None
    d = np.clip(np.nan_to_num(distance[mask], nan=.5, posinf=.5, neginf=.5), 0, .5)
    coverage = touch[mask].mean(0)
    longest = max_contact_run(touch & mask[:, None])
    longest_expected = max_contact_run(mask[:, None])[0]
    required = max(1, min(longest_expected, round(config.persistence_s * fps)))
    sustained = np.minimum(longest / required, 1.)
    return dict(distance_m=d.mean(0), coverage=coverage, sustained=sustained,
                cost=d.mean(0) / config.distance_scale_m + 2 * (1 - coverage) + (1 - sustained))


def transfer_quality(receiver_touch, giver_touch, giver_expected, receiver, giver, fps, *, bounds=None, receiver_expected=None):
    receiver_touch, giver_touch = np.asarray(receiver_touch, bool), np.asarray(giver_touch, bool)
    expected = np.asarray(giver_expected, bool)
    if receiver_touch.shape != giver_touch.shape or expected.shape != (len(receiver_touch),):
        raise ValueError('receiver/giver touch and expected-contact shapes disagree')
    mask = contact_phase_mask(len(expected) + 1, receiver, fps, bounds=bounds, expected=receiver_expected)
    mask &= (np.arange(len(expected)) > giver.end + round(.15 * fps)) & ~expected
    if not mask.any():
        return dict(release_quality=np.zeros(receiver_touch.shape[1]), release_penalty=np.zeros(receiver_touch.shape[1]))
    return dict(release_quality=(receiver_touch[mask] & ~giver_touch[mask]).mean(0),
                release_penalty=(receiver_touch[mask] & giver_touch[mask]).mean(0))


def choose_contact_failure_window(contacts, expected, segments, fps, *, config=WindowConfig(),
                                  focus=(), retry=0, feedback_config=ContactFeedbackConfig()):
    """Find persistent hand/object contact failures even when object pose is good.

    ``expected`` is explicit [F,object] intent for every hand; ``contacts`` is
    complete physical [F-1,world,object] evidence in those same named identities.
    The adapter must establish complete collider coverage before calling this
    pure selector. Short brushes do not establish acquisition. A giver release
    can be selected only in worlds where the receiver has sustained contact and
    currently touches, and only outside ALL intended bimanual overlap.

    Persistence, replica risk, retry lookback, focus and maximum window bounds
    reuse the pose-window settings. Returned windows protect episode endpoints,
    other-object intent for the affected hand(s), and later giver regrasp.
    Labels and observed arrays are never modified.
    """
    if not contacts or set(contacts) != set(expected):
        raise ValueError('explicit expected and observed contacts for the same named hands are required')
    observed = {side: np.asarray(value) for side, value in contacts.items()}
    labels = {side: np.asarray(value) for side, value in expected.items()}
    first = next(iter(observed.values()))
    if first.ndim != 3 or not all(first.shape) or not np.isfinite(fps) or fps <= 0 or retry < 0:
        raise ValueError('nonempty [stepped frame,world,object] contacts and positive FPS are required')
    steps, _, objects = first.shape
    if (any(value.shape != first.shape or value.dtype.kind not in 'biuf' for value in observed.values())
            or any(value.shape != (steps + 1, objects) or value.dtype.kind not in 'biuf'
                   for value in labels.values())):
        raise ValueError('contact observations and full reference labels must have matching frame/object axes')
    observed = {side: np.isfinite(value) & (value > 0) for side, value in observed.items()}
    labels = {side: np.isfinite(value) & (value > 0) for side, value in labels.items()}
    segments = tuple(segments)
    if isinstance(focus, str): focus = [key for key in focus.split(',') if key]
    focus = set(focus)
    keys = {key for s in segments for key in (s.key, f'{s.hand}_seg{s.index}')}
    if focus - keys:
        raise ValueError(f'unknown contact segments: {sorted(focus - keys)}')
    minimum = max(1, round(config.persistence_s * fps))
    choices = []
    for segment in segments:
        if (segment.hand not in labels or not 0 <= segment.object_id < objects
                or not 0 <= segment.start <= segment.end <= steps):
            raise ValueError('contact segment is outside its named hand/object trajectory')
        if focus and segment.key not in focus and f'{segment.hand}_seg{segment.index}' not in focus:
            continue
        own = observed[segment.hand][:, :, segment.object_id]
        intended = labels[segment.hand][:-1, segment.object_id]
        phase = intended & (np.arange(steps) >= segment.start) & (np.arange(steps) <= segment.end)
        run = np.zeros(own.shape[1], int); acquired = np.zeros_like(own)
        reached = np.zeros(own.shape[1], bool)
        for frame in range(segment.start, min(segment.end + 1, steps)):
            run = np.where(own[frame] & phase[frame], run + 1, 0)
            reached |= run >= minimum
            acquired[frame] = reached
        mature = phase & (np.arange(steps) >= segment.start + round(feedback_config.acquisition_grace_s * fps))
        giver = find_giver(segment, segments, fps)
        cases = [('receiver_acquisition_missing', ~acquired, 2),
                 ('receiver_contact_lost', acquired & ~own, 2)]
        next_giver = steps + 1
        if giver is not None:
            next_giver = min((s.start for s in segments if s.hand == giver.hand
                              and s.object_id == segment.object_id and s.start > giver.end), default=steps + 1)
            sole = mature.copy()
            for side, values in labels.items():
                if side != segment.hand: sole &= ~values[:-1, segment.object_id]
            sole &= np.arange(steps) > giver.end + round(feedback_config.acquisition_grace_s * fps)
            sole &= np.arange(steps) < next_giver
            cases.append(('giver_release_missing', observed[giver.hand][:, :, segment.object_id]
                          & own & acquired & sole[:, None], 1))
        for reason, failed, priority in cases:
            trace = np.zeros((steps, objects))
            trace[:, segment.object_id] = 2 * np.quantile((failed & mature[:, None]).astype(float),
                                                        config.replica_quantile, axis=1)
            selection, _ = choose_refinement_window(trace, [segment], fps, config=config, retry=retry)
            if selection is None: continue
            a, b = selection.bounds
            affected = [segment.hand] if priority == 2 else [segment.hand, giver.hand]
            protected = np.zeros(steps, bool)
            other = np.arange(objects) != segment.object_id
            for side in affected: protected |= labels[side][:-1, other].any(1)
            onset = selection.failure_start
            if protected[onset]: continue
            before = np.flatnonzero(protected[:onset])
            after = np.flatnonzero(protected[onset + 1:])
            if len(before): a = max(a, int(before[-1]))
            if len(after): b = min(b, onset + 1 + int(after[0]))
            if priority == 1: b = min(b, next_giver)
            # An initial-state failure can precede the protected left boundary.
            # Keep its true onset, but require a persistent editable continuation
            # rather than moving the failure to an invented later frame.
            editable_start = max(onset, a + 1)
            editable_end = min(selection.failure_end, b - 1)
            if not 0 < a < b < steps or editable_end - editable_start + 1 < minimum: continue
            selection = replace(selection, bounds=(a, b))
            diagnosis = dict(reason=reason, receiver=segment.hand, object_id=segment.object_id,
                giver=giver.hand if giver is not None else None,
                release_eligible=priority == 1, affected_hands=affected,
                editable_failure_start=editable_start,
                receiver_coverage=float(own[mature].mean()), required_contact_frames=minimum)
            choices.append((priority, selection, diagnosis))
    if not choices:
        return None, dict(reason='no_persistent_contact_failure', candidate_count=0)
    choices.sort(key=lambda item: (item[0], item[1].score, -item[1].failure_start), reverse=True)
    _, selection, diagnosis = choices[0]
    return selection, dict(detect='contact_failures', reason='persistent_contact_failure',
        candidate_count=len(choices), retry=int(retry), contact_diagnosis=diagnosis)


def acquisition_window(selection, contacts, segments, frames, fps, *, max_window_s=3.,
                       expected=None, giver_expected=None, config=ContactFeedbackConfig()):
    """Keep the acquisition editable when a nearby receiver still never touches.

    An accepted pose-only improvement must not erase unresolved contact context.
    Long established grasps retain their local error windows.
    """
    if selection is None or selection.segment.hand not in contacts:
        return None
    segment = selection.segment
    elapsed = selection.failure_start - segment.start
    if elapsed < -round(config.acquisition_context_s * fps):
        return None
    touch = np.asarray(contacts[segment.hand])[:, :, segment.object_id]
    mature = contact_phase_mask(frames, segment, fps, expected=expected, config=config)
    mature &= np.arange(frames - 1) <= max(selection.failure_start, segment.start + round(.5 * fps))
    if mature.sum() < max(1, round(config.persistence_s * fps)):
        return None
    coverage = float(touch[mature].mean())
    sustained = max_contact_run(touch & mature[:, None]) >= max(1, round(config.persistence_s * fps))
    if coverage >= .5 or sustained.mean() >= .5:
        giver = find_giver(segment, segments, fps)
        recent = mature & (np.arange(frames - 1) >= max(selection.failure_start, segment.start + round(.5 * fps)) - round(.5 * fps))
        if giver is None or giver.hand not in contacts or not recent.any() or touch[recent].mean() < .5:
            return None
        release = mature & (np.arange(frames - 1) > giver.end + round(config.acquisition_grace_s * fps))
        if giver_expected is not None:
            release &= ~(np.asarray(giver_expected) > 0)
        other = np.asarray(contacts[giver.hand])[:, :, segment.object_id]
        if release.sum() < max(1, round(config.persistence_s * fps)) or other[release].mean() < .5:
            return None
        a = max(1, giver.end - round(config.acquisition_context_s * fps))
        b = min(frames - 2, max(selection.bounds[1], giver.end + round(.75 * fps)), a + round(max_window_s * fps))
        return (a, b), dict(reason='giver_release_missing', receiver_coverage=coverage,
                            acquisition_frame=segment.start, release_frame=giver.end, giver=giver.hand,
                            observed_failure_frame=selection.failure_start,
                            failure_after_edit_window=selection.failure_start > b)
    # If acquisition never happened, its cause stays relevant even after a
    # pose-only edit delays the visible failure. An established long grasp is
    # excluded by the sustained-contact check above.
    a = max(1, segment.start - round(config.acquisition_context_s * fps))
    minimum_end = segment.start + round(.75 * fps)
    giver = find_giver(segment, segments, fps)
    maximum = round(max_window_s * fps)
    if giver is not None and giver.end + round(config.persistence_s * fps) - a <= maximum:
        minimum_end = max(minimum_end, giver.end + round(config.persistence_s * fps))
    if minimum_end - a > maximum:
        return None  # Never silently exceed the configured intervention budget.
    a = max(min(selection.bounds[0], a), minimum_end - maximum, 1)
    b = min(frames - 2, max(selection.bounds[1], minimum_end), a + maximum)
    if not 0 < a < b < frames - 1:
        return None
    return (a, b), dict(reason='receiver_acquisition_missing', receiver_coverage=coverage,
                        acquisition_frame=segment.start, giver=giver.hand if giver else None,
                        observed_failure_frame=selection.failure_start,
                        failure_after_edit_window=selection.failure_start > b)


def palm_reach_correction(actual_palm, actual_object, reference_palm, reference_object, supported,
                          frame, *, radius=3):
    """Correct achieved hand reach in the actual object's frame, not object error alone.

    Poses are xyzw; actual arrays have [frame,world,7], references [frame,7].
    Only supported finite object states are usable, so falling poses are rejected.
    """
    palm, obj = np.asarray(actual_palm, float), np.asarray(actual_object, float)
    rp, ro = np.asarray(reference_palm, float), np.asarray(reference_object, float)
    support = np.asarray(supported, bool)
    if (palm.shape != obj.shape or palm.ndim != 3 or palm.shape[-1] != 7
            or rp.shape != (len(palm), 7) or ro.shape != rp.shape or support.shape != palm.shape[:2]):
        raise ValueError('invalid actual/reference pose or support shapes')
    ids = np.arange(max(0, frame - radius), min(len(palm), frame + radius + 1))
    pp, op = palm[ids], obj[ids]
    valid = support[ids] & np.isfinite(pp).all(2) & np.isfinite(op).all(2)
    valid &= np.linalg.norm(pp[..., 3:], axis=-1) > 1e-8
    valid &= np.linalg.norm(op[..., 3:], axis=-1) > 1e-8
    valid &= (np.isfinite(rp[ids]).all(1) & np.isfinite(ro[ids]).all(1)
              & (np.linalg.norm(rp[ids, 3:], axis=1) > 1e-8)
              & (np.linalg.norm(ro[ids, 3:], axis=1) > 1e-8))[:, None]
    f, w = np.nonzero(valid)
    if not len(f): return None
    r_object = Rotation.from_quat(op[f, w, 3:])
    r_reference = Rotation.from_quat(ro[ids[f], 3:])
    transform = r_object * r_reference.inv()
    desired_p = op[f, w, :3] + transform.apply(rp[ids[f], :3] - ro[ids[f], :3])
    desired_r = transform * Rotation.from_quat(rp[ids[f], 3:])
    dp = np.median(desired_p - pp[f, w, :3], axis=0)
    rv = (desired_r * Rotation.from_quat(pp[f, w, 3:]).inv()).mean().as_rotvec()
    return dp, rv
