"""Complete finger releases with explicit timing and joint-limit contracts.

The embodiment supplies the open posture; this module does not infer opening
directions from joint names or closure weights. Only the specified finger
channels change. No object pose, contact label, or wrist channel is an input.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from flash_chord.retargeting.contact_refinement import HandChannels


@dataclass(frozen=True)
class ReleaseProposal:
    """An exact full release, or a timing diagnosis without a partial proposal."""

    joint_q: np.ndarray | None
    diagnostic: dict

    @property
    def feasible(self) -> bool:
        return self.joint_q is not None


@dataclass(frozen=True)
class ProcessedFingerProposal:
    """Actuator targets and their reference/residual split, not physical poses.

    Rows correspond to ``frame_ids`` only. The caller retains all later motion;
    the last row restores its explicitly supplied scheduled command/reference.
    An infeasible proposal has no partial tracks.
    """

    frame_ids: np.ndarray | None
    command: np.ndarray | None
    reference: np.ndarray | None
    residual: np.ndarray | None
    diagnostic: dict

    @property
    def feasible(self) -> bool:
        return self.command is not None


def _smoothstep(t):
    """Quintic interpolation has zero endpoint velocity and acceleration."""
    return t * t * t * (10. + t * (-15. + 6. * t))


def _ramp(start, end, steps):
    if steps == 0:
        return np.asarray(start, float)[None].copy()
    weight = _smoothstep(np.arange(steps + 1, dtype=float) / steps)
    values = start + weight[:, None] * (end - start)
    # Preserve exact boundary contracts even with floating-point cancellation.
    values[0], values[-1] = start, end
    return values


def _minimum_steps(delta, fps, speed):
    # The maximum derivative of quintic smoothstep on [0,1] is 15/8.
    return int(np.ceil(1.875 * np.max(np.abs(delta), initial=0.) * fps / speed))


def plan_full_release(
    current,
    hand: "HandChannels | Sequence[int]",
    open_target,
    lower,
    upper,
    *,
    fps: float,
    bounds: tuple[int, int],
    event_frame: int,
    hold_until: int,
    next_contact_frame: int | None = None,
    delay_s: float = 0.,
    max_speed_rad_s: float = 3.,
    max_window_extension_s: float = 0.,
    complete_by: int | None = None,
) -> ReleaseProposal:
    """Open every requested joint fully, hold open, then restore the reference.

    ``event_frame + round(delay_s * fps)`` is the opening onset. Its command
    remains unchanged; the duration is derived from the requested joint delta
    and a smooth absolute finger-velocity bound. The exact adapter-supplied
    ``open_target`` is held from completion through ``hold_until`` (inclusive).
    ``complete_by`` optionally specifies the first required no-contact frame.

    Bounds are inclusive protected boundary frames. Only their interior may
    change, except that the right boundary may extend by the explicitly allowed
    ``max_window_extension_s``. The original reference is restored as early as
    its velocity bound permits after the hold, at or before ``next_contact_frame``
    and the allowed right boundary. All commands at and after that restoration,
    including a later intended regrasp, remain exact. The first and final
    trajectory frames are always protected.

    ``hand`` may be HandChannels or explicit finger q indices. Target and limits
    are one-dimensional arrays in that finger order. Targets outside the supplied
    physical limits are rejected, never clipped. Insufficient timing returns an
    infeasible result, never a contracted, partially opened hand.

    Velocity is the absolute command velocity, not velocity of the correction
    relative to ``current``. Both peaks are reported so a caller can avoid a
    later generic projection silently shrinking this validated full release.
    """
    base = np.asarray(current)
    if (base.ndim != 2 or len(base) < 3 or not np.issubdtype(base.dtype, np.number) or not np.isrealobj(base)
            or not np.isfinite(base).all()):
        raise ValueError('a finite [frame, joint] reference with at least three frames is required')
    finger_values = getattr(hand, 'fingers', hand)
    fingers = tuple(finger_values)
    if (not fingers or any(not isinstance(i, (int, np.integer)) for i in fingers)
            or len(set(fingers)) != len(fingers) or min(fingers) < 0 or max(fingers) >= base.shape[1]):
        raise ValueError('distinct valid finger q indices are required')
    target, lo, hi = (np.asarray(x, dtype=float) for x in (open_target, lower, upper))
    if any(x.shape != (len(fingers),) or not np.isfinite(x).all() for x in (target, lo, hi)):
        raise ValueError('one finite open target and lower/upper bound per finger is required')
    if np.any(lo > hi) or np.any(target < lo) or np.any(target > hi):
        raise ValueError('open target must satisfy the supplied physical joint limits')
    if np.any(base[:, fingers] < lo - 1e-6) or np.any(base[:, fingers] > hi + 1e-6):
        raise ValueError('current finger reference violates physical joint limits')
    if (not np.isfinite([fps, delay_s, max_speed_rad_s, max_window_extension_s]).all()
            or fps <= 0 or max_speed_rad_s <= 0 or delay_s < 0 or max_window_extension_s < 0):
        raise ValueError('positive fps/speed and nonnegative finite delays/extensions are required')
    if len(bounds) != 2:
        raise ValueError('two inclusive protected window boundaries are required')
    a, b = bounds
    indices = (a, b, event_frame, hold_until)
    if next_contact_frame is not None:
        indices += (next_contact_frame,)
    if complete_by is not None:
        indices += (complete_by,)
    if any(not isinstance(i, (int, np.integer)) for i in indices):
        raise ValueError('window, event, hold, and contact frames must be integers')
    if not 0 <= a < b < len(base):
        raise ValueError('window boundaries must lie in the reference')
    if not 0 <= hold_until < len(base) or not 0 <= event_frame < len(base):
        raise ValueError('event and hold frames must lie in the reference')
    if next_contact_frame is not None and not 0 <= next_contact_frame < len(base):
        raise ValueError('next intended contact must lie in the reference')
    if complete_by is not None and not 0 <= complete_by <= hold_until:
        raise ValueError('completion deadline must precede or equal the last required open frame')

    start = int(event_frame + round(delay_s * fps))
    limit = min(len(base) - 1, b + round(max_window_extension_s * fps))
    if next_contact_frame is not None:
        limit = min(limit, next_contact_frame)
    info = dict(
        feasible=False, finger_qids=list(map(int, fingers)), requested_bounds=[int(a), int(b)],
        opening_start=start, hold_until=int(hold_until), latest_restore_frame=int(limit),
        next_contact_frame=int(next_contact_frame) if next_contact_frame is not None else None,
        delay_s=float(delay_s), max_speed_rad_s=float(max_speed_rad_s),
        open_target=target.tolist(),
    )

    def infeasible(reason, **details):
        return ReleaseProposal(None, dict(info, reason=reason, **details))

    if not a <= start < limit:
        return infeasible('opening_onset_outside_window')
    opening_steps = _minimum_steps(target - base[start, fingers], fps, max_speed_rad_s)
    complete = start + opening_steps
    info.update(opening_complete=complete, opening_duration_s=opening_steps / fps,
                minimum_opening_frames=opening_steps)
    deadline = min(hold_until, complete_by if complete_by is not None else hold_until)
    if complete > deadline:
        return infeasible('insufficient_opening_time', available_opening_frames=max(0, deadline - start),
                          required_completion_frame=complete, completion_deadline=int(deadline))
    if hold_until > limit:
        return infeasible('hold_overlaps_protected_motion', required_open_through=int(hold_until))

    # The future original posture can move, so determine return duration against
    # each actual endpoint instead of assuming the opening delta is symmetric.
    restore = None
    for frame in range(hold_until, limit + 1):
        if _minimum_steps(base[frame, fingers] - target, fps, max_speed_rad_s) <= frame - hold_until:
            restore = frame
            break
    if restore is None:
        required_steps = _minimum_steps(base[limit, fingers] - target, fps, max_speed_rad_s)
        return infeasible('insufficient_restore_time', available_restore_frames=int(limit - hold_until),
                          minimum_restore_frames_at_boundary=required_steps)

    # Float64 construction preserves exact unchanged rows even for a float32
    # input. The adapter can cast after checking its physical projection.
    out = np.array(base, dtype=float, copy=True)
    out[np.ix_(np.arange(start, complete + 1), fingers)] = _ramp(base[start, fingers], target, opening_steps)
    out[np.ix_(np.arange(complete, hold_until + 1), fingers)] = target
    out[np.ix_(np.arange(hold_until, restore + 1), fingers)] = _ramp(target, base[restore, fingers], restore - hold_until)
    command_speed = np.abs(np.diff(out[start:restore + 1, fingers], axis=0)).max(initial=0.) * fps
    added = out[:, fingers] - base[:, fingers]
    added_speed = np.abs(np.diff(added, axis=0)).max(initial=0.) * fps
    if command_speed > max_speed_rad_s + 1e-9:
        return infeasible('velocity_contract_failed', peak_command_speed_rad_s=float(command_speed))
    info.update(
        feasible=True, reason='full_release', restore_frame=int(restore),
        edit_bounds=[int(start), int(restore)],
        window_extension_frames=max(0, int(restore - b)),
        open_interval=[int(complete), int(hold_until)],
        return_duration_s=(restore - hold_until) / fps,
        peak_command_speed_rad_s=float(command_speed), peak_added_speed_rad_s=float(added_speed),
        max_finger_delta_rad=float(np.abs(added).max(initial=0.)),
    )
    return ReleaseProposal(out, info)


def plan_release_by(current, hand, open_target, lower, upper, *, fps, bounds,
                    complete_by, hold_until, next_contact_frame=None,
                    max_speed_rad_s=3.) -> ReleaseProposal:
    """Find the latest feasible opening onset before a withdrawal deadline.

    Search the actual changing reference, rather than subtracting a duration
    measured at the deadline: the posture at an earlier onset can differ.
    Every candidate must fully open, hold, and restore within the existing
    release contract. No partial opening is returned when timing is infeasible.
    Bounds protect the preceding skill and a later regrasp independently.
    """
    # Validate all caller inputs with the same public contract before searching.
    initial = plan_full_release(current, hand, open_target, lower, upper,
        fps=fps, bounds=bounds, event_frame=complete_by, complete_by=complete_by,
        hold_until=hold_until, next_contact_frame=next_contact_frame,
        max_speed_rad_s=max_speed_rad_s)
    attempts = 1
    if initial.feasible:
        return ReleaseProposal(initial.joint_q, dict(initial.diagnostic,
            onset_search_attempts=attempts, completion_deadline=int(complete_by)))
    for onset in range(complete_by - 1, bounds[0] - 1, -1):
        candidate = plan_full_release(current, hand, open_target, lower, upper,
            fps=fps, bounds=bounds, event_frame=onset, complete_by=complete_by,
            hold_until=hold_until, next_contact_frame=next_contact_frame,
            max_speed_rad_s=max_speed_rad_s)
        attempts += 1
        if candidate.feasible:
            return ReleaseProposal(candidate.joint_q, dict(candidate.diagnostic,
                onset_search_attempts=attempts, completion_deadline=int(complete_by)))
    return ReleaseProposal(None, dict(initial.diagnostic,
        reason='no_feasible_opening_before_deadline', onset_search_attempts=attempts,
        completion_deadline=int(complete_by)))


def plan_finger_transition(current, hand, target, lower, upper, **timing) -> ReleaseProposal:
    """Plan the same bounded pose/hold/restore operation for any finger target.

    For example, ``target`` may be an embodiment-fitted receiving grasp rather
    than an open posture. Timing and diagnostics follow ``plan_full_release``;
    no opening direction or closure-weight assumption is applied to the target.
    """
    return plan_full_release(current, hand, target, lower, upper, **timing)


def plan_processed_finger_transition(
    start_command,
    target,
    lower,
    upper,
    residual_clip,
    *,
    initial_reference,
    restore_command,
    restore_reference,
    target_reference=None,
    fps: float,
    bounds: tuple[int, int],
    event_frame: int,
    hold_until: int,
    restore_frame: int | None = None,
    next_contact_frame: int | None = None,
    complete_by: int | None = None,
    max_speed_rad_s: float = 3.,
    max_reference_speed_rad_s: float | None = None,
) -> ProcessedFingerProposal:
    """Release captured processed PD targets without discarding their preload.

    All vectors use the caller's explicit finger order. A processed command is
    ``reference + residual``. Its captured value may exceed physical joint
    limits by the allowed residual; it must never be clipped into a physical
    posture before planning. References remain within ``lower/upper`` and
    residuals within the scalar or per-finger ``residual_clip``.

    Inclusive ``bounds`` identify the controlled interval. The captured command
    AND ``initial_reference`` remain exact through ``event_frame``. Shared
    quintic ramps then bring the processed command to the exact ``target``, hold
    it through ``hold_until``, and restore the explicitly supplied scheduled
    command/reference. ``target_reference`` defaults to the nearest physical
    reference; a physical open target therefore has zero residual. An open
    processed target outside a physical stop is also permitted if its reference
    split respects the residual limit. Sharing the
    interpolation weights keeps the residual within its limits by convexity.
    Opening duration accounts for both command and reference velocity; neither
    is a velocity of the correction relative to another trajectory.

    By default restoration ends at the right bound, or the next intended
    contact if earlier. An explicit ``restore_frame`` cannot overrun either.
    Only rows from the left bound through restoration are returned, so the
    caller can preserve every subsequent intended contact and endpoint. Future
    command/reference values must not be inferred from the captured grasp.

    Endpoint feasibility and timing failures return diagnostics with no tracks.
    Structural errors raise ValueError. A 1e-7-radian numerical tolerance admits
    float32 snapshot addition/rounding without weakening a captured preload.
    This validates reference/residual ranges and slew, not the reachability of
    a policy input through a particular EMA or action mapping. The caller must
    still encode the residual through its ordinary actuator interface.
    """
    values = [np.asarray(value) for value in
              (start_command, target, lower, upper, initial_reference,
               restore_command, restore_reference)]
    if any(value.ndim != 1 or not np.isrealobj(value) or not np.issubdtype(value.dtype, np.number)
           for value in values):
        raise ValueError('finite one-dimensional finger endpoint vectors are required')
    start, goal, lo, hi, initial, returned, returned_reference = [value.astype(float) for value in values]
    if not len(start) or any(value.shape != start.shape or not np.isfinite(value).all()
                             for value in (start, goal, lo, hi, initial, returned, returned_reference)):
        raise ValueError('matching nonempty finite finger endpoint vectors are required')
    clip = np.asarray(residual_clip)
    if not np.isrealobj(clip) or not np.issubdtype(clip.dtype, np.number):
        raise ValueError('a finite nonnegative scalar or per-finger residual limit is required')
    try:
        clip = np.broadcast_to(clip.astype(float), start.shape)
    except ValueError as exc:
        raise ValueError('a scalar or matching per-finger residual limit is required') from exc
    if np.any(lo > hi) or not np.isfinite(clip).all() or np.any(clip < 0):
        raise ValueError('ordered joint bounds and finite nonnegative residual limits are required')
    goal_reference = np.clip(goal, lo, hi) if target_reference is None else np.asarray(target_reference)
    if (goal_reference.shape != start.shape or not np.isrealobj(goal_reference)
            or not np.issubdtype(goal_reference.dtype, np.number) or not np.isfinite(goal_reference).all()):
        raise ValueError('a matching finite target reference is required')
    goal_reference = goal_reference.astype(float)
    reference_speed = max_speed_rad_s if max_reference_speed_rad_s is None else max_reference_speed_rad_s
    if (not np.isfinite([fps, max_speed_rad_s, reference_speed]).all()
            or fps <= 0 or max_speed_rad_s <= 0 or reference_speed <= 0):
        raise ValueError('positive finite fps and command/reference speeds are required')
    if len(bounds) != 2:
        raise ValueError('two inclusive frame bounds are required')
    a, b = bounds
    frame_values = (a, b, event_frame, hold_until)
    frame_values += tuple(i for i in (restore_frame, next_contact_frame, complete_by) if i is not None)
    if any(not isinstance(i, (int, np.integer)) for i in frame_values) or not 0 <= a <= b:
        raise ValueError('nonnegative integer frame bounds and integer event frames are required')
    if any(i < 0 for i in frame_values):
        raise ValueError('frame indices must be nonnegative')

    tolerance = 1e-7
    limit = min(b, next_contact_frame) if next_contact_frame is not None else b
    restore = limit if restore_frame is None else restore_frame
    info = dict(feasible=False, target_space='processed_PD_finger_targets',
                requested_bounds=[int(a), int(b)], opening_start=int(event_frame),
                hold_until=int(hold_until), restore_frame=int(restore),
                next_contact_frame=int(next_contact_frame) if next_contact_frame is not None else None,
                max_speed_rad_s=float(max_speed_rad_s),
                max_reference_speed_rad_s=float(reference_speed), residual_clip=clip.tolist(),
                numerical_tolerance_rad=tolerance,
                reference_split='physical_joint_reference_plus_bounded_policy_residual')

    def infeasible(reason, **details):
        return ProcessedFingerProposal(None, None, None, None, dict(info, reason=reason, **details))

    for label, reference in (('initial', initial), ('target', goal_reference), ('restore', returned_reference)):
        if np.any(reference < lo - tolerance) or np.any(reference > hi + tolerance):
            return infeasible(label + '_reference_outside_joint_limits')
    for label, command, reference in (('start', start, initial), ('target', goal, goal_reference),
                                       ('restore', returned, returned_reference)):
        if np.any(np.abs(command - reference) > clip + tolerance):
            return infeasible(label + '_command_exceeds_residual_limit')
    if not a <= event_frame <= b:
        return infeasible('opening_onset_outside_window')
    if not event_frame <= restore <= b:
        return infeasible('restore_outside_window')
    if restore > limit:
        return infeasible('restore_overlaps_protected_contact')
    if hold_until > restore:
        return infeasible('hold_overlaps_protected_motion')
    if hold_until < event_frame:
        return infeasible('hold_precedes_opening_onset')

    opening_steps = max(_minimum_steps(goal - start, fps, max_speed_rad_s),
                        _minimum_steps(goal_reference - initial, fps, reference_speed))
    complete = event_frame + opening_steps
    restore_steps = max(_minimum_steps(returned - goal, fps, max_speed_rad_s),
                        _minimum_steps(returned_reference - goal_reference, fps, reference_speed))
    info.update(opening_complete=int(complete), minimum_opening_frames=opening_steps,
                opening_duration_s=opening_steps / fps, minimum_restore_frames=restore_steps)
    deadline = min(hold_until, complete_by) if complete_by is not None else hold_until
    if complete > deadline:
        return infeasible('insufficient_opening_time', available_opening_frames=max(0, int(deadline - event_frame)),
                          completion_deadline=int(deadline))
    if restore - hold_until < restore_steps:
        return infeasible('insufficient_restore_time', available_restore_frames=int(restore - hold_until))

    frames = np.arange(a, restore + 1, dtype=np.int64)
    commands = np.broadcast_to(start, (len(frames), len(start))).copy()
    references = np.broadcast_to(initial, commands.shape).copy()
    opening = slice(event_frame - a, complete - a + 1)
    commands[opening] = _ramp(start, goal, opening_steps)
    references[opening] = _ramp(initial, goal_reference, opening_steps)
    commands[complete - a:hold_until - a + 1] = goal
    references[complete - a:hold_until - a + 1] = goal_reference
    returning = slice(hold_until - a, restore - a + 1)
    commands[returning] = _ramp(goal, returned, restore - hold_until)
    references[returning] = _ramp(goal_reference, returned_reference, restore - hold_until)
    residual = commands - references
    command_speed = float(np.abs(np.diff(commands, axis=0)).max(initial=0.) * fps)
    achieved_reference_speed = float(np.abs(np.diff(references, axis=0)).max(initial=0.) * fps)
    if (command_speed > max_speed_rad_s + 1e-9 or achieved_reference_speed > reference_speed + 1e-9
            or np.any(references < lo - tolerance) or np.any(references > hi + tolerance)
            or np.any(np.abs(residual) > clip + tolerance)):
        return infeasible('processed_target_contract_failed')
    info.update(feasible=True, reason='processed_target_transition',
                edit_bounds=[int(event_frame), int(restore)], open_interval=[int(complete), int(hold_until)],
                peak_processed_target_speed_rad_s=command_speed,
                peak_reference_speed_rad_s=achieved_reference_speed,
                peak_residual_rad=float(np.abs(residual).max(initial=0.)),
                peak_processed_joint_limit_overdrive_rad=float(np.maximum(lo - commands, commands - hi).max(initial=0.)),
                initial_preload_rad=(start - initial).tolist(),
                target_preload_rad=(goal - goal_reference).tolist(),
                restore_preload_rad=(returned - returned_reference).tolist())
    return ProcessedFingerProposal(frames, commands, references, residual, info)
