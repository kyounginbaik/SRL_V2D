"""Failure timing is a search hint, never a diagnosis of the responsible hand."""
import numpy as np


def first_streak(mask, length):
    mask = np.asarray(mask, bool)
    if len(mask) < length:
        return None
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(length, int), 'valid') == length)
    return int(hits[0]) if len(hits) else None


def failure_timing(position, orientation, fps, *, position_limit=.05,
                   angle_limit=np.deg2rad(20), persistence_seconds=.15):
    """[frame, object] errors, using the reference-command frame's timestamp.

    T is the beginning of the first sustained violation by either object.
    S is the last departure from a sustained half-threshold region before T.
    This hysteresis heuristic finds a lead-up, not a proven causal onset.
    """
    p, a = np.asarray(position), np.asarray(orientation)
    if p.ndim != 2 or p.shape != a.shape or not len(p) or fps <= 0:
        raise ValueError('expected matching nonempty [frame, object] errors and positive fps')
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError('cannot select windows from nonfinite object errors')
    if position_limit <= 0 or angle_limit <= 0 or persistence_seconds <= 0:
        raise ValueError('positive failure thresholds required')
    score = np.maximum(p/position_limit, a/angle_limit)
    length = max(1, int(np.ceil(persistence_seconds*fps-1e-9)))
    per_object = [first_streak(score[:, i] > 1, length) for i in range(p.shape[1])]
    failures = [i for i in per_object if i is not None]
    t = min(failures) if failures else None
    end = t if t is not None else len(p)-1
    # Both objects must be in the low-error region to call the prefix recovered.
    low = np.max(score[:end], axis=1) <= .5
    stable = (np.flatnonzero(np.convolve(low.astype(int), np.ones(length, int), 'valid') == length)
              if len(low) >= length else np.array([], int))
    s = min(end, int(stable[-1]+length)) if len(stable) else 0
    return dict(failed=t is not None, failure_frame=t, onset_frame=s,
                failure_seconds=None if t is None else t/fps, onset_seconds=s/fps,
                detected_seconds=None if t is None else (t+length-1)/fps,
                per_object_failure_seconds=[None if i is None else i/fps for i in per_object],
                position_threshold_m=float(position_limit), orientation_threshold_rad=float(angle_limit),
                persistence_frames=length, onset_rule='last exit from sustained half-threshold region')


def propose_windows(timing, frames, fps, rng, visits, *, width=1., count=4):
    duration = (frames-1)/fps
    # A failure in the first transition has no causal history. Still permit a
    # minimal initial correction after the immutable reset frame, and log it.
    end = max(1/fps, timing['failure_seconds']) if timing['failed'] else duration
    end = min(end, duration)
    width = min(width, end)
    max_start = max(0., end-width)
    selected = []

    def add(start, reason):
        start = float(np.clip(round(start*fps)/fps, 0, max_start))
        if any(abs(start-w['start_seconds']) < .5/fps for w in selected):
            return
        selected.append(dict(window_id=len(selected), start_seconds=start,
                             end_seconds=start+width, reason=reason, focus='both_hands_all_objects'))

    if timing['failed']:
        add(timing['onset_seconds']-width, 'before_error_growth')
        add(end-width, 'before_threshold_failure')
    else:
        add(max_start, 'no_failure_final_window')
    grid = np.arange(int(np.floor(max_start*fps))+1)/fps
    # Random alternatives span all available history, downweighting old visits.
    # Prefer separated starts, but gracefully reduce count in very short prefixes.
    separation = max(1/fps, min(.25, max_start/max(count-1, 1)))
    while len(selected) < count:
        available = np.array([s for s in grid if all(abs(s-w['start_seconds']) >= separation-1e-8 for w in selected)])
        if not len(available):
            break
        weights = np.array([1/(1+.25*visits.get(str(round(float(s)*4)), 0)) for s in available])
        add(rng.choice(available, p=weights/weights.sum()), 'sampled_earlier_history' if timing['failed'] else 'sampled_full_episode')
    for w in selected:
        key = str(round(w['start_seconds']*4))
        visits[key] = visits.get(key, 0)+1
    return selected


def trajectory_metrics(position, orientation, fps, prefix_frames, **timing_options):
    p, a = np.asarray(position), np.asarray(orientation)
    timing = failure_timing(p, a, fps, **timing_options)
    duration = len(p)/fps
    n = min(int(prefix_frames), len(p))
    return dict(first_failure_seconds=duration if not timing['failed'] else timing['failure_seconds'],
                failure_detected=timing['failed'],
                per_object_first_failure_seconds=[duration if t is None else t for t in timing['per_object_failure_seconds']],
                protected_prefix_frames=n,
                prefix_position_mean_m=(p[:n].mean(axis=0) if n else np.zeros(p.shape[1])).tolist(),
                prefix_orientation_mean_rad=(a[:n].mean(axis=0) if n else np.zeros(a.shape[1])).tolist(),
                prefix_position_p95_m=(np.percentile(p[:n], 95, axis=0) if n else np.zeros(p.shape[1])).tolist(),
                prefix_orientation_p95_rad=(np.percentile(a[:n], 95, axis=0) if n else np.zeros(a.shape[1])).tolist())


def prefix_gate(candidate, baseline, *, time_tolerance=.10):
    if candidate['protected_prefix_frames'] != baseline['protected_prefix_frames']:
        return False
    for key, tolerance in [('prefix_position_mean_m', .002), ('prefix_orientation_mean_rad', .01),
                           ('prefix_position_p95_m', .005), ('prefix_orientation_p95_rad', .03)]:
        if np.any(np.asarray(candidate[key]) > np.asarray(baseline[key])+tolerance):
            return False
    return bool(np.all(np.asarray(candidate['per_object_first_failure_seconds']) >=
                       np.asarray(baseline['per_object_first_failure_seconds'])-time_tolerance))


def object_pose_cost(metrics):
    return (metrics['object_position_robust_m']/.03 + metrics['object_orientation_robust_rad']/.3
            + 2*(1-metrics['object_success_robust_fraction']))
