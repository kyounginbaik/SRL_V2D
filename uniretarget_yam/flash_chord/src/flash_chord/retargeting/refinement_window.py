"""Local failure windows inside hand/object contact phases.

Contact labels provide intervention context, not manipulation-stage boundaries.
Persistent failures contribute onset windows. A later sustained escalation
after a stable error plateau can form another event within the same contact.
Re-simulation selects the next window.
All reported frame numbers refer to commands: rollout row i executes frame i.
"""
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    persistence_s: float = .25
    recovery_s: float = .2
    lead_s: float = .5
    tail_s: float = .5
    max_window_s: float = 3.
    max_lookback_s: float = 2.
    replica_quantile: float = .75
    blend_s: float = .25
    escalation_ratio: float = 2.
    escalation_margin: float = 1.
    plateau_relative_range: float = .25

    def __post_init__(self):
        durations = [self.persistence_s, self.lead_s, self.tail_s,
                     self.max_window_s, self.max_lookback_s, self.blend_s,
                     self.escalation_ratio, self.escalation_margin, self.plateau_relative_range]
        if (not np.isfinite(durations).all() or min(durations) <= 0
                or not np.isfinite(self.recovery_s) or self.recovery_s < 0
                or not 0 < self.replica_quantile <= 1
                or self.escalation_ratio <= 1
                or self.max_window_s < self.lead_s + self.persistence_s + self.tail_s
                or self.max_lookback_s < self.lead_s):
            raise ValueError('invalid local-window durations or replica quantile')


def robust_tracking_error(position, orientation, *, position_m=.03,
                          orientation_rad=np.pi / 6, quantile=.75):
    """Normalize within each world before computing a replica risk quantile.

    Unlike a median, the default quantile exposes failures affecting a sizeable
    minority of worlds. Nonfinite errors remain failures instead of vanishing.
    """
    p, r = np.asarray(position, float), np.asarray(orientation, float)
    if (p.shape != r.shape or p.ndim != 3 or not all(p.shape)
            or not np.isfinite([position_m, orientation_rad, quantile]).all()
            or min(position_m, orientation_rad) <= 0 or not 0 < quantile <= 1):
        raise ValueError('expected nonempty [stepped frame, world, object] errors and positive thresholds')
    error = np.maximum(p / position_m, r / orientation_rad)
    invalid = ~np.isfinite(p) | ~np.isfinite(r) | (p < 0) | (r < 0)
    error = np.where(invalid, 100., np.minimum(error, 100.))
    return np.quantile(error, quantile, axis=1)


def persistent_failures(trace, fps, config=WindowConfig()):
    """Return inclusive reference-frame bounds, merging only brief recoveries.

    Each component must first satisfy continuous persistence. A series of short
    impact spikes cannot become a sustained failure merely by gap merging.
    """
    x = np.asarray(trace, float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('finite one-dimensional error trace and positive FPS required')
    edges = np.flatnonzero(np.diff(np.r_[False, x > 1, False]))
    minimum = max(1, round(config.persistence_s * fps))
    recovery = round(config.recovery_s * fps)
    runs = []
    for start, stop in edges.reshape(-1, 2):
        if stop - start < minimum:
            continue
        a, b = int(start), int(stop - 1)
        if runs and a - runs[-1][1] - 1 <= recovery:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def window_mask(frames, bounds, fps, *, blend_s=.25):
    """Smooth edit weights, exactly zero at/outside both window boundaries."""
    a, b = bounds
    if not 0 < a < b < frames - 1 or fps <= 0 or blend_s <= 0:
        raise ValueError('a window requires a nonempty interior and protected episode endpoints')
    ramp = max(1., min(blend_s * fps, (b - a) / 2))
    t = np.arange(frames, dtype=float)
    weight = np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0, 1)
    return .5 - .5 * np.cos(np.pi * weight)


def failure_events(trace, fps, config=WindowConfig()):
    """Split a sustained failure only at distinct, stable-to-worse transitions.

    This prevents mild early drift from starving a later failed manipulation.
    A rapid cascade within one maximum window retains its original onset.
    Escalations require a stable preceding plateau and continuously elevated
    following errors; isolated impact spikes cannot create them.
    """
    x = np.asarray(trace, float)
    stable = max(1, round(config.persistence_s * fps))
    separation = max(stable, round((config.max_window_s - config.lead_s) * fps))
    events = []
    for start, end in persistent_failures(x, fps, config):
        starts = [start]
        values = x[start:end + 1]
        if len(values) >= 2 * stable:
            windows = np.lib.stride_tricks.sliding_window_view(values, stable)
            before = windows[:-stable]
            after = windows[stable:]
            level = before.mean(1)
            # Only a clear additional deficit creates another event. These
            # dimensionless criteria apply equally to position/orientation.
            raised = ((before.max(1) - before.min(1) <= config.plateau_relative_range * np.maximum(level, 1))
                      & (after.min(1) >= np.maximum(config.escalation_ratio * level,
                                                    level + config.escalation_margin)))
            edges = np.flatnonzero(np.diff(np.r_[False, raised].astype(int)) == 1)
            for edge in edges:
                onset = start + stable + int(edge)
                if onset - starts[-1] >= separation:
                    starts.append(onset)
        for i, onset in enumerate(starts):
            stop = starts[i + 1] - 1 if i + 1 < len(starts) else end
            events.append((onset, stop, start))
    return events


@dataclass(frozen=True)
class LocalWindow:
    segment: object
    failure_start: int
    failure_end: int
    bounds: tuple[int, int]
    score: float
    lead_s: float
    transition: bool
    failure_origin: int

    def metadata(self):
        s = self.segment
        return dict(detect='local_failures', key=s.key, window=list(self.bounds),
                    start=s.start, end=s.end, object_id=s.object_id, receiver=s.hand,
                    failure_start=self.failure_start, failure_end=self.failure_end,
                    failure_origin=self.failure_origin,
                    event='error_escalation' if self.failure_start != self.failure_origin else 'failure_onset',
                    score=self.score, lead_s=self.lead_s, contact_transition=self.transition)


def choose_refinement_window(error, segments, fps, *, config=WindowConfig(), focus=(), retry=0):
    """Choose a bounded onset window, retaining the original contact identity.

    Independent failures compete by local severity. Rapid failure cascades are
    addressed from onset; distinct later escalations may get their own window.
    Rejected attempts widen preceding context, up to max_lookback_s; they never
    authorize editing an entire long grasp. Contact lookup is indexed by object
    and hand, avoiding a sliding-window scan or all segment/failure pairs.
    """
    x = np.asarray(error, float)
    if (x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all()
            or not np.isfinite(fps) or fps <= 0 or retry < 0):
        raise ValueError('finite [stepped frame, object] errors, positive FPS and nonnegative retry required')
    frames = len(x) + 1
    if isinstance(focus, str):
        focus = [s for s in focus.split(',') if s]
    focus = set(focus)
    indexed = {}
    valid_keys = set()
    for segment in segments:
        if not 0 <= segment.object_id < x.shape[1] or not 0 <= segment.start <= segment.end < frames:
            raise ValueError('contact segment is outside the error trajectory')
        simple = f'{segment.hand}_seg{segment.index}'
        valid_keys.update((segment.key, simple))
        if not focus or segment.key in focus or simple in focus:
            indexed.setdefault((segment.object_id, segment.hand), []).append(segment)
    if focus - valid_keys:
        raise ValueError(f'unknown contact segments: {sorted(focus - valid_keys)}')
    by_object = {}
    for (obj, _), group in indexed.items():
        group.sort(key=lambda s: s.start)
        by_object.setdefault(obj, []).append(([s.start for s in group], group))
    lead_s = min(config.max_lookback_s, config.lead_s * 2 ** min(int(retry), 8))
    lead = max(1, round(lead_s * fps))
    tail = max(1, round(config.tail_s * fps))
    maximum = max(2, round(config.max_window_s * fps))
    lookback = round(config.max_lookback_s * fps)
    ahead = round(config.lead_s * fps)
    choices = []; failures = 0
    for obj in range(x.shape[1]):
        for onset, end, origin in failure_events(x[:, obj], fps, config):
            failures += 1
            nearby = []
            for starts, group in by_object.get(obj, []):
                j = bisect_right(starts, onset)
                # Current/most recent contact and the next contact onset.
                for k in (j - 1, j):
                    if (0 <= k < len(group) and group[k].start <= onset + ahead
                            and group[k].end >= onset - lookback):
                        nearby.append(group[k])
            for s in nearby:
                a = max(1, onset - lead)
                # Include a nearby acquisition approach or the last release
                # that can still affect a currently unsupported object.
                if onset - lead <= s.start <= onset + ahead:
                    a = min(a, max(1, s.start - round(config.lead_s * fps)))
                if s.end < onset:
                    a = min(a, max(1, s.end - round(config.blend_s * fps)))
                # Leave room to observe the initial sustained failure even
                # after expanding the preceding intervention context.
                required = min(end, onset + max(1, round(config.persistence_s * fps)) - 1)
                a = max(a, required + tail - maximum)
                b = min(frames - 2, end + tail, a + maximum)
                if not 0 < a < b < frames - 1 or onset > b:
                    continue
                values = np.maximum(x[onset:min(end + 1, onset + maximum), obj] - 1, 0)
                score = float(values.mean() + .25 * np.quantile(values, .95))
                transition = a < s.start < b
                choices.append(LocalWindow(s, onset, end, (a, b), score, lead_s, transition, origin))
    if not choices:
        return None, dict(reason='no_persistent_failure' if not failures else 'no_contact_near_failure',
                          failure_count=failures, candidate_count=0)
    choices.sort(key=lambda w: (w.score, w.transition, w.segment.start, -w.failure_start), reverse=True)
    return choices[0], dict(failure_count=failures, candidate_count=len(choices), retry=int(retry),
                            alternatives=[w.metadata() for w in choices[:8]])
