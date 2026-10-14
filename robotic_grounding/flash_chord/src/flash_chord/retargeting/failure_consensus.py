"""Locate repeatable failure onsets, without counting persistent errors twice."""
import numpy as np

from flash_chord.retargeting.failure_windows import failure_timing


def failure_consensus(position, orientation, fps, *, minimum_fraction=.5,
                      cluster_seconds=.5, **timing_options):
    """Use first sustained onsets per world AND object, never time-in-failure.

    Prefer the earliest cluster supported by minimum_fraction of ALL worlds.
    If none qualifies, use the largest cluster (earliest on ties) and explicitly
    report weak consensus. Successful worlds stay in the denominator. A cluster
    contains onsets within cluster_seconds, not a chain of pairwise-near onsets.
    The median onset time targets the recurring event; the lower-quartile error
    growth time conservatively selects its lead-up. Neither identifies causality.
    """
    p, a = np.asarray(position), np.asarray(orientation)
    if p.ndim != 3 or p.shape != a.shape or min(p.shape) < 1:
        raise ValueError('expected nonempty [frame, world, object] errors')
    if not np.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError('minimum_fraction must be in (0, 1]')
    if not np.isfinite(cluster_seconds) or cluster_seconds <= 0:
        raise ValueError('cluster_seconds must be positive')
    # failure_timing checks the physical thresholds, fps and finite trajectories.
    worlds, objects = p.shape[1:]
    timings = [[failure_timing(p[:, w, o:o+1], a[:, w, o:o+1], fps,
                              **timing_options) for o in range(objects)]
               for w in range(worlds)]
    required = max(1, int(np.ceil(worlds * minimum_fraction)))
    span = int(np.floor(cluster_seconds * fps + 1e-9))
    clusters = []
    for obj in range(objects):
        events = sorted((timings[w][obj]['failure_frame'], w) for w in range(worlds)
                        if timings[w][obj]['failed'])
        for start in sorted({frame for frame, _ in events}):
            members = [(frame, w) for frame, w in events if start <= frame <= start+span]
            clusters.append(dict(object_index=obj, start=start,
                                 end=max(frame for frame, _ in members), members=members))
    strong = [c for c in clusters if len(c['members']) >= required]
    chosen = (min(strong, key=lambda c: (c['start'], -len(c['members']), c['object_index']))
              if strong else min(clusters, key=lambda c: (-len(c['members']), c['start'], c['object_index']))
              if clusters else None)
    base = failure_timing(np.zeros((len(p), objects)), np.zeros((len(p), objects)),
                          fps, **timing_options)
    per_object = [[timings[w][o]['failure_seconds'] for o in range(objects)] for w in range(worlds)]
    base.update(detector='baseline_onset_consensus', baseline_worlds=worlds,
                consensus_minimum_fraction=float(minimum_fraction), consensus_required_worlds=required,
                consensus_cluster_seconds=float(cluster_seconds), consensus_reached=bool(strong),
                consensus_fraction=0., selected_object_index=None, selected_worlds=[],
                baseline_failure_fraction=sum(any(t['failed'] for t in row) for row in timings)/worlds,
                per_world_object_failure_seconds=per_object,
                per_world_object_onset_seconds=[[t['onset_seconds'] for t in row] for row in timings],
                selection_status='no_failure', cluster_start_seconds=None, cluster_end_seconds=None)
    if chosen is None:
        return base
    obj = chosen['object_index']
    members = [w for _, w in chosen['members']]
    t = int(np.floor(np.median([frame for frame, _ in chosen['members']])))
    s = min(t, int(np.floor(np.quantile([timings[w][obj]['onset_frame'] for w in members], .25))))
    base.update(failed=True, failure_frame=t, onset_frame=s,
                failure_seconds=t/fps, onset_seconds=s/fps,
                detected_seconds=(t+base['persistence_frames']-1)/fps,
                consensus_fraction=len(members)/worlds, selected_object_index=obj,
                selected_worlds=members, selection_status='consensus' if strong else 'weak_consensus',
                cluster_start_seconds=chosen['start']/fps, cluster_end_seconds=chosen['end']/fps,
                onset_rule='lower quartile of selected worlds last exit from sustained half-threshold region')
    base['per_object_failure_seconds'] = [
        float(np.median([row[o] for row in per_object if row[o] is not None]))
        if any(row[o] is not None for row in per_object) else None for o in range(objects)]
    return base


def perturb_initial_joints(initial, names, qids, lower, upper, worlds, seed,
                           *, translation_m=.001, angle_degrees=.5):
    """Independent bounded reset perturbations; reference and objects untouched."""
    initial = np.asarray(initial, dtype=np.float32)
    qids = np.asarray(qids, dtype=int)
    if worlds < 2 or len(names) != len(qids) or len(set(qids)) != len(qids):
        raise ValueError('multiple worlds and matching unique scalar joints required')
    if (not np.isfinite(initial).all() or translation_m <= 0 or angle_degrees <= 0
            or not np.isfinite([translation_m, angle_degrees]).all()):
        raise ValueError('finite initial pose and positive perturbation bounds required')
    rng = np.random.default_rng(seed)
    bounds = np.array([translation_m if name.endswith(('wrist_x', 'wrist_y', 'wrist_z'))
                       else np.deg2rad(angle_degrees) for name in names])
    q = np.broadcast_to(initial, (worlds, len(initial))).copy()
    q[:, qids] = np.clip(q[:, qids] + rng.uniform(-1., 1., (worlds, len(qids)))*bounds,
                        lower, upper)
    return q
