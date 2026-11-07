"""Nominal-start, failure-local acceptance for reconstructing one demonstration.

The scoring interval is shared by every candidate in a search round. Editing
windows may precede it: a failed lift can require fixing the preceding grasp.
These are empirical acceptance rules, not a convergence certificate.
"""
import numpy as np
from flash_chord.retargeting.chord_upper import WrenchSupportAccumulator


def scoring_region(timing, frames, fps, *, lookahead_seconds=1.):
    if frames < 2 or fps <= 0 or not np.isfinite(lookahead_seconds) or lookahead_seconds <= 0:
        raise ValueError('invalid local scoring horizon')
    samples = frames - 1
    if timing['failed']:
        start = min(max(0, int(timing['onset_frame'])), samples-1)
        stop = min(samples, max(start+1, int(np.ceil(
            (timing['failure_seconds']+lookahead_seconds)*fps-1e-9))))
    else:
        start, stop = 0, samples
    return dict(start_frame=start, stop_frame_exclusive=stop,
                start_seconds=start/fps, end_seconds_exclusive=stop/fps,
                lookahead_seconds=float(lookahead_seconds),
                rule='error_onset_to_failure_plus_lookahead' if timing['failed'] else 'full_episode_no_failure')


def local_pose_metrics(position, orientation, region):
    p, a = np.asarray(position), np.asarray(orientation)
    lo, hi = region['start_frame'], region['stop_frame_exclusive']
    if p.ndim != 2 or a.shape != p.shape or not 0 <= lo < hi <= len(p):
        raise ValueError('local scoring interval does not match rollout')
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise FloatingPointError('nonfinite local object feedback')
    lp, la = p[lo:hi], a[lo:hi]
    return dict(local_start_frame=lo, local_stop_frame_exclusive=hi,
                local_object_position_robust_m=float(lp.mean()),
                local_object_orientation_robust_rad=float(la.mean()),
                local_object_success_robust_fraction=float(np.all((lp<.1)&(la<.5), axis=-1).mean()),
                local_per_object_position_mean_m=lp.mean(axis=0).tolist(),
                local_per_object_orientation_mean_rad=la.mean(axis=0).tolist(),
                tail_frames=len(p)-hi,
                tail_position_mean_m=(p[hi:].mean(axis=0) if hi<len(p) else np.zeros(p.shape[1])).tolist(),
                tail_orientation_mean_rad=(a[hi:].mean(axis=0) if hi<len(p) else np.zeros(a.shape[1])).tolist())


class LocalWrenchAccumulator:
    """Collect full-episode diagnostics and exactly the shared scoring interval."""
    def __init__(self, worlds, hands, objects, basis, *, region, tolerance=.1):
        self.full = WrenchSupportAccumulator(worlds, hands, objects, basis, tolerance=tolerance)
        self.local = WrenchSupportAccumulator(worlds, hands, objects, basis, tolerance=tolerance)
        self.region = dict(region)

    def add(self, current, reference):
        frame = self.full.frames
        self.full.add(current, reference)
        if self.region['start_frame'] <= frame < self.region['stop_frame_exclusive']:
            self.local.add(current, reference)

    def summaries(self):
        if self.local.frames != self.region['stop_frame_exclusive']-self.region['start_frame']:
            raise ValueError('incomplete local wrench interval')
        return [dict(full, **{'local_'+k:v for k,v in local.items()})
                for full, local in zip(self.full.summaries(), self.local.summaries())]


def local_guard_reasons(candidate, baseline, *, position_regression_m=.05,
                        orientation_regression_rad=np.deg2rad(20)):
    """Protect the preceding prefix; reject large per-object downstream regressions.

    No perturbed-reset gate, no first-failure-time veto, and no requirement that
    every wrench submetric improve separately. The local weighted cost decides
    improvement. Full and tail regressions tolerate one detection-threshold unit.
    """
    if (not np.isfinite(position_regression_m) or position_regression_m < 0 or
            not np.isfinite(orientation_regression_rad) or orientation_regression_rad < 0):
        raise ValueError('invalid regression tolerances')
    reasons = []
    for key in ('all_errors_finite', 'full_horizon'):
        if not candidate[key] or not baseline[key]:
            reasons.append(key)
    for key in ('local_start_frame', 'local_stop_frame_exclusive', 'protected_prefix_frames', 'tail_frames'):
        if candidate[key] != baseline[key]:
            reasons.append(key+'_mismatch')
    bounds = [('prefix_position_mean_m', .002), ('prefix_orientation_mean_rad', .01),
              ('prefix_position_p95_m', .005), ('prefix_orientation_p95_rad', .03),
              ('per_object_position_mean_m', position_regression_m),
              ('per_object_orientation_mean_rad', orientation_regression_rad),
              ('tail_position_mean_m', position_regression_m),
              ('tail_orientation_mean_rad', orientation_regression_rad),
              ('unstable_world_fraction', 0.)]
    for key, tolerance in bounds:
        after, before = np.asarray(candidate[key]), np.asarray(baseline[key])
        if (after.shape != before.shape or not np.isfinite(after).all() or
                not np.isfinite(before).all() or np.any(after > before+tolerance)):
            reasons.append(key)
    return reasons
