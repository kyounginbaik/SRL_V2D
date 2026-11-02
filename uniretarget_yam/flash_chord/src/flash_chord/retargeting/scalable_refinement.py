"""Physical-feedback phase diagnosis and independent reference acceptance.

No simulator imports. Poses use XYZW; rows are executed commands 0..F-2.
Retention is a kinematic observation of a moving grasp, not a force certificate.
"""
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_segment_window import aligned_pose_errors


@dataclass(frozen=True)
class FeedbackConfig:
    position_m: float = .03
    orientation_rad: float = np.pi / 6
    slip_m: float = .02
    slip_rad: float = np.pi / 12
    settle_s: float = .25
    persistence_s: float = .25
    retention_fraction: float = .8
    alpha: float = .05
    minimum_gain: float = .01

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError('finite feedback settings required')
        if min(self.position_m, self.orientation_rad, self.slip_m, self.slip_rad,
               self.settle_s, self.persistence_s) <= 0:
            raise ValueError('positive physical thresholds required')
        if not 0 < self.retention_fraction <= 1 or not 0 < self.alpha < .5 or self.minimum_gain < 0:
            raise ValueError('invalid retention fraction, significance or minimum gain')


def official_scores(poses, config=FeedbackConfig()):
    p, r = aligned_pose_errors(poses)
    finite = np.isfinite(p).all((0, 2)) & np.isfinite(r).all((0, 2))
    raw_pm, raw_rm = p.mean(0), r.mean(0)
    # Official success uses uncapped episode means. Cap only the final robust
    # cost summaries; clipping frames first could hide a brief large excursion.
    passed = finite & ((raw_pm < config.position_m) & (raw_rm < config.orientation_rad)).all(1)
    pm = np.minimum(np.nan_to_num(raw_pm, nan=1., posinf=1.), 1.)
    rm = np.minimum(np.nan_to_num(raw_rm, nan=np.pi, posinf=np.pi), np.pi)
    loss = np.max(pm / config.position_m + rm / config.orientation_rad, axis=1) + 2 * ~passed
    return dict(position=pm, orientation=rm, passed=passed, finite=finite, loss=loss)


def phase_feedback(poses, palms, hand_index, segment, segments, fps, config=FeedbackConfig(), *, reference_palms=None):
    """Measure carry retention after closure, and receiver retention after release.

    A stationary object on its support cannot pass lift-and-hold. Every world,
    including invalid/exploded worlds, contributes to the denominator.
    """
    from .contact_refinement import find_giver
    x, h = np.asarray(poses, float), np.asarray(palms, float)
    if (x.ndim != 4 or x.shape[-1] != 14 or h.ndim != 4 or h.shape[-1] != 7
            or h.shape[:2] != x.shape[:2] or not np.isfinite(fps) or fps <= 0):
        raise ValueError('matching object/palm rollout buffers and positive fps required')
    obj = x[:, :, segment.object_id, :7]
    palm = h[:, :, hand_index]
    n, worlds = obj.shape[:2]
    giver = find_giver(segment, segments, fps)
    a = min(n - 1, segment.start + max(1, round(config.settle_s * fps)))
    b = min(n, segment.end + 1)
    valid = (np.isfinite(obj).all(-1) & np.isfinite(palm).all(-1)
             & (np.linalg.norm(obj[..., 3:], axis=-1) > 1e-8)
             & (np.linalg.norm(palm[..., 3:], axis=-1) > 1e-8))
    safe_obj = np.where(valid[..., None], obj, [0, 0, 0, 0, 0, 0, 1])
    safe_palm = np.where(valid[..., None], palm, [0, 0, 0, 0, 0, 0, 1])
    ro = Rotation.from_quat(safe_obj[..., 3:].reshape(-1, 4))
    rh = Rotation.from_quat(safe_palm[..., 3:].reshape(-1, 4))
    relative_p = rh.inv().apply((safe_obj[..., :3] - safe_palm[..., :3]).reshape(-1, 3)).reshape(n, worlds, 3)
    relative_q = (rh.inv() * ro).as_quat().reshape(n, worlds, 4)
    if reference_palms is not None:
        planned = np.asarray(reference_palms, float)
        if planned.shape not in ((n, 7), (n, worlds, 7)) or not np.isfinite(planned).all():
            raise ValueError('one finite reference palm pose per executed command required')
        # Remove the intended in-hand motion. Otherwise a correct functional
        # grasp transition would be diagnosed as slip merely for rotating.
        if planned.ndim == 2:
            planned = np.broadcast_to(planned[:, None], (n, worlds, 7))
        rp = Rotation.from_quat(planned[..., 3:].reshape(-1, 4))
        target = x[..., segment.object_id, 7:14]
        safe_target = np.where(valid[..., None], target, [0, 0, 0, 0, 0, 0, 1])
        target_good = np.isfinite(safe_target).all(-1) & (np.linalg.norm(safe_target[..., 3:], axis=-1) > 1e-8)
        safe_target = np.where(target_good[..., None], safe_target, [0, 0, 0, 0, 0, 0, 1])
        valid &= target_good
        expected_p = rp.inv().apply((safe_target[..., :3] - planned[..., :3]).reshape(-1, 3)).reshape(n, worlds, 3)
        expected_r = rp.inv() * Rotation.from_quat(safe_target[..., 3:].reshape(-1, 4))
        relative_p -= expected_p
        relative_q = (Rotation.from_quat(relative_q.reshape(-1, 4)) * expected_r.inv()).as_quat().reshape(n, worlds, 4)
    anchor_q = np.broadcast_to(relative_q[a], relative_q.shape).reshape(-1, 4)
    drift_p = np.linalg.norm(relative_p - relative_p[a], axis=-1)
    drift_r = (Rotation.from_quat(relative_q.reshape(-1, 4))
               * Rotation.from_quat(anchor_q).inv()).magnitude().reshape(n, worlds)
    held = valid & valid[a] & (drift_p < config.slip_m) & (drift_r < config.slip_rad)
    start = max(a, giver.end + max(1, round(.15 * fps))) if giver else a
    p, r = aligned_pose_errors(x)
    run = max(1, round(config.persistence_s * fps))
    retained = np.zeros(worlds, bool)
    moving = np.zeros(worlds, bool)
    target_moves = np.zeros(worlds, bool)
    if b - start >= run:
        target = x[start:b, :, segment.object_id, 7:10]
        target_move = np.linalg.norm(target - x[a, :, segment.object_id, 7:10], axis=-1).max(0)
        actual_move = np.linalg.norm(obj[start:b, :, :3] - obj[a, :, :3], axis=-1).max(0)
        target_moves = target_move >= .01
        moving = target_moves & (actual_move >= .5 * target_move)
        # Require retention throughout the loaded interval, including its tail.
        retained = (held[start:b].mean(0) >= config.retention_fraction) & held[max(start, b-run):b].all(0)
        retained &= moving & valid[a:b].all(0)
    local_p, local_r = p[a:b, :, segment.object_id], r[a:b, :, segment.object_id]
    if b <= a:
        local_loss = np.full(worlds, 100.)
        persistent = True
    else:
        norm = np.maximum(local_p / config.position_m, local_r / config.orientation_rad)
        local_loss = np.minimum(np.nan_to_num(norm, nan=100., posinf=100.), 100.).mean(0)
        bad = (norm > 1).mean(1) >= .5
        persistent = len(bad) >= run and bool((np.convolve(bad.astype(int), np.ones(run), 'valid') >= run).any())
    fraction = float(retained.mean())
    # No required movement means retention is unproven; precision edits are
    # allowed for static holds only when their tracking already stays bounded.
    static = not target_moves.any() and not persistent and bool(valid[a:b].all())
    mode = ('handover' if giver else 'acquisition') if fraction < config.retention_fraction and not static else 'precision'
    passed = bool(not persistent and (fraction >= config.retention_fraction or static))
    return dict(segment=segment.key, mode=mode, passed=passed, retained=retained,
                retention_fraction=fraction, required_motion_fraction=float(moving.mean()),
                local_loss=local_loss, interval=[a, b], giver=giver.hand if giver else None,
                reason='tracking_error' if mode == 'precision' else 'lift_or_transfer_retention_failed')


def feedback_costs(poses, phase=None, config=FeedbackConfig()):
    result = official_scores(poses, config)['loss'].copy()
    if phase is not None:
        result += phase['local_loss']
        if phase['mode'] != 'precision':
            result += 3 * ~phase['retained']
    return result


def independent_gate(poses, costs, candidate_count, replicas, *, prefix_end=0,
                     retained=None, config=FeedbackConfig()):
    """A preselected shortlist is measured once on NEW physics rollouts.

    Bonferroni correct the shortlist, never fall back to discovery scores.
    Fresh baseline worlds protect each object, official success and prefix.
    """
    scores = official_scores(poses, config)
    costs = np.asarray(costs, float)
    n = candidate_count * replicas
    if (candidate_count < 1 or replicas < 2 or len(costs) != poses.shape[1]
            or len(costs) - n < 2 or not np.isfinite(costs).all()):
        raise ValueError('finite costs, candidate replicas and fresh baseline replicas required')
    c = costs[:n].reshape(candidate_count, replicas); b = costs[n:]
    se = np.sqrt(c.var(1, ddof=1) / replicas + b.var(ddof=1) / len(b))
    z = NormalDist().inv_cdf(1 - config.alpha / candidate_count)
    gain = b.mean() - c.mean(1)
    passed = (gain > np.maximum(config.minimum_gain, z * se))
    passed &= scores['finite'][:n].reshape(candidate_count, replicas).all(1)
    for key, tolerance in (('position', .002), ('orientation', .01)):
        v = scores[key]
        passed &= (v[:n].reshape(candidate_count, replicas, -1).mean(1) <= v[n:].mean(0) + tolerance).all(1)
    sr = scores['passed']
    passed &= sr[:n].reshape(candidate_count, replicas).mean(1) >= sr[n:].mean() - .01
    prefix_record = None
    if prefix_end > 0:
        from .contact_segment_window import prefix_gate
        ok, prefix_record = prefix_gate(poses, candidate_count, replicas, prefix_end)
        passed &= ok
    retention = None
    if retained is not None:
        retained = np.asarray(retained, bool)
        if retained.shape != costs.shape:
            raise ValueError('one retention result per validation world required')
        retention = retained[:n].reshape(candidate_count, replicas).mean(1)
        passed &= retention >= retained[n:].mean() - .01
    return passed, dict(independent=True, candidates=candidate_count, replicas=replicas,
        baseline_replicas=len(b), significance_z=z, improvement=gain.tolist(), standard_error=se.tolist(),
        candidate_cost=c.mean(1).tolist(), baseline_cost=float(b.mean()),
        candidate_mp_sr=sr[:n].reshape(candidate_count, replicas).mean(1).tolist(),
        baseline_mp_sr=float(sr[n:].mean()), retention=retention.tolist() if retention is not None else None,
        prefix=prefix_record, passed=passed.tolist())
