"""Contact-segment windows and prefix protection, independent of the simulator.

Error arrays have one row per executed command (row zero = reference frame zero).
An isolated threshold crossing must not starve a later sustained failure.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def contact_segments(contacts, gap_frames=11):
    out = []
    for side, active in contacts.items():
        idx = np.flatnonzero(np.asarray(active, bool).reshape(len(active), -1).any(1))
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > gap_frames) + 1)
        for number, group in enumerate(g for g in groups if len(g)):
            out.append(dict(key=f'{side}_seg{number}', start=int(group[0]), end=int(group[-1])))
    return sorted(out, key=lambda d: (d['start'], d['key']))


def select_segment_window(err_pos, err_ori, contacts, frames, fps, *, focus=(),
                          position_m=.03, orientation_rad=np.pi / 6,
                          persistence_s=.25, lead_s=.5, release_s=.5):
    """Choose the contact segment with the largest persistent normalized deficit.

    The whole receiving segment, approach and release can change. Errors must
    persist for persistence_s; ranking uses the mean deficit, not its first peak.
    Returning None means no eligible failure, never unrestricted editing.
    """
    p, r = np.asarray(err_pos, float), np.asarray(err_ori, float)
    if p.shape != (frames - 1,) or r.shape != p.shape:
        raise ValueError('error rows must correspond to reference commands 0..F-2')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or fps <= 0:
        raise ValueError('finite errors and positive fps required')
    segments = contact_segments(contacts)
    wanted = set(focus)
    if wanted - {s['key'] for s in segments}:
        raise ValueError(f'unknown contact segments: {sorted(wanted - {s["key"] for s in segments})}')
    run = max(1, int(round(persistence_s * fps)))
    deficit = np.maximum(p / position_m, r / orientation_rad) - 1
    candidates = []
    for seg in segments:
        if wanted and seg['key'] not in wanted:
            continue
        a = max(1, seg['start'] - int(round(lead_s * fps)))
        b = min(frames - 2, seg['end'] + int(round(release_s * fps)))
        values = deficit[a:b + 1]
        if len(values) < run:
            continue
        persistent = np.convolve((values > 0).astype(int), np.ones(run, int), 'valid') >= run
        if persistent.any():
            score = float(np.maximum(values, 0).mean())
            candidates.append(dict(**seg, window=[a, b], score=score,
                                   first_persistent_fail=a + int(np.flatnonzero(persistent)[0])))
    if not candidates:
        return None
    best = max(candidates, key=lambda d: d['score'])
    a, b = best['window']
    t = np.arange(frames, dtype=float)
    ramp = max(1., .25 * fps)
    # Approach/release padding is part of the editable window. Prefix < a and
    # suffix > b are bitwise protected, including the published endpoints.
    mask = np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0., 1.)
    mask = .5 - .5 * np.cos(np.pi * mask)
    mask[0] = mask[-1] = 0
    return mask, dict(best, detect='contact_segments', candidates=candidates)


def aligned_pose_errors(poses):
    """Official first-pose alignment; bad poses remain failures, not identity."""
    x = np.asarray(poses, float)
    if x.ndim != 4 or x.shape[-1] != 14:
        raise ValueError('expected [stepped frame, world, object, 14]')
    finite = np.isfinite(x).all(-1)
    aq, tq = x[..., 3:7], x[..., 10:14]
    good = finite & (np.linalg.norm(aq, axis=-1) > 1e-8) & (np.linalg.norm(tq, axis=-1) > 1e-8)
    aq = np.where(good[..., None], aq, [0, 0, 0, 1])
    tq = np.where(good[..., None], tq, [0, 0, 0, 1])
    p = np.empty(x.shape[:-1]); r = np.empty_like(p)
    for w in range(x.shape[1]):
        rotation = Rotation.from_quat(tq[0, w, 0]) * Rotation.from_quat(aq[0, w, 0]).inv()
        offset = x[0, w, 0, 7:10] - rotation.apply(x[0, w, 0, :3])
        for o in range(x.shape[2]):
            p[:, w, o] = np.linalg.norm(rotation.apply(x[:, w, o, :3]) + offset - x[:, w, o, 7:10], axis=-1)
            r[:, w, o] = (rotation * Rotation.from_quat(aq[:, w, o]) * Rotation.from_quat(tq[:, w, o]).inv()).magnitude()
        good[:, w] &= bool(good[0, w, 0])
    return np.where(good, p, np.inf), np.where(good, r, np.inf)


def prefix_gate(poses, candidate_count, replicas, prefix_end_frame, *, position_tolerance=.002,
                orientation_tolerance=.01):
    """Per-object prefix gate using the same batch's untouched baseline worlds."""
    x = np.asarray(poses)
    n = candidate_count * replicas
    if n >= x.shape[1] or candidate_count <= 0 or replicas < 2:
        raise ValueError('candidate replicas and independent baseline worlds required')
    if prefix_end_frame <= 0:
        return np.ones(candidate_count, bool), dict(frames=0)
    # Row i executes reference command i; protect frames strictly before a.
    p, r = aligned_pose_errors(x[:prefix_end_frame])
    p, r = p.mean(0), r.mean(0)
    bp, br = p[n:].mean(0), r[n:].mean(0)
    cp = p[:n].reshape(candidate_count, replicas, -1).mean(1)
    cr = r[:n].reshape(candidate_count, replicas, -1).mean(1)
    ok = (np.isfinite(cp).all(1) & np.isfinite(cr).all(1)
          & (cp <= bp + position_tolerance).all(1)
          & (cr <= br + orientation_tolerance).all(1))
    return ok, dict(frames=prefix_end_frame, baseline_position_m=bp.tolist(),
                    baseline_orientation_rad=br.tolist(), candidate_position_m=cp.tolist(),
                    candidate_orientation_rad=cr.tolist(), passed=ok.tolist())


def task_success_gate(poses, candidate_count, replicas, *, position_m=.03,
                      orientation_rad=np.pi / 6, position_tolerance=.002,
                      orientation_tolerance=.01, success_tolerance=.01):
    """Protect strict task success and each object's episode error separately.

    In particular, a passing single-object episode cannot be traded for an
    average improvement on another object or a looser per-frame success score.
    """
    p, r = aligned_pose_errors(poses)
    n = candidate_count * replicas
    if not 0 < n < p.shape[1]:
        raise ValueError('independent baseline worlds are required')
    pp, rr = p.mean(0), r.mean(0)
    bp, br = pp[n:].mean(0), rr[n:].mean(0)
    passed = ((pp < position_m) & (rr < orientation_rad)).all(1)
    baseline_success = float(passed[n:].mean())
    candidate_success = passed[:n].reshape(candidate_count, replicas).mean(1)
    cp = pp[:n].reshape(candidate_count, replicas, -1).mean(1)
    cr = rr[:n].reshape(candidate_count, replicas, -1).mean(1)
    ok = (np.isfinite(cp).all(1) & np.isfinite(cr).all(1)
          & (cp <= bp + position_tolerance).all(1)
          & (cr <= br + orientation_tolerance).all(1)
          & (candidate_success >= baseline_success - success_tolerance))
    return ok, dict(baseline_mp_sr=baseline_success, candidate_mp_sr=candidate_success.tolist(),
                    baseline_position_m=bp.tolist(), candidate_position_m=cp.tolist(),
                    baseline_orientation_rad=br.tolist(), candidate_orientation_rad=cr.tolist(), passed=ok.tolist())


def task_frontier_gate(poses, candidate_count, replicas, *, position_m=.03,
                       orientation_rad=np.pi / 6, normalized_tolerance=.05, success_tolerance=.01,
                       anchor_normalized=None, anchor_tolerance=.1):
    """Permit bounded tradeoffs inside passing components of a failing task.

    Preserve position/orientation pass rates separately for every object, each
    object's joint success rate, and complete task success. A contact-progress
    gate and uncertainty test must additionally pass before accepting an edit.
    This avoids freezing a failed grasp merely because orientation changes a
    little while remaining safely inside its task threshold.
    """
    p, r = aligned_pose_errors(poses)
    n = candidate_count * replicas
    if not 0 < n < p.shape[1] or replicas < 2 or p.shape[1] - n < 2:
        raise ValueError('replicated candidates and independent baseline worlds required')
    pp, rr = p.mean(0), r.mean(0)
    normalized = np.maximum(pp / position_m, rr / orientation_rad)
    components = np.stack([pp < position_m, rr < orientation_rad], axis=-1)
    passed_objects = components.all(-1)
    passed_task = passed_objects.all(1)
    cc = components[:n].reshape(candidate_count, replicas, *components.shape[1:]).mean(1)
    bc = components[n:].mean(0)
    co = passed_objects[:n].reshape(candidate_count, replicas, -1).mean(1)
    bo = passed_objects[n:].mean(0)
    ct = passed_task[:n].reshape(candidate_count, replicas).mean(1)
    bt = float(passed_task[n:].mean())
    cn = normalized[:n].reshape(candidate_count, replicas, -1).mean(1)
    bn = normalized[n:].mean(0)
    ok = (np.isfinite(cn).all(1) & (cn <= bn + normalized_tolerance).all(1)
          & (cc >= bc - success_tolerance).all((1, 2))
          & (co >= bo - success_tolerance).all(1) & (ct >= bt - success_tolerance))
    if anchor_normalized is not None:
        anchor = np.asarray(anchor_normalized, float)
        if anchor.shape != bn.shape or np.isnan(anchor).any():
            raise ValueError('one initial normalized error bound per object required')
        ok &= (cn <= anchor + anchor_tolerance).all(1)
    return ok, dict(baseline_mp_sr=bt, candidate_mp_sr=ct.tolist(),
        baseline_normalized_error=bn.tolist(), candidate_normalized_error=cn.tolist(),
        baseline_component_success=bc.tolist(), candidate_component_success=cc.tolist(),
        baseline_object_success=bo.tolist(), candidate_object_success=co.tolist(),
        passed=ok.tolist(), mode='task_frontier', normalized_tolerance=normalized_tolerance,
        anchor_normalized=list(anchor_normalized) if anchor_normalized is not None else None,
        anchor_tolerance=anchor_tolerance)
