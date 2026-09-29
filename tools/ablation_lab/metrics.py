"""Diagnostic metrics, with explicit coordinate alignment and validity masks.

These are not challenge scores. No held-out data is used. Trajectory alignment
fits one rigid transform on all valid public frames; it does not fit scale.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rigid_align(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 3:
        raise ValueError("Need at least three corresponding points")
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ fix @ u.T
    return r, b - r @ a


def trajectory_metrics(predicted, truth, valid):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    valid = np.asarray(valid, bool) & np.isfinite(predicted).all((1, 2)) & np.isfinite(truth).all((1, 2))
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return {}, None
    p, q = predicted[ids, :3, 3], truth[ids, :3, 3]
    r, t = rigid_align(p, q)
    error = np.linalg.norm(p @ r.T + t - q, axis=1)
    # Pose-anchored alignment uses exactly the first visible object pose.
    # Unlike the best-fit translation metric it tests subsequent rotation too.
    first = ids[0]
    anchor = truth[first] @ np.linalg.inv(predicted[first])
    aligned = anchor @ predicted[ids]
    ae = np.linalg.norm(aligned[:, :3, 3] - truth[ids, :3, 3], axis=1)
    angle = Rotation.from_matrix(np.einsum('nji,njk->nik', truth[ids, :3, :3], aligned[:, :3, :3])).magnitude()
    spread = np.linalg.svd(p - p.mean(0), compute_uv=False)
    series = [None] * len(predicted)
    for i, e in zip(ids, error):
        series[int(i)] = round(float(e * 1000), 3)
    return {
        'gt_frames': len(ids), 'object_rigid_rmse_mm': float(np.sqrt(np.mean(error ** 2)) * 1000),
        'object_rigid_p95_mm': float(np.percentile(error, 95) * 1000),
        'object_anchor_rmse_mm': float(np.sqrt(np.mean(ae ** 2)) * 1000),
        'object_anchor_rotation_mean_deg': float(np.rad2deg(angle).mean()),
        'alignment_degenerate': bool(spread[1] < 1e-5),
        'object_error_mm': series,
    }, {'rotation': r.tolist(), 'translation': t.tolist()}


def hand_metrics(translations, valid, fps):
    translations, valid = np.asarray(translations), np.asarray(valid, bool)
    valid = valid & np.isfinite(translations).all(1)
    distances = np.linalg.norm(translations, axis=1)
    velocity = np.linalg.norm(np.diff(translations, axis=0), axis=1) * fps
    pairs = valid[1:] & valid[:-1]
    return {
        'valid_frames': int(valid.sum()), 'coverage': float(valid.mean()),
        'max_camera_distance_m': float(distances[valid].max()) if valid.any() else None,
        'far_frames': np.flatnonzero(valid & (distances > 1.1)).tolist(),
        'speed_p95_m_s': float(np.percentile(velocity[pairs], 95)) if pairs.any() else None,
        'camera_distance_m': [round(float(x), 4) if ok else None for x, ok in zip(distances, valid)],
    }


def bundle_checks(arrays, expected_frames):
    n = len(arrays['camera_to_world_transform'])
    problems = []
    if n != expected_frames:
        problems.append(f'Frame count {n} differs from source {expected_frames}')
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            problems.append(f'Non-finite values: {key}')
    for prefix in ['object', 'hand_left', 'hand_right']:
        middle = '_wrist' if prefix.startswith('hand') else ''
        expected = arrays['camera_to_world_transform'] @ arrays[f'{prefix}{middle}_to_camera_transform']
        valid = arrays.get(f'{prefix}_is_valid', np.ones(n, bool)).astype(bool)
        if valid.any():
            error = np.max(np.abs(expected[valid] - arrays[f'{prefix}{middle}_to_world_transform'][valid]))
            if error > 1e-4:
                problems.append(f'{prefix} camera/world inconsistency {error:.4g}')
    return problems
