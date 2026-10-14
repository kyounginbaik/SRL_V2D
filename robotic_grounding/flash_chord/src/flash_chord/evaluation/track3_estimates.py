"""Track 3 checkpoint estimates against an immutable reconstructed reference.

These are NOT held-out leaderboard scores. The local reconstructed mesh also
stands in for the confidential scan: CD-O is therefore a mesh self-comparison.
Pose arrays use metres and xyzw quaternions, shape (time, world, object, 7).
Definitions checked against the official submission kit dated 2026-10-06:
https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _rotations(poses):
    return Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix().reshape(*poses.shape[:-1], 3, 3)


def estimate_track3(achieved, reference, vertices, *, axes=None, orders=None):
    """Five official-style pose metrics; mesh coordinates are already centred.

    Mesh anchors/axes/surface samples are local proxies, since host geometry is
    confidential. One rigid initial-scene alignment is shared by all objects.
    The optional axes and orders implement half-turn lid and axial cup symmetry.
    """
    a, r = np.asarray(achieved, float).copy(), np.asarray(reference, float).copy()
    if r.ndim == 3:
        r = r[:, None]
    if a.ndim != 4 or a.shape[-1] != 7 or r.shape != (a.shape[0], 1, a.shape[2], 7):
        raise ValueError('Expected achieved (T,W,O,7) and reference (T,O,7) or (T,1,O,7)')
    if not np.isfinite(a).all() or not np.isfinite(r).all() or min(a.shape[:3]) == 0:
        raise ValueError('Non-finite or empty pose sequence')
    for pose in (a, r):
        norms = np.linalg.norm(pose[..., 3:], axis=-1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError('Zero quaternion')
        pose[..., 3:] /= norms
    bodies = a.shape[2]
    axes = np.zeros((bodies, 3)) if axes is None else np.asarray(axes, float)
    orders = np.ones(bodies, int) if orders is None else np.asarray(orders, int)
    if axes.shape != (bodies, 3) or orders.shape != (bodies,) or len(vertices) != bodies:
        raise ValueError('Geometry must match object count')
    for i in range(bodies):
        if orders[i] > 1:
            if np.linalg.norm(axes[i]) < 1e-8:
                raise ValueError('Symmetric object requires a nonzero axis')
            axes[i] = axes[i] / np.linalg.norm(axes[i])
    ra, rr = _rotations(a), _rotations(r)
    fixed = np.flatnonzero(orders == 1)
    anchor = int(fixed[0]) if len(fixed) else 0
    for w in range(a.shape[1]):
        if len(fixed) or bodies == 1:
            align = rr[0, 0, anchor] @ ra[0, w, anchor].T
        else:
            def basis(axis, direction):
                x = direction - axis * np.dot(direction, axis)
                if np.linalg.norm(x) < 1e-8:
                    x = np.eye(3)[np.argmin(abs(axis))]
                    x -= axis * np.dot(x, axis)
                x /= np.linalg.norm(x)
                return np.column_stack((x, np.cross(axis, x), axis))
            align = basis(rr[0, 0, 0] @ axes[0], r[0, 0, 1, :3]-r[0, 0, 0, :3]) @ basis(
                ra[0, w, 0] @ axes[0], a[0, w, 1, :3]-a[0, w, 0, :3]).T
        offset = r[0, 0, 0, :3] - align @ a[0, w, 0, :3]
        a[:, w, :, :3] = a[:, w, :, :3] @ align.T + offset
        ra[:, w] = align @ ra[:, w]
    distance = np.linalg.norm(a[..., :3]-r[..., :3], axis=-1).mean(axis=0)
    orientation, adds, key_errors = [], [], []
    symmetries = []
    keys = np.concatenate((np.eye(3), -np.eye(3))) * .05
    for body in range(bodies):
        sym = (Rotation.from_rotvec(np.arange(orders[body])[:, None] *
               (2*np.pi/orders[body]) * axes[body]).as_matrix()
               if orders[body] > 1 else np.eye(3)[None])
        symmetries.append(sym)
        angular = np.full(a.shape[:2], np.inf)
        for s in sym:
            relative = np.swapaxes(ra[:, :, body], -1, -2) @ rr[:, :, body] @ s
            angle = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(a.shape[:2])
            angular = np.minimum(angular, angle)
        orientation.append(angular.mean(axis=0))
        def point_error(points):
            points = np.asarray(points, float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Invalid object sample')
            result = np.empty(a.shape[:2])
            for start in range(0, len(a), 32):
                sl = slice(start, start+32)
                moved = np.einsum('twij,vj->twvi', ra[sl, :, body], points) + a[sl, :, body, None, :3]
                best = np.full(moved.shape[:2], np.inf)
                for s in sym:
                    target = np.einsum('twij,vj->twvi', rr[sl, :, body] @ s, points) + r[sl, :, body, None, :3]
                    best = np.minimum(best, np.linalg.norm(moved-target, axis=-1).mean(axis=-1))
                result[sl] = best
            return result
        adds.append(point_error(vertices[body]))
        key_errors.append(point_error(keys))
    orientation = np.stack(orientation, axis=-1)
    thresholds = np.arange(.01, .10, .01)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    auc = np.mean([trapezoid([np.mean(add < t) for t in thresholds], x=np.linspace(0, 1, 9)) for add in adds])
    relative_errors = []
    for first in range(bodies):
        for second in range(first+1, bodies):
            local_a = np.einsum('twji,twj->twi', ra[:, :, first], a[:, :, second, :3]-a[:, :, first, :3])
            local_r = np.einsum('twji,twj->twi', rr[:, :, first], r[:, :, second, :3]-r[:, :, first, :3])
            relative_errors.append(np.minimum.reduce([np.linalg.norm(local_a-local_r @ s, axis=-1) for s in symmetries[first]]))
    return dict(auc=float(auc), sp_sr=float(np.mean((distance.mean(-1) <= .1) & (orientation.mean(-1) <= .5))),
                mp_sr=float(np.mean(np.all((distance < .03) & (orientation < np.pi/6), axis=-1))),
                mppe_cm=float(np.mean(key_errors)*100), rpe_cm=float(np.mean(relative_errors)*100) if relative_errors else 0.)


def score_recording(recording, parquet, output):
    """Score a saved checkpoint rollout, using frozen poses and visual meshes."""
    import pyarrow.parquet as pq
    import trimesh
    row = pq.read_table(parquet).to_pylist()[0]
    data = np.load(recording)
    frames = data['frame_indices'].astype(int)
    if not np.array_equal(frames, np.arange(len(row['object_body_position']))):
        raise ValueError('Metric recording must cover every reference frame exactly once')
    achieved = data['achieved_pose_w'].astype(float)
    reference = np.concatenate((row['object_body_position'], np.asarray(row['object_body_wxyz'])[..., [1,2,3,0]]), axis=-1)
    points, axes, orders = [], [], []
    for body, (name, filename) in enumerate(zip(row['object_body_names'], row['object_mesh_paths'], strict=True)):
        path = Path(filename)
        if not path.is_absolute():
            path = Path(parquet).parent / path
        mesh = trimesh.load(path, force='mesh', process=False)
        center = np.asarray(mesh.centroid)
        samples = np.asarray(mesh.vertices)[np.random.default_rng(0).choice(len(mesh.vertices), 500, replace=len(mesh.vertices)<500)] - center
        points.append(samples)
        for poses in (achieved, reference):
            rotation = _rotations(poses[..., body, :])
            poses[..., body, :3] += np.einsum('...ij,j->...i', rotation, center)
        # Public symmetry declaration; the confidential scan's axis is unavailable.
        # Lid normal uses the least-variance surface axis of our proxy mesh.
        axis, order = np.zeros(3), 1
        if name == 'white_pot_lid':
            surface, _ = trimesh.sample.sample_surface(mesh, 10000, seed=0)
            axis = np.linalg.eigh(np.cov(surface.T))[1][:, 0]; order = 2
        elif name in ('beige_cup', 'blue_cup'):
            raise ValueError('Supply a verified cup symmetry axis before evaluating cup episodes')
        axes.append(axis); orders.append(order)
    metrics = estimate_track3(achieved, reference, points, axes=axes, orders=orders)
    # Same mesh on both sides means exact surface distance zero, before/after Sim(3).
    metrics['cd_o_cm'] = 0.0
    report = dict(metrics=metrics, estimated=True, ground_truth='immutable reconstructed reference',
                  cd_o_basis='identical reference/submitted mesh: self-comparison, not scan accuracy',
                  mesh_registration='identity; same proxy mesh and body coordinates on both sides',
                  geometry_anchor='surface centroid of proxy mesh', symmetry_axes=np.asarray(axes).tolist(),
                  symmetry_orders=orders, symmetry_axis_source='proxy mesh PCA for lid',
                  frames=len(frames), worlds=achieved.shape[1], rpe_applicable=len(points)>1,
                  voc_scale=float(data['voc_scale']), reference_parquet=str(parquet))
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report
