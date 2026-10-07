# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keypoint tracking objectives (motion imitation).

Object keypoints = the 6 object-frame axis unit vectors (±X,±Y,±Z) transformed to world by the object
pose; hand keypoints = the wrist + fingertip bodies. Each keypoint's squared error feeds the shared
:func:`~flash_chord.objectives.shaping.shaped_objective`; the per-world objective is the mean over keypoints.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from pathlib import Path

from flash_chord.assets.registry import resolve_asset_path

from flash_chord.objectives.shaping import shaped_objective

# 6 principal axis unit vectors in the object frame.
KEYPOINT_VECS_NP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)


def object_local_keypoints(reference, *, mode: str, count: int, seed: int) -> tuple[np.ndarray, int]:
    """Return deterministic per-body object-frame points for pose tracking."""
    body_count = (
        len(reference.object_body_names())
        if hasattr(reference, "object_body_names")
        else int(np.asarray(reference.object_body_pos_w()).shape[1])
    )
    if mode == "axis6":
        return np.tile(KEYPOINT_VECS_NP, (body_count, 1)).astype(np.float32), 6
    if mode != "mesh_surface":
        raise ValueError(f"unsupported object keypoint mode: {mode!r}")

    import trimesh

    paths = reference.object_mesh_paths()
    if len(paths) != body_count:
        raise ValueError(f"reference has {len(paths)} object meshes for {body_count} bodies")
    sampled = []
    for body_index, raw_path in enumerate(paths):
        mesh_path = Path(resolve_asset_path(raw_path, reference_path=reference.metadata.source_path))
        if not mesh_path.is_file():
            raise FileNotFoundError(f"object mesh for ADD-style keypoints not found: {mesh_path}")
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"object mesh must be triangular: {mesh_path}")
        triangles = vertices[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        valid = areas > 1.0e-12
        triangles, areas = triangles[valid], areas[valid]
        if len(triangles) == 0 or not np.isfinite(areas).all():
            raise ValueError(f"object mesh has no valid surface triangles: {mesh_path}")
        rng = np.random.default_rng(seed + body_index)
        face_ids = rng.choice(len(triangles), size=count, replace=True, p=areas / areas.sum())
        chosen = triangles[face_ids]
        uv = rng.random((count, 2))
        root = np.sqrt(uv[:, 0])
        bary = np.column_stack((1.0 - root, root * (1.0 - uv[:, 1]), root * uv[:, 1]))
        sampled.append(np.einsum("ni,nij->nj", bary, chosen).astype(np.float32))
    return np.concatenate(sampled, axis=0), count


@wp.kernel
def object_keypoints_objective(
    body_q: wp.array(dtype=wp.transform),
    object_body_ids: wp.array(dtype=wp.int32),  # [n_obj] per-world object body ids
    target_pos: wp.array(dtype=wp.vec3),  # [world_count * n_obj] command object positions
    target_quat: wp.array(dtype=wp.quat),  # [world_count * n_obj] command object quats (xyzw)
    keypoint_vecs: wp.array(dtype=wp.vec3),  # [n_obj * n_points] object-frame points
    n_obj: int,
    n_points: int,
    bodies_per_world: int,
    var: float,
    kind: int,
    aggregate_add: int,
    objective_sum: wp.array(dtype=wp.float32),  # [world_count] atomic accumulator; caller divides by n_obj*6
) -> None:
    """Mean shaped point reward per object, accumulated per world."""
    tid = wp.tid()
    body_local = tid % n_obj
    world = tid // n_obj
    xf = body_q[world * bodies_per_world + object_body_ids[body_local]]
    idx = world * n_obj + body_local
    sim_pos = wp.transform_get_translation(xf)
    sim_rot = wp.transform_get_rotation(xf)
    cmd_pos = target_pos[idx]
    cmd_rot = target_quat[idx]
    point_reward_sum = float(0.0)
    add_distance_sum = float(0.0)
    for kp in range(n_points):
        v = keypoint_vecs[body_local * n_points + kp]
        kp_sim = sim_pos + wp.quat_rotate(sim_rot, v)
        kp_cmd = cmd_pos + wp.quat_rotate(cmd_rot, v)
        d = kp_sim - kp_cmd
        distance_squared = wp.dot(d, d)
        if aggregate_add != 0:
            add_distance_sum += wp.sqrt(distance_squared)
        else:
            point_reward_sum += shaped_objective(distance_squared, var, kind, 0.0)
    if aggregate_add != 0:
        mean_add = add_distance_sum / float(n_points)
        point_reward_sum = shaped_objective(mean_add * mean_add, var, kind, 0.0)
    else:
        point_reward_sum /= float(n_points)
    wp.atomic_add(objective_sum, world, point_reward_sum)


@wp.kernel
def hand_keypoints_objective(
    body_q: wp.array(dtype=wp.transform),
    keypoint_body_ids: wp.array(dtype=wp.int32),  # [n_hands * n_kp] retained per-world bodies
    keypoint_local_pos: wp.array(dtype=wp.vec3),  # [n_hands * n_kp] body-frame semantic offsets
    target_pos: wp.array(dtype=wp.vec3),  # [world_count * n_hands * n_kp] command keypoint positions
    n_kp: int,  # keypoints per hand (wrist + fingertips)
    n_hands: int,
    bodies_per_world: int,
    var: float,
    kind: int,
    threshold: float,
    objective_sum: wp.array(dtype=wp.float32),  # [world_count * n_hands] atomic accumulator; caller divides by n_kp
) -> None:
    """Accumulate semantic-frame position tracking per hand; launch over W*n_hands*n_kp."""
    tid = wp.tid()
    kp = tid % n_kp
    hand = (tid // n_kp) % n_hands
    world = tid // (n_kp * n_hands)
    keypoint = hand * n_kp + kp
    body = body_q[world * bodies_per_world + keypoint_body_ids[keypoint]]
    sim = wp.transform_get_translation(body) + wp.quat_rotate(
        wp.transform_get_rotation(body),
        keypoint_local_pos[keypoint],
    )
    d = sim - target_pos[tid]
    wp.atomic_add(objective_sum, world * n_hands + hand, shaped_objective(wp.dot(d, d), var, kind, threshold))
