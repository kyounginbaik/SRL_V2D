# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np


class _MeshReference:
    def __init__(self, mesh_path):
        self._mesh_path = str(mesh_path)
        self.metadata = SimpleNamespace(source_path=str(mesh_path))

    def object_body_names(self):
        return ["box"]

    def object_mesh_paths(self):
        return [self._mesh_path]


def test_mesh_surface_keypoints_are_fixed_deterministic_surface_samples(tmp_path):
    import trimesh

    from flash_chord.objectives.keypoints import object_local_keypoints

    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.2, 0.1, 0.08)).export(mesh_path)
    reference = _MeshReference(mesh_path)

    points_a, count_a = object_local_keypoints(reference, mode="mesh_surface", count=500, seed=19)
    points_b, count_b = object_local_keypoints(reference, mode="mesh_surface", count=500, seed=19)

    assert count_a == count_b == 500
    assert points_a.shape == (500, 3)
    np.testing.assert_array_equal(points_a, points_b)
    assert np.all(np.abs(points_a) <= np.array([0.1, 0.05, 0.04]) + 1.0e-6)


def test_axis6_mode_preserves_original_unit_offsets():
    from flash_chord.objectives.keypoints import KEYPOINT_VECS_NP, object_local_keypoints

    points, count = object_local_keypoints(_MeshReference("unused.obj"), mode="axis6", count=500, seed=0)

    assert count == 6
    np.testing.assert_array_equal(points, KEYPOINT_VECS_NP)


def test_mesh_add_shaping_uses_mean_point_distance_before_shaping():
    import warp as wp

    from flash_chord.objectives.keypoints import object_keypoints_objective

    points = wp.array(np.array([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cpu")
    body_ids = wp.array([0], dtype=wp.int32, device="cpu")
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu")
    target_pos = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu")
    target_quat = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device="cpu")
    score = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        object_keypoints_objective,
        dim=1,
        inputs=[body_q, body_ids, target_pos, target_quat, points, 1, 2, 1, 0.1, 0, 1],
        outputs=[score],
        device="cpu",
    )
    wp.synchronize_device("cpu")

    np.testing.assert_allclose(score.numpy(), [1.0], atol=1.0e-6)
