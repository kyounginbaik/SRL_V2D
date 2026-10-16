"""Unknown hand targets must not trigger tracking resets; object guards remain active."""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout
from flash_chord.lifecycle.termination import (
    ThresholdTerminationTermConfig, TrackingTermination, TrackingTerminationConfig,
)
from test_termination import _hand, _robot_reference


def fixture():
    layout = EmbodimentLayout(num_joint_q=0, num_joint_dof=0,
        hands=(_hand('left', BodyFrame('left_palm', 0)),
               _hand('right', BodyFrame('right_palm', 1))))
    binding = _robot_reference(layout, np.zeros((2, 2, 3), np.float32),
        np.tile([0., 0., 0., 1.], (2, 2, 1)))
    scene = SimpleNamespace(layout=layout, model=SimpleNamespace(body_count=12),
        robot_reference=binding, objects=[object()], world_count=4)
    with wp.ScopedDevice('cpu'):
        term = TrackingTermination.build(scene,
            config=TrackingTerminationConfig(
                wrist_position=ThresholdTerminationTermConfig(threshold=.1),
                wrist_orientation=ThresholdTerminationTermConfig(threshold=.1),
                object_position=ThresholdTerminationTermConfig(threshold=.1)),
            object_body_ids_w=wp.array(np.arange(4)*3+2, dtype=wp.int32),
            object_ref_pos_w=wp.zeros(4, dtype=wp.vec3),
            object_ref_quat_w=wp.array(np.tile([0., 0., 0., 1.], (4, 1)), dtype=wp.quat),
            num_objects=1)
        bodies = []
        for world in range(4):
            bodies.extend([
                wp.transform(wp.vec3(1., 0., 0.), wp.quat(0., 0., np.sin(.3), np.cos(.3))),
                wp.transform(wp.vec3(float(world == 2), 0., 0.), wp.quat_identity()),
                wp.transform(wp.vec3(float(world == 3), 0., 0.), wp.quat_identity()),
            ])
        body_q = wp.array(bodies, dtype=wp.transform)
        frames = wp.array([0, 1, 0, 0], dtype=wp.int32)
    return term, binding, body_q, frames


def test_invalid_hand_errors_do_not_reset_but_valid_hands_objects_and_horizon_still_do():
    term, binding, bodies, frames = fixture()
    term.evaluate(bodies, frames)
    np.testing.assert_array_equal(term.buffers.wrist_done.numpy(), [1, 1, 1, 1])
    original_targets = term.reference.wrist_pos_w.numpy().copy()
    validity = {'left': np.array([False, True]), 'right': np.array([True, True])}
    term.bind_reference_validity(SimpleNamespace(num_frames=2, hand_valid=validity.__getitem__))
    term.evaluate(bodies, frames)
    np.testing.assert_array_equal(term.buffers.wrist_done.numpy(), [0, 1, 1, 0])
    np.testing.assert_array_equal(term.buffers.object_done.numpy(), [0, 0, 0, 1])
    np.testing.assert_array_equal(term.terminated.numpy(), [0, 1, 1, 1])
    np.testing.assert_array_equal(term.truncated.numpy(), [0, 1, 0, 0])
    np.testing.assert_allclose(term.buffers.wrist_pos_err.numpy().reshape(4, 2),
                               [[0., 0.], [1., 0.], [0., 1.], [0., 0.]])
    np.testing.assert_allclose(term.buffers.wrist_ori_err.numpy().reshape(4, 2)[[0, 2, 3], 0], 0.)
    np.testing.assert_allclose(term.buffers.wrist_ori_err.numpy().reshape(4, 2)[1, 0], .6, atol=1e-6)
    np.testing.assert_array_equal(term.reference.wrist_pos_w.numpy(), original_targets)
    np.testing.assert_array_equal(binding.hand('left').wrist_pos_w, np.zeros((2, 3)))


def test_missing_validity_restores_legacy_all_valid_tracking_and_mismatched_frames_fail():
    term, _, bodies, frames = fixture()
    term.bind_reference_validity(SimpleNamespace(num_frames=2,
        hand_valid=lambda side: np.zeros(2, bool)))
    term.evaluate(bodies, frames)
    np.testing.assert_array_equal(term.buffers.wrist_done.numpy(), [0, 0, 0, 0])
    term.bind_reference_validity(SimpleNamespace(num_frames=2))
    term.evaluate(bodies, frames)
    np.testing.assert_array_equal(term.buffers.wrist_done.numpy(), [1, 1, 1, 1])
    with pytest.raises(ValueError, match='shape'):
        term.bind_reference_validity(SimpleNamespace(num_frames=3))
