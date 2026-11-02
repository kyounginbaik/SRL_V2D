"""Measured grasp conditioning and exact residual-input convention checks."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.grasp_feedback import (
    capture_object_relative_grasp, conditioned_hand_target, encode_filtered_residual, hand_target_residual,
    rebase_grasp_motion,
    handover_hand_target,
    correct_planned_hand_target,
    decode_processed_hand_target,
)


class GraspFeedbackTests(unittest.TestCase):
    def test_processed_quaternion_order_does_not_invent_a_rotation_bias(self):
        wrist = np.r_[[.1, -.2, .3], Rotation.from_euler('xyz', [.3, -.5, 1.2]).as_quat()]
        processed = np.r_[wrist[:3], wrist[[6, 3, 4, 5]], [.4, 1.7]]
        decoded = decode_processed_hand_target(processed, finger_count=2)
        np.testing.assert_array_equal(decoded.wrist_pose_xyzw, wrist)
        np.testing.assert_array_equal(decoded.fingers, [.4, 1.7])
        self.assertLess((Rotation.from_quat(decoded.wrist_pose_xyzw[3:])
                        * Rotation.from_quat(wrist[3:]).inv()).magnitude(), 1e-14)

    def test_processed_batch_preserves_dtype_and_overdrive_without_aliases(self):
        values = np.array([[0., 0., 0., 1., 0., 0., 0., -.3, 1.8]], dtype=np.float32)
        decoded = decode_processed_hand_target(values, finger_count=2)
        self.assertEqual(decoded.wrist_pose_xyzw.dtype, np.float32)
        self.assertEqual(decoded.fingers.dtype, np.float32)
        np.testing.assert_array_equal(decoded.fingers, values[:, 7:])
        decoded.fingers[0, 0] = 0.
        decoded.wrist_pose_xyzw[0, 0] = 1.
        self.assertEqual(values[0, 0], 0.)
        self.assertEqual(values[0, 7], np.float32(-.3))

    def test_processed_slice_dimensions_and_unit_quaternion_are_checked(self):
        valid = np.array([0., 0., 0., 1., 0., 0., 0., .2])
        for invalid in [valid[:-1], np.r_[valid, .3], np.zeros(8), valid.astype(int),
                        valid * np.nan]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    decode_processed_hand_target(invalid, finger_count=1)

    def test_processed_target_requires_an_explicit_valid_finger_count(self):
        values = np.array([0., 0., 0., 1., 0., 0., 0., .2])
        for count in [True, np.bool_(True), 1.5, 0, -1]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    decode_processed_hand_target(values, finger_count=count)

    def test_measured_grasp_moves_and_rotates_with_the_actual_object(self):
        object_pose = np.r_[[.2, .3, .4], Rotation.from_euler('y', .4).as_quat()]
        wrist = np.r_[[.3, .2, .6], Rotation.from_euler('xyz', [.3, -.1, .2]).as_quat()]
        grasp = capture_object_relative_grasp(object_pose, wrist, [.2, .4])
        target, fingers = conditioned_hand_target(grasp, object_pose)
        np.testing.assert_allclose(target, wrist, atol=1e-14)
        np.testing.assert_array_equal(fingers, [.2, .4])
        delta = Rotation.from_euler('z', .7)
        moved = np.r_[delta.apply(object_pose[:3]) + [.5, 0, 0],
                      (delta * Rotation.from_quat(object_pose[3:])).as_quat()]
        target, _ = conditioned_hand_target(grasp, moved)
        np.testing.assert_allclose(target[:3], delta.apply(wrist[:3]) + [.5, 0, 0], atol=1e-14)
        self.assertLess((Rotation.from_quat(target[3:]).inv() * delta
                         * Rotation.from_quat(wrist[3:])).magnitude(), 1e-14)

    def test_optional_prediction_is_bounded_for_fast_motion(self):
        pose = np.array([0., 0, 0, 0, 0, 0, 1])
        grasp = capture_object_relative_grasp(pose, pose, [0.])
        wrist, _ = conditioned_hand_target(grasp, pose, linear_velocity=[10., 0, 0],
            angular_velocity=[0, 0, 20.], lead_s=.2)
        self.assertAlmostEqual(np.linalg.norm(wrist[:3]), .02)
        self.assertAlmostEqual(Rotation.from_quat(wrist[3:]).magnitude(), .2)

    def test_a_changed_grasp_preserves_reference_in_hand_rotation_and_finger_motion(self):
        obj = np.array([0., 0, 0, 0, 0, 0, 1])
        initial_hand = np.r_[[.1, 0, 0], Rotation.from_euler('z', .4).as_quat()]
        physical_hand = np.r_[[0., .2, 0], Rotation.from_euler('z', 1.).as_quat()]
        anchor = capture_object_relative_grasp(obj, physical_hand, [.7, .8])
        ref_delta = Rotation.from_euler('x', .6)
        current_hand = np.r_[initial_hand[:3] + Rotation.from_quat(initial_hand[3:]).apply([.02, 0, 0]),
            (Rotation.from_quat(initial_hand[3:]) * ref_delta).as_quat()]
        rebased = rebase_grasp_motion(anchor, obj, initial_hand, [.1, .2], obj, current_hand, [.3, .1])
        target, fingers = conditioned_hand_target(rebased, obj)
        np.testing.assert_allclose(target[:3], physical_hand[:3]
            + Rotation.from_quat(physical_hand[3:]).apply([.02, 0, 0]), atol=1e-14)
        expected = Rotation.from_quat(physical_hand[3:]) * ref_delta
        self.assertLess((Rotation.from_quat(target[3:]).inv() * expected).magnitude(), 1e-14)
        np.testing.assert_allclose(fingers, [.9, .7])

    def test_rebasing_at_acquisition_returns_the_actual_grasp(self):
        obj = np.array([0., 0, 0, 0, 0, 0, 1])
        hand = np.array([.1, .2, .3, 0, 0, 0, 1])
        actual = np.array([.3, .1, .2, 0, 0, 1., 0])
        anchor = capture_object_relative_grasp(obj, actual, [.8])
        rebased = rebase_grasp_motion(anchor, obj, hand, [.2], obj, hand, [.2])
        target, fingers = conditioned_hand_target(rebased, obj)
        np.testing.assert_allclose(target, actual, atol=1e-14)
        np.testing.assert_allclose(fingers, [.8])

    def test_rotation_residual_uses_reference_local_rpy(self):
        reference = np.r_[[.2, .3, .4], Rotation.from_euler('xyz', [.7, -.5, .3]).as_quat()]
        delta = np.array([.1, -.2, .3])
        target = np.r_[reference[:3] + [.01, -.02, .03],
            (Rotation.from_quat(reference[3:]) * Rotation.from_euler('xyz', delta)).as_quat()]
        result = hand_target_residual(reference, [.2], target, [.5])
        np.testing.assert_allclose(result, [.01, -.02, .03, *delta, .3], atol=1e-14)

    def test_matched_pre_step_feedback_preserves_a_fast_planned_command_exactly(self):
        before = np.r_[[.1, .2, .3], Rotation.from_euler('z', .5).as_quat()]
        planned = np.r_[[.2, .3, .4], Rotation.from_euler('z', .9).as_quat()]
        result = correct_planned_hand_target(planned, before, before.copy())
        np.testing.assert_array_equal(result, planned)
        moved = before.copy(); moved[:3] += [.01, 0, 0]
        result = correct_planned_hand_target(planned, before, moved)
        np.testing.assert_allclose(result[:3], planned[:3] + [.01, 0, 0])
        np.testing.assert_allclose(result[3:], planned[3:], atol=1e-14)

    def test_receiver_catches_actual_pose_then_carries_the_unchanged_reference(self):
        actual = np.array([.02, 0., 0, 0, 0, 0, 1])
        reference = np.r_[[.1, .2, .3], Rotation.from_euler('z', .8).as_quat()]
        saved = reference.copy()
        anchor = capture_object_relative_grasp(actual, np.r_[[.02, .1, 0], actual[3:]], [.5])
        catch, _ = handover_hand_target(anchor, actual, reference, carry_weight=0.)
        carry, _ = handover_hand_target(anchor, actual, reference, carry_weight=1.)
        np.testing.assert_allclose(catch[:3], [.02, .1, 0])
        np.testing.assert_allclose(carry[:3], reference[:3]
            + Rotation.from_quat(reference[3:]).apply([0, .1, 0]), atol=1e-14)
        np.testing.assert_array_equal(reference, saved)

    def test_rational_inverse_reproduces_the_desired_physical_residual(self):
        desired = np.array([[.015, -.02, .2], [-.01, .01, -.1]])
        previous = np.array([.01, -.005, .03]); scale = np.array([.05, .05, .15])
        clip = np.array([.03, .03, .35]); endpoint = clip / (.7 * scale)
        result = encode_filtered_residual(desired, previous, scale, clip,
                                         ema=.3, mapping='rational', input_scale=endpoint)
        action = result.action
        mapped = action / (1 - np.abs(action) + np.abs(action) / endpoint)
        actual = np.clip(.3 * previous + .7 * scale * mapped, -clip, clip)
        np.testing.assert_allclose(actual, desired, atol=1e-14)
        self.assertFalse(result.saturated.any())

    def test_current_ema_and_mapping_limits_report_unreachable_targets(self):
        result = encode_filtered_residual([1., -.1], [-.1, .1], [.1, .1], [.1, .1],
                                         ema=.9, mapping='linear', input_scale=[1., 1.])
        np.testing.assert_allclose(result.action, [1., -1.])
        np.testing.assert_allclose(result.filtered, [-.08, .08])
        self.assertTrue(result.saturated.all())

    def test_raw_input_mapping_inverts_ema_without_normalized_clipping(self):
        result = encode_filtered_residual([.2], [.1], [.1], [.5], ema=.5, mapping='identity')
        np.testing.assert_allclose(result.action, [3.])
        np.testing.assert_allclose(result.filtered, [.2])

    def test_invalid_pose_prediction_and_frozen_filter_are_rejected(self):
        with self.assertRaises(ValueError):
            capture_object_relative_grasp(np.zeros(7), np.zeros(7), [0.])
        with self.assertRaises(ValueError):
            encode_filtered_residual([0.], [0.], [1.], [1.], ema=1., mapping='identity')
