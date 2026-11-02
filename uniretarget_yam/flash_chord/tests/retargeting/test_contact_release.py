"""Physical and timing contracts of a complete, reusable finger release."""
import unittest

import numpy as np

from flash_chord.retargeting.contact_refinement import HandChannels
from flash_chord.retargeting.contact_release import (
    plan_finger_transition,
    plan_full_release,
    plan_release_by,
    plan_processed_finger_transition,
)


class ContactReleaseTests(unittest.TestCase):
    def setUp(self):
        self.fps = 30.
        self.hand = HandChannels('giver', (0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10), (0., 1., 0., 1.))
        self.q = np.zeros((180, 14), dtype=np.float32)
        self.q[:, 6] = 1.
        self.q[:, self.hand.fingers] = [1.6, 1.4, -.4, .7]
        self.q[:, 11] = np.arange(len(self.q)) * .01
        self.target = np.array([0., 0., .2, -.2])
        self.lower = np.array([0., 0., -.5, -.3])
        self.upper = np.array([1.8, 1.8, .3, 1.])

    def release(self, **kwargs):
        options = dict(fps=self.fps, bounds=(10, 150), event_frame=30,
                       hold_until=100, next_contact_frame=140, max_speed_rad_s=3.)
        options.update(kwargs)
        return plan_full_release(self.q, self.hand, self.target, self.lower, self.upper, **options)

    def test_deadline_release_preserves_latest_possible_loaded_posture(self):
        result = plan_release_by(self.q, self.hand, self.target, self.lower, self.upper,
            fps=self.fps, bounds=(10, 150), complete_by=65, hold_until=100,
            next_contact_frame=140)
        self.assertTrue(result.feasible, result.diagnostic)
        onset = result.diagnostic['opening_start']
        self.assertLessEqual(result.diagnostic['opening_complete'], 65)
        self.assertFalse(self.release(event_frame=onset + 1, complete_by=65).feasible)
        np.testing.assert_array_equal(result.joint_q[:onset + 1], self.q[:onset + 1])
        np.testing.assert_array_equal(result.joint_q[:, :7], self.q[:, :7])
        np.testing.assert_array_equal(result.joint_q[140:], self.q[140:])

    def test_deadline_release_uses_posture_at_actual_onset(self):
        self.q[10:66, 7] = np.linspace(.2, 1.6, 56)
        result = plan_release_by(self.q, self.hand, self.target, self.lower, self.upper,
            fps=self.fps, bounds=(10, 150), complete_by=65, hold_until=100)
        self.assertTrue(result.feasible, result.diagnostic)
        onset = result.diagnostic['opening_start']
        np.testing.assert_array_equal(result.joint_q[onset], self.q[onset])
        self.assertFalse(self.release(event_frame=onset + 1, complete_by=65).feasible)
        self.assertLessEqual(result.diagnostic['peak_command_speed_rad_s'], 3.)

    def test_impossible_deadline_has_no_partial_release(self):
        result = plan_release_by(self.q, self.hand, self.target, self.lower, self.upper,
            fps=self.fps, bounds=(60, 150), complete_by=65, hold_until=100)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.joint_q)
        self.assertEqual(result.diagnostic['reason'], 'no_feasible_opening_before_deadline')

    def test_full_target_reached_including_joints_with_zero_closure_weight(self):
        result = self.release(complete_by=65)
        self.assertTrue(result.feasible, result.diagnostic)
        first, last = result.diagnostic['open_interval']
        expected = np.broadcast_to(self.target, (last - first + 1, len(self.target)))
        np.testing.assert_array_equal(result.joint_q[first:last + 1, self.hand.fingers], expected)
        self.assertLess(first, 65)
        # A full 1.6-radian opening survives; it is not clipped to a .4/.8 delta.
        self.assertGreaterEqual(result.diagnostic['max_finger_delta_rad'], 1.6)

    def test_every_command_respects_limits_and_absolute_velocity(self):
        result = self.release()
        fingers = result.joint_q[:, self.hand.fingers]
        self.assertTrue((fingers >= self.lower - 1e-12).all())
        self.assertTrue((fingers <= self.upper + 1e-12).all())
        speed = np.abs(np.diff(fingers, axis=0)) * self.fps
        self.assertLessEqual(speed.max(), 3. + 1e-10)
        self.assertAlmostEqual(result.diagnostic['peak_command_speed_rad_s'], speed.max())

    def test_only_fingers_in_allowed_interior_change(self):
        original = self.q.copy()
        result = self.release()
        nonfingers = [i for i in range(self.q.shape[1]) if i not in self.hand.fingers]
        np.testing.assert_array_equal(result.joint_q[:, nonfingers], self.q[:, nonfingers])
        np.testing.assert_array_equal(result.joint_q[:31], self.q[:31])
        np.testing.assert_array_equal(result.joint_q[150:], self.q[150:])
        np.testing.assert_array_equal(result.joint_q[[0, -1]], self.q[[0, -1]])
        np.testing.assert_array_equal(self.q, original)

    def test_next_intended_regrasp_and_all_later_motion_are_exact(self):
        # The later contact has its own changing posture. Restore it as recorded.
        self.q[140:, 7] = np.linspace(1.6, 1., 40)
        result = self.release()
        self.assertTrue(result.feasible, result.diagnostic)
        self.assertLessEqual(result.diagnostic['restore_frame'], 140)
        np.testing.assert_array_equal(result.joint_q[140:], self.q[140:])

    def test_insufficient_opening_time_produces_diagnosis_and_no_partial_edit(self):
        result = self.release(complete_by=40)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.joint_q)
        self.assertEqual(result.diagnostic['reason'], 'insufficient_opening_time')
        self.assertGreater(result.diagnostic['minimum_opening_frames'],
                           result.diagnostic['available_opening_frames'])

    def test_insufficient_return_time_does_not_overwrite_next_contact(self):
        result = self.release(next_contact_frame=110)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'insufficient_restore_time')
        self.assertEqual(result.diagnostic['latest_restore_frame'], 110)

    def test_hold_cannot_overlap_next_intended_contact(self):
        result = self.release(next_contact_frame=90)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'hold_overlaps_protected_motion')

    def test_only_explicit_limited_window_extension_allows_longer_release(self):
        result = self.release(bounds=(10, 120))
        self.assertFalse(result.feasible)
        result = self.release(bounds=(10, 120), max_window_extension_s=.4)
        self.assertTrue(result.feasible, result.diagnostic)
        self.assertGreater(result.diagnostic['window_extension_frames'], 0)
        self.assertLessEqual(result.diagnostic['window_extension_frames'], 12)
        np.testing.assert_array_equal(result.joint_q[132:], self.q[132:])

    def test_release_delay_changes_onset_without_retiming_other_channels(self):
        result = self.release(delay_s=.4)
        self.assertTrue(result.feasible, result.diagnostic)
        self.assertEqual(result.diagnostic['opening_start'], 42)
        np.testing.assert_array_equal(result.joint_q[:43], self.q[:43])
        np.testing.assert_array_equal(result.joint_q[140:], self.q[140:])

    def test_explicit_finger_indices_are_supported(self):
        by_hand = self.release()
        by_ids = plan_full_release(self.q, self.hand.fingers, self.target, self.lower, self.upper,
                                  fps=self.fps, bounds=(10, 150), event_frame=30,
                                  hold_until=100, next_contact_frame=140)
        np.testing.assert_array_equal(by_ids.joint_q, by_hand.joint_q)

    def test_generic_transition_accepts_a_fitted_receiving_grasp(self):
        target = np.array([1.7, 1.6, -.1, .9])
        result = plan_finger_transition(self.q, self.hand, target, self.lower, self.upper,
                                       fps=self.fps, bounds=(10, 150), event_frame=30, hold_until=100)
        self.assertTrue(result.feasible, result.diagnostic)
        start, end = result.diagnostic['open_interval']
        np.testing.assert_array_equal(result.joint_q[start:end + 1, self.hand.fingers],
                                      np.broadcast_to(target, (end - start + 1, len(target))))

    def test_full_target_outside_physical_limits_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, 'physical joint limits'):
            plan_full_release(self.q, self.hand, [-.1, 0., .2, -.2], self.lower, self.upper,
                              fps=30, bounds=(10, 150), event_frame=30, hold_until=100)

    def test_return_duration_uses_actual_future_reference(self):
        # The recorded regrasp has already opened these joints. Returning to a
        # fictitious initial grasp would reject this otherwise feasible window.
        self.q[100:, self.hand.fingers] = self.target
        result = self.release(bounds=(10, 105), next_contact_frame=105)
        self.assertTrue(result.feasible, result.diagnostic)
        # One frame may be needed to return the float64 target to its exact
        # float32 representation in the original command stream.
        self.assertLessEqual(result.diagnostic['restore_frame'], 101)
        np.testing.assert_array_equal(result.joint_q[101:], self.q[101:])

    def test_added_speed_is_reported_separately_from_command_speed(self):
        # During full opening the original command closes. Full opening remains
        # physically slow, while its correction changes faster than either hand.
        self.q[45:76, 8] = np.linspace(1.4, 1.8, 31)
        self.q[76:, 8] = 1.8
        result = self.release()
        self.assertTrue(result.feasible, result.diagnostic)
        correction = result.joint_q[:, self.hand.fingers] - self.q[:, self.hand.fingers]
        expected = np.abs(np.diff(correction, axis=0)).max() * self.fps
        self.assertAlmostEqual(result.diagnostic['peak_added_speed_rad_s'], expected)
        self.assertLessEqual(result.diagnostic['peak_command_speed_rad_s'], 3.)


class ProcessedFingerTransitionTests(unittest.TestCase):
    def setUp(self):
        # Captured PD preload intentionally exceeds the physical posture range.
        self.initial = np.array([1.7453, 1.4, -.4, .7])
        self.command = self.initial + [.3, .3, -.2, .1]
        self.target = np.array([0., 0., .2, -.2])
        self.lower = np.array([0., 0., -.5, -.3])
        self.upper = np.array([1.7453, 1.8, .3, 1.])
        self.clip = np.array([.3, .3, .2, .2])
        self.return_reference = np.array([1.2, .9, -.25, .35])
        self.return_command = self.return_reference + [.1, .2, -.1, .1]

    def plan(self, **kwargs):
        options = dict(initial_reference=self.initial, restore_command=self.return_command,
                       restore_reference=self.return_reference, fps=20., bounds=(5, 130),
                       event_frame=15, hold_until=85, next_contact_frame=120,
                       max_speed_rad_s=3.)
        options.update(kwargs)
        return plan_processed_finger_transition(self.command, self.target, self.lower, self.upper,
                                                self.clip, **options)

    def test_captured_preload_is_preserved_instead_of_clipped_to_joint_limits(self):
        result = self.plan()
        self.assertTrue(result.feasible, result.diagnostic)
        prefix = result.frame_ids <= 15
        np.testing.assert_array_equal(result.command[prefix], np.broadcast_to(self.command, (prefix.sum(), 4)))
        np.testing.assert_array_equal(result.reference[prefix], np.broadcast_to(self.initial, (prefix.sum(), 4)))
        np.testing.assert_allclose(result.residual[prefix],
                                   np.broadcast_to(self.command - self.initial, (prefix.sum(), 4)), atol=1e-15)
        self.assertGreater(result.command[0, 0], self.upper[0])
        self.assertGreater(result.diagnostic['peak_processed_joint_limit_overdrive_rad'], .29)

    def test_exact_open_hold_and_scheduled_restore_are_achievable(self):
        result = self.plan()
        first, last = result.diagnostic['open_interval']
        held = (result.frame_ids >= first) & (result.frame_ids <= last)
        expected = np.broadcast_to(self.target, (held.sum(), 4))
        np.testing.assert_array_equal(result.command[held], expected)
        np.testing.assert_array_equal(result.reference[held], expected)
        np.testing.assert_array_equal(result.residual[held], 0.)
        np.testing.assert_array_equal(result.command[-1], self.return_command)
        np.testing.assert_array_equal(result.reference[-1], self.return_reference)
        np.testing.assert_allclose(result.reference + result.residual, result.command, atol=1e-15)
        self.assertEqual(result.frame_ids[-1], 120)

    def test_reference_limits_residual_limits_and_both_absolute_speeds(self):
        result = self.plan()
        self.assertTrue(np.all(result.reference >= self.lower - 1e-12))
        self.assertTrue(np.all(result.reference <= self.upper + 1e-12))
        self.assertTrue(np.all(np.abs(result.residual) <= self.clip + 1e-12))
        for values in (result.command, result.reference):
            self.assertLessEqual(np.abs(np.diff(values, axis=0)).max() * 20, 3. + 1e-10)

    def test_slower_reference_speed_is_part_of_duration_not_a_later_projection(self):
        result = self.plan(max_reference_speed_rad_s=.7, hold_until=110,
                           next_contact_frame=130, bounds=(5, 180), restore_frame=175)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'restore_overlaps_protected_contact')
        result = self.plan(max_reference_speed_rad_s=.7, hold_until=110,
                           next_contact_frame=180, bounds=(5, 180), restore_frame=180)
        self.assertTrue(result.feasible, result.diagnostic)
        self.assertGreater(result.diagnostic['minimum_opening_frames'], 26)
        self.assertLessEqual(np.abs(np.diff(result.reference, axis=0)).max() * 20, .7 + 1e-10)
        self.assertGreater(result.diagnostic['opening_complete'], 41)

    def test_insufficient_opening_time_returns_no_partial_command_or_split(self):
        saved = self.command.copy()
        result = self.plan(complete_by=40)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'insufficient_opening_time')
        self.assertGreater(result.diagnostic['minimum_opening_frames'],
                           result.diagnostic['available_opening_frames'])
        for name in ('frame_ids', 'command', 'reference', 'residual'):
            self.assertIsNone(getattr(result, name))
        np.testing.assert_array_equal(self.command, saved)

    def test_insufficient_restore_time_does_not_clip_preload_or_overwrite_contact(self):
        result = self.plan(next_contact_frame=90)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'insufficient_restore_time')
        self.assertIsNone(result.command)
        result = self.plan(next_contact_frame=80)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'hold_overlaps_protected_motion')

    def test_future_moving_grasp_is_untouched_when_interval_is_stitched(self):
        result = self.plan()
        commands = np.broadcast_to(self.command, (160, 4)).copy()
        references = np.broadcast_to(self.initial, commands.shape).copy()
        commands[120:] = self.return_command + np.arange(40)[:, None] * .001
        references[120:] = self.return_reference + np.arange(40)[:, None] * .001
        changed_command, changed_reference = commands.copy(), references.copy()
        changed_command[result.frame_ids] = result.command
        changed_reference[result.frame_ids] = result.reference
        np.testing.assert_array_equal(changed_command[120:], commands[120:])
        np.testing.assert_array_equal(changed_reference[120:], references[120:])
        np.testing.assert_array_equal(changed_command[:5], commands[:5])

    def test_unrepresentable_endpoints_are_diagnosed_without_silent_clipping(self):
        result = self.plan(initial_reference=self.initial - [0., .1, 0., 0.])
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'start_command_exceeds_residual_limit')
        result = self.plan(restore_command=self.return_reference + .5)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'restore_command_exceeds_residual_limit')
        result = self.plan(restore_reference=self.upper + .1)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'restore_reference_outside_joint_limits')

    def test_unachievable_target_is_never_projected_into_a_partial_open(self):
        result = plan_processed_finger_transition(self.command, [-.4, 0., .2, -.2],
            self.lower, self.upper, self.clip, initial_reference=self.initial,
            restore_command=self.return_command, restore_reference=self.return_reference,
            fps=20., bounds=(5, 130), event_frame=15, hold_until=85)
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostic['reason'], 'target_command_exceeds_residual_limit')

    def test_processed_target_beyond_physical_stop_has_a_valid_reference_split(self):
        target = np.array([-.1, 0., .2, -.2])
        result = plan_processed_finger_transition(self.command, target,
            self.lower, self.upper, self.clip, initial_reference=self.initial,
            restore_command=self.return_command, restore_reference=self.return_reference,
            fps=20., bounds=(5, 130), event_frame=15, hold_until=85)
        self.assertTrue(result.feasible, result.diagnostic)
        first, last = result.diagnostic['open_interval']
        held = (result.frame_ids >= first) & (result.frame_ids <= last)
        np.testing.assert_array_equal(result.command[held], np.broadcast_to(target, (held.sum(), 4)))
        np.testing.assert_array_equal(result.reference[held, 0], 0.)
        np.testing.assert_array_equal(result.residual[held, 0], -.1)
        self.assertTrue(np.all(np.abs(result.residual) <= self.clip + 1e-12))

    def test_float32_captured_preload_rounding_preserves_both_exact_start_values(self):
        command = np.array([1.976505994796753], np.float32)
        reference = np.array([1.6765059232711792], np.float32)
        result = plan_processed_finger_transition(command, [0.], [0.], [1.7453], .3,
            initial_reference=reference, restore_command=command, restore_reference=reference,
            fps=20., bounds=(0, 120), event_frame=10, hold_until=60)
        self.assertTrue(result.feasible, result.diagnostic)
        np.testing.assert_array_equal(result.command[0], command)
        np.testing.assert_array_equal(result.reference[0], reference)
        self.assertLessEqual(result.diagnostic['peak_residual_rad'], .3 + 1e-7)

    def test_zero_residual_and_zero_duration_endpoints(self):
        result = plan_processed_finger_transition([.2], [.2], [0.], [1.], 0.,
            initial_reference=[.2], restore_command=[.2], restore_reference=[.2],
            fps=20., bounds=(7, 7), event_frame=7, hold_until=7)
        self.assertTrue(result.feasible, result.diagnostic)
        np.testing.assert_array_equal(result.frame_ids, [7])
        np.testing.assert_array_equal(result.command, [[.2]])
        np.testing.assert_array_equal(result.reference, result.command)
        np.testing.assert_array_equal(result.residual, 0.)

    def test_invalid_shapes_and_noninteger_frames_raise(self):
        with self.assertRaisesRegex(ValueError, 'scalar or matching'):
            plan_processed_finger_transition(self.command, self.target, self.lower, self.upper,
                [.3, .3], initial_reference=self.initial, restore_command=self.return_command,
                restore_reference=self.return_reference, fps=20., bounds=(5, 130), event_frame=15, hold_until=85)
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.plan(event_frame=15.5)


if __name__ == '__main__':
    unittest.main()
