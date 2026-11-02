"""Falsification cases for static contact-wrench projection."""
import unittest
import numpy as np
from flash_chord.retargeting.contact_wrench import ContactPatch, project_contact_wrench


def point(normal=(0., 0., 1.), friction=(.5, .5), position=(0., 0., 0.), group='hand'):
    return ContactPatch(position, normal, (1., 0., 0.), (0., 1., 0.), friction, group)


class ProjectionTests(unittest.TestCase):
    def test_pressing_into_table_cannot_be_mistaken_for_lifting(self):
        r = project_contact_wrench([point(normal=(0., 0., -1.))], [0, 0, 1, 0, 0, 0])
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.contact_forces, 0, atol=1e-8)
        np.testing.assert_allclose(r.missing_wrench, [0, 0, 1, 0, 0, 0], atol=1e-8)

    def test_attainable_upward_load_has_zero_missing_force(self):
        r = project_contact_wrench([point()], [0, 0, 1, 0, 0, 0])
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.missing_wrench, 0, atol=1e-6)

    def test_missing_torque_is_preserved_instead_of_zeroing_all_forces(self):
        r = project_contact_wrench([point()], [0, 0, 1, 0, 0, .02])
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.achieved_wrench, [0, 0, 1, 0, 0, 0], atol=1e-6)
        self.assertAlmostEqual(r.missing_wrench[5], .02)

    def test_friction_and_normal_force_budgets_are_simultaneously_enforced(self):
        r = project_contact_wrench([point()], [2, 0, 1, 0, 0, 0], normal_force_limits={'hand': 1})
        self.assertTrue(r.converged)
        self.assertLessEqual(r.normal_force_per_group['hand'], 1 + 1e-7)
        self.assertLessEqual(abs(r.contact_forces[0, 0]), .5 * r.contact_forces[0, 2] + 1e-7)
        self.assertGreater(r.missing_wrench[0], 1.4)

    def test_joint_effort_budget_limits_an_otherwise_feasible_lift(self):
        # 0.1 m moment arm: each 1 N normal load costs 0.1 Nm.
        r = project_contact_wrench([point()], [0, 0, 1, 0, 0, 0],
            effort_matrix=np.full((1, 4), .1), effort_lower=[-.04], effort_upper=[.04])
        self.assertTrue(r.converged)
        self.assertLessEqual(r.joint_effort[0], .04 + 1e-7)
        self.assertAlmostEqual(r.achieved_wrench[2], .4, places=5)
        self.assertAlmostEqual(r.missing_wrench[2], .6, places=5)

    def test_inconsistent_effort_bounds_are_not_a_valid_projection(self):
        r = project_contact_wrench([point()], [0, 0, 1, 0, 0, 0],
            normal_force_limits={'hand': 0}, effort_matrix=np.ones((1, 4)), effort_lower=[1], effort_upper=[2])
        self.assertFalse(r.converged)

    def test_empty_cloud_returns_missing_load_without_claiming_support(self):
        r = project_contact_wrench([], [0, 0, 1, 0, 0, 0])
        self.assertTrue(r.converged)
        np.testing.assert_array_equal(r.achieved_wrench, np.zeros(6))
        self.assertGreater(r.scaled_loss, 0)

    def test_invalid_effort_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            project_contact_wrench([point()], np.zeros(6), effort_matrix=np.zeros((1, 3)), effort_lower=[0], effort_upper=[1])

if __name__ == '__main__':
    unittest.main()
