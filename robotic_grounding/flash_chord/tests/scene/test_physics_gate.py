"""Physics-gate diagnostics detect overflow and use the correct post-step clock."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import warp as wp

path = Path(__file__).resolve().parents[2]/'experiments/online_critic/cluster/physics_gate.py'
if not path.is_file():  # Prepared snapshots keep launchers beside flash_chord/.
    path = Path(__file__).resolve().parents[3]/'cluster/physics_gate.py'
spec = importlib.util.spec_from_file_location('uniretarget_physics_gate', path)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def test_sustained_failure_preserves_post_step_physical_clock():
    position = np.zeros((10, 2, 2))
    angle = np.zeros_like(position)
    angle[2:5, 1, 1] = np.deg2rad(21.)
    position[:2, 0, 0] = .06  # Only two transient bad samples.
    report = gate.summarize_errors(position, angle, 20.)
    assert not report['passed']
    assert report['objects'][0]['failed_worlds'] == 0
    lid = report['objects'][1]
    assert lid['failed_worlds'] == 1
    assert lid['first_sustained_failure_command_frame'] == 2
    assert lid['first_sustained_failure_physical_seconds'] == .15
    assert lid['failure_confirmed_physical_seconds'] == .25


def test_pose_errors_use_quaternion_sign_and_norm_invariance():
    actual = np.array([[[.01, 0., 0., 0., 0., 0., -2.],
                        [0., 0., 0., 0., 0., np.sin(.1), np.cos(.1)]]])
    position, angle = gate.pose_errors(actual, np.zeros((2, 3)), np.array([[0,0,0,1.]]*2))
    np.testing.assert_allclose(position, [[.01, 0.]])
    np.testing.assert_allclose(angle, [[0., .2]], atol=1e-14)


def test_substep_audit_detects_saturation_and_nonfinite_states_on_cpu():
    with wp.ScopedDevice('cpu'):
        q = wp.array(np.tile([0,0,0,0,0,0,1.], (3, 1)), dtype=wp.transform)
        velocity = np.zeros((3, 6)); velocity[2, 3] = np.nan
        qd = wp.array(velocity, dtype=wp.spatial_vector)
        counters = wp.zeros(5, dtype=wp.int32)
        wp.launch(gate.audit_substep, dim=3, inputs=[q, qd, wp.array([4], dtype=wp.int32),
            wp.array([2, 7], dtype=wp.int32), 4, 7, 2, counters])
        np.testing.assert_array_equal(counters.numpy(), [4, 7, 1, 1, 1])


def test_support_force_readout_excludes_object_pair_and_unused_slots():
    with wp.ScopedDevice('cpu'):
        force = np.zeros((4, 6), dtype=np.float32)
        force[:, 2] = [-5., -2., 100., 1000.]
        maxima = wp.zeros(2, dtype=wp.float32)
        net = wp.zeros((1, 3), dtype=wp.float32)
        wp.launch(gate.sample_contact_forces, dim=4, inputs=[wp.array([3], dtype=wp.int32),
            wp.array([0, -1, 1, 0], dtype=wp.int32), wp.array([1, 2, 2, 1], dtype=wp.int32),
            wp.array([2, 0, 1], dtype=wp.int32), wp.array(force, dtype=wp.spatial_vector),
            2, 3, maxima, net])
        np.testing.assert_array_equal(maxima.numpy(), [100., 5.])
        np.testing.assert_array_equal(net.numpy(), [[0., 0., 7.]])
