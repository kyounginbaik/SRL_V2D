from types import SimpleNamespace

import numpy as np

from flash_chord.retargeting.penetration import HandObjectPenetration, convex_hull_interior_depth, penetration_summary
from flash_chord.scene.collision import HandCollisionShapes, SceneCollisionLayout, ShapeSpan
from flash_chord.retargeting.ttsa_editor import upper_cost, passes_study_gate


def test_contact_pair_selector_and_intended_contact_tolerance(monkeypatch):
    import newton

    builder = newton.ModelBuilder()
    for label in ("finger", "object"):
        body = builder.add_body(mass=1.0, label=label)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    q = np.asarray(builder.joint_q, dtype=np.float32)
    q[7] = 0.09  # ten millimetres of overlap
    builder.joint_q = q.tolist()
    model = builder.finalize(device="cpu")
    state = model.state()
    state.joint_q.assign(q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = model.contacts()
    model.collide(state, contacts)
    layout = SceneCollisionLayout(
        robot=ShapeSpan(0, 1), objects=ShapeSpan(1, 2), support=ShapeSpan(2, 2),
        hands=(HandCollisionShapes("left", (0,), (0,)),),
    )
    scene = SimpleNamespace(model=model, collision_layout=layout, world_count=1)
    monitor = HandObjectPenetration(scene, tolerance_m=0.002)
    assert len(monitor.selected_gaps(state, contacts)) > 0
    # Newton versions differ in whether reduced mesh contacts retain depth;
    # inject a measured gap to test the selector's depth/tolerance semantics.
    monkeypatch.setattr(monitor, "selected_gaps", lambda *_: [(0, 0, -0.01)])
    np.testing.assert_allclose(monitor.frame_depths(state, contacts), [[0.01]], atol=1e-5)
    np.testing.assert_allclose(monitor.frame_depths(state, contacts, [True]), [[0.008]], atol=1e-5)
    assert penetration_summary(np.array([[[0.01]], [[0.0]]]))["penetration_mean_m"] == 0.005


def test_penetration_changes_upper_ranking_and_gate():
    baseline = dict(
        object_position_robust_m=0.02, object_orientation_robust_rad=0.1,
        object_success_robust_fraction=0.9, wrench_robust_mse=1.0,
        robot_position_robust_mse=0.0001, unstable_world_fraction=0.0,
        reference_penetration_mean_m=0.006, reference_penetration_peak_m=0.020,
        rollout_penetration_mean_m=0.004,
    )
    better = dict(baseline, reference_penetration_mean_m=0.002,
                  reference_penetration_peak_m=0.012, rollout_penetration_mean_m=0.002)
    delta = np.zeros((10, 2))
    assert upper_cost(better, delta, 20, 1, robust=True, penetration_weight=1) < upper_cost(
        baseline, delta, 20, 1, robust=True, penetration_weight=1)
    assert passes_study_gate(better, baseline, penetration_aware=True)
    assert not passes_study_gate(dict(better, reference_penetration_peak_m=0.024),
                                 baseline, penetration_aware=True)
    assert not passes_study_gate(dict(better, rollout_penetration_mean_m=baseline["rollout_penetration_mean_m"]+0.0002),
                                 baseline, penetration_aware=True)


def test_convex_visual_proxy_distinguishes_inside_from_outside():
    normals = np.concatenate((np.eye(3), -np.eye(3)))
    offsets = np.full(6, -0.05)
    samples = np.array([[0, 0, 0], [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0]])
    np.testing.assert_allclose(convex_hull_interior_depth(samples, normals, offsets),
                               [0.05, 0.01, 0.0, 0.0], atol=1e-10)
