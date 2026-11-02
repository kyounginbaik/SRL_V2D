"""Contracts for the experimental tissue manipulation upper loop."""
from dataclasses import replace
import numpy as np
import pytest


def test_wrench_changes_candidate_ranking_not_just_reported_loss():
    from flash_chord.retargeting.manipulation_bilevel import candidate_score
    a = dict(robot_position_mse=.0001,robot_rotation_mse=.01,wrench_mse=1.)
    b = dict(robot_position_mse=.00012,robot_rotation_mse=.01,wrench_mse=.2)
    assert candidate_score(a,0) < candidate_score(b,0)
    assert candidate_score(a,1) > candidate_score(b,1)


def test_object_regression_rejected_even_with_better_robot_tracking():
    from flash_chord.retargeting.manipulation_bilevel import passes_task_gate
    baseline = dict(object_position_mean_m=.01,object_position_p95_m=.02,
                    object_orientation_mean_rad=.1,object_success_fraction=.99)
    assert passes_task_gate(baseline,baseline)
    assert not passes_task_gate(dict(baseline,object_position_mean_m=.03),baseline)
    assert not passes_task_gate(dict(baseline,object_success_fraction=.9),baseline)
    assert not passes_task_gate(dict(baseline,object_position_mean_m=float("nan")),baseline)


def test_zero_map_exact_fk_and_snapshot_roundtrip(tmp_path):
    import jax.numpy as jnp
    import newton
    from flash_chord.embodiments.vega_sharpa import VegaSharpa, VegaSharpaConfig
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.retargeting.manipulation_bilevel import binding_motion
    from flash_chord.retargeting.reactor import NominalAlignment, map_motion, zero_parameters
    from flash_chord.retargeting.reference_snapshot import save_snapshot,load_snapshot
    builder=newton.ModelBuilder()
    layout=VegaSharpa(VegaSharpaConfig()).build(builder)
    q=np.stack([np.asarray(builder.joint_q,dtype=np.float32)]*2)
    binding=build_robot_reference_binding(builder,layout,q,q,20.)
    source=binding_motion(binding)
    b=source.position.shape[1]
    mapped=map_motion(source,NominalAlignment(1.,jnp.zeros((b,3)),jnp.broadcast_to(jnp.eye(3),(b,3,3))),
                      zero_parameters(b,1),jnp.zeros(2,dtype=jnp.int32),jnp.zeros(1))
    np.testing.assert_allclose(mapped.position,source.position,atol=1e-7)
    np.testing.assert_allclose(mapped.rotation,source.rotation,atol=1e-7)
    path=tmp_path/"reference.npz"
    sha=save_snapshot(path,binding,layout,provenance={"test":True})
    loaded=load_snapshot(path,layout,binding,sha256=sha)
    np.testing.assert_array_equal(loaded.joint_target,binding.joint_target)
    np.testing.assert_array_equal(loaded.hands[0].fingertip_pos_w,binding.hands[0].fingertip_pos_w)
    with pytest.raises(ValueError,match="checksum"):
        load_snapshot(path,layout,binding,sha256="wrong")
    with pytest.raises(ValueError,match="FPS"):
        load_snapshot(path,layout,replace(binding,fps=30.))


def test_projection_changes_targets_preserves_initial_reset_and_joint_limits():
    import newton
    from flash_chord.embodiments.vega_sharpa import VegaSharpa,VegaSharpaConfig
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.retargeting.manipulation_bilevel import LocalProjector,binding_motion
    embodiment=VegaSharpa(VegaSharpaConfig())
    builder=newton.ModelBuilder()
    layout=embodiment.build(builder)
    q=np.stack([np.asarray(builder.joint_q,dtype=np.float32)]*12)
    binding=build_robot_reference_binding(builder,layout,q,q,20.)
    projector=LocalProjector(embodiment,binding,stride=11)
    source=binding_motion(binding)
    target=replace(source,position=source.position.at[:,:,0].add(.002))
    candidate=projector.project(target)
    np.testing.assert_array_equal(candidate.joint_q[0],binding.joint_q[0])
    assert np.max(np.abs(candidate.joint_q-binding.joint_q))>1e-6
    assert np.all(candidate.joint_q[:,projector.qids]>=projector.lower-1e-6)
    assert np.all(candidate.joint_q[:,projector.qids]<=projector.upper+1e-6)
    assert np.max(np.abs(candidate.hands[0].wrist_pos_w-binding.hands[0].wrist_pos_w))>1e-6
