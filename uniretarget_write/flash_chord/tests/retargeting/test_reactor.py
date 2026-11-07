# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric and gradient checks for the ReActor outer reference map."""

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.retargeting.reactor import (
    BodyMotion,
    ReactorParameters,
    calibrate_nominal,
    map_motion,
    outer_loss,
    outer_step,
    project_parameters,
    rotation_exp,
    stack_nominal_alignments,
    zero_parameters,
)


def _fixture():
    identity = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    source_nominal = np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 0.7]])
    target_nominal = np.array([[0.0, 0.0, 0.5], [0.15, 0.02, 0.36]])
    nominal = calibrate_nominal(source_nominal, identity, target_nominal, identity, 1.0, 0.5)
    source = BodyMotion(
        jnp.asarray(source_nominal[None]),
        jnp.asarray(identity[None]),
        jnp.zeros((1, 2, 3)),
        jnp.zeros((1, 2, 3)),
    )
    return source, nominal, jnp.array([0]), jnp.array([0.0])


def test_zero_offsets_align_nominal_bodies():
    source, nominal, ids, heights = _fixture()
    target = map_motion(source, nominal, zero_parameters(2, 1), ids, heights)
    np.testing.assert_allclose(target.position, [[[0.0, 0.0, 0.5], [0.15, 0.02, 0.36]]], atol=1e-6)
    np.testing.assert_allclose(target.rotation, np.broadcast_to(np.eye(3), (1, 2, 3, 3)), atol=1e-6)


def test_rotating_local_offset_contributes_angular_velocity_cross_product():
    source, nominal, ids, heights = _fixture()
    turning = BodyMotion(source.position, source.rotation, source.linear_velocity, jnp.array([[[0.0, 0.0, 1.0]] * 2]))
    params = ReactorParameters(jnp.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]), jnp.zeros((2, 3)), jnp.zeros((1,)))
    target = map_motion(turning, nominal, params, ids, heights)
    np.testing.assert_allclose(target.linear_velocity[0, 0], [0.0, 0.1, 0.0], atol=1e-6)


def test_projected_step_reduces_error_and_stops_at_bounds():
    source, nominal, ids, heights = _fixture()
    params = zero_parameters(2, 1)
    initial = map_motion(source, nominal, params, ids, heights)
    achieved = BodyMotion(initial.position.at[:, 0, 0].add(0.1), initial.rotation, initial.linear_velocity, initial.angular_velocity)
    updated, loss = outer_step(params, source, achieved, nominal, ids, heights, jnp.array([True]), learning_rate=0.01)
    assert np.isfinite(loss)
    assert updated.position[0, 0] > 0
    assert outer_loss(updated, source, achieved, nominal, ids, heights, jnp.array([True])) < loss
    oversized = ReactorParameters(jnp.ones((2, 3)), jnp.ones((2, 3)), jnp.array([3.0]))
    bounded = project_parameters(oversized)
    np.testing.assert_allclose(np.linalg.norm(bounded.position, axis=-1), 0.5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(bounded.orientation, axis=-1), 0.5, atol=1e-6)
    np.testing.assert_allclose(bounded.height, [0.5], atol=1e-6)


def test_zero_weight_feedback_padding_preserves_upper_step():
    source, nominal, ids, heights = _fixture()
    params = zero_parameters(2, 1)
    achieved = map_motion(source, nominal, params, ids, heights)
    achieved = BodyMotion(
        achieved.position.at[0, 0, 0].add(0.1), achieved.rotation,
        achieved.linear_velocity, achieved.angular_velocity,
    )
    expected, expected_loss = outer_step(
        params, source, achieved, nominal, ids, heights, jnp.array([1.0]),
        learning_rate=0.01,
    )
    padded_source = jax.tree.map(lambda field: jnp.repeat(field, 64, axis=0), source)
    padded_achieved = jax.tree.map(lambda field: jnp.repeat(field, 64, axis=0), achieved)
    padded, padded_loss = outer_step(
        params, padded_source, padded_achieved, nominal, jnp.zeros(64, dtype=jnp.int32),
        heights, jnp.concatenate((jnp.ones(1), jnp.zeros(63))), learning_rate=0.01,
    )
    np.testing.assert_allclose(padded_loss, expected_loss, atol=1e-6)
    for actual, original in zip(jax.tree.leaves(padded), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, original, atol=1e-6)


def test_orientation_gradient_matches_finite_difference():
    source, nominal, ids, heights = _fixture()
    params = zero_parameters(2, 1)
    achieved = map_motion(source, nominal, params, ids, heights)
    achieved = BodyMotion(achieved.position, achieved.rotation.at[0, 0].set(rotation_exp(jnp.array([0.2, 0.0, 0.0]))), achieved.linear_velocity, achieved.angular_velocity)

    def loss(angle):
        changed = ReactorParameters(params.position, params.orientation.at[0, 0].set(angle), params.height)
        return outer_loss(changed, source, achieved, nominal, ids, heights, jnp.array([True]))

    analytic = jax.grad(loss)(0.0)
    step = 1e-3
    finite_difference = (loss(step) - loss(-step)) / (2 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-3, atol=1e-4)


def test_per_clip_nominal_alignment_selects_source_shape():
    source, nominal, _, _ = _fixture()
    second = calibrate_nominal(
        np.asarray(source.position[0]) * 2,
        np.asarray(source.rotation[0]),
        np.array([[0.0, 0.0, 0.5], [0.15, 0.02, 0.36]]),
        np.asarray(source.rotation[0]),
        2.0,
        0.5,
    )
    stacked = stack_nominal_alignments((nominal, second))
    source_both = BodyMotion(
        jnp.concatenate((source.position, source.position * 2)),
        jnp.concatenate((source.rotation, source.rotation)),
        jnp.zeros((2, 2, 3)),
        jnp.zeros((2, 2, 3)),
    )
    mapped = map_motion(source_both, stacked, zero_parameters(2, 2), jnp.array([0, 1]), jnp.zeros(2))
    np.testing.assert_allclose(mapped.position[0], mapped.position[1], atol=1e-6)


def test_outer_loss_accepts_fractional_per_motion_feedback_weights():
    source, nominal, _, heights = _fixture()
    doubled = BodyMotion(*(
        jnp.concatenate((value, value), axis=0)
        for value in (source.position, source.rotation, source.linear_velocity, source.angular_velocity)
    ))
    ids = jnp.array([0, 0])
    target = map_motion(doubled, nominal, zero_parameters(2, 1), ids, heights)
    achieved = BodyMotion(
        target.position.at[0, 0, 0].add(0.1).at[1, 0, 0].add(0.2),
        target.rotation, target.linear_velocity, target.angular_velocity,
    )
    params = zero_parameters(2, 1)
    first_only = outer_loss(params, doubled, achieved, nominal, ids, heights, jnp.array([1.0, 0.0]))
    second_only = outer_loss(params, doubled, achieved, nominal, ids, heights, jnp.array([0.0, 1.0]))
    balanced = outer_loss(params, doubled, achieved, nominal, ids, heights, jnp.array([0.5, 0.5]))
    np.testing.assert_allclose(balanced, (first_only + second_only) / 2, atol=1e-7)
