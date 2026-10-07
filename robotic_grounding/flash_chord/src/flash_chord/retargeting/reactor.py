# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ReActor's sparse reference map and stop-gradient outer update.

Coordinates are world-frame metres, radians, and seconds. Rotation matrices map
body-local vectors to world coordinates. A motion batch has shape [N, B, ...],
where N indexes samples and B indexes source/target body correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ReactorLimits:
    position: float = 0.5
    orientation: float = 0.5
    height: float = 0.5


@dataclass(frozen=True)
class ReactorWeights:
    position: float = 10.0
    orientation: float = 1.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ReactorParameters:
    """Body offsets are shared across clips; height is indexed by clip ID."""

    position: jax.Array  # [B, 3], target nominal body coordinates
    orientation: jax.Array  # [B, 3], axis-angle in target nominal body coordinates
    height: jax.Array  # [M], one scalar per motion

    def tree_flatten(self):
        return (self.position, self.orientation, self.height), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@dataclass(frozen=True)
class NominalAlignment:
    scale: float | jax.Array  # scalar or [M], one per source skeleton
    position: jax.Array  # [B, 3] or [M, B, 3], in source body coordinates
    rotation: jax.Array  # [B, 3, 3] or [M, B, 3, 3]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BodyMotion:
    position: jax.Array  # [N, B, 3]
    rotation: jax.Array  # [N, B, 3, 3]
    linear_velocity: jax.Array  # [N, B, 3]
    angular_velocity: jax.Array  # [N, B, 3]

    def tree_flatten(self):
        return (self.position, self.rotation, self.linear_velocity, self.angular_velocity), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def calibrate_nominal(
    source_position: np.ndarray,
    source_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    source_root_height: float,
    target_root_height: float,
) -> NominalAlignment:
    """Compute the paper's nominal alignment (equations 8 and 9)."""
    source_position = np.asarray(source_position, dtype=np.float64)
    source_rotation = np.asarray(source_rotation, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    if source_position.shape != target_position.shape or source_position.ndim != 2 or source_position.shape[1] != 3:
        raise ValueError("nominal positions must both have shape [B, 3]")
    body_count = source_position.shape[0]
    if source_rotation.shape != (body_count, 3, 3) or target_rotation.shape != (body_count, 3, 3):
        raise ValueError("nominal rotations must both have shape [B, 3, 3]")
    if not np.isfinite(source_root_height) or source_root_height <= 0:
        raise ValueError("source root height must be positive and finite")
    if not np.isfinite(target_root_height) or target_root_height <= 0:
        raise ValueError("target root height must be positive and finite")
    for name, value in (("source", source_rotation), ("target", target_rotation)):
        if not np.all(np.isfinite(value)) or not np.allclose(value @ np.swapaxes(value, -1, -2), np.eye(3), atol=1e-5):
            raise ValueError(f"{name} nominal rotations must be orthonormal")
        if not np.allclose(np.linalg.det(value), 1.0, atol=1e-5):
            raise ValueError(f"{name} nominal rotations must have determinant +1")
    scale = float(target_root_height / source_root_height)
    source_inverse = np.swapaxes(source_rotation, -1, -2)
    position = np.einsum("bij,bj->bi", source_inverse, target_position - scale * source_position)
    rotation = source_inverse @ target_rotation
    return NominalAlignment(scale, jnp.asarray(position), jnp.asarray(rotation))


def zero_parameters(body_count: int, motion_count: int) -> ReactorParameters:
    if body_count <= 0 or motion_count <= 0:
        raise ValueError("body_count and motion_count must be positive")
    return ReactorParameters(
        jnp.zeros((body_count, 3)), jnp.zeros((body_count, 3)), jnp.zeros((motion_count,))
    )


def stack_nominal_alignments(alignments: tuple[NominalAlignment, ...]) -> NominalAlignment:
    """Collect per-clip nominal maps while retaining shared learned offsets."""
    if not alignments:
        raise ValueError("at least one nominal alignment is required")
    body_count = alignments[0].position.shape[0]
    if any(
        alignment.position.shape != (body_count, 3)
        or alignment.rotation.shape != (body_count, 3, 3)
        for alignment in alignments
    ):
        raise ValueError("nominal alignments must have the same ordered body correspondence")
    return NominalAlignment(
        jnp.asarray([alignment.scale for alignment in alignments]),
        jnp.stack([alignment.position for alignment in alignments]),
        jnp.stack([alignment.rotation for alignment in alignments]),
    )


def project_parameters(parameters: ReactorParameters, limits: ReactorLimits = ReactorLimits()) -> ReactorParameters:
    """Euclidean projection onto independent L2 balls and height intervals."""
    if min(limits.position, limits.orientation, limits.height) < 0:
        raise ValueError("ReActor limits must be nonnegative")

    def ball(values, radius):
        norm = jnp.linalg.norm(values, axis=-1, keepdims=True)
        return values * jnp.minimum(1.0, radius / jnp.maximum(norm, 1e-12))

    return ReactorParameters(
        ball(parameters.position, limits.position),
        ball(parameters.orientation, limits.orientation),
        jnp.clip(parameters.height, -limits.height, limits.height),
    )


def _skew(vector: jax.Array) -> jax.Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), axis=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def rotation_exp(vector: jax.Array) -> jax.Array:
    """SO(3) exponential with a stable zero-angle Taylor branch."""
    squared = jnp.sum(vector * vector, axis=-1, keepdims=True)
    theta = jnp.sqrt(jnp.maximum(squared, 1e-12))
    a = jnp.where(squared < 1e-8, 1.0 - squared / 6.0 + squared * squared / 120.0, jnp.sin(theta) / theta)
    b = jnp.where(
        squared < 1e-8,
        0.5 - squared / 24.0 + squared * squared / 720.0,
        (1.0 - jnp.cos(theta)) / (theta * theta),
    )
    skew = _skew(vector)
    return jnp.eye(3, dtype=vector.dtype) + a[..., None] * skew + b[..., None] * (skew @ skew)


def map_motion(
    source: BodyMotion,
    nominal: NominalAlignment,
    parameters: ReactorParameters,
    clip_ids: jax.Array,
    nominal_heights: jax.Array,
) -> BodyMotion:
    """Map source body pose and twist to robot reference (equations 10–13)."""
    if nominal.position.ndim == 2:
        nominal_position = nominal.position[None]
        nominal_rotation = nominal.rotation[None]
        scale = jnp.asarray(nominal.scale)
    else:
        nominal_position = nominal.position[clip_ids]
        nominal_rotation = nominal.rotation[clip_ids]
        scale = jnp.asarray(nominal.scale)[clip_ids, None, None]
    offset = jnp.einsum("nbij,bj->nbi", nominal_rotation, parameters.position) + nominal_position
    world_offset = jnp.einsum("nbij,nbj->nbi", source.rotation, offset)
    vertical = nominal_heights[clip_ids] + parameters.height[clip_ids]
    position = scale * source.position + world_offset + vertical[:, None, None] * jnp.array([0.0, 0.0, 1.0])
    rotation = source.rotation @ nominal_rotation @ rotation_exp(parameters.orientation)[None]
    velocity = scale * source.linear_velocity + jnp.cross(source.angular_velocity, world_offset)
    return BodyMotion(position, rotation, velocity, source.angular_velocity)


def mink_anchored_mocap_delta(
    mapped: BodyMotion, nominal_mapped: BodyMotion, base: BodyMotion,
) -> BodyMotion:
    """Keep Mink FK at zero offset while applying ReActor's source-frame delta."""
    # BVH/nominal rotations can be slightly non-orthogonal after float32
    # conversion. A transpose is then not the inverse: even zero parameters
    # would deform the Mink FK reference and fail Isaac's SO(3) validation.
    delta_rotation = jnp.linalg.solve(nominal_mapped.rotation, mapped.rotation)
    return BodyMotion(
        jnp.asarray(base.position) + mapped.position - jnp.asarray(nominal_mapped.position),
        jnp.asarray(base.rotation) @ delta_rotation,
        jnp.asarray(base.linear_velocity) + mapped.linear_velocity
        - jnp.asarray(nominal_mapped.linear_velocity),
        jnp.asarray(base.angular_velocity) + mapped.angular_velocity
        - jnp.asarray(nominal_mapped.angular_velocity),
    )


def outer_loss(
    parameters: ReactorParameters,
    source: BodyMotion,
    achieved: BodyMotion,
    nominal: NominalAlignment,
    clip_ids: jax.Array,
    nominal_heights: jax.Array,
    valid: jax.Array,
    weights: ReactorWeights = ReactorWeights(),
    target_transform: Callable[[ReactorParameters, BodyMotion, BodyMotion, jax.Array], BodyMotion] | None = None,
) -> jax.Array:
    """Reference tracking loss evaluated against a stopped simulated rollout."""
    target = map_motion(source, nominal, parameters, clip_ids, nominal_heights)
    if target_transform is not None:
        target = target_transform(parameters, source, target, clip_ids)
    achieved = jax.tree.map(jax.lax.stop_gradient, achieved)
    position_error = jnp.sum((target.position - achieved.position) ** 2, axis=-1)
    # Squared geodesic distance is stable around zero with atan2 of sine/cosine.
    relative = jnp.swapaxes(achieved.rotation, -1, -2) @ target.rotation
    vee = jnp.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    sine_squared = 0.25 * jnp.sum(vee * vee, axis=-1)
    sine = jnp.sqrt(jnp.maximum(sine_squared, 1e-12))
    cosine = 0.5 * (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0)
    orientation_error = jnp.where(
        (sine_squared < 1e-8) & (cosine > 0.0),
        sine_squared,
        jnp.arctan2(sine, cosine) ** 2,
    )
    velocity_error = jnp.sum((target.linear_velocity - achieved.linear_velocity) ** 2, axis=-1)
    angular_error = jnp.sum((target.angular_velocity - achieved.angular_velocity) ** 2, axis=-1)
    per_body = (
        weights.position * position_error
        + weights.orientation * orientation_error
        + weights.linear_velocity * velocity_error
        + weights.angular_velocity * angular_error
    )
    sample_mask = jnp.asarray(valid, dtype=per_body.dtype)
    return jnp.sum(per_body * sample_mask[:, None]) / jnp.maximum(jnp.sum(sample_mask) * per_body.shape[1], 1.0)


def outer_step(
    parameters: ReactorParameters,
    source: BodyMotion,
    achieved: BodyMotion,
    nominal: NominalAlignment,
    clip_ids: jax.Array,
    nominal_heights: jax.Array,
    valid: jax.Array,
    learning_rate: float,
    weights: ReactorWeights = ReactorWeights(),
    limits: ReactorLimits = ReactorLimits(),
    target_transform: Callable[[ReactorParameters, BodyMotion, BodyMotion, jax.Array], BodyMotion] | None = None,
) -> tuple[ReactorParameters, jax.Array]:
    """One projected semi-gradient step; the lower policy is not differentiated."""
    loss, gradient = jax.value_and_grad(outer_loss)(
        parameters, source, achieved, nominal, clip_ids, nominal_heights, valid, weights, target_transform
    )
    updated = jax.tree.map(lambda parameter, grad: parameter - learning_rate * grad, parameters, gradient)
    return project_parameters(updated, limits), loss
