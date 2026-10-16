"""Optimizer parity and independent finite differences for the actor response."""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.retargeting.actor_response import unroll_adam_actor
from flash_chord.training.flash_sac.config import TrainingConfig, NetworkConfig, OptimizationConfig
from flash_chord.training.flash_sac.learner import (
    OptimizedNetworkState, _create_adam, _apply_adam_if_finite, _scale_residual_mean_updates,
)
from flash_chord.training.flash_sac.network import project_unit_parameters
from flash_chord.training.flash_sac.update import learning_rate


def config(mean_head='unit_gain'):
    return TrainingConfig(world_count=4, total_environment_steps=3_000_000,
        network=NetworkConfig(actor_head_mode='residual', residual_mean_head=mean_head),
        optimization=OptimizationConfig(learning_rate_initial=.002, learning_rate_peak=.004,
            learning_rate_end=.001, learning_rate_schedule_updates=100,
            warmup_fraction=.05, decay_fraction=.8, actor_learning_rate_multiplier=1.7,
            residual_mean_learning_rate_multiplier=2.3))


def state_for(params, training, *, history=13, schedule=17):
    optimizer = _create_adam(training)
    moments = optimizer.init(params)
    for index in range(history):
        _, moments = optimizer.update(jax.tree.map(lambda p: -.2-jnp.ones_like(p)*.03*index, params),
                                      moments, params)
    return OptimizedNetworkState(params=params, batch_stats={'unchanged': jnp.asarray(2.)},
        optimizer_state=moments, schedule_step=jnp.asarray(schedule, jnp.int32))


@pytest.mark.parametrize('mean_head', ['unit_gain', 'zero_dense'])
def test_response_matches_production_adam_moments_schedule_scaling_and_projection(mean_head):
    training = config(mean_head)
    mean = ({'gain': jnp.asarray([.7, -.4])} if mean_head == 'unit_gain'
            else {'kernel': jnp.asarray([[.7, -.4], [.2, .8]])})
    params = dict(trunk={'unit_kernel': jnp.asarray([[.8, .6], [.6, -.8]])},
                  mean=mean, mean_bias=jnp.asarray([.1, -.2]))
    state = state_for(params, training)
    before = jax.tree.map(np.array, state)
    theta = jnp.asarray([.2, -.3])
    batches = {'offset': jnp.asarray([.1, -.2, .3])}
    def loss(p, t, batch):
        return sum(jnp.sum((leaf-t[0]-batch['offset'])**2)
                   for leaf in jax.tree.leaves(p))
    adapted = unroll_adam_actor(theta, state, batches, loss, training, steps=3)
    optimizer, expected, moments = _create_adam(training), params, state.optimizer_state
    transform = partial(_scale_residual_mean_updates,
        multiplier=training.optimization.residual_mean_learning_rate_multiplier, mean_head=mean_head)
    for index in range(3):
        gradient = jax.grad(loss)(expected, theta, {'offset': batches['offset'][index]})
        rate = learning_rate(state.schedule_step+index, training.optimization,
                             training.learning_rate_schedule_updates)
        rate *= training.optimization.actor_learning_rate_multiplier
        expected, moments = _apply_adam_if_finite(optimizer, expected, moments,
            gradient, rate, jnp.asarray(True), transform)
        expected = project_unit_parameters(expected, training.network.projection_epsilon)
    for actual, target in zip(jax.tree.leaves(adapted), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, target, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(adapted['trunk']['unit_kernel'], axis=0), 1., atol=1e-6)
    for actual, original in zip(jax.tree.leaves(state), jax.tree.leaves(before)):
        np.testing.assert_array_equal(actual, original)
    # A fresh optimizer is not a substitute for the incoming policy's moments.
    reset = state.replace(optimizer_state=optimizer.init(params))
    reset_result = unroll_adam_actor(theta, reset, batches, loss, training, steps=3)
    assert max(float(np.max(abs(a-b))) for a, b in
               zip(jax.tree.leaves(adapted), jax.tree.leaves(reset_result))) > 1e-4


def test_hypergradient_matches_central_difference_through_adam_state_and_mixed_derivatives():
    with jax.enable_x64():
        training = replace(config(), network=NetworkConfig(),
            optimization=replace(config().optimization, residual_mean_learning_rate_multiplier=1.))
        state = state_for({'weight': jnp.asarray([.3, -.2], jnp.float64)}, training)
        theta = jnp.asarray([.13, -.17], jnp.float64)
        batches = {'offset': jnp.asarray([[.01, -.03], [.04, .02], [-.02, .01]], jnp.float64)}
        lower = lambda p, t, b: .5*jnp.sum((p['weight']-2*t-b['offset'])**2)
        def objective(t):
            adapted = unroll_adam_actor(t, state, batches, lower, training, steps=3)
            return jnp.sum((adapted['weight']-jnp.asarray([1., -.6]))**2)
        derivative = jax.jit(jax.grad(objective))(theta)
        h = 1e-4
        numerical = [(float(objective(theta+h*jnp.eye(2)[i]))-
                      float(objective(theta-h*jnp.eye(2)[i])))/(2*h) for i in range(2)]
        assert np.isfinite(np.asarray(derivative)).all()
        assert np.linalg.norm(derivative) > 1e-5
        np.testing.assert_allclose(derivative, numerical, rtol=1e-5, atol=1e-8)


def test_positive_root_epsilon_keeps_zero_moment_hypergradient_finite():
    training = TrainingConfig(world_count=4, total_environment_steps=3_000_000)
    params = {'weight': jnp.zeros(2)}
    state = state_for(params, training, history=0, schedule=0)
    batches = {'offset': jnp.zeros((3, 2))}
    lower = lambda p, t, b: .5*jnp.sum((p['weight']-t-b['offset'])**2)
    def objective(t):
        return jnp.sum(unroll_adam_actor(t, state, batches, lower, training, steps=3)['weight'])
    assert np.isfinite(np.asarray(jax.grad(objective)(jnp.zeros(2)))).all()
    np.testing.assert_array_equal(unroll_adam_actor(jnp.zeros(2), state, batches, lower,
                                                   training, steps=3)['weight'], np.zeros(2))
    # The production zero-root Adam has a singular AD path at exactly zero;
    # this is why the response explicitly documents a regularized denominator.
    optimizer = _create_adam(training)
    def unsmoothed(t):
        updates, _ = optimizer.update({'weight': -t}, state.optimizer_state, params)
        return jnp.sum(updates['weight'])
    assert not np.isfinite(np.asarray(jax.grad(unsmoothed)(jnp.zeros(2)))).all()


@pytest.mark.parametrize('kwargs', [dict(steps=0), dict(steps=True), dict(steps=1, eps_root=0.),
    dict(steps=1, eps_root=float('nan')), dict(steps=4)])
def test_invalid_response_configuration_or_missing_update_batch_fails(kwargs):
    training = TrainingConfig(world_count=4, total_environment_steps=3_000_000)
    state = state_for({'weight': jnp.zeros(2)}, training)
    with pytest.raises(ValueError):
        unroll_adam_actor(jnp.zeros(2), state, {'offset': jnp.zeros((3, 2))},
            lambda p, t, b: jnp.sum(p['weight']**2), training, **kwargs)
