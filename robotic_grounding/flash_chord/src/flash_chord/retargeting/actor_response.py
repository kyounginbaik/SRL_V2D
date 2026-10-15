"""Differentiable finite actor response using the incoming FlashSAC Adam state.

The replay loss supplied by the caller can freeze the critic, temperature,
statistics and sampled states. Matching the actor's optimizer does not turn
that surrogate into differentiation through complete FlashSAC training.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import optax

from flash_chord.training.flash_sac.learner import (
    _apply_adam_if_finite, _scale_residual_mean_updates,
)
from flash_chord.training.flash_sac.network import project_unit_parameters
from flash_chord.training.flash_sac.update import gradients_are_finite, learning_rate


def unroll_adam_actor(theta, actor_state, batches, loss, training, *, steps, eps_root=1e-12):
    """Return actor parameters after K differentiable, projected Adam updates.

    ``loss(params, theta, batch)`` is scalar. Every batch leaf has its update
    index on axis zero. Incoming parameters, Adam moments/count and the actor's
    independent schedule clock are read without mutation. Each temporary update
    uses production betas, epsilon, learning-rate schedule/multipliers, residual
    mean scaling, finite-update behavior and parameter projection.

    Production Adam uses sqrt(v_hat) + epsilon. Differentiating through sqrt at
    zero moments can yield NaNs even for an inactive coordinate. This response
    instead uses sqrt(v_hat + eps_root) + epsilon, with a positive configurable
    1e-12 default. This is a smooth approximation to production Adam, not an
    exactly identical optimizer at tiny second moments. No simulator steps,
    critic/temperature updates or running-statistic updates occur here.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('positive integer actor response steps required')
    if not math.isfinite(eps_root) or eps_root <= 0:
        raise ValueError('finite positive actor response Adam eps_root required')
    leaves = jax.tree.leaves(batches)
    if not leaves or any(value.ndim < 1 or value.shape[0] < steps for value in leaves):
        raise ValueError('actor response batches must provide every requested update')
    optimization = training.optimization
    optimizer = optax.scale_by_adam(b1=optimization.adam_beta1,
        b2=optimization.adam_beta2, eps=optimization.adam_epsilon, eps_root=eps_root)
    transform = None
    if (training.network.actor_head_mode == 'residual'
            and optimization.residual_mean_learning_rate_multiplier != 1.):
        transform = partial(_scale_residual_mean_updates,
            multiplier=optimization.residual_mean_learning_rate_multiplier,
            mean_head=training.network.residual_mean_head)

    def body(carry, index):
        params, moments, schedule_step = carry
        batch = jax.tree.map(lambda value: value[index], batches)
        value, gradients = jax.value_and_grad(loss)(params, theta, batch)
        finite = jnp.isfinite(value) & gradients_are_finite(gradients)
        rate = learning_rate(schedule_step, optimization, training.learning_rate_schedule_updates)
        rate *= optimization.actor_learning_rate_multiplier
        params, moments = _apply_adam_if_finite(optimizer, params, moments,
            gradients, rate, finite, transform)
        params = project_unit_parameters(params, training.network.projection_epsilon)
        # Production advances this clock on every attempted actor update, even
        # if its nonfinite gradient causes parameters and moments to be retained.
        return (params, moments, schedule_step+1), None

    initial = (actor_state.params, actor_state.optimizer_state, actor_state.schedule_step)
    (params, _, _), _ = jax.lax.scan(body, initial, jnp.arange(steps))
    return params
