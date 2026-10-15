"""Finite-step SAC actor response and a learned upper wrench-cost objective.

This is a truncated-unrolling hypergradient, not implicit differentiation.
The lower critic, temperature, batch statistics, sampled states and noise are
fixed within a meta update. All lower ACTOR parameters take K projected SGD
steps on the SAC loss. The upper cost critic is a learned physics surrogate;
its gradient is always subject to subsequent simulator validation.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from flash_chord.training.flash_sac.network import (
    Actor, Critic, deterministic_action, policy_sample_from_noise, project_unit_parameters,
)
from flash_chord.training.flash_sac.update import actor_loss


class UpperActor(nn.Module):
    dimensions: int = 56
    radius: float = .25

    @nn.compact
    def __call__(self, context):
        x = nn.LayerNorm()(context)
        for _ in range(2):
            x = nn.tanh(nn.Dense(64)(x))
        return self.radius * nn.tanh(nn.Dense(
            self.dimensions, kernel_init=nn.initializers.zeros_init())(x))


class WrenchCostCritic(nn.Module):
    """Twin discounted cost critics; max is pessimistic for a cost."""
    width: int = 128

    @nn.compact
    def __call__(self, observation, context, action):
        x = jnp.concatenate((observation, context, action), axis=-1)
        outputs = []
        for member in range(2):
            h = nn.LayerNorm(name=f'norm_{member}')(x)
            for layer in range(2):
                h = nn.silu(nn.Dense(self.width, name=f'h_{member}_{layer}')(h))
            outputs.append(nn.Dense(1, name=f'out_{member}')(h)[..., 0])
        return jnp.stack(outputs)


@dataclass(frozen=True)
class HypergradientConfig:
    inner_steps: int = 3
    inner_rate: float = 1e-4
    upper_rate: float = 1e-3
    upper_steps: int = 4
    radius: float = .25
    regularization: float = .01
    discount: float = .99
    critic_rate: float = 3e-4
    critic_updates: int = 64
    batch_size: int = 128
    target_rate: float = .01

    def __post_init__(self):
        if min(self.inner_steps, self.upper_steps, self.critic_updates, self.batch_size) < 1:
            raise ValueError('positive update counts and batch size required')
        values = (self.inner_rate, self.upper_rate, self.radius, self.regularization,
                  self.critic_rate, self.discount, self.target_rate)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('finite positive hypergradient settings required')
        if not 0 < self.discount < 1 or not 0 < self.target_rate <= 1 or self.radius > 1:
            raise ValueError('invalid discount, target rate or normalized trust radius')


def unroll_actor(theta, params, batches, loss, *, steps, rate, project=lambda x: x):
    """Differentiate through every gradient step, including mixed derivatives."""
    def body(p, i):
        batch = jax.tree.map(lambda x: x[i], batches)
        gradient = jax.grad(loss, argnums=0)(p, theta, batch)
        p = jax.tree.map(lambda value, g: value - rate*g, p, gradient)
        return project(p), None
    return jax.lax.scan(body, params, jnp.arange(steps))[0]


def gradient_parts(theta, initial, batches, loss, outer, *, steps, rate, project=lambda x: x):
    """Return total and frozen-response gradients with identical inner work."""
    def composed(t):
        adapted = unroll_actor(t, initial, batches, loss, steps=steps, rate=rate, project=project)
        return outer(t, adapted), adapted
    (value, adapted), total = jax.value_and_grad(composed, has_aux=True)(theta)
    fixed = jax.tree.map(jax.lax.stop_gradient, adapted)
    direct = jax.grad(lambda t: outer(t, fixed))(theta)
    return value, total, direct, total-direct, adapted


class ReferenceReplay:
    """Bounded, version-isolated observations and measured wrench-cost labels."""
    def __init__(self, capacity=32768):
        if capacity < 1:
            raise ValueError('positive replay capacity required')
        self.capacity, self.version, self.size, self.cursor = capacity, None, 0, 0
        self.storage = {}

    def clear(self, version):
        self.version, self.size, self.cursor = version, 0, 0

    def add(self, version, **row):
        if self.version is None:
            self.clear(version)
        if version != self.version:
            raise ValueError('stale reference data: clear replay before changing versions')
        values = {k: np.asarray(v) for k, v in row.items()}
        n = len(next(iter(values.values())))
        if not n or any(len(v) != n or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('finite equally sized transition fields required')
        if self.storage and values.keys() != self.storage.keys():
            raise ValueError('replay schema changed')
        for k, v in values.items():
            if k not in self.storage:
                self.storage[k] = np.empty((self.capacity, *v.shape[1:]), dtype=v.dtype)
            if v.shape[1:] != self.storage[k].shape[1:]:
                raise ValueError('replay field dimensions changed')
        if n > self.capacity:
            values = {k: v[-self.capacity:] for k, v in values.items()}
            n = self.capacity
        ids = (self.cursor + np.arange(n)) % self.capacity
        for k, v in values.items():
            self.storage[k][ids] = v
        self.cursor = (self.cursor+n) % self.capacity
        self.size = min(self.size+n, self.capacity)

    def sample(self, rng, size, *, frame_range=None, exclude=None, source=None):
        ids = np.arange(self.size)
        if exclude is not None:
            ids = np.setdiff1d(ids, exclude)
        if frame_range is not None:
            f = self.storage['frame'][ids]
            ids = ids[(f >= frame_range[0]) & (f < frame_range[1])]
        if source is not None:
            ids = ids[self.storage['source'][ids] == source]
        if not len(ids):
            raise ValueError('no replay rows for requested window/split')
        ids = rng.choice(ids, size=size, replace=len(ids) < size)
        return {k: jnp.asarray(v[ids]) for k, v in self.storage.items()}, ids


class WrenchHypergradient:
    def __init__(self, training, observation_dim, context_dim, upper_context_dim, config, seed):
        self.config, self.training = config, training
        self.lower_actor = Actor(action_dim=56, config=training.network, dtype=jnp.float32)
        lo, hi = training.reward_normalization.value_support
        self.lower_critic = Critic(lo, hi, training.network, dtype=jnp.float32)
        self.cost_model = WrenchCostCritic()
        self.upper_model = UpperActor(radius=config.radius)
        a, b = jax.random.split(jax.random.PRNGKey(seed))
        self.cost_params = self.cost_model.init(a, jnp.zeros((1, observation_dim)),
            jnp.zeros((1, context_dim)), jnp.zeros((1, 56)))['params']
        self.target_params = self.cost_params
        self.upper_params = self.upper_model.init(b, jnp.zeros((1, upper_context_dim)))['params']
        self.cost_opt = optax.adam(config.critic_rate)
        self.upper_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(config.upper_rate))
        self.cost_state = self.cost_opt.init(self.cost_params)
        self.upper_state = self.upper_opt.init(self.upper_params)
        self._critic_update = jax.jit(self._update_critic)

    def _action(self, params, stats, observation):
        distribution = self.lower_actor.apply({'params': params, 'batch_stats': stats},
                                               observation, training=False)
        return deterministic_action(distribution, action_transform=self.training.network.action_transform,
                                    action_scale=self.training.network.action_scale)

    def _update_critic(self, params, target, state, lower_params, lower_stats, batch):
        next_action = self._action(lower_params, lower_stats, batch['next_observation'])
        future = self.cost_model.apply({'params': target}, batch['next_observation'],
                                      batch['next_context'], next_action).max(axis=0)
        # Both true termination and the finite demonstration's end stop return.
        target_cost = jax.lax.stop_gradient(batch['cost'] + self.config.discount*(1-batch['done'])*future)
        def loss(p):
            pred = self.cost_model.apply({'params': p}, batch['observation'], batch['context'], batch['action'])
            return jnp.mean(optax.huber_loss(pred, target_cost[None], delta=1.))
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = self.cost_opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        target = optax.incremental_update(params, target, self.config.target_rate)
        return params, target, state, value

    def fit_critic(self, replay, learner, rng):
        # Hold out complete rows from every optimization batch in this round.
        validation, held_ids = replay.sample(rng, min(self.config.batch_size, max(1, replay.size//5)))
        losses = []
        for _ in range(self.config.critic_updates):
            batch, _ = replay.sample(rng, self.config.batch_size, exclude=held_ids)
            self.cost_params, self.target_params, self.cost_state, value = self._critic_update(
                self.cost_params, self.target_params, self.cost_state,
                learner.actor.params, learner.actor.batch_stats, batch)
            losses.append(float(value))
        prediction = self.cost_model.apply({'params': self.cost_params}, validation['observation'],
                                          validation['context'], validation['action']).max(axis=0)
        next_action = self._action(learner.actor.params, learner.actor.batch_stats, validation['next_observation'])
        future = self.cost_model.apply({'params': self.target_params}, validation['next_observation'],
                                      validation['next_context'], next_action).max(axis=0)
        target = validation['cost'] + self.config.discount*(1-validation['done'])*future
        metrics = dict(critic_train_loss=float(np.mean(losses)),
                       critic_heldout_td_mae=float(jnp.mean(abs(prediction-target))), replay_rows=replay.size)
        if not all(np.isfinite(x) for x in metrics.values()):
            raise FloatingPointError('nonfinite wrench critic fit')
        return metrics

    def objectives(self, learner, relabel, validation):
        config = self.config
        def lower(params, theta, batch):
            obs = relabel(batch['observation'], batch['frame'], theta)
            distribution = self.lower_actor.apply({'params': params, 'batch_stats': learner.actor.batch_stats},
                                                   obs, training=False)
            sample = policy_sample_from_noise(distribution, batch['noise'],
                                               action_transform=self.training.network.action_transform,
                                               action_scale=self.training.network.action_scale)
            q = self.lower_critic.apply({'params': learner.critic.params,
                'batch_stats': learner.critic.batch_stats}, jnp.concatenate((obs, batch['context']), axis=-1),
                sample.action, training=False).q.min(axis=0)
            return actor_loss(sample.log_probability, q, sample.action, batch['action'],
                jnp.exp(learner.temperature.log_temperature),
                behavior_cloning_coefficient=self.training.optimization.behavior_cloning_coefficient).loss
        def outer(theta, params):
            obs = relabel(validation['observation'], validation['frame'], theta)
            action = self._action(params, learner.actor.batch_stats, obs)
            cost = self.cost_model.apply({'params': self.cost_params}, obs, validation['context'], action).max(axis=0)
            return jnp.mean(cost) + config.regularization*jnp.mean(theta**2)
        return lower, outer

    def propose(self, context, learner, batches, validation, relabel):
        lower, outer = self.objectives(learner, relabel, validation)
        cfg = self.config
        project = lambda p: project_unit_parameters(p, self.training.network.projection_epsilon)
        evaluate = jax.jit(lambda theta: gradient_parts(theta, learner.actor.params, batches,
            lower, outer, steps=cfg.inner_steps, rate=cfg.inner_rate, project=project))
        context = jnp.asarray(context)[None]
        def proposal(p):
            raw = self.upper_model.apply({'params': p}, context)[0]
            return relabel.parameterize(raw, cfg.radius) if hasattr(relabel, 'parameterize') else raw
        for _ in range(cfg.upper_steps):
            theta, pullback = jax.vjp(proposal, self.upper_params)
            value, total, direct, response, _ = evaluate(theta)
            if not np.isfinite(np.asarray(total)).all() or not np.isfinite(float(value)):
                raise FloatingPointError('nonfinite unrolled reference gradient')
            gradients = pullback(total)[0]
            updates, self.upper_state = self.upper_opt.update(gradients, self.upper_state, self.upper_params)
            self.upper_params = optax.apply_updates(self.upper_params, updates)
        theta = proposal(self.upper_params)
        value, total, direct, response, params = evaluate(theta)
        control_value, _, _, _, control_params = evaluate(jnp.zeros_like(theta))
        return np.asarray(theta), params, control_params, dict(
            predicted_cost=float(value), predicted_noedit_cost=float(control_value),
            predicted_wrench_return_gain=float(control_value-value), gradient_norm=float(jnp.linalg.norm(total)),
            direct_gradient_norm=float(jnp.linalg.norm(direct)),
            response_gradient_norm=float(jnp.linalg.norm(response)),
            inner_steps=cfg.inner_steps, upper_actor_updates=cfg.upper_steps)
