"""Finite-step SAC actor response and a learned upper wrench-cost objective.

This is a truncated-unrolling hypergradient, not implicit differentiation.
The lower critic, temperature, batch statistics, sampled states and noise are
fixed within a meta update. All lower ACTOR parameters take K projected Adam
steps from their incoming optimizer state (or explicitly selected legacy SGD).
The upper cost critic is a learned physics surrogate;
its gradient is always subject to subsequent simulator validation.
"""
from dataclasses import dataclass
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from flash_chord.training.flash_sac.network import (
    Actor, Critic, deterministic_action, policy_sample_from_noise, project_unit_parameters,
)
from flash_chord.training.flash_sac.update import actor_loss
from flash_chord.retargeting.actor_response import unroll_adam_actor


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
    inner_optimizer: str = 'adam'
    inner_adam_eps_root: float = 1e-12
    upper_rate: float = 1e-3
    upper_steps: int = 4
    radius: float = .25
    regularization: float = .01
    discount: float = .99
    critic_rate: float = 3e-4
    critic_updates: int = 64
    trial_critic_updates: int = 32
    trial_mc_weight: float = 1.  # Complete rollouts supply an observed finite-horizon return.
    cost_scale: float = .01  # All critic targets/objectives share these dimensionless units.
    batch_size: int = 128
    target_rate: float = .01

    def __post_init__(self):
        if min(self.inner_steps, self.upper_steps, self.critic_updates,
               self.trial_critic_updates, self.batch_size) < 1:
            raise ValueError('positive update counts and batch size required')
        values = (self.inner_rate, self.upper_rate, self.radius, self.regularization,
                  self.critic_rate, self.discount, self.target_rate, self.trial_mc_weight,
                  self.cost_scale, self.inner_adam_eps_root)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('finite positive hypergradient settings required')
        if (not 0 < self.discount < 1 or not 0 < self.target_rate <= 1 or self.radius > 1
                or self.trial_mc_weight > 1):
            raise ValueError('invalid discount, target rate or normalized trust radius')
        if self.inner_optimizer not in ('adam', 'sgd'):
            raise ValueError('inner_optimizer must be adam or legacy sgd')


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

    def sample(self, rng, size, *, frame_range=None, exclude=None, source=None, generation=None,
               critic_valid=None):
        ids = np.arange(self.size)
        if exclude is not None:
            ids = np.setdiff1d(ids, exclude)
        if frame_range is not None:
            f = self.storage['frame'][ids]
            ids = ids[(f >= frame_range[0]) & (f < frame_range[1])]
        if source is not None:
            ids = ids[self.storage['source'][ids] == source]
        if generation is not None:
            ids = ids[self.storage['generation'][ids] == generation]
        if critic_valid is not None and 'critic_valid' in self.storage:
            ids = ids[self.storage['critic_valid'][ids] == critic_valid]
        if not len(ids):
            raise ValueError('no replay rows for requested window/split')
        ids = rng.choice(ids, size=size, replace=len(ids) < size)
        return {k: jnp.asarray(v[ids]) for k, v in self.storage.items()}, ids


def actor_policy_digest(learner):
    """Identify the exact inference policy responsible for measured trial data."""
    leaves, structure = jax.tree.flatten((learner.actor.params, learner.actor.batch_stats))
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        value = np.asarray(leaf)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


class TrialRollout:
    """Isolated on-policy trial evidence; never inserted into reference replay.

    Each add contains a contiguous physical step for the selected world IDs.
    Both reference and inference-policy identities are immutable. Only completed
    finite-horizon/terminated segments receive Monte Carlo labels; an unfinished
    collection tail is discarded, rather than incorrectly declared terminal.
    A rollout can contain multiple episodes when the environment resets.
    """
    def __init__(self, reference_key, policy_key):
        if not reference_key or not policy_key:
            raise ValueError('trial reference and policy identities required')
        self.reference_key, self.policy_key = reference_key, policy_key
        self._rows, self._active, self._completed = [], {}, []
        self._schema = None

    @property
    def size(self):
        return len(self._rows)

    def add(self, reference_key, policy_key, *, world_ids=None, **row):
        if reference_key != self.reference_key or policy_key != self.policy_key:
            raise ValueError('mixed reference or policy identities in trial feedback')
        required = {'observation', 'context', 'action', 'next_observation',
                    'next_context', 'cost', 'done', 'frame'}
        if not required <= row.keys():
            raise ValueError('incomplete trial transition schema')
        values = {k: np.array(v, copy=True) for k, v in row.items()}
        n = len(values['cost'])
        if not n or any(len(v) != n or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('finite equally sized trial transition fields required')
        if any(values[k].shape != (n,) for k in ('cost', 'done', 'frame')):
            raise ValueError('trial cost, done and frame must be scalar per world')
        if not np.isin(values['done'], (0, 1)).all():
            raise ValueError('trial done must be zero or one')
        schema = {k: v.shape[1:] for k, v in values.items()}
        if self._schema is not None and schema != self._schema:
            raise ValueError('trial transition schema changed')
        self._schema = schema
        worlds = np.arange(n) if world_ids is None else np.asarray(world_ids)
        if worlds.shape != (n,) or len(np.unique(worlds)) != n:
            raise ValueError('unique trial world IDs required')
        # Validate all rows before mutation, including continuity across steps.
        for i, world in enumerate(worlds):
            active = self._active.get(int(world), [])
            if active and values['frame'][i] != self._rows[active[-1]]['frame']+1:
                raise ValueError('trial feedback must contain consecutive physical frames')
        for i, world in enumerate(worlds):
            index = len(self._rows)
            item = {k: v[i].copy() for k, v in values.items()}
            self._rows.append(item)
            active = self._active.setdefault(int(world), [])
            active.append(index)
            if item['done']:
                self._completed.append(tuple(active))
                self._active[int(world)] = []

    def completed_batch(self, discount):
        if not 0 < discount < 1:
            raise ValueError('trial return discount must be between zero and one')
        selected, returns, trajectory_ids = [], [], []
        for trajectory_id, indices in enumerate(self._completed):
            accumulated = 0.
            trajectory_returns = []
            for index in reversed(indices):
                accumulated = float(self._rows[index]['cost'])+discount*accumulated
                trajectory_returns.append(accumulated)
            selected.extend(indices)
            returns.extend(reversed(trajectory_returns))
            trajectory_ids.extend([trajectory_id]*len(indices))
        if not selected:
            return {}, dict(trial_feedback_rows=self.size, trial_feedback_complete_rows=0,
                            trial_feedback_discarded_rows=self.size, trial_feedback_trajectories=0)
        result = {k: np.stack([self._rows[index][k] for index in selected])
                  for k in self._schema}
        result['measured_return'] = np.asarray(returns, np.float32)
        result['trajectory_id'] = np.asarray(trajectory_ids, np.int32)
        for value in result.values():
            value.setflags(write=False)
        return result, dict(trial_feedback_rows=self.size,
                            trial_feedback_complete_rows=len(selected),
                            trial_feedback_discarded_rows=self.size-len(selected),
                            trial_feedback_trajectories=len(self._completed))


class WrenchHypergradient:
    def __init__(self, training, observation_dim, context_dim, upper_context_dim, config, seed,
                 *, upper_action_dim=56):
        if upper_action_dim not in (12, 56):
            raise ValueError('upper action must contain 12 wrist or 56 full edit channels')
        self.upper_action_dim = upper_action_dim
        self.config, self.training = config, training
        self.lower_actor = Actor(action_dim=56, config=training.network, dtype=jnp.float32)
        lo, hi = training.reward_normalization.value_support
        self.lower_critic = Critic(lo, hi, training.network, dtype=jnp.float32)
        self.cost_model = WrenchCostCritic()
        self.upper_model = UpperActor(dimensions=upper_action_dim, radius=config.radius)
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
        self._trial_critic_update = jax.jit(self._update_trial_critic)

    def _action(self, params, stats, observation):
        distribution = self.lower_actor.apply({'params': params, 'batch_stats': stats},
                                               observation, training=False)
        return deterministic_action(distribution, action_transform=self.training.network.action_transform,
                                    action_scale=self.training.network.action_scale)

    def _update_critic(self, params, target, state, lower_params, lower_stats, batch):
        next_action = self._action(lower_params, lower_stats, batch['next_observation'])
        future = self.cost_model.apply({'params': target}, batch['next_observation'],
                                      batch['next_context'], next_action).max(axis=0)
        # Only the finite demonstration's end stops this full-horizon return.
        # Early sampling cutoffs are excluded by fit_critic (stale next phase).
        target_cost = jax.lax.stop_gradient(self.config.cost_scale*batch['cost'] +
                                           self.config.discount*(1-batch['done'])*future)
        def loss(p):
            pred = self.cost_model.apply({'params': p}, batch['observation'], batch['context'], batch['action'])
            return jnp.mean(optax.huber_loss(pred, target_cost[None], delta=1.))
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = self.cost_opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        target = optax.incremental_update(params, target, self.config.target_rate)
        return params, target, state, value

    def fit_critic(self, replay, learner, rng, *, updates=None, source=None, generation=None):
        # Hold out complete rows from every optimization batch in this round.
        count = self.config.critic_updates if updates is None else updates
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('positive integer critic update count required')
        eligible = np.ones(replay.size, bool)
        if source is not None:
            eligible &= replay.storage['source'][:replay.size] == source
        if generation is not None:
            eligible &= replay.storage['generation'][:replay.size] == generation
        valid = eligible & replay.storage.get('critic_valid', np.ones(replay.size, bool))[:replay.size]
        valid_count = int(np.count_nonzero(valid))
        cutoff_count = int(np.count_nonzero(eligible & ~valid))
        if valid_count < 2:
            return dict(critic_skipped=1, replay_rows=replay.size,
                        critic_updates=0, critic_valid_rows=valid_count, critic_cutoff_rows=cutoff_count)
        # Replay is unchanged throughout this fit. Build its eligible split once,
        # preserving the ascending row order and RNG draws used by sample().
        # Refiltering/sorting the whole replay for each small batch needlessly
        # makes a critic burst scale with replay capacity times update count.
        valid_ids = np.flatnonzero(valid)
        held_ids = rng.choice(valid_ids,
            size=min(self.config.batch_size, max(1, valid_count//5)), replace=False)
        train_ids = np.setdiff1d(valid_ids, held_ids, assume_unique=True)
        def batch(ids):
            return {key: jnp.asarray(value[ids]) for key, value in replay.storage.items()}
        validation = batch(held_ids)
        losses = []
        for _ in range(count):
            ids = rng.choice(train_ids, size=self.config.batch_size,
                             replace=len(train_ids) < self.config.batch_size)
            self.cost_params, self.target_params, self.cost_state, value = self._critic_update(
                self.cost_params, self.target_params, self.cost_state,
                learner.actor.params, learner.actor.batch_stats, batch(ids))
            # Keep scalar losses on device until the burst finishes instead of
            # synchronizing the GPU after every optimizer update.
            losses.append(value)
        prediction = self.cost_model.apply({'params': self.cost_params}, validation['observation'],
                                          validation['context'], validation['action']).max(axis=0)
        next_action = self._action(learner.actor.params, learner.actor.batch_stats, validation['next_observation'])
        future = self.cost_model.apply({'params': self.target_params}, validation['next_observation'],
                                      validation['next_context'], next_action).max(axis=0)
        target = self.config.cost_scale*validation['cost'] + self.config.discount*(1-validation['done'])*future
        losses = np.asarray(jax.device_get(jnp.stack(losses)), dtype=np.float64)
        metrics = dict(critic_train_loss=float(np.mean(losses)),
                       critic_heldout_td_mae=float(jnp.mean(abs(prediction-target))),
                       critic_cost_scale=self.config.cost_scale, replay_rows=replay.size,
                       critic_skipped=0, critic_updates=count, critic_valid_rows=valid_count,
                       critic_cutoff_rows=cutoff_count)
        metrics.update(critic_train_loss_normalized=metrics['critic_train_loss'],
            critic_heldout_td_mae_normalized=metrics['critic_heldout_td_mae'],
            critic_heldout_td_mae_raw=metrics['critic_heldout_td_mae']/self.config.cost_scale)
        if not all(np.isfinite(x) for x in metrics.values()):
            raise FloatingPointError('nonfinite wrench critic fit')
        return metrics

    def score_trial(self, trial, *, params=None, frame_range=None):
        """Score complete real trajectories before fitting, with explicit weights.

        This checks Q values on observed state/actions, not the counterfactual
        fixed-state hypergradient. MC targets follow each trial's own policy.
        A shared frozen critic permits paired comparisons without fitting the
        control first and accidentally testing the candidate with newer weights.
        """
        data, counts = trial.completed_batch(self.config.discount)
        if not data:
            return dict(scored_rows=0, complete_trajectories=0), {}
        weights = self.cost_params if params is None else params
        twins = np.asarray(self.cost_model.apply({'params': weights},
            data['observation'], data['context'], data['action']))
        predicted = twins.max(axis=0)
        measured = self.config.cost_scale*data['measured_return']
        mask = np.ones(len(predicted), bool)
        if frame_range is not None:
            lo, hi = frame_range
            if lo >= hi:
                raise ValueError('nonempty half-open frame range required')
            mask &= (data['frame'] >= lo) & (data['frame'] < hi)
        if not mask.any():
            return dict(scored_rows=0, complete_trajectories=counts['trial_feedback_trajectories']), {}
        stats = dict(scored_rows=int(mask.sum()),
            complete_trajectories=counts['trial_feedback_trajectories'],
            prediction_mean=float(predicted[mask].mean()), measured_return_mean=float(measured[mask].mean()),
            return_mae=float(np.abs(predicted[mask]-measured[mask]).mean()),
            twin_disagreement_mean=float(np.abs(twins[0, mask]-twins[1, mask]).mean()),
            cost_scale=self.config.cost_scale, discount=self.config.discount)
        arrays = dict(data, predicted_return=predicted, measured_return_normalized=measured,
                      twin_predictions=twins, scoring_mask=mask)
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError('nonfinite frozen critic diagnostic')
        return stats, arrays

    def compare_trials(self, candidate, control, *, params, frame_range):
        """Compare mean discounted support returns on identical physical frames."""
        after, a = self.score_trial(candidate, params=params, frame_range=frame_range)
        before, b = self.score_trial(control, params=params, frame_range=frame_range)
        if not after['scored_rows'] or not before['scored_rows']:
            return dict(available=False, reason='no_complete_rows_in_window')
        af = np.sort(a['frame'][a['scoring_mask']])
        bf = np.sort(b['frame'][b['scoring_mask']])
        if not np.array_equal(af, bf):
            return dict(available=False, reason='physical_frame_coverage_differs')
        predicted_gain = before['prediction_mean']-after['prediction_mean']
        measured_gain = before['measured_return_mean']-after['measured_return_mean']
        return dict(available=True, frame_range=list(frame_range), candidate=after, control=before,
            predicted_support_gain=predicted_gain, measured_support_gain=measured_gain,
            gain_absolute_error=abs(predicted_gain-measured_gain),
            sign_agreement=bool(np.sign(predicted_gain) == np.sign(measured_gain)),
            candidate_policy_sha256=candidate.policy_key, control_policy_sha256=control.policy_key,
            interpretation='same frozen critic on realized trajectories; not a physical hypergradient test')

    def _update_trial_critic(self, params, target, state, lower_params, lower_stats, batch):
        next_action = self._action(lower_params, lower_stats, batch['next_observation'])
        future = self.cost_model.apply({'params': target}, batch['next_observation'],
                                      batch['next_context'], next_action).max(axis=0)
        td_target = self.config.cost_scale*batch['cost']+self.config.discount*(1-batch['done'])*future
        measured = self.config.cost_scale*batch['measured_return']
        weight = self.config.trial_mc_weight
        target_cost = jax.lax.stop_gradient((1-weight)*td_target+weight*measured)
        def loss(p):
            prediction = self.cost_model.apply({'params': p}, batch['observation'],
                                               batch['context'], batch['action'])
            # MC labels connect a rejected edit's observed outcome directly to
            # its prediction, without treating the physical gate's mixed-unit
            # score as a wrench reward or bootstrapping from another policy.
            # Blend targets BEFORE robustification. Averaging separately clipped
            # TD and MC losses can cancel their opposite saturated gradients,
            # leaving a grossly underestimated measured return uncorrected.
            # Complete rollout calibration defaults to the observed MC target.
            return jnp.mean(optax.huber_loss(prediction, target_cost[None], delta=1.))
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = self.cost_opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        target = optax.incremental_update(params, target, self.config.target_rate)
        return params, target, state, value

    def fit_trial_critic(self, trial, learner, rng):
        """Learn from accepted AND rejected branch rollouts in their own context.

        Complete rollouts default to direct measured-return calibration. An
        optional TD blend uses precisely the branch policy that generated the
        evidence. The shared critic remains an approximate policy-value model:
        it is not conditioned on network weights and cannot provide a guaranteed
        value for an arbitrary subsequent policy. Held-out row errors below are
        calibration diagnostics, not independent-episode generalization tests.
        Unsuffixed prediction/error metrics are normalized critic units; explicit
        *_raw aliases recover the unscaled dimensionless wrench-support return.
        """
        if actor_policy_digest(learner) != trial.policy_key:
            raise ValueError('trial feedback policy does not match bootstrap learner')
        data, metrics = trial.completed_batch(self.config.discount)
        n = metrics['trial_feedback_complete_rows']
        if n < 2:
            return dict(metrics, trial_critic_skipped=1)
        held_ids = rng.choice(n, size=min(self.config.batch_size, max(1, n//5)), replace=False)
        train_ids = np.setdiff1d(np.arange(n), held_ids)
        def batch(ids):
            return {k: jnp.asarray(value[ids]) for k, value in data.items()}
        validation = batch(held_ids)
        def prediction():
            return self.cost_model.apply({'params': self.cost_params},
                validation['observation'], validation['context'], validation['action']).max(axis=0)
        before = prediction()
        before_params = self.cost_params
        losses = []
        for _ in range(self.config.trial_critic_updates):
            ids = rng.choice(train_ids, size=self.config.batch_size,
                             replace=len(train_ids) < self.config.batch_size)
            self.cost_params, self.target_params, self.cost_state, value = self._trial_critic_update(
                self.cost_params, self.target_params, self.cost_state,
                learner.actor.params, learner.actor.batch_stats, batch(ids))
            losses.append(float(value))
        after = prediction()
        measured = self.config.cost_scale*validation['measured_return']
        delta = sum(float(jnp.sum((a-b)**2)) for a, b in
                    zip(jax.tree.leaves(self.cost_params), jax.tree.leaves(before_params)))**.5
        metrics.update(trial_critic_skipped=0, trial_critic_train_loss=float(np.mean(losses)),
            trial_critic_cost_scale=self.config.cost_scale,
            trial_critic_updates=self.config.trial_critic_updates, trial_critic_training_rows=len(train_ids),
            trial_critic_heldout_rows=len(held_ids),
            trial_critic_heldout_return_mae_before=float(jnp.mean(abs(before-measured))),
            trial_critic_heldout_return_mae_after=float(jnp.mean(abs(after-measured))),
            trial_critic_prediction_before=float(jnp.mean(before)),
            trial_critic_prediction_after=float(jnp.mean(after)),
            trial_critic_measured_return=float(jnp.mean(measured)),
            trial_critic_parameter_delta_l2=delta)
        metrics['trial_critic_train_loss_normalized'] = metrics['trial_critic_train_loss']
        for name in ('heldout_return_mae_before', 'heldout_return_mae_after',
                     'prediction_before', 'prediction_after', 'measured_return'):
            key = 'trial_critic_'+name
            metrics[key+'_normalized'] = metrics[key]
            metrics[key+'_raw'] = metrics[key]/self.config.cost_scale
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('nonfinite measured trial critic fit')
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
            return jnp.mean(cost) + config.cost_scale*config.regularization*jnp.mean(theta**2)
        return lower, outer

    def propose(self, context, learner, batches, validation, relabel):
        lower, outer = self.objectives(learner, relabel, validation)
        cfg = self.config
        project = lambda p: project_unit_parameters(p, self.training.network.projection_epsilon)
        def adam_gradient(theta):
            def composed(t):
                adapted = unroll_adam_actor(t, learner.actor, batches, lower,
                    self.training, steps=cfg.inner_steps, eps_root=cfg.inner_adam_eps_root)
                return outer(t, adapted), adapted
            (value, adapted), total = jax.value_and_grad(composed, has_aux=True)(theta)
            fixed = jax.tree.map(jax.lax.stop_gradient, adapted)
            direct = jax.grad(lambda t: outer(t, fixed))(theta)
            return value, total, direct, total-direct, adapted
        evaluate = jax.jit(adam_gradient if cfg.inner_optimizer == 'adam' else
            lambda theta: gradient_parts(theta, learner.actor.params, batches,
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
        metrics = dict(
            predicted_cost=float(value), predicted_noedit_cost=float(control_value),
            predicted_wrench_return_gain=float(control_value-value), gradient_norm=float(jnp.linalg.norm(total)),
            direct_gradient_norm=float(jnp.linalg.norm(direct)),
            response_gradient_norm=float(jnp.linalg.norm(response)),
            response_gradient_fraction=float(jnp.linalg.norm(response)/jnp.maximum(jnp.linalg.norm(total),1e-12)),
            inner_optimizer=cfg.inner_optimizer,
            inner_optimizer_state_preserved=cfg.inner_optimizer == 'adam',
            inner_adam_eps_root=cfg.inner_adam_eps_root,
            critic_cost_scale=cfg.cost_scale, inner_steps=cfg.inner_steps, upper_actor_updates=cfg.upper_steps)
        for name in ('predicted_cost', 'predicted_noedit_cost', 'predicted_wrench_return_gain',
                     'gradient_norm', 'direct_gradient_norm', 'response_gradient_norm'):
            metrics[name+'_normalized'] = metrics[name]
            metrics[name+'_raw'] = metrics[name]/cfg.cost_scale
        return np.asarray(theta), params, control_params, metrics
