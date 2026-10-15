"""Numerical derivatives and production-network checks for the new upper path."""
from types import SimpleNamespace as NS
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.retargeting.wrench_hypergradient import (
    HypergradientConfig, ReferenceReplay, TrialRollout, WrenchHypergradient,
    actor_policy_digest, gradient_parts, unroll_actor,
)


def test_unrolling_matches_finite_difference_and_includes_learning_response():
    # Closed-form response after K steps on .5*(phi-2*theta)^2.
    # The physical objective has NO direct theta term. Detaching the lower
    # response therefore stalls; the full hypergradient must still improve it.
    theta = jnp.array([.1, -.2], dtype=jnp.float32)
    initial = jnp.zeros(2)
    batches = {'offset': jnp.zeros((3, 2))}
    lower = lambda p, t, b: .5*jnp.sum((p-2*t-b['offset'])**2)
    outer = lambda t, p: jnp.sum((p-jnp.array([1., -.5]))**2)
    value, total, direct, response, adapted = gradient_parts(
        theta, initial, batches, lower, outer, steps=3, rate=.2)
    factor = 2*(1-.8**3)
    expected = 2*(factor*np.asarray(theta)-np.array([1., -.5]))*factor
    np.testing.assert_allclose(total, expected, rtol=2e-6)
    np.testing.assert_array_equal(direct, np.zeros(2))
    np.testing.assert_allclose(response, total)
    def objective(t):
        return float(outer(t, unroll_actor(t, initial, batches, lower, steps=3, rate=.2)))
    eps = .001
    numerical = [(objective(theta+eps*jnp.eye(2)[i])-objective(theta-eps*jnp.eye(2)[i]))/(2*eps)
                 for i in range(2)]
    np.testing.assert_allclose(total, numerical, atol=1e-4, rtol=1e-4)
    assert objective(theta-.1*total) < float(value)


def test_reference_replay_copies_labels_rejects_stale_and_splits_sources():
    replay = ReferenceReplay(5)
    original = np.arange(4, dtype=np.float32)
    replay.add(7, frame=np.arange(4), source=np.array([0, 1, 0, 1]), cost=original)
    original[:] = -100
    sample, _ = replay.sample(np.random.default_rng(1), 10, source=1)
    assert set(np.asarray(sample['cost'])) == {1., 3.}
    with pytest.raises(ValueError, match='stale'):
        replay.add(8, frame=[0], source=[0], cost=[1.])
    replay.clear(8)
    assert replay.size == 0
    replay.add(8, frame=[8, 9], source=[0, 0], cost=[2., 3.])
    with pytest.raises(ValueError, match='no replay'):
        replay.sample(np.random.default_rng(1), 1, frame_range=(0, 3))


def make_network_case():
    from flash_chord.training.flash_sac.config import TrainingConfig, NetworkConfig
    from flash_chord.training.flash_sac.network import Actor, Critic, project_unit_parameters
    config = TrainingConfig(world_count=4, total_environment_steps=3000000,
        network=NetworkConfig(actor_head_mode='residual', actor_hidden_dim=16, critic_hidden_dim=16,
                              actor_blocks=1, critic_blocks=1))
    obs = jax.random.normal(jax.random.PRNGKey(2), (8, 12))
    context = jnp.ones((8, 3))
    actor = Actor(56, config.network)
    variables = actor.init(jax.random.PRNGKey(3), obs, training=False)
    lo, hi = config.reward_normalization.value_support
    critic = Critic(lo, hi, config.network)
    cv = critic.init(jax.random.PRNGKey(4), jnp.concatenate((obs, context), -1), jnp.zeros((8, 56)), training=False)
    learner = NS(actor=NS(params=project_unit_parameters(variables['params']), batch_stats=variables['batch_stats']),
        critic=NS(params=project_unit_parameters(cv['params']), batch_stats=cv['batch_stats']),
        temperature=NS(log_temperature=jnp.array(-2.)))
    batch = dict(observation=obs, context=context, action=jnp.zeros((8, 56)),
                 frame=jnp.arange(8), noise=jax.random.normal(jax.random.PRNGKey(5), (8, 56)))
    return config, learner, batch


def test_real_flashsac_actor_unroll_updates_upper_and_has_nonzero_response_gradient():
    config, learner, batch = make_network_case()
    settings = HypergradientConfig(inner_steps=2, inner_rate=.01, upper_steps=1,
                                   critic_updates=2, batch_size=8)
    engine = WrenchHypergradient(config, 12, 3, 7, settings, 12)
    # Known differentiable observation contract; production uses native FK.
    relabel = lambda obs, frame, theta: obs.at[:, :3].add(theta[:3])
    batches = jax.tree.map(lambda x: jnp.stack([x, x]), batch)
    before = jax.tree.map(np.array, engine.upper_params)
    theta, params, control, metrics = engine.propose(jnp.ones(7), learner, batches, batch, relabel)
    assert metrics['gradient_norm'] > 1e-8
    assert metrics['response_gradient_norm'] > 1e-8
    assert np.linalg.norm(theta) > 0
    assert abs(theta).max() <= settings.radius
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(engine.upper_params)))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(control), jax.tree.leaves(params)))


def test_critic_trains_on_measured_costs_and_bootstraps_terminal_correctly():
    config, learner, batch = make_network_case()
    engine = WrenchHypergradient(config, 12, 3, 7,
        HypergradientConfig(critic_updates=16, batch_size=8), 11)
    replay = ReferenceReplay(64)
    replay.add(1, observation=np.tile(batch['observation'], (4, 1)),
        context=np.ones((32, 3), np.float32), action=np.zeros((32, 56), np.float32),
        next_observation=np.tile(batch['observation'], (4, 1)), next_context=np.ones((32, 3), np.float32),
        cost=np.full(32, 2., np.float32), done=np.ones(32, np.float32),
        frame=np.arange(32), source=np.zeros(32, np.int32))
    metrics = engine.fit_critic(replay, learner, np.random.default_rng(2))
    assert np.isfinite(metrics['critic_heldout_td_mae'])
    assert metrics['critic_heldout_td_mae'] < 1.5


def trial_rows(frame, cost, done, *, observation_dim=12, context_dim=3):
    n = len(frame)
    return dict(observation=np.zeros((n, observation_dim), np.float32),
                context=np.ones((n, context_dim), np.float32),
                action=np.zeros((n, 56), np.float32),
                next_observation=np.ones((n, observation_dim), np.float32),
                next_context=np.ones((n, context_dim), np.float32),
                frame=np.asarray(frame), cost=np.asarray(cost, np.float32),
                done=np.asarray(done, np.float32))


def test_trial_returns_are_terminal_bounded_copy_capture_and_drop_incomplete_tail():
    trial = TrialRollout('candidate-A', 'policy-A')
    first = trial_rows([0, 0], [1, 10], [0, 0])
    trial.add('candidate-A', 'policy-A', **first)
    first['cost'][:] = 999
    first['observation'][:] = 999
    trial.add('candidate-A', 'policy-A', **trial_rows([1, 1], [2, 20], [1, 0]))
    # World 0 has reset. Its final incomplete tail must NOT bootstrap/contaminate
    # either previous completed trajectory's observed return.
    trial.add('candidate-A', 'policy-A', **trial_rows([0, 2], [100, 30], [0, 1]))
    batch, metrics = trial.completed_batch(.9)
    np.testing.assert_allclose(batch['measured_return'], [2.8, 2, 52.3, 47, 30])
    np.testing.assert_array_equal(batch['trajectory_id'], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(batch['observation'], np.zeros((5, 12)))
    assert metrics == dict(trial_feedback_rows=6, trial_feedback_complete_rows=5,
                          trial_feedback_discarded_rows=1, trial_feedback_trajectories=2)
    with pytest.raises(ValueError):
        batch['cost'][0] = 1


@pytest.mark.parametrize('reference,policy', [('candidate-B', 'policy-A'), ('candidate-A', 'policy-B')])
def test_trial_rejects_mixed_references_and_policies(reference, policy):
    trial = TrialRollout('candidate-A', 'policy-A')
    with pytest.raises(ValueError, match='mixed reference or policy'):
        trial.add(reference, policy, **trial_rows([0], [1], [1]))
    assert trial.size == 0


def test_trial_rejects_capture_gaps_instead_of_fabricating_monte_carlo_return():
    trial = TrialRollout('candidate-A', 'policy-A')
    trial.add('candidate-A', 'policy-A', **trial_rows([0], [1], [0]))
    with pytest.raises(ValueError, match='consecutive physical frames'):
        trial.add('candidate-A', 'policy-A', **trial_rows([2], [1], [1]))
    assert trial.size == 1
    data, metrics = trial.completed_batch(.99)
    assert data == {}
    assert metrics['trial_feedback_complete_rows'] == 0


def test_bad_trial_improves_critic_calibration_without_touching_main_replay_or_lower_actor():
    config, learner, batch = make_network_case()
    engine = WrenchHypergradient(config, 12, 3, 7,
        HypergradientConfig(trial_critic_updates=32, batch_size=8), 11)
    policy_key = actor_policy_digest(learner)
    trial = TrialRollout('rejected-candidate', policy_key)
    row = trial_rows(np.zeros(32, int), np.full(32, 2.), np.ones(32))
    row['observation'] = np.tile(np.asarray(batch['observation']), (4, 1))
    trial.add('rejected-candidate', policy_key, **row)
    main = ReferenceReplay(64)
    main.add('accepted-reference', **trial_rows([0], [.1], [1]))
    main_copy = {key: value.copy() for key, value in main.storage.items()}
    learner_before = actor_policy_digest(learner)
    metrics = engine.fit_trial_critic(trial, learner, np.random.default_rng(6))
    assert metrics['trial_critic_parameter_delta_l2'] > 0
    assert metrics['trial_critic_prediction_after'] > metrics['trial_critic_prediction_before']
    assert metrics['trial_critic_heldout_return_mae_after'] < metrics['trial_critic_heldout_return_mae_before']
    assert metrics['trial_critic_measured_return'] == 2.
    assert metrics['trial_feedback_complete_rows'] == 32
    assert metrics['trial_critic_updates'] == 32
    assert actor_policy_digest(learner) == learner_before
    assert main.version == 'accepted-reference' and main.size == 1
    for key in main_copy:
        np.testing.assert_array_equal(main.storage[key], main_copy[key])
    mismatched = NS(actor=NS(params=jax.tree.map(lambda x: x+.1, learner.actor.params),
                             batch_stats=learner.actor.batch_stats))
    with pytest.raises(ValueError, match='does not match bootstrap learner'):
        engine.fit_trial_critic(trial, mismatched, np.random.default_rng(6))


def test_trial_fit_never_optimizes_heldout_rows():
    config, learner, _ = make_network_case()
    engine = WrenchHypergradient(config, 12, 3, 7,
        HypergradientConfig(trial_critic_updates=3, batch_size=8), 11)
    policy_key = actor_policy_digest(learner)
    trial = TrialRollout('candidate-A', policy_key)
    row = trial_rows(np.zeros(40, int), np.full(40, 2.), np.ones(40))
    row['observation'][:, 0] = np.arange(40)
    trial.add('candidate-A', policy_key, **row)
    # Reproduce only the holdout split; inspect what actual optimizer calls see.
    heldout = np.random.default_rng(17).choice(40, size=8, replace=False)
    optimized_ids = set()
    original = engine._trial_critic_update
    def capture(*args):
        optimized_ids.update(np.asarray(args[-1]['observation'][:, 0], dtype=int))
        return original(*args)
    engine._trial_critic_update = capture
    metrics = engine.fit_trial_critic(trial, learner, np.random.default_rng(17))
    assert optimized_ids
    assert optimized_ids.isdisjoint(heldout)
    assert metrics['trial_critic_training_rows'] == 32
    assert metrics['trial_critic_heldout_rows'] == 8


@pytest.mark.parametrize('kwargs', [{'inner_steps': 0}, {'discount': 1.}, {'radius': np.nan},
                                   {'trial_critic_updates': 0}, {'trial_mc_weight': 0.},
                                   {'trial_mc_weight': 1.1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        HypergradientConfig(**kwargs)
