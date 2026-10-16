import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.retargeting.lower_adaptation import (
    AdaptationConfig, FlashSACAdaptation, learning_snapshot, install_learning,
    training_readiness, comparison_stable, comparison_instability_reasons,
)


def test_episode_aware_budgets_do_not_stop_before_the_grasp():
    settings = AdaptationConfig()
    assert settings.budgets(2048, 401) == (819200, 1638400, 3276800)
    assert settings.budgets(64, 401) == (50048, 100032, 200000)


def test_bounded_window_budgets_avoid_training_four_whole_horizons():
    settings = AdaptationConfig(budget_mode='bounded_window')
    region = dict(start_frame=20, stop_frame_exclusive=53)
    assert settings.budgets(2048, 335, region=region) == (141312, 206848, 272384)
    # An unrelated later tail must not multiply all candidate adaptation work.
    assert settings.budgets(2048, 1335, region=region) == settings.budgets(2048, 335, region=region)
    report = settings.budget_report(2048, 335, region=region)
    assert report['vector_steps'] == [69, 101, 133]
    assert report['first_checkpoint_can_reach_scoring_end']
    assert report['final_checkpoint_includes_requested_tail']
    assert not report['cap_reached']


def test_bounded_budgets_allow_replay_warmup_and_equal_complete_collections():
    settings = AdaptationConfig(budget_mode='bounded_window')
    region = dict(start_frame=20, stop_frame_exclusive=53)
    budgets = settings.budgets(64, 335, region=region)
    assert budgets == (9216, 11264, 13312)
    assert budgets[0] > settings.replay_minimum_size
    assert all(b % 64 == 0 for b in budgets)
    assert all(b-a == 32*64 for a, b in zip(budgets, budgets[1:]))
    with pytest.raises(ValueError, match='replay warmup'):
        settings.budgets(2, 335, region=region)


def test_late_window_exposes_cap_without_claiming_coverage_or_growing_budget():
    settings = AdaptationConfig(budget_mode='bounded_window')
    region = dict(start_frame=270, stop_frame_exclusive=300)
    report = settings.budget_report(2048, 335, region=region)
    assert report['vector_steps'] == [192, 224, 256]
    assert report['cap_reached']
    assert not report['final_checkpoint_can_reach_scoring_end']
    assert not report['final_checkpoint_includes_requested_tail']
    # A large aggregate sample count is not evidence that the needed frames
    # occurred. The existing readiness guard must reject this capped branch.
    training = dict(actor_updates=100, critic_updates=200, actor_parameter_delta_l2=.01,
        critic_parameter_delta_l2=.1, frame_samples=[2048]*256, max_reference_frame=255)
    assert training_readiness(training, settings, region) == ['insufficient_failure_region_experience']


def test_bounded_schedule_does_not_confuse_reachable_end_with_requested_tail():
    settings = AdaptationConfig(budget_mode='bounded_window')
    report = settings.budget_report(2048, 335,
        region=dict(start_frame=230, stop_frame_exclusive=250))
    assert report['final_checkpoint_can_reach_scoring_end']
    assert not report['final_checkpoint_includes_requested_tail']


@pytest.mark.parametrize('region', [None, dict(start_frame=3, stop_frame_exclusive=3),
    dict(start_frame=-1, stop_frame_exclusive=10), dict(start_frame=0, stop_frame_exclusive=336),
    dict(start_frame=0, stop_frame_exclusive=3.5)])
def test_bounded_schedule_requires_a_valid_shared_scoring_region(region):
    with pytest.raises(ValueError):
        AdaptationConfig(budget_mode='bounded_window').budgets(2048, 335, region=region)


@pytest.mark.parametrize('kwargs', [dict(checkpoints=[1]), dict(stable_comparisons=3),
    dict(minimum_reference_passes=[1, 1, 2]), dict(replay_capacity=1),
    dict(position_tolerance_m=float('nan')), dict(checkpoints=[1., 2., 2.5]),
    dict(budget_mode='unsupported'), dict(budget_mode='bounded_window', maximum_vector_steps=96),
    dict(checkpoint_stride_vector_steps=0)])
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        AdaptationConfig(**kwargs)


def checkpoint(cost, position=.02):
    return dict(cost=cost, metrics=dict(local_per_object_position_mean_m=[position, .01],
        local_per_object_orientation_mean_rad=[.1, .05], local_object_success_robust_fraction=.9,
        local_cws_tolerance_error=.1))


def test_stability_checks_both_objects_and_the_comparison():
    settings = AdaptationConfig()
    before = dict(candidate=checkpoint(.8), control=checkpoint(1.))
    after = dict(candidate=checkpoint(.79), control=checkpoint(1.01))
    assert comparison_stable(before, after, settings)
    after['candidate']['metrics']['local_per_object_position_mean_m'][1] = .05
    assert not comparison_stable(before, after, settings)
    assert comparison_instability_reasons(before, after, settings) == [
        'candidate_local_per_object_position_mean_m_not_stable']
    # Small scalar movement that reverses the winner is not stable evidence.
    before = dict(candidate=checkpoint(.99), control=checkpoint(1.))
    after = dict(candidate=checkpoint(1.01), control=checkpoint(1.))
    assert not comparison_stable(before, after, settings)
    assert comparison_instability_reasons(before, after, settings) == ['candidate_advantage_changed_sign']


def scored_checkpoint(cost, success, start=20, stop=50):
    result = checkpoint(cost)
    result['metrics'].update(local_object_success_robust_fraction=success,
        local_start_frame=start, local_stop_frame_exclusive=stop)
    return result


@pytest.mark.parametrize('side', ['candidate', 'control'])
def test_one_scored_frame_of_success_change_is_stable_but_two_are_not(side):
    settings = AdaptationConfig()
    before = dict(candidate=scored_checkpoint(.8, 20/30), control=scored_checkpoint(1., 20/30))
    after = dict(candidate=scored_checkpoint(.8, 20/30), control=scored_checkpoint(1., 20/30))
    after[side]['metrics']['local_object_success_robust_fraction'] = 21/30
    assert comparison_stable(before, after, settings)
    after[side]['metrics']['local_object_success_robust_fraction'] = 22/30
    assert comparison_instability_reasons(before, after, settings) == [
        side+'_local_object_success_robust_fraction_not_stable']


def test_success_resolution_does_not_relax_other_stability_guards():
    settings = AdaptationConfig()
    before = dict(candidate=scored_checkpoint(.8, 20/30), control=scored_checkpoint(1., 20/30))
    after = dict(candidate=scored_checkpoint(.8, 21/30), control=scored_checkpoint(1., 20/30))
    after['candidate']['metrics']['local_per_object_position_mean_m'][1] += .004
    assert comparison_instability_reasons(before, after, settings) == [
        'candidate_local_per_object_position_mean_m_not_stable']


def test_success_resolution_requires_identical_scored_frame_intervals():
    settings = AdaptationConfig()
    before = dict(candidate=scored_checkpoint(.8, .5), control=scored_checkpoint(1., .5))
    after = dict(candidate=scored_checkpoint(.8, .5, start=21, stop=51),
                 control=scored_checkpoint(1., .5))
    # Same number of frames is insufficient when the interval itself changes.
    assert comparison_instability_reasons(before, after, settings) == [
        'candidate_local_scoring_interval_changed']


@pytest.mark.parametrize('missing', ['local_start_frame', 'local_stop_frame_exclusive'])
def test_missing_interval_metadata_keeps_configured_success_tolerance(missing):
    settings = AdaptationConfig()
    before = dict(candidate=scored_checkpoint(.8, 20/30), control=scored_checkpoint(1., 20/30))
    after = dict(candidate=scored_checkpoint(.8, 21/30), control=scored_checkpoint(1., 20/30))
    del after['candidate']['metrics'][missing]
    assert comparison_instability_reasons(before, after, settings) == [
        'candidate_local_object_success_robust_fraction_not_stable']


def test_training_must_reach_the_failure_region_and_update_both_networks():
    report = dict(actor_updates=50, critic_updates=100, actor_parameter_delta_l2=.01,
        critic_parameter_delta_l2=.1, frame_samples=[2048]*26, max_reference_frame=25)
    region = dict(start_frame=95, stop_frame_exclusive=120)
    assert training_readiness(report, AdaptationConfig(), region) == ['insufficient_failure_region_experience']
    # An unchanged reference that keeps failing early remains a valid control.
    assert not training_readiness(report, AdaptationConfig(), region, require_coverage=False)
    report.update(frame_samples=[2048]*140, max_reference_frame=139)
    assert not training_readiness(report, AdaptationConfig(), region)
    report['actor_updates'] = 0
    assert 'insufficient_lower_actor_updates' in training_readiness(report, AdaptationConfig(), region)


class TinyEnv:
    """Action-dependent deterministic transitions, with real FlashSAC learning."""
    world_count=2; action_dim=2; observation_dim=3; frame_dt=.05
    objective_term_names=('tracking', 'termination')
    termination_cause_names=(); tracking_error_names=()
    critic_context_names=('applied_voc_scale', 'target_voc_scale', 'normalized_settling_progress')

    def set_curriculum(self, stage): self.stage=stage

    def reset(self):
        self.count=0
        self.initial_critic_context=jnp.tile(jnp.array([self.stage.voc_scale, self.stage.voc_scale, 0.]), (2,1))
        self.observation=jnp.zeros((2,3))
        return self.observation

    def frames(self): return np.full(2, self.count%8)

    def step(self, action):
        from flash_chord.training.environment import VectorStep
        self.count += 1
        next_observation=self.observation.at[:,:2].add(.1*action).at[:,2].set((self.count%8)/8.)
        done=jnp.full(2, self.count%8 == 0, jnp.int32)
        self.observation=jnp.where(done[:,None], 0., next_observation)
        objective=jnp.stack((1.-jnp.sum((action-.3)**2, axis=-1), -done.astype(jnp.float32)), axis=-1)
        return VectorStep(observation=self.observation, reward=None, terminated=jnp.zeros(2,jnp.int32),
            episode_return=None, episode_length=None, episode_reference_progress=None,
            truncated=done, objective_terms=objective, terminal_observation=next_observation,
            critic_context=self.initial_critic_context, terminal_critic_context=self.initial_critic_context)

    def transition_logging_inputs(self): return jnp.zeros(2), jnp.zeros((2,0))


def test_full_sac_adaptation_is_fresh_reproducible_and_does_not_mutate_incoming():
    from flash_chord.training.flash_sac.config import TrainingConfig, ReplayConfig, NetworkConfig
    from flash_chord.training.flash_sac.learner import create_learner
    from flash_chord.lifecycle.curriculum import CurriculumStage
    training=TrainingConfig(device='cpu', world_count=2, total_environment_steps=128,
        mixed_precision=False, updates_per_collection=2., replay=ReplayConfig(capacity=32, minimum_size=4, batch_size=4, n_step=1),
        network=NetworkConfig(actor_blocks=1, critic_blocks=1, actor_hidden_dim=8,
                              critic_hidden_dim=8, expansion=2, atom_count=11))
    stage=CurriculumStage(.25, {'tracking':1., 'termination':-1.})
    env=TinyEnv(); env.set_curriculum(stage); obs=env.reset()
    incoming, _=create_learner(training, obs, env.initial_critic_context, action_dim=2, objective_term_count=2)
    original=learning_snapshot(incoming)
    settings=AdaptationConfig(checkpoints=(16,32,64), replay_capacity=32, replay_minimum_size=4,
        minimum_actor_updates=1, minimum_critic_updates=1, minimum_window_samples=1)
    control=FlashSACAdaptation(env, training, settings, original, stage, 42)
    cache=control.cache_template()
    assert int(control.state.replay.size) == 0
    control.advance(16, env.frames)
    report=control.advance(32, env.frames)
    assert report['actor_updates'] > 3 and report['critic_updates'] > 3
    assert report['actor_parameter_delta_l2'] > 0 and report['critic_parameter_delta_l2'] > 0
    assert report['environment_steps'] == 32 and report['replay_rows'] == 32
    tested=learning_snapshot(control.state)
    # The incoming state remains readable, untouched, and undonated.
    for a,b in zip(jax.tree.leaves(original), jax.tree.leaves(learning_snapshot(incoming))):
        np.testing.assert_array_equal(a,b)
    assert int(incoming.environment_steps) == 0 and int(incoming.replay.size) == 0
    # Same policy/reference/start/RNG must yield exactly the same adaptation.
    other_env=TinyEnv()
    other=FlashSACAdaptation(other_env, training, settings, original, stage, 42, executables=cache)
    other.advance(32, other_env.frames)
    for a,b in zip(jax.tree.leaves(tested), jax.tree.leaves(learning_snapshot(other.state))):
        np.testing.assert_allclose(a,b, atol=1e-7, rtol=1e-6)
    # Publish the full tested learning state, never its side-training clock or replay.
    published=install_learning(incoming, tested)
    for field in ('actor','critic','target_critic','temperature'):
        for a,b in zip(jax.tree.leaves(getattr(published,field)), jax.tree.leaves(tested[field])):
            np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(published.reward_normalizer.discounted_return, 0.)
    assert int(published.environment_steps) == 0 and published.replay is incoming.replay
    assert int(published.global_update_step) > int(incoming.global_update_step)
    assert not np.array_equal(tested['temperature'].log_temperature, original['temperature'].log_temperature)
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(tested['target_critic']),
                                                    jax.tree.leaves(original['target_critic'])))


@pytest.mark.parametrize('costs,expected_budget,stable', [
    ([.8,.79,.78], 100000, True), ([.8,.6,.4], 200000, False), ([1.,.8,.79], 200000, True)])
def test_matched_comparison_extends_training_until_stable_or_rejects(costs, expected_budget, stable):
    from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor
    editor=WrenchHypergradientEditor.__new__(WrenchHypergradientEditor)
    editor.adaptation_config=AdaptationConfig(acceptance_mode='plateau')
    editor.scoring_region=dict(start_frame=1, stop_frame_exclusive=3)
    report=dict(actor_updates=50, critic_updates=100, actor_parameter_delta_l2=.01,
        critic_parameter_delta_l2=.1, frame_samples=[2048]*4, max_reference_frame=3)
    budgets=(50000,100000,200000)
    controls={b:dict(**checkpoint(1.1), training=report) for b in budgets}
    visited=[]
    def branch(*args):
        for b,c in zip(budgets,costs):
            visited.append(b)
            yield dict(**checkpoint(c), training=report, environment_steps=b, snapshot='trained', feedback={})
    editor._adaptation_branch=branch
    result, control, diagnostics=editor._matched_adaptation(None,None,None,None,43,budgets,controls,'test',0)
    assert result['environment_steps'] == expected_budget
    assert visited[-1] == expected_budget
    assert diagnostics['stable'] == stable
    assert bool(diagnostics['rejection_reasons']) != stable


def test_early_failing_control_does_not_veto_a_stable_improving_candidate():
    from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor
    editor=WrenchHypergradientEditor.__new__(WrenchHypergradientEditor)
    editor.adaptation_config=AdaptationConfig(acceptance_mode='plateau')
    editor.scoring_region=dict(start_frame=10, stop_frame_exclusive=13)
    trained=dict(actor_updates=50, critic_updates=100, actor_parameter_delta_l2=.01,
        critic_parameter_delta_l2=.1, frame_samples=[2048]*20, max_reference_frame=19)
    early=dict(trained, frame_samples=[2048]*5, max_reference_frame=4)
    budgets=(50000,100000,200000)
    controls={b:dict(**checkpoint(1.1), training=early) for b in budgets}
    def branch(*args):
        for b in budgets:
            yield dict(**checkpoint(.8), training=trained, environment_steps=b, snapshot='trained', feedback={})
    editor._adaptation_branch=branch
    _, _, decision=editor._matched_adaptation(None,None,None,None,43,budgets,controls,'test',0)
    assert decision['stable'] and not decision['rejection_reasons']
