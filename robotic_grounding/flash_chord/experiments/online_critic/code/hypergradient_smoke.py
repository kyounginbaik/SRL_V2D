"""GPU integration evidence for the learned wrench hypergradient engine."""
import json
from pathlib import Path

import numpy as np


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, (float, np.ndarray)):
        return bool(np.isfinite(value).all())
    return True


def validate_feedback_and_budget(row, config):
    """Reject a disconnected critic loop or a supposedly fast unbounded trial."""
    budget = row['adaptation_budget']
    assert budget['mode'] == 'bounded_window'
    assert budget['world_count'] == config['worlds']
    assert budget['maximum_vector_steps'] == config['adaptation_maximum_vector_steps']
    expected = budget['environment_steps']
    assert len(expected) >= 2 and expected == row['adaptation_checkpoints']
    assert all(0 < count <= config['adaptation_maximum_transitions_per_branch'] for count in expected)
    assert all(count == vector*config['worlds'] for count, vector in zip(expected, budget['vector_steps']))
    feedback = row['critic_feedback']
    assert feedback and all(_finite(record) for record in feedback)
    required = {('baseline', 0)} | {('noedit', count) for count in expected}
    for candidate in row['candidates']:
        if 'gradient_norm' not in candidate:
            continue
        branch = f'window_{candidate["window"]["window_id"]}'
        required.update((branch, checkpoint['environment_steps'])
                        for checkpoint in candidate['adaptation']['checkpoints'])
    indexed = {(record['branch'], record['environment_steps']): record for record in feedback}
    assert len(indexed) == len(feedback), 'Duplicate measured-feedback checkpoint'
    assert required == indexed.keys(), 'Missing baseline/candidate/control measured-feedback checkpoint'
    for record in feedback:
        assert record['trial_feedback_rows'] > 0
        assert record['trial_feedback_complete_rows'] > 0, 'No complete physical rollout labels'
        assert not record['trial_critic_skipped'], 'Measured trial failed to train the upper critic'
        assert record['trial_critic_updates'] > 0
        assert record['trial_critic_parameter_delta_l2'] > 0, 'Trial feedback did not change the upper critic'
        assert np.isclose(record['trial_critic_cost_scale'], .01), 'Trial critic used inconsistent cost units'
    return dict(critic_feedback_checkpoints=len(feedback),
                critic_feedback_complete_rows=sum(record['trial_feedback_complete_rows'] for record in feedback),
                critic_feedback_parameter_delta_l2_max=max(record['trial_critic_parameter_delta_l2'] for record in feedback))


def validate_current_proposal(row, candidate):
    """A fresh baseline and preserved Adam state are part of the response contract."""
    assert row['outer_validation_generation'] == row['upper_round'], 'Stale outer validation generation'
    assert candidate['validation_generation'] == row['upper_round'], 'Stale candidate validation generation'
    assert candidate['inner_optimizer'] == 'adam', 'Proposal used a different inner optimizer'
    if 'inner_optimizer_state_preserved' in candidate:
        assert candidate['inner_optimizer_state_preserved'], 'Incoming Adam state was not preserved'
    assert np.isclose(candidate['critic_cost_scale'], .01), 'Unexpected upper cost scale'


def validate_finite_acceptance(row):
    """Publish only a physically admissible, confirmed reference/controller pair.

    Checkpoint stability is diagnostic: a finite training response may improve
    the task while its learning curve is still moving.
    """
    confirmation = row['confirmation']
    assert isinstance(confirmation['attempted'], bool)
    assert isinstance(confirmation['passed'], bool)
    assert confirmation['attempted'] == bool(confirmation['attempts']), 'Missing frozen confirmation attempts'
    assert len(confirmation['attempts']) <= 2, 'Frozen confirmation exceeded its two-candidate cap'
    attempts = {(check['window_id'], check['environment_steps']): check
                for check in confirmation['attempts']}
    assert len(attempts) == len(confirmation['attempts']), 'Duplicate frozen confirmation attempt'
    accepted = []
    for candidate in row['candidates']:
        if 'gradient_norm' not in candidate:
            assert not candidate['accepted'], 'Skipped candidate was published'
            continue
        adaptation = candidate['adaptation']
        assert adaptation['acceptance_mode'] == 'finite_response', 'Unexpected adaptation acceptance mode'
        assert [checkpoint['environment_steps'] for checkpoint in adaptation['checkpoints']] == \
            row['adaptation_checkpoints'], 'Finite response did not test all matched budgets'
        budget = adaptation['selected_environment_steps']
        selected = [checkpoint for checkpoint in adaptation['checkpoints']
                    if checkpoint['environment_steps'] == budget]
        assert len(selected) == 1, 'Selected checkpoint was not measured exactly once'
        for branch in ('candidate', 'control'):
            assert selected[0][branch]['training']['environment_steps'] == budget, 'Unmatched selected training budgets'
        if candidate['accepted']:
            accepted.append(candidate)
            assert candidate['physical_passed_at_selected_checkpoint'], 'Published candidate failed physical checks'
            assert not candidate['rejection_reasons'], 'Rejected candidate was published'
            check = candidate['confirmation']
            assert check['attempted'] and check['passed'], 'Published candidate lacks frozen confirmation'
            assert check['environment_steps'] == budget, 'Confirmation used a different adaptation checkpoint'
            attempt = attempts.get((candidate['window']['window_id'], budget), {})
            assert attempt.get('attempted') and attempt.get('passed'), 'Published candidate missing from round confirmation'
    assert len(accepted) == int(row['accepted']), 'Published candidate count does not match round acceptance'
    assert confirmation['passed'] == bool(accepted), 'Round confirmation disagrees with publication'
    if accepted:
        assert confirmation['attempted'], 'Accepted round did not attempt confirmation'


def validate(run, config, manifest, rows):
    """Acceptance is not required; real training and complete finite evidence are."""
    from flax.serialization import msgpack_restore
    from safetensors.numpy import load_file

    train = Path(run)/'training'
    assert config['engine'] == manifest['engine'] == 'wrench_hypergradient'
    expected_upper_dim = 12 if config.get('upper_edit_scope') == 'wrists_only' else 56
    assert manifest['upper_action_dimension'] == config['upper_action_dimension'] == expected_upper_dim
    assert manifest['nominal_candidates_per_window'] == 1
    assert manifest['adaptation_worlds'] == config['worlds']
    assert manifest['adaptation']['acceptance_mode'] == config['adaptation_acceptance_mode'] == 'finite_response'
    assert manifest['hypergradient']['inner_optimizer'] == config['proposal_inner_optimizer'] == 'adam'
    assert np.isclose(manifest['hypergradient']['cost_scale'], config['upper_cost_scale'])
    assert np.isclose(config['upper_cost_scale'], .01)
    assert np.isclose(manifest['hypergradient']['trial_mc_weight'], 1.)
    assert len(rows) >= 2 and all(row['engine'] == 'wrench_hypergradient' for row in rows)
    assert manifest['critic_online_interval_vector_steps'] == 16
    assert manifest['critic_online_updates'] == 4
    online = [json.loads(line) for line in (train/'online_critic_updates.jsonl').read_text().splitlines()]
    assert online and all(_finite(record) for record in online)
    rounds = {row['upper_round']: row for row in rows}
    for record in online:
        assert record['critic_skipped'] == 0 and record['critic_updates'] == 4
        assert record['critic_valid_rows'] >= 2*manifest['hypergradient']['batch_size']
        assert record['generation'] in rounds
        assert record['step'] > rounds[record['generation']]['step']
    assert any(rows[0]['step'] < record['step'] < rows[1]['step'] for record in online), \
        'No online critic fitting between the first two real upper rounds'
    measured = []
    feedback_checks = []
    for row in rows:
        assert _finite(row)
        assert row['adaptation_transitions'] > 0, 'No full lower adaptation was exercised'
        feedback_checks.append(validate_feedback_and_budget(row, config))
        validate_finite_acceptance(row)
        assert len(row['windows']) <= 4
        assert all(window['candidate_count'] == 1 for window in row['windows'])
        assert row['gradient_estimator'] == 'short_actor_adam_unrolling_proposal_then_bounded_adaptation_validation'
        assert row['baseline_metrics'] and row['baseline_critic']['trial_feedback_complete_rows'] > 0
        assert not row['baseline_critic']['trial_critic_skipped']
        assert np.isclose(row['baseline_critic']['trial_critic_cost_scale'], .01)
        failure = row['failure_context']
        assert failure['detector'] == 'fixed_start_nominal' and failure['baseline_worlds'] == 1
        tested = [candidate for candidate in row['candidates'] if 'gradient_norm' in candidate]
        assert tested, 'All windows skipped: upper proposal was not exercised'
        for candidate in tested:
            validate_current_proposal(row, candidate)
            assert np.asarray(candidate['theta']).shape == (expected_upper_dim,)
            assert _finite(candidate)
            for key in ('gradient_norm', 'direct_gradient_norm', 'response_gradient_norm'):
                assert candidate[key] >= 0
            comparisons = candidate['adaptation']['checkpoints']
            assert len(comparisons) >= 2, 'Matched longer training comparison was not exercised'
            for comparison in comparisons:
                assert comparison['environment_steps'] <= config['adaptation_maximum_transitions_per_branch']
                for branch in ('candidate', 'control'):
                    report = comparison[branch]['training']
                    assert report['actor_updates'] > 0 and report['critic_updates'] > 0
                    assert report['actor_parameter_delta_l2'] > 0
                    assert report['critic_parameter_delta_l2'] > 0
        measured.extend(tested)
        search = train/'search_batches'/f'step_{row["step"]}'
        assert (search/'selected_reference.npz').is_file()
        assert _finite(msgpack_restore((search/'upper_state.msgpack').read_bytes()))
    progress = [json.loads(line) for line in (train/'adaptation_progress.jsonl').read_text().splitlines()]
    assert any(row['branch'] == 'noedit' for row in progress)
    assert any(row['branch'].startswith('window_') for row in progress)
    assert all(_finite(row) for row in progress)
    checkpoints = list(train.glob('policy_*.safetensors'))
    assert checkpoints and all(_finite(load_file(str(path))) for path in checkpoints)
    audits = [json.loads(line) for line in (train/'initial_state_checks.jsonl').read_text().splitlines()]
    assert len({row['initial_state_sha256'] for row in audits}) == 1
    assert all(row['frame'] == 0 and row['joint_velocity_max_error'] == 0
               and row['joint_position_max_error'] <= 1e-6 for row in audits)
    contexts = {row['context'] for row in audits}
    assert 'training_start' in contexts and 'adaptation_noedit' in contexts
    assert any(context.startswith('adaptation_window_') for context in contexts)
    best = json.loads((train/'best_demo.json').read_text())
    assert Path(best['policy']).is_file() and Path(best['reference']).is_file()
    return dict(engine='wrench_hypergradient', physics_proposals=len(measured),
                online_critic_bursts=len(online),
                online_critic_optimizer_updates=sum(record['critic_updates'] for record in online),
                adaptation_transitions=sum(row['adaptation_transitions'] for row in rows),
                accepted=sum(row['accepted'] for row in rows), searches=len(rows),
                maximum_adaptation_vector_steps=config['adaptation_maximum_vector_steps'],
                critic_feedback_checkpoints=sum(check['critic_feedback_checkpoints'] for check in feedback_checks),
                critic_feedback_complete_rows=sum(check['critic_feedback_complete_rows'] for check in feedback_checks),
                critic_feedback_parameter_delta_l2_max=max(check['critic_feedback_parameter_delta_l2_max'] for check in feedback_checks))
