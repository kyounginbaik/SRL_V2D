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


def validate(run, config, manifest, rows):
    """Acceptance is not required; real training and complete finite evidence are."""
    from flax.serialization import msgpack_restore
    from safetensors.numpy import load_file

    train = Path(run)/'training'
    assert config['engine'] == manifest['engine'] == 'wrench_hypergradient'
    assert manifest['upper_action_dimension'] == 56
    assert manifest['nominal_candidates_per_window'] == 1
    assert manifest['adaptation_worlds'] == config['worlds']
    assert rows and all(row['engine'] == 'wrench_hypergradient' for row in rows)
    measured = []
    for row in rows:
        assert _finite(row)
        assert row['adaptation_transitions'] > 0, 'No full lower adaptation was exercised'
        assert len(row['windows']) <= 4
        assert row['gradient_estimator'] == 'short_actor_unrolling_proposal_then_long_adaptation_validation'
        failure = row['failure_context']
        assert failure['detector'] == 'fixed_start_nominal' and failure['baseline_worlds'] == 1
        tested = [candidate for candidate in row['candidates'] if 'gradient_norm' in candidate]
        assert tested, 'All windows skipped: upper proposal was not exercised'
        for candidate in tested:
            assert np.asarray(candidate['theta']).shape == (56,)
            assert _finite(candidate)
            for key in ('gradient_norm', 'direct_gradient_norm', 'response_gradient_norm'):
                assert candidate[key] >= 0
            comparisons = candidate['adaptation']['checkpoints']
            assert len(comparisons) >= 2, 'Matched longer training comparison was not exercised'
            for comparison in comparisons:
                for branch in ('candidate', 'control'):
                    report = comparison[branch]['training']
                    assert report['actor_updates'] > 0 and report['critic_updates'] > 0
                    assert report['actor_parameter_delta_l2'] > 0
                    assert report['critic_parameter_delta_l2'] > 0
            if candidate['accepted']:
                assert candidate['adaptation']['stable']
                assert not candidate['rejection_reasons']
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
                adaptation_transitions=sum(row['adaptation_transitions'] for row in rows),
                accepted=sum(row['accepted'] for row in rows), searches=len(rows))
