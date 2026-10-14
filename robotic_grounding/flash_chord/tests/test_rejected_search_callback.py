"""Exercise the editor-to-training-runner contract on a rejected search."""
from types import SimpleNamespace as NS

import numpy as np


def test_rejected_search_does_not_request_environment_or_replay_reset(monkeypatch, tmp_path):
    import flash_chord.retargeting.parallel_editor as module

    metrics = dict(cost=1., all_errors_finite=True, full_horizon=True)
    worse = dict(metrics, cost=2.)
    monkeypatch.setattr(module.jax, 'block_until_ready', lambda value: value)
    monkeypatch.setattr(module, 'upper_cost', lambda metric, *args, **kwargs: metric['cost'])
    monkeypatch.setattr(module, 'passes_study_gate', lambda *args: True)
    monkeypatch.setattr(module, 'contact_proximity_gate', lambda *args: True)
    monkeypatch.setattr(module, 'rollout_candidates', lambda *args: ([worse, worse], None, None))
    monkeypatch.setattr(module, 'save_learner_checkpoint', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'command_digest', lambda reference: 'fixed')
    upper = NS(global_update_step=1)
    reference = NS(joint_q=np.zeros((3, 1)), joint_target=np.zeros((3, 1)), fps=20.)
    editor = NS(next_update=10, interval=10, upper_round=0, validation_steps=0,
                training=NS(total_environment_steps=1000, seed=43),
                rollout=lambda *args, **kwargs: (metrics, None, None, {}),
                mode='reactor_wrench', options={}, contact_proximity_weight=0.,
                binding=reference, baseline=reference, upper_worlds=2,
                observation=lambda *args: module.jnp.zeros((1, 1)),
                prepare_upper=lambda obs: (module.jnp.zeros((2, 1)), module.jnp.zeros((2, 0))),
                guided_contact_proposal=False, candidate_joint_q=lambda action: reference.joint_q.copy(),
                projector=NS(dids=[0], qids=[0], ball_joints=[]), upper=upper,
                executables=NS(consume_transition=lambda *args: (upper, None, None)),
                version=0, command_sha256='fixed', setup=NS(reference=object()),
                output=tmp_path/'references')
    env = object()
    replay = object()
    learner = NS(replay=replay)
    replacement = module.ParallelReferenceEditor.__call__(editor, env, learner, 10, None)
    assert replacement is None  # A non-None result triggers reset + clear_replay in the runner.
    assert editor.binding is reference and editor.version == 0
    assert learner.replay is replay
    assert editor.upper_round == 1
