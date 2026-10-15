from dataclasses import replace
from types import SimpleNamespace as NS
import json

import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor, support_cost
from flash_chord.retargeting.wrench_hypergradient import ReferenceReplay, HypergradientConfig
from flash_chord.retargeting.lower_adaptation import AdaptationConfig


def test_support_target_is_fixed_and_both_objects_contribute():
    target = np.ones((2, 2, 2, 3), np.float32)
    before = target.copy()
    actual = target.copy(); actual[1, :, 1] = 0.
    value = support_cost(actual, target)
    assert value[0] == 0 and value[1] > 0
    np.testing.assert_array_equal(target, before)


def test_hydra_recipe_selects_new_engine_and_preserves_runtime_reset_contract():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[2]/'src/flash_chord/configs'
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(config_name='train_flash_sac', overrides=[
            '+bilevel=wrench_hypergradient', 'reset.fixed_initial_state=true', 'reset.voc_decay_steps=0'])
    assert cfg.bilevel.engine == 'wrench_hypergradient'
    assert cfg.bilevel.eval_worlds == cfg.scene.world_count
    assert cfg.bilevel.hypergradient.inner_steps == 3
    assert list(cfg.bilevel.adaptation.checkpoints) == [50000,100000,200000]
    assert cfg.reset.fixed_initial_state and cfg.reset.voc_decay_steps == 0


def test_capture_materializes_pre_step_values_and_penalizes_termination(monkeypatch):
    import flash_chord.retargeting.wrench_hypergradient_editor as module
    monkeypatch.setattr(module, 'to_jax', lambda value, shape: jnp.asarray(value).reshape(shape))
    editor = WrenchHypergradientEditor.__new__(WrenchHypergradientEditor)
    editor.replay = ReferenceReplay(10); editor.version = 4; editor.wrench_tolerance = .1
    editor.binding = NS(num_frames=10)
    obs = np.ones((2, 3), np.float32)
    context = np.ones((2, 2), np.float32)
    action = np.ones((2, 56), np.float32)
    editor._before(None, obs, context, action, np.array([2, 3]), np.array([0, 1]), 0)
    obs[:] = 77; action[:] = 77; context[:] = 77
    objective = NS(num_hands=2, num_bodies=2, num_basis=3,
                   current_wrench_support=np.ones(24), wrench_support_target=np.ones(24))
    adapter = NS(env=NS(world_count=2, objective=objective))
    transition = NS(terminated=jnp.array([0, 1]), truncated=jnp.array([1, 0]),
                    terminal_observation=jnp.full((2, 3), 9.), terminal_critic_context=jnp.full((2, 2), 8.))
    editor._after(adapter, transition)
    np.testing.assert_array_equal(editor.replay.storage['observation'][:2], 1.)
    np.testing.assert_array_equal(editor.replay.storage['action'][:2], 1.)
    np.testing.assert_array_equal(editor.replay.storage['next_observation'][:2], 9.)
    np.testing.assert_array_equal(editor.replay.storage['done'][:2], 1.)
    assert editor.replay.storage['cost'][0] == 0
    assert editor.replay.storage['cost'][1] >= 20
    assert editor.pending is None


@pytest.mark.parametrize('improves', [False, True])
def test_publication_uses_measured_physics_and_commits_the_tested_pair(monkeypatch, tmp_path, improves):
    import flash_chord.retargeting.wrench_hypergradient_editor as module
    import flash_chord.retargeting.manipulation_bilevel as base
    import flash_chord.envs.rl as rl
    import flash_chord.training.environment as environment
    class FakeEnv:
        def __init__(self, *a, **kw): pass
        def reset(self): pass
        def capture_step(self): pass
    monkeypatch.setattr(rl, 'RLEnv', FakeEnv)
    monkeypatch.setattr(environment, 'WarpRLEnv', lambda env, **kw: env)
    monkeypatch.setattr(base, 'assert_device_binding', lambda *a: None)
    monkeypatch.setattr(module, 'command_digest', lambda *a: 'immutable')
    monkeypatch.setattr(module, 'save_snapshot', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'binding_for_q', lambda editor, q: NS(joint_q=q))
    monkeypatch.setattr(module.ReferenceRelabel, 'from_editor', lambda *a: NS(validate_zero=lambda *a: 0.))
    editor = WrenchHypergradientEditor.__new__(WrenchHypergradientEditor)
    editor.next_update=10; editor.interval=10; editor.upper_round=0; editor.validation_steps=0
    editor.training=NS(total_environment_steps=1000, seed=43)
    editor.training.replay=NS(batch_size=4)
    editor.adaptation_config=AdaptationConfig(checkpoints=(8,16,32), replay_capacity=32,
        replay_minimum_size=4, minimum_actor_updates=1, minimum_critic_updates=1, minimum_window_samples=1)
    editor.adaptation_transitions=0
    editor.scoring_region=dict(start_frame=0, stop_frame_exclusive=2)
    editor.hg_config=HypergradientConfig(inner_steps=1, batch_size=4)
    editor.hyper_rng=np.random.default_rng(2); editor.version=0
    editor.output=tmp_path/'references'; editor.output.mkdir()
    editor.command_sha256='immutable'; editor.metadata={}
    editor.replay=ReferenceReplay(1024); editor.replay.clear(0)
    editor.replay.add(0, observation=np.ones((400, 12), np.float32), context=np.zeros((400, 3), np.float32),
        action=np.zeros((400, 56), np.float32), frame=np.ones(400, np.int32), source=np.arange(400)%2)
    editor.binding=NS(joint_q=np.zeros((3, 58)), num_frames=3)
    original_binding=editor.binding
    editor.baseline=NS(joint_q=np.zeros((3, 58)))
    editor.projector=NS(layout=NS(hands=[], scalar_joints=NS(names=[])), qids=[])
    # dataclasses.replace must see a real dataclass, as in production scenes.
    from dataclasses import make_dataclass
    Scene=make_dataclass('Scene', [('robot_reference', object)])
    editor.setup=NS(scene=Scene(original_binding), reference=object())
    editor.window_contexts=[dict(window_id=0, start_seconds=0., end_seconds=.1)]
    editor.window_ids=np.zeros(4, int); editor.failure_context={'failed':True}
    editor.basis=np.array([[0.], [1.], [0.]])
    editor.select_candidate_context=lambda *a: None
    editor.observation=lambda *a: jnp.ones((4, 5))
    editor._archive_pair=lambda *a, **kw: {'current_first_failure_seconds': .1}
    editor._cost=lambda m, q: m['cost']
    editor._local_rejections=lambda *a: []
    editor.candidate_joint_q=lambda theta: np.ones((3, 58))*.001
    policy=NS(critic_context=jnp.zeros((1, 3)))
    candidate_policy=object(); control_policy=object()
    monkeypatch.setattr(module, 'install_learning', lambda learner, snapshot: candidate_policy)
    trace={'frame_per_object_position_m':np.zeros((2, 1, 2)),
           'frame_per_object_orientation_rad':np.zeros((2, 1, 2))}
    calls=[]
    def rollout(binding, learner, seed, **kw):
        calls.append((binding, learner))
        cost=(.5 if improves else 2.) if learner is candidate_policy else 1.
        result=({'cost':cost}, None, None)
        return (*result, trace) if kw.get('return_feedback') else result
    editor.rollout=rollout
    branch_calls=[]
    def branch(binding, learner, env, stage, seed, budgets, label, step):
        assert learner is policy  # Both branches start from the exact same learner.
        branch_calls.append(label)
        for budget in budgets:
            cost=(.5 if improves else 2.) if label != 'noedit' else 1.
            metrics=dict(cost=cost, local_per_object_position_mean_m=[.01,.02],
                local_per_object_orientation_mean_rad=[.1,.1],
                local_object_success_robust_fraction=.9, local_cws_tolerance_error=.1)
            report=dict(actor_updates=10, critic_updates=20, actor_parameter_delta_l2=.01,
                critic_parameter_delta_l2=.1, frame_samples=[20]*3, max_reference_frame=2)
            yield dict(cost=cost, metrics=metrics, training=report, environment_steps=budget,
                       snapshot='full_trained_state', feedback=trace)
    editor._adaptation_branch=branch
    editor.publish=lambda step: None
    # Always predict improvement: rejection must depend on actual simulation.
    diag=dict(predicted_wrench_return_gain=10., gradient_norm=1., direct_gradient_norm=.4, response_gradient_norm=.6)
    editor.hyper=NS(fit_critic=lambda *a: {'replay_rows':400},
        propose=lambda *a: (np.zeros(56), 'candidate', 'control', diag),
        upper_params={}, upper_state={}, cost_params={}, target_params={}, cost_state={})
    env=NS(world_count=4, env=NS(observation_strategy=object(), config={}))
    result=editor(env, policy, 10, NS(voc_scale=0.))
    assert editor.version == int(improves)
    assert result.learner is (candidate_policy if improves else policy)
    assert (result.environment is not None) == improves
    assert (editor.replay.size == 0) == improves
    assert (editor.binding is original_binding) != improves
    assert branch_calls == ['noedit', 'window_0']
    report=json.loads((tmp_path/'upper_updates.jsonl').read_text())
    assert report['accepted'] == improves
    assert report['candidates'][0]['predicted_wrench_return_gain'] > 0
    assert report['candidates'][0]['adaptation']['selected_environment_steps'] == 16
