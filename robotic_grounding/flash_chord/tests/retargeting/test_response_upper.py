import numpy as np
import pytest

from flash_chord.retargeting.response_upper import ResponseUpperFlashSAC


def test_real_flashsac_learns_solved_cost_and_refreshes_stage_replay():
    import jax
    upper = ResponseUpperFlashSAC(43, 8, updates=4)
    obs = np.arange(8*12, dtype=np.float32).reshape(8,12)/100
    actions = upper.propose(obs)
    assert actions.shape == (8,56)
    np.testing.assert_array_equal(actions[0], 0)
    with pytest.raises(RuntimeError, match='previous'):
        upper.propose(obs)
    costs = np.array([3.,2.,4.,1.,7.,0.,10.,2.])
    report = upper.learn(costs)
    assert report['upper_optimizer_updates_this_round'] == 4
    assert report['upper_actor_updates_this_round'] == 2
    assert report['upper_actor_parameter_max_change'] > 0
    assert report['upper_critic_parameter_max_change'] > 0
    assert report['upper_replay_size'] == 8
    expected = np.clip(costs[0]-costs, -5, 5)
    np.testing.assert_allclose(upper.state.replay.storage.discounted_objective_terms[:8,0], expected)
    np.testing.assert_allclose(upper.state.replay.storage.action[:8], actions)
    actor_before = jax.tree.map(lambda x: np.array(x), upper.state.actor.params)
    upper.propose(obs+.1)
    assert int(upper.state.replay.size) == 0
    for before, after in zip(jax.tree.leaves(actor_before), jax.tree.leaves(upper.state.actor.params)):
        np.testing.assert_array_equal(before, after)
    second = upper.learn(np.linspace(0,2,8))
    assert second['upper_optimizer_updates'] == 8
    assert second['upper_replay_size'] == 8
    assert second['upper_actor_parameter_max_change'] > 0


def test_invalid_costs_fail_before_upper_update():
    upper = ResponseUpperFlashSAC(44,8,updates=2)
    upper.propose(np.ones((8,3),np.float32))
    with pytest.raises(ValueError, match='Finite solved-response'):
        upper.learn([0.,1.,np.nan,0.,0.,0.,0.,0.])
    assert int(upper.state.global_update_step) == 0


def test_no_update_without_candidate_batch():
    upper = ResponseUpperFlashSAC(44,8)
    with pytest.raises(RuntimeError, match='No pending'):
        upper.learn(np.zeros(8))
