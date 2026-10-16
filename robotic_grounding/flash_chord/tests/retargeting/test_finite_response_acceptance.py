"""Acceptance regressions for short matched lower responses, without physics.

The rollout results are controlled; the production selection, physical guards,
learning snapshot installation, and inference-policy identity checks run here.
"""
from types import SimpleNamespace as NS

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.retargeting.lower_adaptation import AdaptationConfig, learning_snapshot
from flash_chord.retargeting.wrench_hypergradient_editor import WrenchHypergradientEditor


def metrics(cost=10.):
    return dict(cost=cost, all_errors_finite=True, full_horizon=True,
        local_start_frame=1, local_stop_frame_exclusive=4, protected_prefix_frames=1,
        tail_frames=6, prefix_position_mean_m=[.01,.01], prefix_position_p95_m=[.01,.01],
        prefix_orientation_mean_rad=[.1,.1], prefix_orientation_p95_rad=[.1,.1],
        per_object_position_mean_m=[.02,.02], per_object_orientation_mean_rad=[.1,.1],
        tail_position_mean_m=[.02,.02], tail_orientation_mean_rad=[.1,.1],
        unstable_world_fraction=0., saved_prefix_rejection_reasons=[],
        local_per_object_position_mean_m=[.02,.02],
        local_per_object_orientation_mean_rad=[.1,.1],
        local_object_success_robust_fraction=1., local_cws_tolerance_error=.1)


def checkpoint(budget, cost):
    return dict(environment_steps=budget, cost=cost, metrics=metrics(cost),
        training=dict(actor_updates=64, critic_updates=128, actor_parameter_delta_l2=.1,
            critic_parameter_delta_l2=.2, frame_samples=[128]*10, max_reference_frame=9),
        snapshot={'which_checkpoint': np.array([budget])}, feedback={'checkpoint': budget})


def editor_for(rows):
    editor=WrenchHypergradientEditor.__new__(WrenchHypergradientEditor)
    editor.adaptation_config=AdaptationConfig()
    editor.scoring_region=dict(start_frame=1, stop_frame_exclusive=4)
    editor.options={}; editor.fixed_start=True
    visited=[]
    def branch(*args):
        try:
            for row in rows:
                visited.append(row['environment_steps'])
                yield row
        finally:
            visited.append('closed')
    editor._adaptation_branch=branch
    return editor, visited


def compare(editor, controls, baseline_cost=10.):
    return editor._matched_adaptation(object(), object(), object(), object(), 43,
        tuple(controls), controls, 'window_0', 500000,
        baseline=metrics(baseline_cost), baseline_cost=baseline_cost)


def test_best_earlier_checkpoint_survives_later_regression_and_unstable_curves():
    rows=[checkpoint(100,5.), checkpoint(200,6.), checkpoint(300,7.)]
    rows[-1]['metrics']['saved_prefix_rejection_reasons']=['earlier_failure_than_saved_demo']
    controls={b:checkpoint(b,c) for b,c in [(100,9.),(200,8.),(300,9.5)]}
    editor,visited=editor_for(rows)
    selected,control,report=compare(editor,controls)
    assert selected is rows[0] and control is controls[100]
    np.testing.assert_array_equal(selected['snapshot']['which_checkpoint'],[100])
    assert visited == [100,200,300,'closed']
    assert not report['stable']
    assert report['rejection_reasons'] == []
    assert report['physical_passed_at_selected_checkpoint']
    assert report['selected_checkpoint_index'] == 0
    assert report['stop_reason'] == 'finite_budget_complete'
    assert report['checkpoints'][0]['instability_reasons'] == ['no_previous_ready_checkpoint']
    assert 'earlier_failure_than_saved_demo' in report['checkpoints'][2]['physical_rejection_reasons']


def test_checkpoint_competes_with_its_own_equal_budget_control():
    rows=[checkpoint(100,1.9),checkpoint(200,5.),checkpoint(300,6.)]
    controls={b:checkpoint(b,c) for b,c in [(100,1.),(200,8.),(300,7.)]}
    editor,_=editor_for(rows)
    selected,control,report=compare(editor,controls)
    # Lowest absolute cost loses to the control trained for those same 100 steps.
    assert selected is rows[1] and control is controls[200]
    assert not report['checkpoints'][0]['finite_response_eligible']
    assert report['checkpoints'][1]['finite_response_eligible']


@pytest.mark.parametrize('veto', ['coverage','saved_prefix','object_tail','lower_updates'])
def test_improvement_still_requires_experience_and_physical_guards(veto):
    rows=[checkpoint(b,5.) for b in (100,200,300)]
    controls={b:checkpoint(b,9.) for b in (100,200,300)}
    for row in rows:
        if veto=='coverage':
            row['training'].update(frame_samples=[128],max_reference_frame=0)
        elif veto=='lower_updates':
            row['training']['actor_updates']=0
        elif veto=='saved_prefix':
            row['metrics']['saved_prefix_rejection_reasons']=['earlier_failure_than_saved_demo']
        else:
            row['metrics']['tail_position_mean_m'][1]=.2
    editor,_=editor_for(rows)
    _,_,report=compare(editor,controls)
    assert report['rejection_reasons']
    assert not report['physical_passed_at_selected_checkpoint']
    assert not any(row['finite_response_eligible'] for row in report['checkpoints'])


def test_unchanged_controller_need_not_reach_region_that_candidate_reaches():
    rows=[checkpoint(b,5.) for b in (100,200,300)]
    controls={b:checkpoint(b,9.) for b in (100,200,300)}
    for row in controls.values():
        row['training'].update(frame_samples=[128],max_reference_frame=0)
    editor,_=editor_for(rows)
    _,_,report=compare(editor,controls)
    assert report['rejection_reasons'] == []
    assert report['physical_passed_at_selected_checkpoint']


@pytest.mark.parametrize('candidate, expected', [(9.96,False),(9.94,True),(.0995,False),(.098,True)])
def test_absolute_and_relative_margin_reject_near_ties(candidate,expected):
    comparator=10. if candidate>1 else .1
    row=checkpoint(100,candidate);control=checkpoint(100,comparator)
    editor,_=editor_for([])
    reasons=editor._physical_pair_rejections(row,control,metrics(comparator),comparator)
    assert (not reasons) == expected


@pytest.mark.parametrize('field', ['candidate','control','baseline'])
@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_nonfinite_comparison_cost_is_never_admissible(field,bad):
    row=checkpoint(100,5.);control=checkpoint(100,9.);baseline_cost=10.
    if field=='candidate':row['cost']=bad
    elif field=='control':control['cost']=bad
    else:baseline_cost=bad
    editor,_=editor_for([])
    assert editor._physical_pair_rejections(row,control,metrics(),baseline_cost)


@struct.dataclass
class TinyActor:
    params: object
    batch_stats: object
    optimizer_state: object


@struct.dataclass
class TinyNormalizer:
    discounted_return: object
    scale: object


@struct.dataclass
class TinyLearner:
    actor: object
    critic: object
    target_critic: object
    temperature: object
    reward_normalizer: object
    loss_scale: object
    global_update_step: object
    environment_steps: object


def learner(marker):
    return TinyLearner(
        actor=TinyActor(params={'weight':jnp.array([marker])},batch_stats={'mean':jnp.zeros(1)},
                        optimizer_state={'moment':jnp.array([marker+.5])}),
        critic={'weight':jnp.array([marker+10.])},target_critic={'weight':jnp.array([marker+20.])},
        temperature=jnp.array(.1),
        reward_normalizer=TinyNormalizer(discounted_return=jnp.array([7.]),scale=jnp.array(2.)),
        loss_scale=jnp.array(1.),global_update_step=jnp.array(99),environment_steps=jnp.array(500000))


def confirmation_setup(costs):
    editor,_=editor_for([])
    editor.binding=NS(joint_q=np.zeros((4,2)))
    candidate=NS(joint_q=np.ones((4,2))*.01)
    editor.initial_state_sha256='fixed-initial-state'
    editor._cost=lambda m,q:m['cost']
    incoming=learner(0.)
    result=checkpoint(100,5.);result['snapshot']=learning_snapshot(learner(1.))
    control=checkpoint(100,9.);control['snapshot']=learning_snapshot(learner(2.))
    calls=[]
    def rollout(binding,state,seed,**kw):
        marker=int(np.asarray(state.actor.params['weight'])[0])
        calls.append((binding,state,seed,kw))
        return metrics(costs[marker]),None,None,{'reference':binding}
    editor.rollout=rollout
    def forbidden(*args,**kwargs):
        raise AssertionError('frozen confirmation must not perform lower learning')
    editor._adaptation_branch=forbidden
    return editor,candidate,incoming,result,control,calls


def confirm(editor,candidate,incoming,result,control):
    return editor._confirm_frozen_pair(candidate,result,control,incoming,43,metrics(10.),10.)


def tree_values(tree):
    return [np.asarray(x).copy() for x in jax.tree.leaves(tree)]


def test_confirmation_uses_exact_frozen_states_without_learning_or_mutation():
    editor,candidate,incoming,result,control,calls=confirmation_setup({1:5.1,2:8.9,0:10.1})
    before=[tree_values(v) for v in (incoming,result['snapshot'],control['snapshot'])]
    report,after,feedback=confirm(editor,candidate,incoming,result,control)
    assert report['passed'] and report['additional_training_steps']==0
    assert report['reset_policy']=='same_fixed_initial_state'
    assert report['initial_state_sha256']=='fixed-initial-state'
    assert report['worst_candidate_cost']==5.1 and report['best_control_cost']==8.9
    assert report['best_incoming_cost']==10.
    assert [int(np.asarray(c[1].actor.params['weight'])[0]) for c in calls]==[1,2,0]
    assert [c[2] for c in calls]==[43,43,43]
    assert calls[0][0] is candidate and all(c[0] is editor.binding for c in calls[1:])
    assert all(c[3]==dict(return_feedback=True) for c in calls)
    assert calls[2][1] is incoming
    for call in calls[:2]:
        np.testing.assert_array_equal(call[1].reward_normalizer.discounted_return,0.)
        assert int(call[1].environment_steps)==500000
    for old,current in zip(before,(incoming,result['snapshot'],control['snapshot'])):
        for a,b in zip(old,tree_values(current)):
            np.testing.assert_array_equal(a,b)
    assert feedback['reference'] is candidate and after['cost']==5.1


def test_confirmation_rejects_inconsistent_winner_even_when_averages_improve():
    editor,candidate,incoming,result,control,_=confirmation_setup({1:8.,2:7.,0:10.})
    assert (result['cost']+8.)/2 < (control['cost']+7.)/2
    report,_,_=confirm(editor,candidate,incoming,result,control)
    assert not report['passed']
    assert report['worst_candidate_cost']==8. and report['best_control_cost']==7.
    assert 'no_improvement_over_adapted_noedit_and_incoming_pair' in report['rejection_reasons']


@pytest.mark.parametrize('marker', [0,1,2])
def test_confirmation_rejects_nonfinite_repeated_cost_before_extrema(marker):
    costs={1:5.,2:9.,0:10.};costs[marker]=float('nan')
    editor,candidate,incoming,result,control,_=confirmation_setup(costs)
    report,_,_=confirm(editor,candidate,incoming,result,control)
    assert not report['passed']


def test_confirmation_detects_inference_parameter_mutation():
    editor,candidate,incoming,result,control,_=confirmation_setup({1:5.,2:9.,0:10.})
    original=editor.rollout
    def mutating_rollout(binding,state,*args,**kwargs):
        output=original(binding,state,*args,**kwargs)
        if binding is candidate:
            state.actor.params['weight']=jnp.array([77.])
        return output
    editor.rollout=mutating_rollout
    with pytest.raises(RuntimeError,match='changed an inference policy'):
        confirm(editor,candidate,incoming,result,control)
