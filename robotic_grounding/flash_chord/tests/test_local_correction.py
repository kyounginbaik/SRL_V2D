from types import SimpleNamespace as NS
import numpy as np
import pytest
from flash_chord.retargeting.local_correction import local_envelope, feasible_increment, local_candidate

def test_increment_preserves_prior_edits_and_local_support():
    w = local_envelope(335,20,6,7)
    current = .005*local_envelope(335,20,2,3)
    out, scale = feasible_increment(current,.02*w,-.03,.03,.1,20)
    np.testing.assert_array_equal(out[w==0],current[w==0])
    assert out[130] == pytest.approx(.02)
    assert scale == pytest.approx(1.)
    assert np.max(abs(np.diff(out)))*20 <= .100001

def test_cumulative_bound_shrinks_only_increment():
    w=local_envelope(335,20,6,7)
    current=.025*w
    out,scale=feasible_increment(current,.02*w,-.03,.03,.1,20)
    assert scale == pytest.approx(.25)
    assert out.max() == pytest.approx(.03)
    zero,_=feasible_increment(current,np.zeros(335),-.03,.03,.1,20)
    np.testing.assert_array_equal(zero,current)

def test_full_hand_quaternion_and_endpoint_invariants():
    base=np.zeros((335,58),np.float32);base[:,6]=1;base[:,35]=1
    qids=list(range(3))+list(range(7,29))+list(range(29,32))+list(range(36,58))
    names=['left_wrist_'+x for x in 'xyz']+['left_finger']*22+['right_wrist_'+x for x in 'xyz']+['right_finger']*22
    channels=qids+[3,4,5,32,33,34]
    basis=np.eye(58)[channels]
    p=NS(qids=qids,lower=np.full(50,-2.),upper=np.full(50,2.),ball_joints=[(3,3),(32,31)],layout=NS(scalar_joints=NS(names=names)))
    e=NS(baseline=NS(joint_q=base,fps=20),binding=NS(joint_q=base.copy()),projector=p,joint_basis=basis,options={},basis=local_envelope(335,20,6,7)[:,None])
    q=local_candidate(e,np.ones(56))
    np.testing.assert_array_equal(q[e.basis[:,0]==0],base[e.basis[:,0]==0])
    np.testing.assert_allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-6)
    assert q[130,0] == pytest.approx(.02)
    e.binding.joint_q=q
    q2=local_candidate(e,np.ones(56))
    assert q2[130,0] == pytest.approx(.03)
    with pytest.raises(ValueError):local_candidate(e,np.full(56,np.nan))


def test_performance_gate_honors_ladder_and_latest_reference_passes():
    from flash_chord.lifecycle.performance_voc import gate_from_options
    gate=gate_from_options(2,dict(scales=[1.,.95,.9,0.],reset_on_reference_change=False))
    assert gate.probe_scale == .95
    metrics=dict(per_object_gate_success_fraction=[.99,.99],per_object_position_p95_m=[.02,.02],
        per_object_orientation_p95_rad=[.1,.1],full_horizon=True,all_errors_finite=True)
    for version in [0,1]: assert not gate.observe(metrics,version)['lowered']
    assert gate.observe(metrics,2)['lowered']
    assert gate.scale == .95
    metrics['per_object_position_p95_m'][1]=.08
    assert not gate.observe(metrics,3)['passed']
    assert gate.streak == 0

def test_finalist_confirmation_uses_paired_nonzero_initial_states():
    from flash_chord.retargeting.local_editor import LocalReferenceEditor
    import copy
    base=dict(all_errors_finite=True,full_horizon=True,object_position_robust_m=.05,object_orientation_robust_rad=.2,
        object_success_robust_fraction=.5,wrench_robust_mse=.4,unstable_world_fraction=0.,
        per_object_position_mean_m=[.05,.05],per_object_orientation_mean_rad=[.2,.2],
        contact_proximity_mean_m=.03,per_object_contact_proximity_mean_m=[.03,.03],
        per_hand_object_contact_proximity_mean_m=[[.03,None],[None,.03]])
    after=copy.deepcopy(base)
    after.update(object_position_robust_m=.04,per_object_position_mean_m=[.04,.04],
        contact_proximity_mean_m=.02,per_object_contact_proximity_mean_m=[.02,.02],
        per_hand_object_contact_proximity_mean_m=[[.02,None],[None,.02]])
    received=[]
    def rollout(binding,learner,seed,initial_q_delta):
        received.append(initial_q_delta.copy())
        return (base if len(received)%2 else after),None,None
    reference=NS(joint_q=np.zeros((10,2)),num_joint_q=2,fps=20)
    e=NS(options={},baseline=reference,binding=reference,contact_proximity_weight=5.,
        projector=NS(qids=[0,1],layout=NS(scalar_joints=NS(names=['left_wrist_x','left_finger']))),rollout=rollout)
    passed,evidence=LocalReferenceEditor.confirm_candidate(e,reference,None,43,None,None)
    assert passed and len(evidence['checks'])==3
    for i in range(0,6,2):np.testing.assert_array_equal(received[i],received[i+1])
    assert np.max(abs(np.stack(received)[:,0])) <= .001
    assert np.max(abs(np.stack(received)[:,1])) <= np.deg2rad(.5)
    assert not np.array_equal(received[0],received[2])

def test_deferred_screening_requires_robust_confirmation():
    from flash_chord.retargeting.parallel_editor import ParallelReferenceEditor
    from flash_chord.retargeting.local_editor import LocalReferenceEditor
    with pytest.raises(ValueError, match='paired robust confirmation'):
        ParallelReferenceEditor(options={'defer_acceptance_gates_to_confirmation':True})
    with pytest.raises(ValueError, match='paired robust confirmation'):
        LocalReferenceEditor(options={'defer_acceptance_gates_to_confirmation':True,'robust_confirmation':False})


def test_local_contact_gate_requires_target_improvement_and_bounds_other_hand():
    from flash_chord.retargeting.local_editor import confirmation_contact_gate
    before=dict(contact_proximity_mean_m=.13,per_object_contact_proximity_mean_m=[.13,.2],
        per_hand_object_contact_proximity_mean_m=[[.13,None],[.05,.2]])
    after=dict(contact_proximity_mean_m=.12,per_object_contact_proximity_mean_m=[.10,.208],
        per_hand_object_contact_proximity_mean_m=[[.10,None],[.04,.208]])
    e=NS(options={'contact_confirmation_tolerance_m':.01},local_context={'focus':'left_pot'})
    assert confirmation_contact_gate(e,after,before)
    e.local_context['focus']='right_lid'
    assert not confirmation_contact_gate(e,after,before)
    e.local_context['focus']='left_pot'
    after['per_hand_object_contact_proximity_mean_m'][1][1]=.211
    assert not confirmation_contact_gate(e,after,before)
