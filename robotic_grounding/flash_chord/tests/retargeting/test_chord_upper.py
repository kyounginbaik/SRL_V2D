from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from flash_chord.retargeting.chord_upper import WrenchSupportAccumulator,wrench_contact_gate


def summarize(current, reference, tolerance=.1):
    a=WrenchSupportAccumulator(*reference.shape,tolerance=tolerance)
    a.add(current,reference)
    return a.summaries()


def test_tolerance_band_and_separate_under_over_support():
    ref=np.ones((1,1,1,2))
    assert summarize(ref*np.array([.95,1.05]),ref)[0]['cws_tolerance_error']==0.
    assert summarize(ref*np.array([.7,1.3]),ref)[0]['cws_tolerance_error']==pytest.approx(.04)
    assert summarize(ref*0,ref)[0]['cws_tolerance_error']==pytest.approx(.81)
    assert summarize(ref*0,ref)[0]['cws_missed_fraction']==1.


def test_no_reference_contact_is_not_a_free_extra_grasp():
    ref=np.zeros((1,2,2,4));ref[0,0,0]=1.
    cur=ref.copy();cur[0,1,1]=1.
    result=summarize(cur,ref)[0]
    assert result['cws_tolerance_error']==0.
    assert result['per_object_cws_unintended_fraction']==[0.,.5]
    assert result['cws_unintended_energy']>0


def test_per_object_gate_does_not_hide_failed_lid_in_pot_average():
    ref=np.ones((1,1,2,10));cur=ref.copy();cur[0,0,1,:5]=0.
    before=summarize(ref,ref)[0];after=summarize(cur,ref)[0]
    assert after['per_object_cws_missed_fraction']==[0.,.5]
    assert not wrench_contact_gate(after,before)


def test_parallel_and_single_world_accumulation_are_identical():
    rng=np.random.default_rng(1)
    ref=rng.uniform(0,1,(5,3,2,2,12));ref[ref<.4]=0.
    cur=rng.uniform(0,1,ref.shape);cur[cur<.4]=0.
    batch=WrenchSupportAccumulator(3,2,2,12)
    singles=[WrenchSupportAccumulator(1,2,2,12) for _ in range(3)]
    for c,r in zip(cur,ref):
        batch.add(c,r)
        for i,a in enumerate(singles):a.add(c[i:i+1],r[i:i+1])
    for i,a in enumerate(singles):assert batch.summaries()[i]==a.summaries()[0]


def test_support_basis_order_does_not_change_score():
    ref=np.array([[[[1.,.8,0.,.2]]]])
    cur=np.array([[[[.6,.9,.1,0.]]]])
    a=summarize(cur,ref)[0];b=summarize(cur[...,::-1],ref[...,::-1])[0]
    for k in a:np.testing.assert_allclose(a[k],b[k])


@pytest.mark.parametrize('bad',[np.nan,np.inf,-1.])
def test_invalid_support_fails_closed(bad):
    with pytest.raises(FloatingPointError):summarize(np.full((1,1,1,1),bad),np.ones((1,1,1,1)))


def test_contact_position_changes_cannot_change_upper_cost_or_gate():
    from flash_chord.retargeting.response_editor import response_cost,ResponseBilevelEditor
    q=np.zeros((20,58));q[:,[6,35]]=1.
    layout=NS(hands=[NS(finger_q_ids=np.arange(7,29)),NS(finger_q_ids=np.arange(36,58))])
    base=dict(object_position_robust_m=.02,object_orientation_robust_rad=.1,
        object_success_robust_fraction=.8,unstable_world_fraction=0.,
        cws_tolerance_error=.2,cws_missed_fraction=.1,cws_unintended_fraction=0.,
        cws_unintended_energy=0.,per_object_cws_missed_fraction=[.1,.1],
        per_object_cws_unintended_fraction=[0.,0.],all_errors_finite=True,full_horizon=True,
        per_object_position_mean_m=[.02,.02],per_object_orientation_mean_rad=[.1,.1],
        protected_prefix_frames=10,prefix_position_mean_m=[.02,.02],
        prefix_orientation_mean_rad=[.1,.1],prefix_position_p95_m=[.03,.03],
        prefix_orientation_p95_rad=[.2,.2],per_object_first_failure_seconds=[3.,4.],
        contact_proximity_mean_m=.001,per_hand_object_contact_proximity_mean_m=[[.001,.001],[.001,.001]])
    moved=deepcopy(base);moved['contact_proximity_mean_m']=2.
    moved['per_hand_object_contact_proximity_mean_m']=[[2.,2.],[2.,2.]]
    assert response_cost(moved,q,q,layout,20.)==response_cost(base,q,q,layout,20.)
    editor=ResponseBilevelEditor.__new__(ResponseBilevelEditor);editor.options={}
    assert editor._gate(moved,base)
    moved['per_object_cws_missed_fraction']=[.1,.5]
    assert not editor._gate(moved,base)
