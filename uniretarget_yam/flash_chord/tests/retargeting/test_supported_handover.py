import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import ContactSegment, HandChannels
from flash_chord.retargeting.physical_response import reference_action_schedule
from flash_chord.retargeting.supported_handover import (
    SupportedAnchor, anchor_validation, approach_motion_limits, connect_supported_contacts, delay_withdrawal, fit_supported_anchor, incoming_approach,
)


def samples():
    target=np.zeros((10,7));target[:,6]=1;target[:,0]=np.linspace(0,.1,10)
    model=SupportedAnchor(np.array([.02,-.01,.03]),np.array([0.,0.,.7]))
    actual=np.repeat(model.predict(target)[:,None],8,axis=1)
    return target,actual,np.ones((10,8),bool),model


def test_supported_anchor_retains_invalid_unloaded_and_wrong_world_denominators():
    target,actual,mask,expected=samples();actual[:,0,:]=np.nan;mask[:,1]=False
    model,fit=fit_supported_anchor(actual,target,mask)
    assert model is not None and fit['passed_fraction']==.75
    np.testing.assert_allclose(model.translation,expected.translation)
    np.testing.assert_allclose(model.rotation,expected.rotation)
    actual[:,2,0]+=.1
    validation=anchor_validation(model,actual,target,mask)
    assert validation['passed_fraction']==5/8
    assert not validation['per_world'][0]['passed']
    assert fit_supported_anchor(actual,target,np.zeros_like(mask))[0] is None


def test_anchor_forecast_is_coordinate_equivariant_and_quaternion_sign_invariant():
    target,actual,mask,_=samples();actual[::2,:,3:]*=-1
    model,_=fit_supported_anchor(actual,target,mask)
    world=Rotation.from_rotvec([.3,-.2,.4]);shift=np.array([2.,3.,-1.])
    def transform(x):
        out=x.copy();out[...,:3]=world.apply(x[...,:3].reshape(-1,3)).reshape(x[...,:3].shape)+shift
        out[...,3:]=(world*Rotation.from_quat(x[...,3:].reshape(-1,4))).as_quat().reshape(x[...,3:].shape)
        return out
    new_target,new_actual=transform(target),transform(actual)
    changed,report=fit_supported_anchor(new_actual,new_target,mask)
    assert report['passed_fraction']==1.
    np.testing.assert_allclose(changed.predict(new_target)[:,:3],transform(model.predict(target))[:,:3],atol=1e-12)
    a=Rotation.from_quat(changed.predict(new_target)[:,3:]);b=Rotation.from_quat(transform(model.predict(target))[:,3:])
    np.testing.assert_allclose((a*b.inv()).magnitude(),0.,atol=1e-12)


def test_unsupported_or_invalid_target_never_supplies_an_anchor():
    target,actual,mask,_=samples();mask[:,:5]=False
    assert fit_supported_anchor(actual,target,mask)[0] is None
    target[0,3:]=0
    with pytest.raises(ValueError):fit_supported_anchor(actual,target,mask)


def test_contact_cluster_requires_independent_ownership_and_never_crosses_object_switch():
    first=ContactSegment('right',0,20,30,0);later=ContactSegment('right',0,40,60,1)
    other=ContactSegment('right',1,70,90,2);same_late=ContactSegment('right',0,100,110,3)
    phases=[first,later,other,same_late];a=np.ones((120,10),bool);b=a.copy()
    merged,editing,report=connect_supported_contacts(first,phases,[a,b])
    assert merged.end==60 and other in editing and same_late in editing
    assert first.end==30 and later in phases and len(report['gaps'])==1
    b[31:40,:3]=False
    merged,_,_=connect_supported_contacts(first,phases,[a,b])
    assert merged==first
    with pytest.raises(ValueError):connect_supported_contacts(first,phases,[a])


def test_contact_only_gap_cannot_connect_when_object_slips_in_the_actual_hand():
    first=ContactSegment('right',0,20,30,0);later=ContactSegment('right',0,40,60,1)
    mask=np.ones((80,10),bool);h=np.zeros((80,10,7));h[...,6]=1
    o=h.copy();o[...,0]=.1
    cluster,_,report=connect_supported_contacts(first,[first,later],[mask,mask],attachment_repeats=[(h,o),(h,o)])
    assert cluster.end==60 and report['gaps'][0]['independent_world_attachment_fraction']==[1.,1.]
    slipped=o.copy();slipped[35:40,:3,1]=.2
    cluster,_,_=connect_supported_contacts(first,[first,later],[mask,mask],attachment_repeats=[(h,o),(h,slipped)])
    assert cluster==first


def reference(frames=200):
    q=np.zeros((frames,10));q[:,0]=.1;q[:,2]=.3;q[:,6]=1;q[:,7:]=.2
    hand=HandChannels('right',(0,1,2),(3,4,5,6),(7,),(1.,))
    target=np.zeros((frames,7));target[:,2]=.3;target[:,6]=1
    segment=ContactSegment('right',0,80,130,1)
    segments=[ContactSegment('right',0,0,30,0),segment,ContactSegment('right',1,160,190,2)]
    return q,hand,target,segment,segments


def test_incoming_approach_preserves_previous_skill_other_channels_and_object_pivot():
    q,hand,target,segment,segments=reference();before=target.copy()
    anchor=SupportedAnchor(np.array([.03,0,0]),np.array([0,0,.8]))
    out,info=incoming_approach(q,hand,target,anchor,segment,segments,20.,project=lambda x:(x,1.))
    assert out is not None,info
    np.testing.assert_array_equal(out[:32],q[:32]);np.testing.assert_array_equal(out[159:],q[159:])
    np.testing.assert_array_equal(out[:,7:],q[:,7:]);np.testing.assert_array_equal(target,before)
    expected=target[100,:3]+anchor.translation+Rotation.from_rotvec(anchor.rotation).apply(q[100,:3]-target[100,:3])
    np.testing.assert_allclose(out[100,:3],expected,atol=1e-12)


def test_approach_backtracking_keeps_translation_rotation_coupled_and_ramps_long():
    q,hand,target,segment,segments=reference()
    anchor=SupportedAnchor(np.array([.08,0,0]),np.array([0,0,.8]))
    def project(x):
        if np.linalg.norm(x[:,:3]-q[:,:3],axis=1).max()>.045:
            out=x.copy();out[:,:3]=q[:,:3]+.5*(x[:,:3]-q[:,:3]);return out,.5
        return x,1.
    out,info=incoming_approach(q,hand,target,anchor,segment,segments,20.,project=project)
    assert out is not None and info['applied_gain']<1
    gain=info['applied_gain']
    expected=target[100,:3]+gain*anchor.translation+Rotation.from_rotvec(gain*anchor.rotation).apply(q[100,:3]-target[100,:3])
    np.testing.assert_allclose(out[100,:3],expected,atol=1e-12)
    assert len(info['feasibility_attempts'])>1
    assert info['required_approach_frames']>=14


def test_rigid_contact_transport_preserves_fast_motion_without_erasing_grasp_placement():
    q,hand,target,segment,segments=reference()
    q[:,3:7]=Rotation.from_rotvec(np.c_[np.zeros((len(q),2)),np.arange(len(q))*.25]).as_quat()
    target[:,3:]=q[:,3:7]
    anchor=SupportedAnchor(np.zeros(3),np.array([.8,0,0]))
    out,info=incoming_approach(q,hand,target,anchor,segment,segments,20.,project=lambda x:(x,1.))
    limits,scale,report=approach_motion_limits(q,out,hand,target,segment,20.)
    assert report['contact_added_rotation_peak_rad_s']>3.
    assert report['free_added_rotation_peak_rad_s']<2.
    assert scale==1. and limits.max_rotation_speed_rad_s>3.
    old=Rotation.from_quat(q[80:131,3:7]);new=Rotation.from_quat(out[80:131,3:7])
    np.testing.assert_allclose((old[1:]*old[:-1].inv()).magnitude(),(new[1:]*new[:-1].inv()).magnitude(),atol=1e-12)
    corrupted=out.copy();corrupted[100,0]+=.02
    with pytest.raises(ValueError):approach_motion_limits(q,corrupted,hand,target,segment,20.)


def test_delayed_withdrawal_preserves_moving_object_grasp_prefix_and_other_channels():
    q,hand,target,_,_=reference()
    target[:,0]=np.linspace(0,.1,len(q));target[:,3:]=Rotation.from_rotvec(np.c_[np.zeros((len(q),2)),np.linspace(0,.5,len(q))]).as_quat()
    ro=Rotation.from_quat(target[:,3:]);relative=np.tile([.1,0,0],(len(q),1));relative[121:,1]=np.linspace(0,.2,len(q)-121)
    q[:,:3]=target[:,:3]+ro.apply(relative);q[:,3:7]=ro.as_quat()
    out,info=delay_withdrawal(q,hand,target,start=100,event=120,end=180,delay_frames=20,fps=20)
    np.testing.assert_array_equal(out[:101],q[:101]);np.testing.assert_array_equal(out[180:],q[180:])
    np.testing.assert_array_equal(out[:,8:],q[:,8:])
    local=ro.inv().apply(out[:,:3]-target[:,:3])
    np.testing.assert_allclose(local[140],[.1,0,0],atol=1e-12)
    assert info['delayed_withdrawal_frame']==140
    assert local[150,1]>0


def test_action_schedule_composes_named_hands_without_changing_other_worlds_or_frames():
    names=['right_finger_0','left_wrist_pos','right_wrist_pos','left_finger_0']
    ranges=[(0,2),(2,5),(5,8),(8,10)]
    candidates=[dict(component='fingers',extra_interventions=[dict(hand='right',component='wrist',bounds=[5,7])]),dict(component='actor')]
    groups=np.array([1,0,1])
    s=reference_action_schedule(names,ranges,10,groups,candidates,12,'left',[1,10])
    assert not s[:,0].any() and not s[:,2].any()
    assert s[1:11,1,8:10].all() and not s[0,1].any() and not s[11,1].any()
    assert s[5:8,1,5:8].all() and not s[4,1,5:8].any() and not s[8,1,5:8].any()
    assert not s[:,:,0:5].any()
    candidates[0]['extra_interventions'][0]['bounds']=[5,12]
    with pytest.raises(ValueError):reference_action_schedule(names,ranges,10,groups,candidates,12,'left',[1,10])
