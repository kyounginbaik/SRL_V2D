import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import HandChannels,RefinementConfig,project_edit
from flash_chord.retargeting.held_carry import fit_attachment,validate_attachment,supported_reference_relation,carry_candidate,feasible_carry_candidate


def fixture(frames=120,worlds=8):
    h=np.zeros((frames,worlds,7));h[...,:3]=[.02,.01,.3];h[...,6]=1
    o=h.copy();o[...,:3]+=[.05,0,0];o[...,3:]=Rotation.from_rotvec([0,0,.8]).as_quat()
    return h,o,np.ones((frames,worlds),bool)


def test_attachment_validation_keeps_slipping_and_unloaded_worlds_in_denominator():
    h,o,mask=fixture();bad=o.copy();bad[60:,0,3:]=Rotation.from_rotvec([0,0,2.8]).as_quat()
    model,report=fit_attachment(h,bad,mask)
    assert report['accepted'] and not model.stable_worlds[0]
    assert report['fit_worlds']==7
    validation=validate_attachment(model,h,bad,mask)
    assert validation['passed_fraction']==7/8
    assert not validation['per_world'][0]['passed']
    assert fit_attachment(h,o,np.zeros_like(mask))[0] is None


def test_inverse_reference_response_preserves_fixed_object_target_and_root_palm_offset():
    h,o,mask=fixture();ref=h[:,0]
    relation,report=supported_reference_relation(ref,o,mask,np.ones(8,bool))
    assert report['first_observed']==0 and report['last_observed']==119
    target=o[:,0].copy();target[:,3:]=Rotation.from_rotvec([0,0,.4]).as_quat();before=target.copy()
    q=np.zeros((120,10));q[:,6]=1;q[:,2]=.3;q[:,7:]=.3
    hand=HandChannels('left',(0,1,2),(3,4,5,6),(7,),(1.,));offset=np.array([.02,.01,0.])
    candidate,info=carry_candidate(q,hand,ref,target,relation,start=20,contact_end=80,restore_limit=118,fps=20)
    assert candidate is not None
    np.testing.assert_array_equal(candidate[:21],q[:21]);np.testing.assert_array_equal(candidate[info['edit_bounds'][1]:],q[info['edit_bounds'][1]:])
    np.testing.assert_array_equal(candidate[:,7:],q[:,7:]);np.testing.assert_array_equal(target,before)
    palm_rotation=Rotation.from_quat(candidate[40,3:7]);palm=candidate[40,:3]+palm_rotation.apply(offset)
    # The learned attachment puts the object at the unchanged task target.
    np.testing.assert_allclose(palm+palm_rotation.apply([.05,0,0]),target[40,:3],atol=1e-12)
    np.testing.assert_allclose((palm_rotation*Rotation.from_rotvec([0,0,.8])).as_quat(),target[40,3:],atol=1e-12)


def test_object_relative_response_is_invariant_to_common_world_transform_and_missing_support():
    h,o,mask=fixture();mask[:10]=False;mask[-10:]=False
    for frame in range(len(o)):o[frame,:,3:]=Rotation.from_rotvec([0,0,.8+.03*np.sin(frame)]).as_quat()
    relation,report=supported_reference_relation(h[:,0],o,mask,np.ones(8,bool))
    transform=Rotation.from_rotvec([.2,-.3,.1]);shift=np.array([2.,-1.,3.])
    for values in (h,o):
        values[...,:3]=transform.apply(values[...,:3].reshape(-1,3)).reshape(values[...,:3].shape)+shift
        values[...,3:]=(transform*Rotation.from_quat(values[...,3:].reshape(-1,4))).as_quat().reshape(values[...,3:].shape)
    changed,new_report=supported_reference_relation(h[:,0],o,mask,np.ones(8,bool))
    assert report['first_observed']==new_report['first_observed']==10
    assert report['last_observed']==new_report['last_observed']==109
    np.testing.assert_allclose(changed[:,:3],relation[:,:3],atol=1e-12)
    np.testing.assert_allclose(np.abs(changed[:,3:]),np.abs(relation[:,3:]),atol=1e-12)
    np.testing.assert_allclose(relation[:10],np.repeat(relation[10:11],10,axis=0))
    np.testing.assert_allclose(relation[110:],np.repeat(relation[109:110],10,axis=0))


def test_carry_preserves_fast_baseline_free_return_without_contracting_grasp_correction():
    h,o,mask=fixture();relation,_=supported_reference_relation(h[:,0],o,mask,np.ones(8,bool))
    q=np.zeros((120,8));q[:,6]=1;q[:,2]=.3
    q[81:,3:7]=Rotation.from_rotvec(np.c_[np.minimum(np.arange(1,40)*.2,np.pi),np.zeros((39,2))]).as_quat()
    ref=np.c_[q[:,:3]+Rotation.from_quat(q[:,3:7]).apply(np.tile([.02,.01,0],(120,1))),q[:,3:7]]
    target=o[:,0].copy();target[:,3:]=Rotation.from_rotvec([0,0,.4]).as_quat()
    hand=HandChannels('left',(0,1,2),(3,4,5,6),(7,),(1.,))
    candidate,_=carry_candidate(q,hand,ref,target,relation,start=20,contact_end=80,restore_limit=118,fps=20)
    projected,scale=project_edit(q,candidate,[hand],[0,1,2,7],np.full(4,-2.),np.full(4,2.),20,
        RefinementConfig(max_rotation_rad=1.8,max_translation_m=.12))
    assert scale>.99999
    np.testing.assert_allclose(projected[40,3:7],candidate[40,3:7],atol=1e-7)


def test_partial_gain_rotates_about_actual_object_not_the_palm():
    h,o,mask=fixture();relation,_=supported_reference_relation(h[:,0],o,mask,np.ones(8,bool))
    q=np.zeros((120,8));q[:,6]=1;q[:,2]=.3
    target=o[:,0].copy();target[:,:3]+=[.02,.01,0];target[:,3:]=Rotation.from_rotvec([0,0,.4]).as_quat()
    hand=HandChannels('left',(0,1,2),(3,4,5,6),(7,),(1.,))
    candidate,_=carry_candidate(q,hand,h[:,0],target,relation,start=20,contact_end=80,restore_limit=118,fps=20,gain=.5)
    rotation=Rotation.from_quat(candidate[40,3:7]);palm=candidate[40,:3]+rotation.apply([.02,.01,0])
    actual_object=palm+rotation.apply([.05,0,0])
    np.testing.assert_allclose(actual_object,(o[40,0,:3]+target[40,:3])/2,atol=1e-12)


def test_feasibility_contracts_coupled_gain_without_moving_object_off_the_response_direction():
    h,o,mask=fixture();relation,_=supported_reference_relation(h[:,0],o,mask,np.ones(8,bool))
    q=np.zeros((120,8));q[:,6]=1;q[:,2]=.3
    target=o[:,0].copy();target[:,0]+=.08*np.sin(np.arange(120)*.7);target[:,3:]=Rotation.from_rotvec([0,0,.4]).as_quat()
    hand=HandChannels('left',(0,1,2),(3,4,5,6),(7,),(1.,))
    def project(candidate):
        return project_edit(q,candidate,[hand],[0,1,2,7],np.full(4,-2.),np.full(4,2.),20,
            RefinementConfig(max_translation_m=1.,max_rotation_rad=1.8))
    candidate,info=feasible_carry_candidate(q,hand,h[:,0],target,relation,project=project,
        start=20,contact_end=80,restore_limit=118,fps=20,gain=1.)
    assert candidate is not None and 0<info['applied_gain']<1 and info['rate_backtracks']>0
    assert info['projection_position_change_m']<=1e-6
    rotation=Rotation.from_quat(candidate[40,3:7]);palm=candidate[40,:3]+rotation.apply([.02,.01,0])
    object_position=palm+rotation.apply([.05,0,0])
    expected=o[40,0,:3]+info['applied_gain']*(target[40,:3]-o[40,0,:3])
    np.testing.assert_allclose(object_position,expected,atol=1e-6)
