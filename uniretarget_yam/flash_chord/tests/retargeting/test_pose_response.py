import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.pose_response import PoseResponse,fit_pose_response,validate_pose_response,inverse_reference_trajectory


def fixture():
    model=PoseResponse([.25,.3,.2],[.0001,0.,-.001],[.35,.34,.36],[0.,.0001,0.])
    rng=np.random.default_rng(72);h=np.zeros((80,8,7));h[...,6]=1;c=h.copy()
    for t in range(1,len(h)):
        c[t,:,:3]=h[t-1,:,:3]+rng.normal(0,.02,(8,3))
        c[t,:,3:]=(Rotation.from_quat(h[t-1,:,3:])*Rotation.from_rotvec(rng.normal(0,.15,(8,3)))).as_quat()
        h[t]=model.predict(h[t-1],c[t])
    return model,h,c,np.ones(h.shape[:2],bool)


def test_identification_uses_current_command_and_previous_after_state():
    expected,h,c,mask=fixture();model,report=fit_pose_response(h,c,mask)
    assert model is not None and report['used_transitions']==79*8
    np.testing.assert_allclose(model.position_gain,expected.position_gain,atol=1e-12)
    np.testing.assert_allclose(model.rotation_gain,expected.rotation_gain,atol=1e-12)
    check=validate_pose_response(model,h,c,mask)
    assert check['model']['position_rmse_m']<1e-12 and check['model']['orientation_rmse_rad']<1e-12
    assert check['instantaneous_tracking']['position_rmse_m']>.01


def test_inverse_reaches_after_goal_in_body_rotation_frame_and_rejects_wrapped_steps():
    model,h,c,_=fixture();before=h[20];after=h[21]
    command=model.inverse(before,after);prediction=model.predict(before,command)
    np.testing.assert_allclose(prediction[:,:3],after[:,:3],atol=1e-12)
    np.testing.assert_allclose((Rotation.from_quat(prediction[:,3:])*Rotation.from_quat(after[:,3:]).inv()).magnitude(),0.,atol=1e-12)
    impossible=before.copy();impossible[:,3:]=(Rotation.from_quat(before[:,3:])*Rotation.from_rotvec(np.tile([2.,0,0],(8,1)))).as_quat()
    with pytest.raises(ValueError):model.inverse(before,impossible)


def test_unloaded_and_invalid_samples_are_reported_and_cannot_fit_a_model():
    expected,h,c,mask=fixture();h[:,0]=np.nan;mask[:,1]=False
    model,report=fit_pose_response(h,c,mask)
    assert model is not None and report['used_transitions']==79*6
    assert report['invalid_pose_fraction']==pytest.approx(1/8)
    assert fit_pose_response(h,c,np.zeros_like(mask))[0] is None
    with pytest.raises(ValueError):PoseResponse([0.,.3,.4],np.zeros(3),np.ones(3),np.zeros(3))


def test_inverse_reference_preserves_geometry_boundaries_and_hits_post_step_goals():
    model,h,c,mask=fixture();desired=h[:,0].copy();before=desired.copy()
    nominal,info=inverse_reference_trajectory(model,desired,bounds=(5,70),contact=(20,50))
    np.testing.assert_array_equal(desired,before)
    np.testing.assert_array_equal(nominal[:6],desired[:6]);np.testing.assert_array_equal(nominal[70:],desired[70:])
    prediction=model.predict(desired[19:50],nominal[20:51])
    np.testing.assert_allclose(prediction[:,:3],desired[20:51,:3],atol=1e-12)
    np.testing.assert_allclose((Rotation.from_quat(prediction[:,3:])*Rotation.from_quat(desired[20:51,3:]).inv()).magnitude(),0.,atol=1e-12)
    position,_=inverse_reference_trajectory(model,desired,bounds=(5,70),contact=(20,50),component='position')
    np.testing.assert_array_equal(position[:,3:],desired[:,3:])
