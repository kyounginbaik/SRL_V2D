from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from flash_chord.runtime.wrist_limits import FloatingWristVelocityLimits


def test_caps_vector_norms_preserves_direction_and_other_dofs():
    layout=SimpleNamespace(hands=[SimpleNamespace(wrist_pos_dof_ids=(0,1,2),wrist_orient_dof_ids=(3,4,5)),SimpleNamespace(wrist_pos_dof_ids=(8,9,10),wrist_orient_dof_ids=(11,12,13))])
    model=SimpleNamespace(joint_dof_count=32,device='cpu')
    limiter=FloatingWristVelocityLimits.build(model,layout,2,.75,6.,device='cpu')
    original=np.array([[3.,4.,0.,0.,-12.,0.,9.,8.,.1,.2,.3,1.,2.,3.,7.,6.],[-3.,-4.,0.,0.,0.,30.,3.,4.,9.,0.,0.,0.,0.,0.,5.,6.]],dtype=np.float32)
    state=SimpleNamespace(joint_qd=wp.array(original.ravel(),dtype=wp.float32,device='cpu'))
    limiter.enforce_velocity(state);actual=state.joint_qd.numpy().reshape(2,16)
    np.testing.assert_allclose(actual[0,:3],[.45,.6,0.],atol=1e-7)
    np.testing.assert_allclose(actual[1,:3],[-.45,-.6,0.],atol=1e-7)
    for start,limit in [(0,.75),(3,6.),(8,.75),(11,6.)]:
        assert np.all(np.linalg.norm(actual[:,start:start+3],axis=1)<=limit+1e-6)
    np.testing.assert_array_equal(actual[:,[6,7,14,15]],original[:,[6,7,14,15]])
    np.testing.assert_array_equal(actual[0,8:14],original[0,8:14])


@pytest.mark.parametrize('linear,angular',[(0,6),(.75,0),(-1,6),(float('nan'),6),(.75,float('inf'))])
def test_invalid_caps_fail(linear,angular):
    with pytest.raises(ValueError):
        FloatingWristVelocityLimits.build(None,None,1,linear,angular)
