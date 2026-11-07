"""Acceleration-level VOC must respect slender objects and changing orientation."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from flash_chord.runtime.object_control import VOCParams,apply_object_control

@pytest.mark.parametrize('device', ['cpu'] + (['cuda:0'] if wp.is_cuda_available() else []))
@pytest.mark.parametrize('angle',[0.,0.7,1.8])
@pytest.mark.parametrize('scale',[0.,0.5,1.])
def test_rotational_damping_is_axis_invariant_in_acceleration(angle,scale,device):
    # Spoon-like anisotropy, including off-diagonal inertia; no torque clamp.
    inertia=np.array([[1.6e-5,-3.7e-7,-3.9e-6],[-3.7e-7,8.14e-4,1.2e-7],[-3.9e-6,1.2e-7,8.24e-4]],dtype=np.float32)
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
    quat=wp.quat_from_axis_angle(wp.vec3(*axis),angle)
    rot=np.asarray(wp.quat_to_matrix(quat)).reshape(3,3)
    omega=np.array([1.2,-0.3,0.7],dtype=np.float32)
    with wp.ScopedDevice(device):
        model=SimpleNamespace(body_mass=wp.array([.1],dtype=wp.float32),body_com=wp.zeros(1,dtype=wp.vec3),body_inertia=wp.array(inertia[None],dtype=wp.mat33))
        state=SimpleNamespace(body_q=wp.array([wp.transform(wp.vec3(0.),quat)],dtype=wp.transform),body_qd=wp.array([[0.,0.,0.,*omega]],dtype=wp.spatial_vector),body_f=wp.zeros(1,dtype=wp.spatial_vector))
        control=SimpleNamespace(joint_target_pos=wp.zeros(0,dtype=wp.float32),joint_target_vel=wp.zeros(0,dtype=wp.float32))
        command=SimpleNamespace(world_count=1,num_joint_dof=0,bodies_per_world=1,layout=SimpleNamespace(num_objects=1,num_articulations=0),voc_body_ids_w=wp.array([0],dtype=wp.int32),voc_object_body_offsets=wp.array([0,1],dtype=wp.int32),voc_object_body_ids=wp.array([0],dtype=wp.int32),voc_target_pos_w=wp.zeros(1,dtype=wp.vec3),voc_target_quat_w=wp.array([quat],dtype=wp.quat),articulation_dof_ids=wp.zeros(0,dtype=wp.int32),articulation_target_pos=wp.zeros(0,dtype=wp.float32))
        apply_object_control(model,state,control,command,scale=wp.array([scale],dtype=wp.float32),params=VOCParams(gravity=0.))
        torque=state.body_f.numpy()[0,3:]
    acceleration=np.linalg.solve(rot@inertia@rot.T,torque)
    np.testing.assert_allclose(acceleration,-40.*scale*omega,rtol=2e-5,atol=2e-5)
    next_omega=omega+.0025*acceleration
    assert np.linalg.norm(next_omega)<=np.linalg.norm(omega)+1e-6
