from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.command_replay import CommandReplay,absorb_finger_commands


class Array:
    def __init__(self,x):self.x=np.asarray(x)
    def numpy(self):return self.x


def fixture():
    frames,worlds=9,3
    # Deliberately different native target order and action hand order.
    left=NS(wrist_pos_dof_ids=(9,10,11),wrist_orient_dof_ids=(12,13,14),finger_dof_ids=(15,))
    right=NS(wrist_pos_dof_ids=(0,1,2),wrist_orient_dof_ids=(3,4,5),finger_dof_ids=(6,7,8))
    layout=NS(sides=('left','right'),num_hands=2,total_fingers=4,hand_finger_starts=(0,1),
        hand_action_starts=(0,7),finger_counts=(1,3))
    p=np.zeros((frames,worlds,2,3));quat=np.zeros((frames,worlds,2,4));quat[...,3]=1
    fingers=np.full((frames,worlds,4),.4)
    reference=NS(sides=('left','right'),num_frames=frames,reference_world_count=worlds,
        wrist_pos_w=Array(p),wrist_quat_w=Array(quat),finger_joint_pos=Array(fingers))
    buffers=NS(layout=layout,reference=reference,scale=Array(np.full(16,.15)),clip=Array(np.full(16,.3)),
        input_scale=Array(np.full(16,.3/(.7*.15))),input_mapping='rational',ema=.3)
    action=NS(buffers=buffers,filtered_action=Array(np.zeros((worlds,16))),delay=None,
        joint_target=Array(np.zeros((worlds,16))))
    env=NS(world_count=worlds,action=action,timestep=Array(np.ones(worlds,int)),
        scene=NS(layout=NS(hand=lambda side:dict(left=left,right=right)[side])))
    targets=np.zeros((2,frames-1,16));targets[...,6:9]=[.7,.2,.5]
    targets[0,:,3:6]=[.1,-.08,.04];targets[...,0]=.02
    candidates=[dict(command_replay=[dict(hand='right',component='all',bounds=[1,6])]),{}]
    return env,targets,candidates,np.array([1,0,1])


def filtered(action,previous,buffer):
    scale,clip,endpoint=(getattr(buffer,k).numpy() for k in ('scale','clip','input_scale'))
    mapped=action/((1-np.abs(action))+np.abs(action)/endpoint)
    return np.clip(buffer.ema*previous+(1-buffer.ema)*scale*mapped,-clip,clip)


def test_replay_preserves_other_worlds_hands_and_exact_actuator_overdrive():
    env,targets,candidates,groups=fixture();controller=CommandReplay(targets,groups,candidates)
    original=np.full((3,16),.01);before=env.action.filtered_action.numpy().copy()
    out=controller.action(original,1,env)
    np.testing.assert_array_equal(out[[0,2]],original[[0,2]])
    np.testing.assert_array_equal(out[1,:7],original[1,:7])
    np.testing.assert_array_equal(env.action.filtered_action.numpy(),before)
    physical=filtered(out,before,env.action.buffers)
    np.testing.assert_allclose(physical[1,13:16]+.4,targets[0,1,6:9],atol=1e-12)
    assert targets[0,1,6]==.7  # The actuator target is never clipped to a pose.
    np.testing.assert_allclose(physical[1,7:10],[.02,0,0],atol=1e-12)
    np.testing.assert_allclose(Rotation.from_euler('xyz',physical[1,10:13]).as_matrix(),
        Rotation.from_rotvec(targets[0,1,3:6]).as_matrix(),atol=1e-12)


def test_replay_reports_unreachable_commands_and_actual_native_target_error():
    env,targets,candidates,groups=fixture();targets[0,:,6]=1.8
    controller=CommandReplay(targets,groups,candidates);out=controller.action(np.zeros((3,16)),1,env)
    assert out[1,13]<=1.
    assert controller.arrays()['command_replay_saturation'][0,2]>0
    env.action.joint_target.x[1]=targets[0,1]
    controller.observe(env,1)
    np.testing.assert_allclose(controller.arrays()['command_replay_errors'][0,2:],0.,atol=1e-12)
    env.action.joint_target.x[1,6]+=.1;controller.observe(env,1)
    assert controller.arrays()['command_replay_errors'][-1,4]==pytest.approx(.1)


def test_finger_only_replay_and_interval_protection():
    env,targets,candidates,groups=fixture();candidates[0]['command_replay'][0]['component']='fingers'
    controller=CommandReplay(targets,groups,candidates)
    action=np.full((3,16),.05)
    out=controller.action(action,1,env)
    np.testing.assert_array_equal(out[1,:13],action[1,:13])
    env.timestep.x[:]=7
    np.testing.assert_array_equal(controller.action(action,7,env),action)


def test_replay_rejects_unknown_delayed_or_misaligned_control_context():
    env,targets,candidates,groups=fixture();controller=CommandReplay(targets,groups,candidates)
    env.action.delay=NS(config=NS(max_steps=2))
    with pytest.raises(ValueError):controller.action(np.zeros((3,16)),1,env)
    env.action.delay=None;env.timestep.x[:]=2
    with pytest.raises(ValueError):controller.action(np.zeros((3,16)),1,env)
    candidates[0]['command_replay'][0]['bounds']=[0,6]
    with pytest.raises(ValueError):CommandReplay(targets,groups,candidates)


def test_command_absorption_preserves_preload_limits_prefix_and_other_channels():
    q=np.zeros((100,8));q[:,[2,6]]=1.;target=np.zeros((100,7));target[:,3]=.1
    command=np.zeros((99,7));command[:,3]=.45;before=command.copy()
    new,nt,info=absorb_finger_commands(q,target,command,[7],[3],[0.],[.3],bounds=(1,90),fps=20.)
    assert info['source_command_max_outside_physical_limits_rad']==pytest.approx(.15)
    assert new[40,7]==pytest.approx(.3)
    assert command[40,3]-new[40,7]==pytest.approx(.15)
    np.testing.assert_array_equal(command,before)
    np.testing.assert_array_equal(new[:2],q[:2]);np.testing.assert_array_equal(new[90:],q[90:])
    np.testing.assert_array_equal(new[:,:7],q[:,:7])
    np.testing.assert_array_equal(nt[:,[0,1,2,4,5,6]],target[:,[0,1,2,4,5,6]])
    np.testing.assert_array_equal(nt[90:],target[90:])
    with pytest.raises(ValueError):absorb_finger_commands(q,target,command,[7],[3],[0.],[.3],bounds=(1,3),fps=20.)
