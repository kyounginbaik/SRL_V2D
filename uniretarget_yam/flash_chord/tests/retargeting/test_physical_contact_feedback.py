from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.retargeting.physical_contact_feedback import ContactFeedbackRecorder,patch_rollout_text


class Array:
    def __init__(self,value):self.value=np.asarray(value)
    def numpy(self):return self.value


def fixture():
    force=np.array([[[0,0,10],[0,0,20],[0,0,5],[0,0,999]],
                    [[20,0,0],[40,0,0],[10,0,0],[999,0,0]]],float)
    active=np.array([[1,1,1,0],[1,1,1,0]])
    observer=SimpleNamespace(sides=('receiver','giver'),slot_hand=(0,0,1,1),slot_object=(0,1,0,1),
        update=lambda env:dict(force=force[:env.world_count],active=active[:env.world_count]))
    env=SimpleNamespace(world_count=2,command=SimpleNamespace(body_ids_w=Array([[0,1],[2,3]])),
        model=SimpleNamespace(body_mass=Array([1.,2.,2.,4.]),gravity=Array([[0,0,-10],[-10,0,0]])))
    return env,observer,force


def test_forces_preserve_hand_object_identity_and_use_actual_gravity_and_mass():
    env,observer,force=fixture();original=force.copy()
    recorder=ContactFeedbackRecorder(lambda env:observer);recorder(env,0)
    data=recorder.arrays()
    np.testing.assert_allclose(data['hand_object_support_mg'][0,:,0],1.)
    np.testing.assert_allclose(data['hand_object_support_mg'][0,:,1,0],.5)
    np.testing.assert_allclose(data['hand_object_support_mg'][0,:,1,1],0.)
    np.testing.assert_array_equal(force,original)
    assert data['hand_object_contact_count'][0,0,1,1]==0


def test_observer_rejects_frame_gaps_and_restarts_for_new_batch():
    env,observer,_=fixture();recorder=ContactFeedbackRecorder(lambda env:observer)
    with pytest.raises(ValueError):recorder(env,1)
    recorder(env,0)
    with pytest.raises(ValueError):recorder(env,2)
    next_env=SimpleNamespace(world_count=1,command=SimpleNamespace(body_ids_w=Array([[0,1]])),
        model=SimpleNamespace(body_mass=env.model.body_mass,gravity=Array([[0,0,-10]])))
    recorder(next_env,0)
    assert recorder.arrays()['hand_object_force_w'].shape[:2]==(1,1)


def test_execution_capture_preserves_robot_order_applied_delay_and_frame_snapshots():
    env,observer,_=fixture()
    env.scene=SimpleNamespace(robot_reference=SimpleNamespace(num_joint_q=3,num_joint_dof=2))
    q=np.arange(20.,dtype=np.float32).reshape(2,10)
    command=np.arange(16.,dtype=np.float32).reshape(2,8)
    applied=command+100.
    env.state_0=SimpleNamespace(joint_q=Array(q))
    env.action=SimpleNamespace(joint_target=Array(command),delay=SimpleNamespace(delayed_target=Array(applied)))
    recorder=ContactFeedbackRecorder(lambda env:observer,record_execution=True);recorder(env,0)
    q_before=q.copy();command_before=command.copy();applied_before=applied.copy()
    q[:]+=50;command[:]+=50;applied[:]+=50;recorder(env,1)
    data=recorder.arrays()
    np.testing.assert_array_equal(data['executed_robot_joint_q'][0],q_before[:,:3])
    np.testing.assert_array_equal(data['commanded_robot_joint_target'][0],command_before[:,:2])
    np.testing.assert_array_equal(data['applied_robot_joint_target'][0],applied_before[:,:2])
    np.testing.assert_array_equal(data['executed_robot_joint_q'][1],q[:,:3])
    np.testing.assert_array_equal(data['execution_frame_ids'],[0,1])
    env.action.delay=None
    recorder(env,2);data=recorder.arrays()
    np.testing.assert_array_equal(data['applied_robot_joint_target'][2],command[:,:2])


def test_rollout_observer_hook_is_explicit_idempotent_and_after_physics():
    source="def run():\n    for frame in range(3):\n        action,key=policy(state,obs,key);tr=adapter.step(action);obs=tr.observation\n        jax.block_until_ready(obs);_clock['policy_and_sim']+=_time.perf_counter()-_t;_t=_time.perf_counter()\n"
    patched=patch_rollout_text(source)
    compile(patched,'rollout','exec')
    assert patched.index('adapter.step')<patched.index('_physical_observer(env,frame)')
    assert patched==patch_rollout_text(patched)
    with pytest.raises(ValueError):patch_rollout_text('def run(): pass')
