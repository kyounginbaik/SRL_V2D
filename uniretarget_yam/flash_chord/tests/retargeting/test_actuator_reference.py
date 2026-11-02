from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
import pytest

from flash_chord.retargeting.actuator_reference import WristNominalReference,patch_nominal_reference_hook


class Array:
    def __init__(self,x):self.x=np.asarray(x)
    def numpy(self):return self.x


@dataclass(frozen=True)
class Ref:
    num_frames:int
    reference_world_count:int
    sides:tuple
    wrist_pos_w:Array
    wrist_quat_w:Array
    finger_joint_pos:Array
    joint_q:Array


def fixture():
    p=np.zeros((10,3,2,3),np.float32);r=np.zeros((10,3,2,4),np.float32);r[...,3]=1
    ref=Ref(10,3,('left','right'),Array(p),Array(r),Array(np.full((10,3,4),.2)),Array(np.ones((10,3,12))))
    left=NS(wrist_pos_dof_ids=(6,7,8),wrist_orient_dof_ids=(9,10,11))
    right=NS(wrist_pos_dof_ids=(0,1,2),wrist_orient_dof_ids=(3,4,5))
    env=NS(world_count=3,timestep=Array(np.zeros(3,int)),
        action=NS(buffers=NS(reference=ref)),observation_strategy=NS(reference=ref),
        scene=NS(layout=NS(hand=lambda side:dict(left=left,right=right)[side])))
    targets=np.zeros((2,9,12),np.float32);targets[0,:,0]=.15;targets[0,:,3]=.3
    candidates=[dict(wrist_nominal_reference=[dict(hand='right',bounds=[2,7])]),{}]
    return env,targets,np.array([1,0,1]),candidates,ref


def test_only_declared_action_nominal_wrist_changes_geometry_fingers_and_other_worlds_exact():
    env,targets,groups,candidates,old=fixture()
    install=WristNominalReference(targets,groups,candidates,array_factory=lambda x,old:Array(x.copy()))
    install(env);new=env.action.buffers.reference
    assert env.observation_strategy.reference is old
    assert new.finger_joint_pos is old.finger_joint_pos and new.joint_q is old.joint_q
    p=new.wrist_pos_w.numpy().reshape(10,3,2,3)
    np.testing.assert_array_equal(p[:,[0,2]],0.)
    np.testing.assert_array_equal(p[:,:,0],0.)
    np.testing.assert_allclose(p[2:8,1,1,0],.15)
    np.testing.assert_array_equal(p[:2],old.wrist_pos_w.numpy()[:2])
    np.testing.assert_array_equal(p[8:],old.wrist_pos_w.numpy()[8:])
    np.testing.assert_array_equal(old.wrist_pos_w.numpy(),0.)
    assert install.installations[0]['controlled_hand_frames']==6


def test_nominal_binding_rejects_late_installation_wrong_population_and_reset_edits():
    env,targets,groups,candidates,_=fixture();install=WristNominalReference(targets,groups,candidates)
    env.timestep.x[:]=1
    with pytest.raises(ValueError):install(env)
    candidates[0]['wrist_nominal_reference'][0]['bounds']=[0,7]
    with pytest.raises(ValueError):WristNominalReference(targets,groups,candidates)


def test_nominal_hook_precedes_graph_capture_and_is_idempotent():
    source="def run():\n    packed=pack_candidate_references(env,q,targets);install_candidate_references(env,packed)\n    env.capture_step()\n"
    patched=patch_nominal_reference_hook(source)
    assert patched.index('_nominal_installer(env)')<patched.index('env.capture_step()')
    assert patch_nominal_reference_hook(patched)==patched
    with pytest.raises(ValueError):patch_nominal_reference_hook('def other(): pass')
