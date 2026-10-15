from types import SimpleNamespace as NS
from copy import deepcopy
import json
import numpy as np
import pytest
from flash_chord.retargeting.fixed_start import assert_initial_state,historical_prefix_check,better_demo
from flash_chord.retargeting.demo_archive import DemoArchive


@pytest.mark.parametrize('seed',[43,44,45])
def test_actual_reset_kernel_keeps_all_worlds_identical_across_repeated_resets(seed):
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceReset,ReferenceResetConfig
    with wp.ScopedDevice('cpu'):
        reset=ReferenceReset.build(2048,335,ReferenceResetConfig(fixed_initial_state=True,
            always_reset_to_first_frame=False,reset_finger_openness=.7,voc_decay_steps=0,seed=seed),device='cpu')
        mask=wp.ones(2048,dtype=wp.int32,device='cpu')
        for scale in [1.,.95,0.]:
            reset.set_reset_to_first_frame_probability(0.)
            reset.sample(mask,wp.array([scale],dtype=wp.float32,device='cpu'))
            np.testing.assert_array_equal(reset.reset_frame.numpy(),0)
            np.testing.assert_array_equal(reset.finger_scale.numpy(),1.)
            np.testing.assert_allclose(reset.applied_voc_scale.numpy(),scale)


def test_legacy_reset_randomization_is_opt_in_unchanged():
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceReset,ReferenceResetConfig
    with wp.ScopedDevice('cpu'):
        reset=ReferenceReset.build(256,335,ReferenceResetConfig(seed=43),device='cpu')
        reset.sample(wp.ones(256,dtype=wp.int32,device='cpu'),wp.ones(1,dtype=wp.float32,device='cpu'))
        assert len(np.unique(reset.reset_frame.numpy()))>1
        assert np.ptp(reset.finger_scale.numpy())>.1


def trace(failure):
    p=np.zeros((100,2));a=p.copy();p[failure:,1]=.08
    return p,a


def test_historical_prefix_blocks_regression_even_when_current_baseline_already_bad():
    best,ba=trace(50);regressed,ra=trace(20);candidate,ca=trace(30)
    # Candidate improves the regressed current controller, but loses saved success.
    assert historical_prefix_check(candidate,ca,regressed,ra,20.)['saved_prefix_preserved']
    check=historical_prefix_check(candidate,ca,best,ba,20.)
    assert not check['saved_prefix_preserved']
    assert 'earlier_failure_than_saved_demo' in check['saved_prefix_rejection_reasons']
    assert not better_demo(check,1.,10.)


def test_advancing_failure_is_progress_and_partial_improvement_does_not_require_solution():
    bp,ba=trace(50);p,a=trace(55)
    check=historical_prefix_check(p,a,bp,ba,20.)
    assert better_demo(check,11.,10.)
    check=historical_prefix_check(bp,ba,bp,ba,20.)
    assert better_demo(check,9.,10.)
    assert not better_demo(check,11.,10.)


def test_saved_prefix_protects_other_object_and_full_success():
    bp,ba=trace(50);p,a=trace(60);p[:40,0]=.02
    assert not historical_prefix_check(p,a,bp,ba,20.)['saved_prefix_preserved']
    complete,ca=trace(100)
    assert not historical_prefix_check(*trace(99),complete,ca,20.,persistence_seconds=.05)['saved_prefix_preserved']


def test_rejected_checkpoint_does_not_overwrite_best_pair(tmp_path,monkeypatch):
    editor=NS(output=tmp_path/'references',baseline=NS(fps=20.),timing_options={},
              initial_state_sha256='fixed',version=0)
    archive=DemoArchive(editor)
    def save(binding,learner,p,a,metrics,cost,step,origin,check):
        return dict(directory=origin,success_prefix_seconds=check['current_success_prefix_frames']/20.,
                    policy=learner,reference=binding,full_horizon_cost=cost)
    monkeypatch.setattr(archive,'_save_pair',save)
    p,a=trace(50)
    archive.consider('ref0','policy0',p,a,{},10.,10,'baseline')
    original=(tmp_path/'best_demo.json').read_bytes()
    p,a=trace(20)
    report=archive.consider('ref1','policy1',p,a,{},1.,20,'checkpoint')
    assert not report['best_pair_updated'] and not report['current_prefix_preserved']
    assert (tmp_path/'best_demo.json').read_bytes()==original
    assert json.loads(original)['policy']=='policy0'


def test_initial_state_signature_covers_robot_objects_and_velocities():
    array=lambda x:NS(numpy=lambda:np.asarray(x,dtype=np.float32))
    q=np.array([[1.,2.,3.,4.],[1.,2.,3.,4.]],np.float32)
    env=NS(world_count=2,state_0=NS(joint_q=array(q),joint_qd=array(np.zeros((2,3)))),
           reset_table=NS(ref_joint_q=array(q[:1]),object_root_position_offsets_w=array(np.zeros((2,0,3))),
                          object_root_q_ids=NS(numpy=lambda:np.array([],dtype=int))),timestep=array([0,0]))
    first=assert_initial_state(env)
    assert assert_initial_state(env,first['initial_state_sha256'])==first
    env.state_0.joint_q=array(q+1)
    with pytest.raises(AssertionError):assert_initial_state(env,first['initial_state_sha256'])
    env.state_0.joint_q=array(q);env.state_0.joint_qd=array(np.ones((2,3)))
    with pytest.raises(AssertionError):assert_initial_state(env,first['initial_state_sha256'])
