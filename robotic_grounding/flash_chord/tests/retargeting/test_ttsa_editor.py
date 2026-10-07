import numpy as np
from flash_chord.retargeting.ttsa_editor import (
    temporal_basis, bounded_correction, upper_cost, study_basis, choose_local_centers,
    robust_world_summary, signed_feedback_summary, passes_study_gate,
    consensus_frame_scores,
)


def test_study_bases_share_endpoints_and_motion_bounds():
    rng=np.random.default_rng(42)
    for variant, centers in (("smooth",None),("per_frame",None),("global_local",[.2,.4,.6,.8])):
        basis=study_basis(100,variant,local_centers=centers)
        np.testing.assert_allclose(basis[[0,-1]],0,atol=1e-6)
        delta=bounded_correction(rng.normal(size=basis.shape[1]*2),basis,np.eye(2),
                                 bound=.12,fps=20,max_velocity=.5)
        assert np.max(np.abs(delta)) <= .120001
        assert np.max(np.abs(np.diff(delta,axis=0)))*20 <= .500001
    assert study_basis(100,"per_frame").shape==(100,98)


def test_local_centers_find_separated_repeated_peaks():
    score=np.zeros(100)
    score[[20,50,80]]=[3,2,1]
    assert np.allclose(choose_local_centers(score,count=3),[20/99,50/99,80/99])


def test_signed_feedback_keeps_opposite_world_spread():
    residual=np.zeros((32,4,2,3),dtype=np.float32)
    residual[:,0,:,0]=.04
    residual[:,1,:,0]=-.04
    pooled=signed_feedback_summary(residual,bins=4)
    assert pooled.shape==(4,2,2,3)
    np.testing.assert_allclose(pooled[:,0],0,atol=1e-7)
    np.testing.assert_allclose(pooled[:,1,:,0],np.sqrt(.0008),atol=1e-6)
    assert np.allclose(consensus_frame_scores(residual),0)
    residual[:,1,:,0]=.04
    residual[:,2,:,0]=.04
    residual[:,3,:,0]=.04
    np.testing.assert_allclose(consensus_frame_scores(residual),.04,atol=1e-6)


def test_robust_world_score_retains_ranking_with_one_explosion():
    pos=np.full((10,4),.02)
    ori=np.full((10,4),.1)
    pos[:,3]=1e4
    summary=robust_world_summary(pos,ori)
    assert .25 < summary['object_position_robust_m'] < .3
    assert summary['unstable_world_fraction']==.25
    base=dict(summary,robot_position_robust_mse=.0001)
    better=dict(base,object_position_robust_m=base['object_position_robust_m']-.01)
    delta=np.zeros((10,2))
    assert upper_cost(better,delta,20,0,robust=True) < upper_cost(base,delta,20,0,robust=True)
    assert passes_study_gate(better,base)
    assert not passes_study_gate(dict(better,unstable_world_fraction=.5),base)
    active=np.zeros_like(pos,dtype=bool)
    active[:2]=True
    wrench=np.zeros_like(pos)
    wrench[:2]=4.
    assert robust_world_summary(pos,ori,wrench,active)['wrench_robust_mse']==4.


def test_joint_clipping_does_not_break_velocity_bound():
    from flash_chord.retargeting.ttsa_editor import project_reference
    base=np.zeros((100,1),np.float32)
    base[50]=.999
    delta=.03*np.sin(np.linspace(0,np.pi,100))[:,None]
    clipped=np.clip(base+delta,-1,1)-base
    assert np.max(np.abs(np.diff(delta,axis=0)))*20 < .5
    assert np.max(np.abs(np.diff(clipped,axis=0)))*20 > .5
    q=project_reference(base,delta,np.array([0]),np.array([-1]),np.array([1]),20)
    assert np.max(np.abs(np.diff(q-base,axis=0)))*20 <= .50001
    np.testing.assert_array_equal(q[[0,-1]],base[[0,-1]])
    assert np.all(np.abs(q)<=1)


def test_saved_failed_run_references():
    import json
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import pytest
    from flash_chord.retargeting.ttsa_editor import project_reference
    root=Path(__file__).resolve().parents[2]
    snapshots=list((root/'outputs/tissue_ttsa_hand_20261006').glob('reactor*/references/reference_*.npz'))
    if not snapshots:
        pytest.skip('production reference artifacts not present')
    urdf=ET.parse(root/'src/flash_chord/assets/urdfs/vega_sharpa/v2/vega_sharpa_58dof.urdf')
    limits={j.attrib['name']:(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
            for j in urdf.findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    rng=np.random.default_rng(123)
    for path in snapshots:
        with np.load(path) as saved, np.load(path.parent/'reference_0000.npz') as original:
            base=original['joint_q']; previous=saved['joint_q']
            header=json.loads(str(saved['header'])); fps=header['fps']
            lower,upper=np.asarray([limits[n] for n in header['joint_names']]).T
        for _ in range(20):
            proposal=bounded_correction(rng.normal(size=3*base.shape[1]),temporal_basis(len(base)),
                np.eye(base.shape[1]),bound=.12,fps=fps,max_velocity=.5)
            q=project_reference(base,.75*(previous-base)+.25*proposal,np.arange(base.shape[1]),lower,upper,fps)
            assert np.max(np.abs(q-base)) <= .12001
            assert np.max(np.abs(np.diff(q-base,axis=0)))*fps <= .50001
            assert np.all(q>=lower-1e-6) and np.all(q<=upper+1e-6)


def test_correction_bounds_and_endpoints():
    basis=temporal_basis(518)
    delta=bounded_correction(np.ones(3*58),basis,np.eye(58),bound=.12,fps=20,max_velocity=.5)
    np.testing.assert_array_equal(delta[[0,-1]],0)
    assert np.max(np.abs(delta)) <= .120001
    assert np.max(np.abs(np.diff(delta,axis=0)))*20 <= .500001
    np.testing.assert_array_equal(bounded_correction(np.zeros(174),basis,np.eye(58),bound=.12,fps=20,max_velocity=.5),0)


def test_wrench_is_in_reward_not_only_gate():
    metrics=dict(object_position_mean_m=.02,object_orientation_mean_rad=.1,
                 object_success_fraction=.9,robot_position_mse=.0001,wrench_mse=2.)
    delta=np.zeros((10,58))
    assert np.isclose(upper_cost(metrics,delta,20,1)-upper_cost(metrics,delta,20,0),2.)
    worse=dict(metrics,object_position_mean_m=.2)
    assert upper_cost(worse,delta,20,0) > upper_cost(metrics,delta,20,0)


def test_real_upper_flashsac_updates():
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from flash_chord.retargeting.ttsa_editor import TTSAReferenceEditor
    from flash_chord.training.flash_sac.learner import Transition
    editor=object.__new__(TTSAReferenceEditor)
    editor.upper=None
    editor.training=SimpleNamespace(seed=42)
    editor.basis=np.zeros((10,3))
    editor.joint_basis=np.eye(2)
    obs=jnp.zeros((1,8),jnp.float32)
    action,context=editor.prepare_upper(obs)
    before=jax.tree.map(lambda x:np.array(x,copy=True),editor.upper.actor.params)
    for i in range(8):
        transition=Transition(action=action,objective_terms=jnp.asarray([[float(i%2)]],jnp.float32),
            terminated=jnp.ones(1,jnp.int32),truncated=jnp.zeros(1,jnp.int32),
            episode_reference_progress=jnp.ones(1,jnp.float32),termination_diagnostics=jnp.zeros((1,0),jnp.float32),
            next_observation=obs,next_critic_context=context,post_reset_observation=obs,post_reset_critic_context=context)
        editor.upper,action,metrics=editor.executables.consume_transition(editor.upper,transition,
            jnp.ones(1,jnp.float32),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
    assert int(editor.upper.global_update_step)>0
    assert np.isfinite(float(metrics.critic_loss_mean_per_update))
    assert max(jax.tree.leaves(jax.tree.map(lambda a,b:np.max(np.abs(np.asarray(a)-b)),editor.upper.actor.params,before)))>0
    old_step=int(editor.upper.global_update_step)
    action,context=editor.prepare_upper(obs)
    assert int(editor.upper.replay.size)==0
    assert int(editor.upper.global_update_step)==old_step
    # A second block must work with the SAME context objects without donation aliasing.
    transition=transition.replace(action=action,next_observation=obs,next_critic_context=context,
        post_reset_observation=obs,post_reset_critic_context=context)
    editor.upper,_,_=editor.executables.consume_transition(editor.upper,transition,
        jnp.ones(1,jnp.float32),jnp.float32(1),jnp.float32(0),jnp.bool_(False),jnp.bool_(True))
    assert int(editor.upper.replay.size)==1
