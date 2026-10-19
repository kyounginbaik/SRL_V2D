"""Small CPU integration check; no training or scheduler submission."""
import json
import os
from pathlib import Path
from dataclasses import replace
import numpy as np
import warp as wp
from omegaconf import OmegaConf
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.retargeting.parallel_rollout import pack_candidate_references,install_candidate_references
b=Path(__file__).resolve().parents[1]
checkpoint=b.parent/'uniretarget_final_ep01_20261013/runs/smoke/episode_000001/chunk/seed_43/training/policy_3145728.safetensors'
saved=checkpoint_config(read_checkpoint_metadata(checkpoint));apply_evaluation_composition(saved,world_count=2)
start_frame=int(os.environ.get('VALIDATE_START_FRAME','0'))
saved['task']['motion_start_frame']=start_frame
saved['task']['motion_end_frame']=start_frame+12
cfg=OmegaConf.create(saved)
with wp.ScopedDevice('cpu'):
 ec=instantiate_typed(cfg.env,RLEnvConfig);emb=instantiate_typed(cfg.embodiment,Embodiment);col=instantiate_typed(cfg.collision,CollisionPolicy)
 setup=setup_scene(parquet=cfg.task.parquet,control_fps=ec.sim.fps,motion_speed=cfg.task.motion_speed,embodiment=emb,collision=col,world_count=2,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,contact_friction=cfg.scene.contact_friction,source_frame_playback=cfg.task.source_frame_playback,motion_start_frame=start_frame,motion_end_frame=start_frame+12)
 env=RLEnv(setup.scene,setup.reference,config=ec);env.reset(frame_id=0)
 ref=setup.scene.robot_reference;F=ref.num_frames
 q=np.repeat(ref.joint_q[None],2,axis=0);target=np.repeat(ref.joint_target[None],2,axis=0)
 joints=setup.scene.layout.scalar_joints;i=list(joints.names).index('left_wrist_x');qi=joints.q_ids[i];di=joints.dof_ids[i]
 delta=.01*np.sin(np.linspace(0,np.pi,F));delta[[0,-1]]=0
 q[0,:,qi]+=delta;q[1,:,qi]-=delta;target[0,:,di]+=delta;target[1,:,di]-=delta
 packed=pack_candidate_references(env,q,target);install_candidate_references(env,packed)
 env.reset(frame_id=0)
 # Opposite translations must reach action processing and tracking target arrays.
 env.timestep.fill_(F//2)
 zero=wp.zeros(2*env.action.action_dim,dtype=wp.float32,device='cpu')
 env.action.process(zero,env.timestep,env.state_0)
 processed=env.action.joint_target.numpy().reshape(2,-1)
 np.testing.assert_allclose(processed[0,di]-processed[1,di],2*delta[F//2],atol=1e-5)
 env.objective.evaluate(env.state_0,env.timestep,env.done)
 objective_before=env.objective.hand_keypoint_target_w.numpy().reshape(2,-1,3).copy()
 np.testing.assert_allclose(objective_before[0,list(env.objective.config.sides).index("left")*6,0]-objective_before[1,list(env.objective.config.sides).index("left")*6,0],2*delta[F//2],atol=1e-5)
 obs_before=env.observation_strategy.compute(env.state_0).numpy().reshape(2,-1).copy()
 assert np.max(np.abs(obs_before[0]-obs_before[1]))>1e-4
 assert env.observation_strategy.reference.reference_world_count==2
 assert env.objective.reference_world_count==2
 # Swap candidate worlds; references and action targets must swap with them.
 before=processed.copy()
 install_candidate_references(env,pack_candidate_references(env,q[::-1],target[::-1]))
 env.action.process(zero,env.timestep,env.state_0)
 np.testing.assert_allclose(env.action.joint_target.numpy().reshape(2,-1),before[::-1],atol=1e-5)
 env.objective.evaluate(env.state_0,env.timestep,env.done)
 np.testing.assert_allclose(env.objective.hand_keypoint_target_w.numpy().reshape(2,-1,3),objective_before[::-1],atol=1e-5)
 obs_after=env.observation_strategy.compute(env.state_0).numpy().reshape(2,-1).copy()
 np.testing.assert_allclose(obs_after,obs_before[::-1],atol=1e-5)
 # Test a complete batched physics rollout and its score reduction with a fixed zero policy.
 from flash_chord.retargeting.contact_proximity import ContactProximity
 from types import SimpleNamespace
 import jax.numpy as jnp
 from flash_chord.retargeting.parallel_rollout import rollout_candidates
 editor=SimpleNamespace(parallel_scene=setup.scene,parallel_reference=setup.reference,parallel_config=ec,training=SimpleNamespace(seed=43),
   parallel_policy=lambda state,obs,key:(jnp.zeros((2,env.action.action_dim)),key),
   frames=tuple(f for h in setup.scene.layout.hands for f in (h.palm_frame,*h.fingertip_frames)),
   contact_proximity=ContactProximity(setup.reference,list(setup.scene.layout.sides)),validation_steps=0)
 learner=SimpleNamespace(actor=SimpleNamespace(params=None,batch_stats=None))
 metrics,_,_=rollout_candidates(editor,learner,q,target)
 assert len(metrics)==2 and all(m['all_errors_finite'] for m in metrics)
 swapped,_,_=rollout_candidates(editor,learner,q[::-1],target[::-1])
 for first,second in zip(metrics,swapped[::-1]):
  for key in ['robot_position_mse','object_position_mean_m','object_orientation_mean_rad','contact_proximity_mean_m']:
   np.testing.assert_allclose(first[key],second[key],atol=1e-5)
 result={'passed' :True,'worlds':2,'frames':F,'opposite_candidate_action_difference_m':float(before[0,di]-before[1,di]),'training_started':False,'source_start_frame':start_frame,'checks':['world-specific action targets','observation and objective permutation','physics rollout score permutation','contact proximity scores']}
 (b/f'validation_frame_{start_frame}.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
