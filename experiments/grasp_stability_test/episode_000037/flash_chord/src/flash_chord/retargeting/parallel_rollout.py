"""Distinct-candidate vectorized rollouts for floating Sharpa hands.

Robot targets carry a [frame, world] axis. Object/contact commands do not.
Endpoints must equal the published reference so resets stay exactly comparable.
"""
from dataclasses import replace
import numpy as np
import newton
import warp as wp
from scipy.spatial.transform import Rotation
from flash_chord.embodiments.binding import RobotReferenceBinding, DeviceRobotReference, _bind_hand, _frame_poses
from flash_chord.objectives.tracking import _hand_reference, _HAND_KEYPOINTS, _HAND_JOINT_POS
from flash_chord.training.environment import WarpRLEnv


def _safe_angle(q_a, q_b, *, scalar_first_a=False):
    """Geodesic angle between quaternion batches; degenerate inputs (exploded worlds) count as pi."""
    q_a = np.asarray(q_a, dtype=np.float64).reshape(-1, 4); q_b = np.asarray(q_b, dtype=np.float64).reshape(-1, 4)
    bad = ~(np.isfinite(q_a).all(1) & np.isfinite(q_b).all(1) & (np.linalg.norm(q_a, axis=1) > 1e-8) & (np.linalg.norm(q_b, axis=1) > 1e-8))
    ident_a = np.array([1., 0., 0., 0.]) if scalar_first_a else np.array([0., 0., 0., 1.])
    q_a = np.where(bad[:, None], ident_a, q_a); q_b = np.where(bad[:, None], np.array([0., 0., 0., 1.]), q_b)
    angle = (Rotation.from_quat(q_a, scalar_first=scalar_first_a) * Rotation.from_quat(q_b).inv()).magnitude()
    return np.where(bad, np.pi, angle)


def pack_candidate_references(env, q, target):
    """FK over all candidate worlds together, once per trajectory frame."""
    q=np.asarray(q,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
    worlds,frames,nq=q.shape;scene=env.scene;layout=scene.layout
    if worlds!=env.world_count or frames!=scene.robot_reference.num_frames:
        raise ValueError('candidate [world, frame] dimensions do not match evaluator')
    if not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError('nonfinite candidate')
    for frame in [0,frames-1]:
        np.testing.assert_allclose(q[:,frame],np.broadcast_to(scene.robot_reference.joint_q[frame],(worlds,nq)),atol=1e-6)
    flatq=q.transpose(1,0,2).reshape(frames*worlds,nq)
    flatt=target.transpose(1,0,2).reshape(frames*worlds,-1)
    # Use the existing multiworld model for FK; do not construct 2048 models.
    base=env.state_0.joint_q.numpy().reshape(worlds,-1).copy()
    fk=env.model.state();fk.joint_qd.zero_();per_body=env.model.body_count//worlds
    # Keep robot body transforms only; object geometry is immutable and is not packed.
    robot_bodies=max(f.body_id for h in layout.hands for f in (h.palm_frame,*h.dp_frames,*h.fingertip_frames))+1
    body=np.empty((frames,worlds,robot_bodies,7),np.float32)
    for frame in range(frames):
        base[:,:nq]=q[:,frame];fk.joint_q.assign(base.reshape(-1))
        newton.eval_fk(env.model,fk.joint_q,fk.joint_qd,fk)
        body[frame]=fk.body_q.numpy().reshape(worlds,per_body,7)[:,:robot_bodies]
    flatbody=body.reshape(frames*worlds,robot_bodies,7)
    packed=RobotReferenceBinding(num_frames=frames*worlds,fps=scene.robot_reference.fps,
        num_joint_q=nq,num_joint_dof=scene.robot_reference.num_joint_dof,
        joint_q=flatq,joint_target=flatt,
        hands=tuple(_bind_hand(flatbody,layout.hand(side),flatq) for side in layout.sides))
    return packed


def install_candidate_references(env, packed, allow_auto_reset=False):
    """Wire the same world-specific reference into actions, observations and reward."""
    worlds=env.world_count;frames=env.scene.robot_reference.num_frames
    if packed.num_frames!=worlds*frames:raise ValueError('packed reference length mismatch')
    if env.config.auto_reset and not allow_auto_reset:raise ValueError('parallel search requires full-horizon non-resetting evaluation')
    from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction
    if not isinstance(env.action, ResidualHandPoseAction):
        raise ValueError('parallel search requires residual hand-pose actions')
    device=env.model.device
    for component in [env.action.buffers,env.observation_strategy]:
        old=component.reference
        ref=DeviceRobotReference.build(packed,env.scene.layout,sides=old.sides,device=device)
        component.reference=replace(ref,num_frames=frames,reference_world_count=worlds)
    objective=env.objective
    keys,_,_,fingers,_=_hand_reference(env.scene.layout,packed,objective.config.sides,
        keypoint_frame=objective.config.hand_keypoint_frame,
        include_keypoints=objective.enabled[_HAND_KEYPOINTS],include_fingers=objective.enabled[_HAND_JOINT_POS])
    objective.ref_hand_keypoints_w=wp.array(keys.reshape(-1,3),dtype=wp.vec3,device=device)
    objective.ref_finger_joint_pos=wp.array(fingers.reshape(-1),dtype=wp.float32,device=device)
    objective.reference_world_count=worlds


def rollout_candidates(editor, learner, q, targets, *, record_indices=()):
    """One simulation pass, one separately reduced score per candidate world."""
    import jax
    import jax.numpy as jnp
    from flash_chord.training.flash_sac.checkpoint import ActorInferenceState
    from flash_chord.training.flash_sac.evaluation import create_actor_template,compile_policy_action
    from flash_chord.retargeting.ttsa_editor import robust_world_summary
    from flash_chord.retargeting.contact_proximity import summarize_contact_proximity
    from copy import deepcopy
    from flash_chord.envs.rl import RLEnv
    # One fresh simulator per whole batch keeps solver warm-start history from
    # a previous batch out of candidate scores. Never one constructor per candidate.
    import time as _time, json as _json
    _clock={'setup':0.,'policy_and_sim':0.,'host_metrics':0.,'contact_proximity':0.,'per_world_summary':0.};_t=_time.perf_counter()
    env=RLEnv(editor.parallel_scene,editor.parallel_reference,config=deepcopy(editor.parallel_config))
    env.reset(frame_id=0)
    packed=pack_candidate_references(env,q,targets);install_candidate_references(env,packed)
    env._step_graph = None  # Recapture once per batch after swapping reference buffers.
    env.reset(frame_id=0)
    if env.device.is_cuda: env.capture_step()
    env.reset(frame_id=0)
    adapter=WarpRLEnv(env,publish_diagnostics=False,publish_step_metrics=False)
    from flash_chord.runtime.jax import to_jax
    obs=jnp.array(to_jax(env.reset(frame_id=0),(env.world_count,env.observation_dim)),copy=True);state=ActorInferenceState(learner.actor.params,learner.actor.batch_stats)
    key=jax.random.PRNGKey(editor.training.seed)
    if editor.parallel_policy is None:
        actor,_=create_actor_template(editor.training,obs.shape[1],adapter.action_dim)
        editor.parallel_policy=compile_policy_action(actor,state,obs,key,deterministic=True,
            action_transform=editor.training.network.action_transform,action_scale=editor.training.network.action_scale)
    B,F=q.shape[:2];O=env.command.layout.num_objects;H=len(env.scene.layout.hands)
    commanded_p=np.stack([packed.hand(s).keypoint_pos_w('fingertip') for s in env.scene.layout.sides],axis=1).reshape(F,B,H*6,3)
    commanded_q=np.concatenate([np.concatenate([packed.hand(s).wrist_quat_w[:,None,:],packed.hand(s).fingertip_quat_w],axis=1) for s in env.scene.layout.sides],axis=1).reshape(F,B,H*6,4)
    pos=[];angles=[];robot_errors=[];robot_angles=[];wrench=[];active=[];signed=[];objpose=[];palmpose=[]
    sums=np.zeros((B,H,O));counts=np.zeros((B,H,O),np.int64);recorded=[]
    _clock['setup']+=_time.perf_counter()-_t
    for frame in range(F-1):
        _t=_time.perf_counter()
        np.testing.assert_array_equal(env.timestep.numpy(),np.full(B,frame))
        action,key=editor.parallel_policy(state,obs,key);tr=adapter.step(action);obs=jnp.array(tr.observation,copy=True)
        jax.block_until_ready(obs);_clock['policy_and_sim']+=_time.perf_counter()-_t;_t=_time.perf_counter()
        bodies=env.state_0.body_q.numpy().reshape(B,-1,7)
        p,r=_frame_poses(bodies,editor.frames)
        robot_errors.append(commanded_p[frame]-p)
        robot_angles.append(_safe_angle(commanded_q[frame], r, scalar_first_a=True).reshape(B,-1))
        actual=env.state_0.body_q.numpy()[env.command.body_ids_w.numpy().reshape(-1)].reshape(B,O,7)
        tp=env.command.body_target_pos_w.numpy().reshape(B,O,3);tq=env.command.body_target_quat_w.numpy().reshape(B,O,4)
        signed.append((actual[:,:,:3]-tp).astype(np.float32));objpose.append(np.concatenate([actual,tp,tq],axis=-1).astype(np.float32));palmpose.append(np.concatenate([p[:,::6],r[:,::6]],axis=-1).astype(np.float32));pos.append(np.linalg.norm(actual[:,:,:3]-tp,axis=-1));angles.append(_safe_angle(actual[:,:,3:], tq).reshape(B,O))
        cur=env.objective.current_wrench_support.numpy().reshape(B,-1);tar=env.objective.wrench_support_target.numpy().reshape(B,-1);mask=np.mean(tar**2,axis=1)>1e-8
        w=np.where(mask,np.mean((cur-tar)**2,axis=1)/np.maximum(np.mean(tar**2,axis=1),1e-6),0);wrench.append(w);active.append(mask)
        _clock['host_metrics']+=_time.perf_counter()-_t;_t=_time.perf_counter()
        if editor.contact_proximity is not None:
            a,c=editor.contact_proximity.frame(np.full(B,frame),p.reshape(B,H,6,3)[:,:,1:],actual)[:2];sums+=a;counts+=c
        if record_indices:recorded.append(bodies[list(record_indices)].copy())
        _clock['contact_proximity']+=_time.perf_counter()-_t
    _t=_time.perf_counter()
    editor._signed_object_offset=np.asarray(signed)  # (F-1, B, O, 3): physics feedback for ILC candidates
    editor._object_pose=np.asarray(objpose); editor._palm_pose=np.asarray(palmpose)  # (F-1,B,O,14) act|tgt pos+quat xyzw, (F-1,B,H,7)
    pos=np.asarray(pos);angles=np.asarray(angles);pe=np.asarray(robot_errors);ra=np.asarray(robot_angles);wrench=np.asarray(wrench);active=np.asarray(active)
    results=[]
    for world in range(B):
        p=pos[:,world];a=angles[:,world];e=pe[:,world];w=wrench[:,world];act=active[:,world]
        m=dict(robot_position_mse=float(np.mean(e**2)),robot_rotation_mse=float(np.mean(ra[:,world]**2)),wrench_mse=float(w[act].mean()) if act.any() else 0.,object_position_mean_m=float(p.mean()),object_position_p95_m=float(np.percentile(p,95)),object_orientation_mean_rad=float(a.mean()),object_success_fraction=float(((p<.1)&(a<.5)).mean()),per_object_position_p95_m=np.percentile(p,95,axis=0).tolist(),per_object_orientation_p95_rad=np.percentile(a,95,axis=0).tolist(),per_object_gate_success_fraction=((p<=.05)&(a<=np.deg2rad(20))).mean(axis=0).tolist(),robot_position_rms_m=float(np.sqrt(np.mean(np.sum(e**2,axis=-1)))),all_errors_finite=bool(np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(e).all()),full_horizon=True,per_object_position_mean_m=p.mean(axis=0).tolist(),per_object_orientation_mean_rad=a.mean(axis=0).tolist(),robot_position_robust_mse=float(np.minimum(e**2,.1**2).mean()))
        m.update(robust_world_summary(p.mean(axis=1)[:,None],a.mean(axis=1)[:,None],w[:,None],act[:,None]))
        if editor.contact_proximity is not None:m.update(summarize_contact_proximity(sums[world:world+1],counts[world:world+1]))
        results.append(m)
    _clock['per_world_summary']+=_time.perf_counter()-_t
    editor.validation_steps+=B*(F-1)
    try:
        with (editor.output.parent/'upper_timing.jsonl').open('a') as _s:_s.write(_json.dumps({'worlds':B,'frames':F,**{k:round(v,2) for k,v in _clock.items()}})+'\n')
    except Exception:pass
    return results,packed,np.asarray(recorded)
