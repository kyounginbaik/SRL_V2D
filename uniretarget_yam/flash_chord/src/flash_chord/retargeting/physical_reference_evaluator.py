"""Fresh full-prefix native-reference tests without policy training or assistance.

This evaluator uses a zero-residual lower controller. Its failures do not prove
that a reference is unlearnable. A policy-conditioned comparison is still needed
before attributing any benefit to the full bilevel method.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import warp as wp
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.retargeting.evaluation_task import apply_task_protocol
from flash_chord.retargeting.native_contact_capture import NativeContactCapture
from flash_chord.retargeting.parallel_rollout import pack_candidate_references, install_candidate_references
from flash_chord.retargeting.reference_snapshot import command_digest
from flash_chord.retargeting.scalable_refinement import official_scores
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


def evaluate_native_references(configuration, manifest, joint_q, joint_target, *,
                               replicas=2, device='cpu', steps=None, output=None):
    """Evaluate [candidate,frame,q/dof] references from the real task reset.

    Every batch constructs a fresh simulator to avoid inherited solver history.
    Full evaluation always uses T-1 transitions. A shorter steps value is an
    explicit smoke test and is NEVER marked an official full-episode result.
    Geometry and native actuator targets are distinct inputs. Physical target
    preload is allowed; this function adds no controller or object assistance.
    The caller must enforce command envelopes and path feasibility separately.
    """
    q=np.asarray(joint_q,np.float32); target=np.asarray(joint_target,np.float32)
    if (q.ndim!=3 or target.ndim!=3 or q.shape[:2]!=target.shape[:2]
            or min(q.shape)<1 or q.shape[1]<2 or not np.isfinite(q).all()
            or not np.isfinite(target).all() or type(replicas) is not int or replicas<1):
        raise ValueError('finite candidate/reference trajectories and positive replica count required')
    if device.startswith('cuda'):
        import os
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('GPU physics requires a real Slurm allocation')
    frames=q.shape[1]; count=frames-1 if steps is None else steps
    if type(count) is not int or not 1<=count<=frames-1: raise ValueError('invalid transition count')
    groups=np.repeat(np.arange(len(q)),replicas); worlds=len(groups)
    resolved=apply_task_protocol(configuration,manifest)
    apply_evaluation_composition(resolved,world_count=worlds)
    resolved['env']['action']['reference_source']='native_targets'
    config=instantiate_typed(OmegaConf.create(resolved['env']),RLEnvConfig)
    embodiment=instantiate_typed(OmegaConf.create(resolved['embodiment']),Embodiment)
    collision=instantiate_typed(OmegaConf.create(resolved['collision']),CollisionPolicy)
    if config.auto_reset: raise ValueError('physical comparison may not reset failed worlds')
    started=time.monotonic()
    with wp.ScopedDevice(device):
        task,scene=resolved['task'],resolved['scene']
        setup=setup_scene(parquet=task['parquet'],control_fps=config.sim.fps,
            motion_speed=task['motion_speed'],embodiment=embodiment,collision=collision,
            world_count=worlds,include_support=scene['support'],decompose_objects=scene['decompose_objects'],
            contact_friction=scene.get('contact_friction',1.),
            object_free_joint_damping=scene.get('object_free_joint_damping',0.),
            robot_reference_path=manifest['reference'],robot_reference_sha256=manifest['reference_sha256'])
        digest=command_digest(setup.reference)
        required=manifest.get('evaluation_task_protocol',{}).get('object_contact_sha256',digest)
        if digest!=required: raise ValueError('immutable task/contact commands changed')
        env=RLEnv(setup.scene,setup.reference,config=config); env.reset(frame_id=0)
        packed=pack_candidate_references(env,q[groups],target[groups])
        install_candidate_references(env,packed)
        env._step_graph=None
        env.reset(frame_id=0)
        if env.device.is_cuda: env.capture_step(); env.reset(frame_id=0)
        capture=NativeContactCapture(np.arange(0,worlds,replicas))
        action=wp.zeros(worlds*env.action.action_dim,dtype=wp.float32,device=device)
        positions=[]
        for frame in range(count):
            np.testing.assert_array_equal(env.timestep.numpy(),np.full(worlds,frame))
            env.step(action); capture(env,frame)
            bodies=env.state_0.body_q.numpy()
            actual=bodies[env.command.body_ids_w.numpy().reshape(-1)].reshape(worlds,-1,7)
            goalp=env.command.body_target_pos_w.numpy().reshape(worlds,-1,3)
            goalq=env.command.body_target_quat_w.numpy().reshape(worlds,-1,4)
            positions.append(np.concatenate([actual,goalp,goalq],axis=-1))
        data=capture.arrays(); poses=np.asarray(positions)
        data.update(object_pose=poses,world_reference_group=groups,
                    fps=np.asarray(config.sim.fps))
        # Pinned native metadata connects force snapshots to original DOFs and
        # body-fixed contact points. These fields are read, never modified.
        native=env.solver.mj_model
        for field in ('jnt_type','jnt_qposadr','jnt_dofadr','jnt_bodyid','body_jntadr','body_jntnum',
                      'body_ipos','body_iquat','body_mass','body_inertia','body_rootid','body_parentid',
                      'dof_bodyid'):
            data['native_model_'+field]=np.asarray(getattr(native,field)).copy()
        data['native_joint_to_newton_joint']=env.solver.mjc_jnt_to_newton_jnt.numpy()
        data['newton_joint_q_start']=env.model.joint_q_start.numpy()
        data['newton_joint_qd_start']=env.model.joint_qd_start.numpy()
        data['newton_body_mass']=env.model.body_mass.numpy()
        data['newton_body_com']=env.model.body_com.numpy()
        data['newton_gravity']=env.model.gravity.numpy()
        data['newton_robot_nq']=np.asarray(env.scene.robot_reference.num_joint_q)
        score=official_scores(poses)
        if command_digest(setup.reference)!=digest: raise AssertionError('task commands were mutated')
        result=dict(full_episode=count==frames-1,steps=count,reference_frames=frames,worlds=worlds,
            controller='zero residual native targets',assistance=False,task_commands_sha256=digest,
            elapsed_seconds=time.monotonic()-started,device=device,
            official_mp_sr=float(score['passed'].mean()) if count==frames-1 else None,
            candidates=[dict(candidate=i,worlds=replicas,
                official_mp_sr=float(score['passed'][groups==i].mean()) if count==frames-1 else None,
                loss=float(np.mean(score['loss'][groups==i]))) for i in range(len(q))])
        if output is not None:
            output=Path(output);output.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(output/'physics.npz',**data)
            np.savez_compressed(output/'references.npz',joint_q=q,joint_target=target)
            result['physics_sha256']=hashlib.sha256((output/'physics.npz').read_bytes()).hexdigest()
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result,data
