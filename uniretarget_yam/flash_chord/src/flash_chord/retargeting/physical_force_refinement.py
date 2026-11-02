"""Generate one auditable force-guided reference-direction falsification batch."""
from dataclasses import asdict
import numpy as np
from scipy.spatial.transform import Rotation
from .contact_force_step import propose_contact_force_step
from .native_contact_analysis import object_contact_snapshot
from .native_contact_kinematics import cached_point_jacobian, scalar_reference_channels
from .physical_response import aligned_signed_pose
from .rigid_wrench_targets import rigid_contact_wrench_targets


def force_direction_candidates(data, base_q, base_target, *, stiffness_n_per_m=1000., seed=51):
    """Locate the first pose failure, explain contacts and propose tiny edits.

    The compliance magnitude is an explicit hypothesis being falsified, not an
    identified material model. Candidate controls are unchanged, opposite,
    equal-normalized-size random, and half step. No candidate is accepted here.
    A missing contact arrangement is explicitly returned to the caller.
    """
    q=np.asarray(base_q,np.float32);target=np.asarray(base_target,np.float32)
    poses=np.asarray(data['object_pose']);fps=float(data['fps'])
    if q.shape[0]!=poses.shape[0]+1 or target.shape[0]!=len(q):
        raise ValueError('full-prefix discovery and full reference trajectories required')
    signed=aligned_signed_pose(poses)
    failed=((np.linalg.norm(signed[...,:3],axis=-1)>1.) |
            (np.linalg.norm(signed[...,3:],axis=-1)>1.)).any(2).mean(1)>.5
    duration=max(1,round(.25*fps))
    starts=[i for i in range(len(failed)-duration+1) if failed[i:i+duration].all()]
    if not starts: raise ValueError('no persistent task failure to refine')
    failure=starts[0]; world=int(data['native_capture_world_ids'][0]); wi=0
    object_id=int(np.argmax(np.linalg.norm(signed[failure,world,:,:3],axis=-1)+
                            np.linalg.norm(signed[failure,world,:,3:],axis=-1)))
    nq=int(data['newton_robot_nq']);qstart=data['newton_joint_q_start']
    nativejoints=data['native_joint_to_newton_joint'][world]
    robot_bodies=set(int(data['native_model_jnt_bodyid'][j]) for j,nj in enumerate(nativejoints)
        if nj>=0 and qstart[nj]<nq)
    parent=np.asarray(data['native_model_body_parentid'])
    def robot_body(body):
        while body>0:
            if body in robot_bodies:return True
            body=int(parent[body])
        return False
    gravity=np.asarray(data['newton_gravity']).reshape(-1,3)[world];up=-gravity/np.linalg.norm(gravity)
    available=[]
    for frame in range(max(0,failure-round(fps)),failure):
        snapshot=object_contact_snapshot(data,frame,world,object_id)
        hand=np.asarray([robot_body(int(b)) for b in snapshot.counterpart_native_bodies],bool)
        if hand.any():
            load=float(snapshot.forces_on_object[hand].sum(0)@up)
            available.append((load,frame,snapshot,hand))
    if not available: raise ValueError('no actuated contact before failure: contact-arrangement generation required')
    _,frame,snapshot,hand=min(available,key=lambda entry:entry[0])
    patches=tuple(p for p,keep in zip(snapshot.patches,hand) if keep)
    force=snapshot.forces_on_object[hand]
    bodies=snapshot.counterpart_native_bodies[hand]
    dof,geometry,target_ids=scalar_reference_channels(data,world)
    jac=np.stack([cached_point_jacobian(data,frame,world,int(body),patch.position)[:,dof]
                  for body,patch in zip(bodies,patches)])
    kinds={int(data['native_model_jnt_dofadr'][j]):int(kind)
           for j,kind in enumerate(data['native_model_jnt_type'])}
    scales=np.asarray([.001 if kinds[int(d)]==2 else .02 for d in dof])
    obj=snapshot.native_body_id
    mass=float(data['native_model_body_mass'][obj])
    # Same first-object rigid alignment used by official scoring, inverted to
    # express the immutable desired motion in the physical simulation frame.
    initial=poses[0,world,0];align=Rotation.from_quat(initial[3:7])*Rotation.from_quat(initial[10:14]).inv()
    offset=initial[:3]-align.apply(initial[7:10])
    desired_p=align.apply(poses[:,world,object_id,7:10])+offset
    desired_r=align*Rotation.from_quat(poses[:,world,object_id,10:14].copy())
    iq=np.asarray(data['native_model_body_iquat'][obj])[[1,2,3,0]]
    demands=rigid_contact_wrench_targets(desired_p,desired_r.as_quat(),fps,mass,
        data['native_model_body_inertia'][obj],iq,data['native_model_body_ipos'][obj],gravity)
    wrench=demands[failure].copy()
    # The next failing motion decides whether an environment contact can
    # remain active. No task/skill name is used for this separation test.
    actual_rotation=Rotation.from_quat(snapshot.quaternion_wxyz[[1,2,3,0]])
    legitimate_external=np.zeros(6);external_report=[]
    for i,patch in enumerate(snapshot.patches):
        if hand[i]:continue
        local=actual_rotation.inv().apply(np.asarray(patch.position)-snapshot.origin_world)
        moved=desired_r[failure].apply(local)+desired_p[failure]
        gap=float((moved-np.asarray(patch.position))@np.asarray(patch.normal))
        valid=gap<=.002
        if valid:
            f=snapshot.forces_on_object[i];legitimate_external[:3]+=f
            legitimate_external[3:]+=np.cross(np.asarray(patch.position)-snapshot.origin_world,f)+snapshot.intrinsic_torques_on_object[i]
        external_report.append(dict(counterpart=int(snapshot.counterpart_native_bodies[i]),
                                    projected_separation_m=gap,retained=bool(valid)))
    wrench[3:]+=np.cross(desired_p[failure]-snapshot.origin_world,wrench[:3])
    wrench-=legitimate_external
    measured_normal=np.asarray([max(0.,float(f@np.asarray(p.normal))) for p,f in zip(patches,force)])
    budgets={p.group:max(2*mass*np.linalg.norm(gravity),1.25*measured_normal[
        [other.group==p.group for other in patches]].sum()) for p in patches}
    step=propose_contact_force_step(patches,force,wrench,jac,scales,-scales,scales,
        normal_force_limits=budgets,contact_stiffness_n_per_m=stiffness_n_per_m,
        center=snapshot.origin_world)
    normalized=step.delta/scales
    random=np.random.default_rng(seed).normal(size=len(scales));random*=np.linalg.norm(normalized)/max(np.linalg.norm(random),1e-15)
    lo=max(1,frame-round(.3*fps));hold=min(len(q)-3,failure+round(.5*fps));end=min(len(q)-2,hold+round(.5*fps))
    if not lo<frame<=hold<end:raise ValueError('no protected reference-edit window')
    mask=np.zeros(len(q));mask[lo:frame+1]=.5-.5*np.cos(np.linspace(0,np.pi,frame-lo+1))
    mask[frame:hold+1]=1.;mask[hold:end+1]=.5+.5*np.cos(np.linspace(0,np.pi,end-hold+1))
    deltas=[np.zeros_like(step.delta),step.delta,-step.delta,random*scales,.5*step.delta]
    names=['unchanged','force_guided','opposite','equal_size_random','half_force_guided']
    targets=[]
    for delta in deltas:
        t=target.copy();t[:,target_ids]+=mask[:,None]*delta[None,:]
        np.testing.assert_array_equal(t[[0,-1]],target[[0,-1]])
        targets.append(t)
    diagnostics=dict(failure_frame=int(failure),contact_frame=int(frame),object_id=object_id,
        edited_geometry=False,edited_native_targets=True,variable_count=len(scales),
        edit_window=[lo,frame,hold,end],missing_wrench=step.missing_wrench.tolist(),
        desired_hand_wrench=wrench.tolist(),observed_hand_force=force.sum(0).tolist(),
        requires_contact_change=step.requires_contact_change,
        normal_force_budgets=budgets,joint_effort_constraints_in_proposal=False,
        model='quasistatic force allocation and cached contact-point Jacobian',
        stiffness_n_per_m=stiffness_n_per_m,stiffness_identified=False,
        external_contact_separation=external_report,normalized_step_norm=float(np.linalg.norm(normalized)),
        candidate_names=names,acceptance_performed=False,task_success_claimed=False)
    return np.repeat(q[None],len(targets),axis=0),np.asarray(targets),diagnostics
