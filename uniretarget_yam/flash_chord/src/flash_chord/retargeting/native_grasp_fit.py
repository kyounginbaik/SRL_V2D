"""Physical contact witnesses, mesh/plane clearance, and native grasp rendering.

These are robot-reference operations. Geometric fits never certify force
closure, collision-free approach trajectories, or a successful physical grasp.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .coupled_transport import transport_poses


def preloaded_joint_bounds(lower,upper,seed_geometry,seed_native,source_commands):
    """Bound physical coordinates separately from measured spring setpoints.

    Servo setpoints may intentionally exceed a mechanical joint stop. Preserve
    that measured preload, but do not enlarge the source command envelope.
    """
    lo,hi,seed,native=[np.asarray(x,float) for x in (lower,upper,seed_geometry,seed_native)];commands=np.asarray(source_commands,float)
    if (lo.ndim!=1 or hi.shape!=lo.shape or seed.shape!=lo.shape or native.shape!=lo.shape or commands.ndim!=2 or commands.shape[1:]!=lo.shape
            or not len(commands) or not all(np.isfinite(x).all() for x in [lo,hi,seed,native,commands]) or np.any(lo>hi)):
        raise ValueError('finite physical limits, preload, and source command envelope required')
    native_lo=np.minimum(lo,commands.min(0));native_hi=np.maximum(hi,commands.max(0));preload=native-seed
    fit_lo=np.maximum(lo,native_lo-preload);fit_hi=np.minimum(hi,native_hi-preload)
    if np.any(fit_lo>fit_hi):raise ValueError('no physical posture preserves preload within the observed native command envelope')
    return fit_lo,fit_hi,native_lo,native_hi


def contact_witnesses(object_pose,landmarks,forces,active):
    obj=np.asarray(object_pose,float);points=np.asarray(landmarks,float);force=np.asarray(forces,float);mask=np.asarray(active,bool)
    if (obj.ndim!=3 or obj.shape[-1]!=7 or points.ndim!=4 or points.shape[:2]!=obj.shape[:2]
            or points.shape[-1]!=3 or force.shape!=points.shape or mask.shape!=points.shape[:-1]
            or not all(np.isfinite(x).all() for x in [obj,points,force]) or np.any(np.linalg.norm(obj[...,3:],axis=-1)<1e-8)):
        raise ValueError('finite corresponding object poses, landmarks and physical forces required')
    result=[]
    for digit in range(points.shape[2]):
        magnitude=np.linalg.norm(force[:,:,digit],axis=-1);selected=mask[:,:,digit]&(magnitude>1e-4)
        t,w=np.nonzero(selected)
        if not len(t):continue
        rotation=Rotation.from_quat(obj[t,w,3:].copy())
        local=rotation.inv().apply(points[t,w,digit]-obj[t,w,:3])
        directions=rotation.inv().apply(force[t,w,digit]/magnitude[t,w,None])
        probability=selected.mean(1);peaks=np.flatnonzero(probability==probability.max())
        peak=int(peaks[np.argmin(np.abs(peaks-np.median(t)))])
        eligible=np.flatnonzero(t==peak);median=np.median(local[eligible],axis=0)
        index=int(eligible[np.argmin(np.linalg.norm(local[eligible]-median,axis=-1))])
        mean=directions.mean(0);norm=np.linalg.norm(mean)
        if norm<1e-8:continue
        result.append(dict(digit=digit,frame=int(t[index]),world=int(w[index]),coverage=float(selected.mean()),peak_world_fraction=float(probability[peak]),
            target_object=local[index].tolist(),force_direction_object=directions[index].tolist(),mean_force_direction_object=(mean/norm).tolist()))
    return result


def mesh_plane_gaps(body_poses,clouds,normals,offsets):
    """Exact halfspace minimum over every vertex of each supplied mesh.

    A support's infinite top halfspace is conservative for its finite shape.
    This does not test self/object collisions or swept approach clearance.
    """
    poses=np.asarray(body_poses,float);normal=np.asarray(normals,float);offset=np.asarray(offsets,float)
    if (poses.ndim!=2 or poses.shape[-1]!=7 or normal.ndim!=2 or normal.shape[-1]!=3 or offset.shape!=(len(normal),)
            or not all(np.isfinite(x).all() for x in [poses,normal,offset]) or not np.allclose(np.linalg.norm(normal,axis=-1),1.,atol=1e-6)
            or np.any(np.linalg.norm(poses[:,3:],axis=-1)<1e-8) or not clouds):raise ValueError('finite bodies, unit planes, and complete mesh clouds required')
    gaps=[]
    for body,vertices in clouds:
        v=np.asarray(vertices,float)
        if not isinstance(body,(int,np.integer)) or not 0<=body<len(poses) or v.ndim!=2 or v.shape[-1]!=3 or not len(v) or not np.isfinite(v).all():
            raise ValueError('nonempty finite body-local mesh vertices required')
        local_normal=Rotation.from_quat(poses[body,3:].copy()).inv().apply(normal.copy())
        gaps.append((v@local_normal.T).min(0)+normal@poses[body,:3]-offset)
    return np.asarray(gaps)


def validate_contact_cluster(object_pose,landmarks,forces,active,witness,*,position_tolerance=.01,minimum_fraction=.5):
    """Validate the fixed discovery event/target across fresh identical worlds.

    GPU replicas have no paired random seed identity. Validation cannot move
    the event or target, or replace the discovery witness with a better sample.
    """
    obj=np.asarray(object_pose,float);points=np.asarray(landmarks,float);force=np.asarray(forces,float);enabled=np.asarray(active,bool)
    if not all(np.isfinite(x).all() for x in [obj,points,force]) or np.any(np.linalg.norm(obj[...,3:],axis=-1)<1e-8):
        return dict(accepted=False,reason='invalid_physical_contact_cluster')
    frame=witness['frame'];digit=witness['digit'];a=max(0,frame-1);b=min(len(obj),frame+2)
    rotation=Rotation.from_quat(obj[a:b,:,3:].reshape(-1,4).copy()).inv()
    local=rotation.apply((points[a:b,:,digit]-obj[a:b,:,:3]).reshape(-1,3)).reshape(b-a,obj.shape[1],3)
    local_force=rotation.apply(force[a:b,:,digit].reshape(-1,3)).reshape(b-a,obj.shape[1],3)
    magnitude=np.linalg.norm(local_force,axis=-1)
    direction=local_force/np.maximum(magnitude[...,None],1e-12)
    hit=(enabled[a:b,:,digit]&(magnitude>1e-4)&(np.linalg.norm(local-np.asarray(witness['target_object']),axis=-1)<=position_tolerance)
        &((direction@np.asarray(witness['mean_force_direction_object']))>=.25))
    fraction=float(hit.any(0).mean())
    return dict(accepted=fraction>=minimum_fraction,world_fraction=fraction,fixed_frame_window=[a,b],position_tolerance_m=position_tolerance,
        reason='fixed_contact_cluster_reproduced' if fraction>=minimum_fraction else 'fixed_contact_cluster_not_reproduced')


def render_native_grasp(q,target,layout,side,object_path,anchor_object,seed_q,fitted_q,seed_native,weight,*,normal=None,remove_inward_bias=False,wrist_preload_scale=1.,finger_preload_scale=1.):
    q=np.asarray(q);target=np.asarray(target);weight=np.asarray(weight,float);path=np.asarray(object_path,float)
    if (q.ndim!=2 or q.shape[1]!=layout.num_joint_q or target.shape!=(len(q),layout.num_joint_dof)
            or weight.shape!=(len(q),) or path.shape!=(len(q),7) or not np.isfinite(weight).all()
            or np.any(weight<0) or np.any(weight>1) or weight[0]!=0 or weight[-1]!=0):raise ValueError('bounded grasp profile with protected endpoints required')
    if not np.isfinite([wrist_preload_scale,finger_preload_scale]).all() or not 0<=wrist_preload_scale<=1 or not 0<=finger_preload_scale<=1:
        raise ValueError('wrist and finger preload scales must lie in [0,1]')
    hand=layout.hand(side);p=hand.wrist_pos_q_ids;qi=hand.wrist_orient_q_id;r=hand.wrist_orient_dof_ids;cp=hand.wrist_pos_dof_ids
    seed=np.asarray(seed_q,float);fit=np.asarray(fitted_q,float);native=np.asarray(seed_native,float)
    if seed.shape!=(layout.num_joint_q,) or fit.shape!=seed.shape or native.shape!=(layout.num_joint_dof,) or not all(np.isfinite(x).all() for x in [q,target,seed,fit,native]):
        raise ValueError('finite full robot geometry and native command prototype required')
    native=native.copy();delta=Rotation.from_quat(fit[qi:qi+4].copy())*Rotation.from_quat(seed[qi:qi+4].copy()).inv()
    bias=wrist_preload_scale*(native[list(cp)]-seed[list(p)]);removed=np.zeros(3)
    if remove_inward_bias:
        normal=np.asarray(normal,float)
        if normal.shape!=(3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1)>1e-6:raise ValueError('unit support normal required for bias ablation')
        removed=min(0.,float(bias@normal))*normal;bias=bias-removed
    native[list(cp)]=fit[list(p)]+bias
    if wrist_preload_scale==1.:
        native[list(r)]=(delta*Rotation.from_rotvec(native[list(r)].copy())).as_rotvec()
    else:
        relative=Rotation.from_quat(seed[qi:qi+4].copy()).inv()*Rotation.from_rotvec(native[list(r)].copy())
        native[list(r)]=(Rotation.from_quat(fit[qi:qi+4].copy())*Rotation.from_rotvec(wrist_preload_scale*relative.as_rotvec())).as_rotvec()
    if finger_preload_scale==1.:
        native[list(hand.finger_dof_ids)]+=fit[list(hand.finger_q_ids)]-seed[list(hand.finger_q_ids)]
    else:
        native[list(hand.finger_dof_ids)]=fit[list(hand.finger_q_ids)]+finger_preload_scale*(native[list(hand.finger_dof_ids)]-seed[list(hand.finger_q_ids)])
    prototype=np.stack([np.r_[fit[list(p)],fit[qi:qi+4]],np.r_[native[list(cp)],Rotation.from_rotvec(native[list(r)].copy()).as_quat()]])
    desired=transport_poses(np.tile(anchor_object,(len(q),1)),path,np.tile(prototype[None],(len(q),1,1)))
    out=q.copy();commands=target.copy()
    out[:,p]=q[:,p]+weight[:,None]*(desired[:,0,:3]-q[:,p])
    commands[:,cp]=target[:,cp]+weight[:,None]*(desired[:,1,:3]-target[:,cp])
    for original,goal,kind in [(q[:,qi:qi+4],desired[:,0,3:],'geometry'),(Rotation.from_rotvec(target[:,r].copy()).as_quat(),desired[:,1,3:],'native')]:
        initial=Rotation.from_quat(original.copy());rotation=Rotation.from_rotvec(weight[:,None]*(Rotation.from_quat(goal.copy())*initial.inv()).as_rotvec())*initial
        if kind=='geometry':out[:,qi:qi+4]=rotation.as_quat()
        else:commands[:,r]=rotation.as_rotvec()
    f=hand.finger_q_ids;fd=hand.finger_dof_ids
    out[:,f]=q[:,f]+weight[:,None]*(fit[list(f)]-q[:,f])
    commands[:,fd]=target[:,fd]+weight[:,None]*(native[list(fd)]-target[:,fd])
    unchanged=weight==0;out[unchanged]=q[unchanged];commands[unchanged]=target[unchanged]
    return out,commands,dict(removed_inward_wrist_bias_m=removed.tolist(),prototype_native=native.tolist(),
        wrist_preload_scale=float(wrist_preload_scale),finger_preload_scale=float(finger_preload_scale))
