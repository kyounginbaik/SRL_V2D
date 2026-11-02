"""Precontact resting-object anchors and robot-reference dwell proposals."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def _quiet(poses,counts,external_mg,invalid,fps):
    p=np.asarray(poses,float);c=np.asarray(counts);ext=np.asarray(external_mg,float);bad=np.asarray(invalid)
    if p.ndim!=3 or p.shape[-1]!=7 or c.ndim!=3 or c.shape[:2]!=p.shape[:2] or ext.shape!=p.shape[:2] or bad.shape!=ext.shape or not np.isfinite(fps) or fps<=0:
        raise ValueError('aligned physical object/contact arrays and positive fps required')
    if not all(np.isfinite(x).all() for x in [p,c,ext,bad]) or np.any(np.linalg.norm(p[...,3:],axis=-1)<1e-8):return None
    velocity=np.full(p.shape[:2],np.inf);angular=velocity.copy()
    velocity[1:]=np.linalg.norm(np.diff(p[:,:,:3],axis=0),axis=-1)*fps
    angular[1:]=(Rotation.from_quat(p[1:,:,3:].reshape(-1,4).copy())*Rotation.from_quat(p[:-1,:,3:].reshape(-1,4).copy()).inv()).magnitude().reshape(len(p)-1,p.shape[1])*fps
    return (velocity<.01)&(angular<.1)&~(c>0).any(-1)&(ext>.5)&(bad==0)


def find_rest_anchor(poses,counts,external_mg,invalid,fps,before,*,population=.8):
    quiet=_quiet(poses,counts,external_mg,invalid,fps)
    if quiet is None:return None
    if not isinstance(before,(int,np.integer)) or not 2<before<=len(quiet) or not 0<population<=1:raise ValueError('precontact frame bound and positive population required')
    good=np.flatnonzero(quiet[:before].mean(1)>=population);minimum=max(2,round(.25*fps))
    runs=[r for r in np.split(good,np.flatnonzero(np.diff(good)>1)+1) if len(r)>=minimum]
    if not runs:return None
    run=runs[-1];a,b=int(run[0]),int(run[-1]+1);eligible=np.flatnonzero(quiet[a:b].mean(0)>=population)
    if not len(eligible):return None
    frame=b-1;p=np.asarray(poses)[frame,eligible];rotation=Rotation.from_quat(p[:,3:].copy());center=rotation.mean()
    cost=np.linalg.norm(p[:,:3]-np.median(p[:,:3],axis=0),axis=-1)/.003+(rotation*center.inv()).magnitude()/.05
    world=int(eligible[np.argmin(cost)])
    return dict(frame=frame,world=world,interval=[a,b],pose=np.asarray(poses)[frame,world].tolist(),minimum_population=population,
        quiet_world_fraction=float((quiet[a:b].mean(0)>=population).mean()),selection='last_precontact_quiet_interval_then_geometric_pose_medoid')


def validate_rest_anchor(anchor,poses,counts,external_mg,invalid,fps):
    quiet=_quiet(poses,counts,external_mg,invalid,fps)
    if quiet is None:return dict(accepted=False,reason='invalid_resting_object_evidence')
    a,b=anchor['interval'];p=np.asarray(poses)[a:b];goal=np.asarray(anchor['pose']);worlds=p.shape[1]
    error=np.linalg.norm(p[:,:,:3]-goal[:3],axis=-1)
    angle=(Rotation.from_quat(p[:,:,3:].reshape(-1,4).copy())*Rotation.from_quat(goal[3:]).inv()).magnitude().reshape(b-a,worlds)
    valid=quiet[a:b]&(error<=.003)&(angle<=.05);fraction=float((valid.mean(0)>=.8).mean())
    return dict(accepted=fraction>=anchor['minimum_population'],world_fraction=fraction,fixed_interval=[a,b],position_tolerance_m=.003,rotation_tolerance_rad=.05,
        reason='fixed_rest_anchor_reproduced' if fraction>=anchor['minimum_population'] else 'fixed_rest_anchor_not_reproduced')


def dwell_reference_path(path,keyframe,dwell_frames,fade_frames):
    """Delay only the robot's rendered path, leaving the task timeline untouched."""
    path=np.asarray(path);n=len(path)
    if path.shape!=(n,7) or n<3 or not np.isfinite(path).all() or np.any(np.linalg.norm(path[:,3:],axis=-1)<1e-8):raise ValueError('finite object-frame robot rendering path required')
    if any(not isinstance(x,(int,np.integer)) for x in (keyframe,dwell_frames,fade_frames)) or not 0<keyframe<n-1 or dwell_frames<0 or fade_frames<1 or keyframe+dwell_frames+fade_frames>=n:
        raise ValueError('bounded interior dwell and catch-up interval required')
    clock=np.arange(n,dtype=float)
    if not dwell_frames:return path.copy(),clock
    u=np.clip((clock-keyframe-dwell_frames)/fade_frames,0,1);smooth=u*u*u*(10+u*(-15+6*u))
    mapped=np.where(clock>=keyframe,np.maximum(keyframe,clock-dwell_frames*(1-smooth)),clock)
    if np.any(np.diff(mapped)<0):raise ValueError('dwell reference clock may not reverse')
    result=np.empty_like(path);result[:,:3]=np.stack([np.interp(mapped,clock,path[:,j]) for j in range(3)],axis=-1)
    result[:,3:]=Slerp(clock,Rotation.from_quat(path[:,3:].copy()))(mapped).as_quat()
    same=mapped==clock;result[same]=path[same]
    return result,mapped
