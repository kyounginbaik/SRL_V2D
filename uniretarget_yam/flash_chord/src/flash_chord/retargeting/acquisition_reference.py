"""Physical-feedback acquisition corrections and independently timed closure.

These functions propose references. Prediction improvement is not acceptance;
fresh physical acquisition, transfer and whole-task evaluation remain required.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AcquisitionCorrection:
    position: np.ndarray
    rotation: np.ndarray


def _residuals(planned,objects,grasp,supported):
    planned=np.asarray(planned,float);objects=np.asarray(objects,float);mask=np.asarray(supported)
    if (objects.ndim!=3 or objects.shape[-1]!=7 or planned.shape!=(len(objects),7)
            or mask.shape!=objects.shape[:2] or mask.dtype!=bool):
        raise ValueError('matched time/world object poses, planned wrist poses and Boolean support required')
    plan=np.broadcast_to(planned[:,None],objects.shape)
    finite=np.isfinite(objects).all(-1)&np.isfinite(plan).all(-1)
    finite&=(np.linalg.norm(objects[...,3:],axis=-1)>1e-8)&(np.linalg.norm(plan[...,3:],axis=-1)>1e-8)
    good=mask&finite;o=objects[good];p=plan[good]
    report=dict(total_samples=int(mask.size),used_samples=int(good.sum()),
        coverage=float(good.mean()),invalid_pose_fraction=float((~finite).mean()))
    if not len(o):return None,None,report
    rotation=Rotation.from_quat(o[:,3:])
    delta=rotation.apply(grasp.wrist_position_o)+o[:,:3]-p[:,:3]
    dr=rotation*Rotation.from_quat(grasp.wrist_quaternion_o)*Rotation.from_quat(p[:,3:]).inv()
    return delta,dr,report


def fit_acquisition_correction(planned,objects,grasp,supported):
    dp,dr,report=_residuals(planned,objects,grasp,supported)
    if report['used_samples']<12 or report['coverage']<.25:
        return None,dict(report,reason='insufficient_supported_seating_feedback')
    model=AcquisitionCorrection(np.median(dp,axis=0),dr.mean().as_rotvec())
    if np.linalg.norm(model.position)>.1 or np.linalg.norm(model.rotation)>.7:
        return None,dict(report,reason='outside_local_acquisition_chart')
    return model,dict(report,reason='measured_object_relative_acquisition_correction')


def validate_acquisition_correction(model,planned,objects,grasp,supported):
    dp,dr,report=_residuals(planned,objects,grasp,supported)
    if not report['used_samples']:return dict(report,reason='no_supported_validation_samples',improved=False)
    before=np.linalg.norm(dp,axis=1)/.03+dr.magnitude()/np.deg2rad(30)
    after=np.linalg.norm(dp-model.position,axis=1)/.03+(dr*Rotation.from_rotvec(model.rotation).inv()).magnitude()/np.deg2rad(30)
    return dict(report,conditional_median_before=float(np.median(before)),conditional_median_after=float(np.median(after)),
        improved=bool(report['coverage']>=.25 and np.median(after)<.9*np.median(before)))


def apply_acquisition_correction(wrist,model,*,bounds,contact,gain=1.):
    base=np.asarray(wrist,float);a,b=bounds;start,stop=contact
    if (base.ndim!=2 or base.shape[1]!=7 or not np.isfinite(base).all()
            or not 0<=a<start<=stop<b<len(base) or not 0<gain<=1):
        raise ValueError('finite poses and protected acquisition interval required')
    t=np.arange(len(base));u=np.clip((t-a)/(start-a),0,1);v=np.clip((b-t)/(b-stop),0,1)
    smooth=lambda x:x*x*x*(10+x*(-15+6*x))
    weight=gain*smooth(u)*smooth(v);out=base.copy()
    out[:,:3]+=weight[:,None]*model.position
    out[:,3:]=(Rotation.from_rotvec(weight[:,None]*model.rotation)*Rotation.from_quat(base[:,3:])).as_quat()
    out[weight==0]=base[weight==0]
    return out


def seating_finger_schedule(current,commands,open_target,closed_target,lower,upper,*,bounds,seat,hold_end,fps,speed=3.):
    """Reach open posture at seating, then close with bounded spring slew.

    Return both joint-limited geometry and native spring commands; preload is
    not clipped away or mistaken for achieved finger positions. All three
    transitions are exact or the proposal is rejected without partial output.
    """
    q,c,opened,closed,lo,hi=[np.asarray(x,float) for x in (current,commands,open_target,closed_target,lower,upper)]
    if (q.ndim!=2 or c.shape!=q.shape or any(x.shape!=(q.shape[1],) for x in (opened,closed,lo,hi))
            or not all(np.isfinite(x).all() for x in (q,c,opened,closed,lo,hi)) or np.any(lo>hi)
            or np.any(opened<lo) or np.any(opened>hi) or fps<=0 or speed<=0):
        raise ValueError('finite matching finger geometry/commands, open posture, limits and rates required')
    a,b=bounds
    if not 0<=a<seat<hold_end<b<len(q):raise ValueError('protected seating/hold/restore times required')
    bounded=np.clip(closed,lo,hi)
    minimum=lambda delta:int(np.ceil(1.875*np.abs(delta).max(initial=0)*fps/speed))
    close_steps=max(1,minimum(closed-opened));complete=seat+close_steps
    transitions=[(a,seat,q[a],opened,c[a],opened),
        (seat,complete,opened,bounded,opened,closed),
        (hold_end,b,bounded,q[b],closed,c[b])]
    for start,end,q0,q1,c0,c1 in transitions:
        if end<=start or end-start<max(minimum(q1-q0),minimum(c1-c0)):
            return None,None,dict(reason='insufficient_full_seating_transition_time',closure_complete=complete)
    if complete>hold_end:return None,None,dict(reason='closure_misses_hold_deadline',closure_complete=complete)
    out=q.copy();native=c.copy()
    out[complete:hold_end+1]=bounded;native[complete:hold_end+1]=closed
    for start,end,q0,q1,c0,c1 in transitions:
        t=np.arange(end-start+1)/(end-start);w=t*t*t*(10+t*(-15+6*t))
        out[start:end+1]=q0+w[:,None]*(q1-q0);native[start:end+1]=c0+w[:,None]*(c1-c0)
        out[start],out[end]=q0,q1;native[start],native[end]=c0,c1
    if np.any(out[a:b+1]<lo-1e-6) or np.any(out[a:b+1]>hi+1e-6):raise ValueError('finger geometry exceeds limits')
    return out,native,dict(reason='open_then_close_after_seating',seat=seat,closure_complete=complete,
        bounds=[a,b],hold_end=hold_end,peak_spring_speed_rad_s=float(np.abs(np.diff(native[a:b+1],axis=0)).max()*fps))
