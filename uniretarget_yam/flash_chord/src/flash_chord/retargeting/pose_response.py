"""Identify/invert a local native actuator-pose response from physical feedback.

Input row i is a command and its AFTER-control-frame pose. Previous row i-1
supplies the BEFORE state. Positions are WORLD; rotation errors/increments are
in that before wrist's CHILD frame. This is a one-step actuator model, not an
analytic simulator gradient or a certificate across contact-mode changes.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _poses(before,other):
    a,b=np.broadcast_arrays(np.asarray(before,float),np.asarray(other,float))
    if (a.ndim<1 or a.shape[-1]!=7 or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(np.linalg.norm(a[...,3:],axis=-1)<1e-8) or np.any(np.linalg.norm(b[...,3:],axis=-1)<1e-8)):
        raise ValueError('matching finite WORLD XYZ / XYZW wrist poses required')
    return a,b,Rotation.from_quat(np.array(a[...,3:].reshape(-1,4),copy=True)),Rotation.from_quat(np.array(b[...,3:].reshape(-1,4),copy=True))


@dataclass(frozen=True)
class PoseResponse:
    position_gain: np.ndarray
    position_offset: np.ndarray
    rotation_gain: np.ndarray
    rotation_offset: np.ndarray

    def __post_init__(self):
        for name in ('position_gain','position_offset','rotation_gain','rotation_offset'):
            x=np.array(getattr(self,name),float,copy=True)
            if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('three finite parameters per response axis required')
            if name.endswith('gain') and (np.any(x<=0) or np.any(x>1)):
                raise ValueError('this overdamped response requires gains in (0,1]')
            x.flags.writeable=False;object.__setattr__(self,name,x)

    def predict(self,before,command):
        before,command,rb,rc=_poses(before,command)
        position=before[...,:3]+self.position_gain*(command[...,:3]-before[...,:3])+self.position_offset
        error=(rb.inv()*rc).as_rotvec()
        step=error*self.rotation_gain+self.rotation_offset
        quaternion=(rb*Rotation.from_rotvec(step)).as_quat().reshape(before[...,3:].shape)
        return np.concatenate([position,quaternion],axis=-1)

    def inverse(self,desired_before,desired_after):
        """Return a command for the specified post-step goal, without clipping.

        A pose-only interface uses principal rotation errors. Reject inverse
        steps outside that domain rather than silently wrapping them. Physical
        force/workspace/contact limits remain the caller's responsibility.
        """
        before,after,rb,ra=_poses(desired_before,desired_after)
        position=before[...,:3]+(after[...,:3]-before[...,:3]-self.position_offset)/self.position_gain
        step=((rb.inv()*ra).as_rotvec()-self.rotation_offset)/self.rotation_gain
        if np.any(np.linalg.norm(step,axis=-1)>np.pi+1e-10):
            raise ValueError('requested after-step rotation exceeds the fitted principal-error domain')
        quaternion=(rb*Rotation.from_rotvec(step)).as_quat().reshape(before[...,3:].shape)
        return np.concatenate([position,quaternion],axis=-1)


def _transitions(actual_after,commands,supported):
    h,c=np.asarray(actual_after,float),np.asarray(commands,float);support=np.asarray(supported)
    if h.ndim!=3 or h.shape[-1]!=7 or c.shape!=h.shape or support.shape!=h.shape[:2] or support.dtype!=bool or len(h)<3:
        raise ValueError('matched [post-control-frame,world,7] poses and Boolean support required')
    finite=np.isfinite(h).all(-1)&np.isfinite(c).all(-1)&(np.linalg.norm(h[...,3:],axis=-1)>1e-8)&(np.linalg.norm(c[...,3:],axis=-1)>1e-8)
    good=support[1:]&support[:-1]&finite[1:]&finite[:-1]
    before=h[:-1][good];after=h[1:][good];command=c[1:][good]
    return before,after,command,dict(total_transitions=int(good.size),used_transitions=int(good.sum()),
        supported_transition_fraction=float(good.mean()),invalid_pose_fraction=float((~finite).mean()),
        row_semantics='command i / after i; before i is after i-1')


def fit_pose_response(actual_after,commands,supported):
    before,after,command,report=_transitions(actual_after,commands,supported)
    if len(before)<12:return None,dict(report,reason='insufficient_supported_transitions')
    _,_,rb,ra=_poses(before,after);_,_,_,rc=_poses(before,command)
    inputs=[command[:,:3]-before[:,:3],(rb.inv()*rc).as_rotvec()]
    outputs=[after[:,:3]-before[:,:3],(rb.inv()*ra).as_rotvec()]
    parameters=[];residuals=[]
    for x,y in zip(inputs,outputs):
        gains=[];offsets=[];rmse=[]
        for axis in range(3):
            if x[:,axis].std()<1e-5:return None,dict(report,reason='insufficient_axis_excitation',axis=axis)
            design=np.c_[x[:,axis],np.ones(len(x))]
            gain,offset=np.linalg.lstsq(design,y[:,axis],rcond=None)[0]
            if not 0<gain<=1:return None,dict(report,reason='not_an_overdamped_local_response',axis=axis,gain=float(gain))
            gains.append(gain);offsets.append(offset);rmse.append(float(np.sqrt(np.mean((y[:,axis]-design@np.r_[gain,offset])**2))))
        parameters.extend([np.asarray(gains),np.asarray(offsets)]);residuals.append(rmse)
    model=PoseResponse(*parameters)
    return model,dict(report,reason='identified_local_actuator_response',position_gain=model.position_gain.tolist(),
        position_offset=model.position_offset.tolist(),rotation_gain=model.rotation_gain.tolist(),rotation_offset=model.rotation_offset.tolist(),
        per_axis_position_step_rmse_m=residuals[0],per_axis_rotation_step_rmse_rad=residuals[1])


def validate_pose_response(model,actual_after,commands,supported):
    before,after,command,report=_transitions(actual_after,commands,supported)
    if not len(before):return dict(report,reason='no_supported_validation_transitions')
    predicted=model.predict(before,command)
    def error(values):
        pe=np.linalg.norm(values[:,:3]-after[:,:3],axis=1)
        re=(Rotation.from_quat(values[:,3:])*Rotation.from_quat(after[:,3:]).inv()).magnitude()
        return dict(position_rmse_m=float(np.sqrt(np.mean(pe**2))),orientation_rmse_rad=float(np.sqrt(np.mean(re**2))))
    return dict(report,reason='one_step_prediction_only',model=error(predicted),instantaneous_tracking=error(command))


def inverse_reference_trajectory(model,desired_after,*,bounds,contact,strength=1.,component='both',static_only=False,position_model=None):
    """Build explicit actuator wrist setpoints for an unchanged geometric path.

    Desired row i is the post-command-i physical goal; desired i-1 is its
    planned before state. The effect ramps to full at contact and returns to
    zero at the protected influence boundaries. No physical goal is edited.
    """
    desired=np.asarray(desired_after,float)
    if (desired.ndim!=2 or desired.shape[1]!=7 or not np.isfinite(desired).all()
            or not np.isfinite(strength) or not 0<strength<=1 or component not in ('both','position','rotation')):
        raise ValueError('finite post-step goals and valid inverse compensation settings required')
    a,b=bounds;start,stop=contact
    if any(not isinstance(x,(int,np.integer)) for x in (a,b,start,stop)) or not 1<=a<start<=stop<b<len(desired):
        raise ValueError('protected influence and inclusive contact interval required')
    active=np.arange(a+1,b)
    before=desired[active-1] if not static_only else desired[active]
    inverse_model=model
    if component=='position':inverse_model=PoseResponse(model.position_gain,model.position_offset,np.ones(3),np.zeros(3))
    if component=='rotation':inverse_model=PoseResponse(np.ones(3),np.zeros(3),model.rotation_gain,model.rotation_offset)
    full=desired.copy();full[active]=inverse_model.inverse(before,desired[active])
    if position_model is not None and component in ('both','position'):
        previous=desired[active-2] if not static_only else desired[active]
        full[active,:3]=position_model.inverse(previous[:,:3],before[:,:3],desired[active,:3])
    t=np.arange(len(desired));rise=np.clip((t-a)/(start-a),0,1);fall=np.clip((b-t)/(b-stop),0,1)
    smooth=lambda u:u*u*u*(10+u*(-15+6*u))
    weight=strength*smooth(rise)*smooth(fall)
    result=desired.copy()
    if component in ('both','position'):result[:,:3]+=weight[:,None]*(full[:,:3]-desired[:,:3])
    delta=(Rotation.from_quat(full[:,3:])*Rotation.from_quat(desired[:,3:]).inv()).as_rotvec()
    if component in ('both','rotation'):
        result[:,3:]=(Rotation.from_rotvec(weight[:,None]*delta)*Rotation.from_quat(desired[:,3:])).as_quat()
    result[weight==0]=desired[weight==0]
    change=np.linalg.norm(result[:,:3]-desired[:,:3],axis=1)
    angle=(Rotation.from_quat(result[:,3:])*Rotation.from_quat(desired[:,3:]).inv()).magnitude()
    return result,dict(component=component,strength=strength,static_only=bool(static_only),bounds=[a,b],contact=[start,stop],
        position_response_kind='inertial' if position_model is not None else 'first_order',
        nominal_position_change_max_m=float(change.max()),nominal_orientation_change_max_rad=float(angle.max()),
        geometric_goals_unchanged=True,goal_clock='AFTER command i')
