"""Carry corrections identified only from independently loaded, steady grasps.

The object target is immutable. The inverse pose map is a local response
hypothesis, not a simulator gradient or an acceptance decision.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter

from .contact_refinement import continuous_rotvec


def _relative(hand,obj):
    rh=Rotation.from_quat(hand[:,3:]);ro=Rotation.from_quat(obj[:,3:])
    return rh.inv().apply(obj[:,:3]-hand[:,:3]),rh.inv()*ro


@dataclass(frozen=True)
class Attachment:
    position: np.ndarray
    quaternion: np.ndarray
    stable_worlds: np.ndarray

    def predict(self,hand):
        h=np.asarray(hand,float);r=Rotation.from_quat(h[...,3:].reshape(-1,4))
        p=h[...,:3]+r.apply(np.broadcast_to(self.position,(r.as_quat().shape[0],3)).copy()).reshape(h[...,:3].shape)
        q=(r*Rotation.from_quat(self.quaternion)).as_quat().reshape(h[...,3:].shape)
        return np.concatenate([p,q],axis=-1)


def fit_attachment(hand,obj,loaded,*,position_m=.02,rotation_rad=np.pi/12,coverage=.8,world_fraction=.5):
    """Fit an actual hand->object transform while retaining rejected denominators."""
    h=np.asarray(hand,float);o=np.asarray(obj,float);mask=np.asarray(loaded,bool)
    if (h.shape!=o.shape or h.ndim!=3 or h.shape[-1]!=7 or mask.shape!=h.shape[:2]
            or len(h)<3 or not np.isfinite([position_m,rotation_rad,coverage,world_fraction]).all()
            or min(position_m,rotation_rad)<=0 or not 0<coverage<=1 or not 0<world_fraction<=1):
        raise ValueError('matching loaded pose intervals and positive model tolerances required')
    valid=np.isfinite(h).all(-1)&np.isfinite(o).all(-1)&(np.linalg.norm(h[...,3:],axis=-1)>1e-8)&(np.linalg.norm(o[...,3:],axis=-1)>1e-8)
    mask=mask&valid;worlds=h.shape[1];positions=[];rotations=[];kept=[];records=[]
    for world in range(worlds):
        good=mask[:,world];record=dict(coverage=float(good.mean()),stable=False)
        if good.mean()>=coverage and good.sum()>=3:
            p,r=_relative(h[good,world],o[good,world]);center=np.median(p,axis=0);mean=r.mean()
            ps=float(np.quantile(np.linalg.norm(p-center,axis=1),.9));rs=float(np.quantile((r*mean.inv()).magnitude(),.9))
            record.update(position_spread_m=ps,rotation_spread_rad=rs)
            if ps<=position_m and rs<=rotation_rad:
                positions.append(center);rotations.append(mean.as_quat());kept.append(world);record['stable']=True
        records.append(record)
    required=max(2,int(np.ceil(world_fraction*worlds)))
    report=dict(worlds=worlds,required_worlds=required,per_world=records,fit_worlds=0,accepted=False)
    if len(kept)<required:return None,dict(report,reason='insufficient_steady_loaded_worlds')
    positions=np.asarray(positions);rotations=Rotation.from_quat(rotations);center=np.median(positions,axis=0)
    distances=np.asarray([(rotations*r.inv()).magnitude() for r in rotations])
    medoid=rotations[int(np.argmin(distances.sum(1)))]
    consensus=(np.linalg.norm(positions-center,axis=1)<=position_m)&((rotations*medoid.inv()).magnitude()<=rotation_rad)
    keep=np.asarray(kept)[consensus]
    if len(keep)<required:return None,dict(report,reason='inconsistent_grasps_across_worlds',fit_worlds=len(keep))
    stable=np.zeros(worlds,bool);stable[keep]=True
    model=Attachment(np.median(positions[consensus],axis=0),rotations[consensus].mean().as_quat(),stable)
    return model,dict(report,accepted=True,reason='steady_loaded_attachment',fit_worlds=len(keep),stable_worlds=stable.tolist(),
        position=model.position.tolist(),quaternion_xyzw=model.quaternion.tolist())


def validate_attachment(model,hand,obj,loaded,*,position_m=.02,rotation_rad=np.pi/12,coverage=.8):
    """Predict a separately recorded interval using the discovery transform."""
    h=np.asarray(hand,float);o=np.asarray(obj,float);loaded=np.asarray(loaded,bool)
    if h.shape!=o.shape or h.shape[:2]!=loaded.shape or h.shape[-1]!=7:raise ValueError('matched validation poses required')
    valid=np.isfinite(h).all(-1)&np.isfinite(o).all(-1)&(np.linalg.norm(h[...,3:],axis=-1)>1e-8)&(np.linalg.norm(o[...,3:],axis=-1)>1e-8)
    good=valid&loaded;passed=np.zeros(h.shape[1],bool);rows=[]
    for w in range(h.shape[1]):
        select=good[:,w];row=dict(coverage=float(select.mean()),passed=False)
        if select.mean()>=coverage and select.sum()>=3:
            prediction=model.predict(h[select,w]);position=np.linalg.norm(prediction[:,:3]-o[select,w,:3],axis=-1)
            angle=(Rotation.from_quat(prediction[:,3:])*Rotation.from_quat(o[select,w,3:]).inv()).magnitude()
            pe=float(np.quantile(position,.9));re=float(np.quantile(angle,.9));passed[w]=pe<=position_m and re<=rotation_rad
            row.update(position_p90_m=pe,orientation_p90_rad=re,passed=bool(passed[w]))
        rows.append(row)
    return dict(passed_fraction=float(passed.mean()),per_world=rows)


def supported_reference_relation(reference,obj,loaded,stable_worlds,*,minimum_worlds=2,smoothing_frames=5):
    """Time-indexed reference-hand pose in the measured actual object frame.

    Missing interior samples are interpolated; endpoint holding is reported.
    A caller must start its edit after the first supported sample and may use
    the last relation as a prediction, never as new physical evidence.
    """
    ref=np.asarray(reference,float);o=np.asarray(obj,float);mask=np.array(loaded,bool,copy=True)
    if (ref.shape!=(len(o),7) or o.ndim!=3 or o.shape[-1]!=7 or mask.shape!=o.shape[:2]
            or np.shape(stable_worlds)!=(o.shape[1],) or minimum_worlds<2 or not np.isfinite(ref).all()):
        raise ValueError('matching reference and supported object poses required')
    mask&=np.asarray(stable_worlds,bool)[None]
    mask&=np.isfinite(o).all(-1)&(np.linalg.norm(o[...,3:],axis=-1)>1e-8)
    frames=[];positions=[];quaternions=[]
    for frame in range(len(o)):
        valid=mask[frame]
        if valid.sum()<minimum_worlds:continue
        ro=Rotation.from_quat(o[frame,valid,3:]);rr=Rotation.from_quat(ref[frame,3:])
        positions.append(np.median(ro.inv().apply(ref[frame,:3]-o[frame,valid,:3]),axis=0))
        quaternions.append((ro.inv()*rr).mean().as_quat());frames.append(frame)
    if len(frames)<3:raise ValueError('insufficient supported reference response samples')
    t=np.arange(len(o));positions=np.asarray(positions);rotations=Rotation.from_quat(quaternions)
    position=np.stack([np.interp(t,frames,positions[:,i]) for i in range(3)],axis=1)
    orientation=Slerp(frames,rotations)(np.clip(t,frames[0],frames[-1]))
    width=min(smoothing_frames,len(frames));width=width if width%2 else width-1
    if width>=3:
        # Smooth only within measured bounds; held endpoints cannot supply
        # synthetic pre-contact evidence to the filter.
        interval=slice(frames[0],frames[-1]+1)
        position[interval]=savgol_filter(position[interval],width,2,axis=0)
        rv=continuous_rotvec(orientation)[interval];rv=savgol_filter(rv,width,2,axis=0)
        q=orientation.as_quat();q[interval]=Rotation.from_rotvec(rv).as_quat()
        q[:frames[0]]=q[frames[0]];q[frames[-1]+1:]=q[frames[-1]];orientation=Rotation.from_quat(q)
        position[:frames[0]]=position[frames[0]];position[frames[-1]+1:]=position[frames[-1]]
    return np.concatenate([position,orientation.as_quat()],axis=-1),dict(first_observed=frames[0],last_observed=frames[-1],
        observed_frames=len(frames),sample_worlds=mask.sum(1).tolist())


def carry_candidate(current,hand,reference_palm,target_object,relation,*,start,contact_end,restore_limit,fps,gain=1.,translation_speed=.3,rotation_speed=2.,minimum_ramp_frames=0):
    """Rate-derived rise/carry/restore; caller projects measured increments.

    A time-varying response can add speed beyond the ramp's bound. The native
    adapter must check/project the resulting actual reference trajectory.
    """
    q=np.asarray(current,float);ref=np.asarray(reference_palm,float);target=np.asarray(target_object,float);rel=np.asarray(relation,float)
    if (ref.shape!=(len(q),7) or target.shape!=ref.shape or rel.shape!=ref.shape or not np.isfinite([fps,gain,translation_speed,rotation_speed]).all()
            or min(fps,translation_speed,rotation_speed)<=0 or not 0<gain<=1 or not 0<=start<contact_end<restore_limit<len(q)
            or not isinstance(minimum_ramp_frames,int) or minimum_ramp_frames<0
            or not all(np.isfinite(v).all() for v in (q,ref,target,rel))):raise ValueError('finite compatible carry references/timing required')
    ro=Rotation.from_quat(target[:,3:]);rr=Rotation.from_quat(ref[:,3:]);root=Rotation.from_quat(q[:,hand.quaternion])
    desired_p=target[:,:3]+ro.apply(rel[:,:3]);desired_r=ro*Rotation.from_quat(rel[:,3:])
    dp=desired_p-ref[:,:3];delta=desired_r*rr.inv();rv=continuous_rotvec(delta)
    object_estimate_r=rr*Rotation.from_quat(rel[:,3:]).inv()
    object_pivot=ref[:,:3]-object_estimate_r.apply(rel[:,:3])
    object_delta=target[:,:3]-object_pivot
    # After the modeled carry interval, restore from the final correction;
    # never chase an unsupported/falling object during the free-hand return.
    dp[contact_end+1:]=dp[contact_end];rv[contact_end+1:]=rv[contact_end]
    full_root=ref[:,:3]+dp+Rotation.from_rotvec(rv).apply(q[:,hand.position]-ref[:,:3])
    peak_position=float(np.linalg.norm(full_root[start:contact_end+1]-q[start:contact_end+1,hand.position],axis=1).max())
    peak_angle=float(np.linalg.norm(rv[start:contact_end+1],axis=1).max())
    # Bound the orbit's arc rather than its endpoint chord. Scaling a palm
    # translation independently would introduce object-position error even for
    # an exactly rigid attachment and an otherwise exact rotation update.
    radius=np.linalg.norm(q[:,hand.position]-object_pivot,axis=1)
    path_bound=np.linalg.norm(object_delta,axis=1)+np.linalg.norm(rv,axis=1)*radius
    travel=float(path_bound[start:contact_end+1].max())
    ramp=max(1,minimum_ramp_frames,int(np.ceil(np.pi/2*gain*max(travel/translation_speed,peak_angle/rotation_speed)*fps))+1)
    if start+ramp>=contact_end or contact_end+ramp>restore_limit:
        return None,dict(reason='insufficient_loaded_carry_or_restore_time',required_ramp_frames=ramp)
    t=np.arange(len(q));rise=np.clip((t-start)/ramp,0,1);fall=np.clip((contact_end+ramp-t)/ramp,0,1)
    weight=gain*(.5-.5*np.cos(np.pi*rise))*(.5-.5*np.cos(np.pi*fall))
    correction=Rotation.from_rotvec(weight[:,None]*rv)
    out=q.copy();out[:,hand.position]=object_pivot+weight[:,None]*object_delta+correction.apply(q[:,hand.position]-object_pivot)
    out[:,hand.quaternion]=(correction*root).as_quat()
    # Preserve the baseline's free-hand return instead of rotating its world
    # angular-velocity axis. A right/body-relative offset adds only its own
    # fade velocity; a fixed world-left rotation would add (D-I)*omega_base.
    tail=np.arange(len(q))>contact_end
    end_delta=out[contact_end,hand.position]-q[contact_end,hand.position]
    body_offset=root[contact_end].inv()*Rotation.from_quat(out[contact_end,hand.quaternion])
    fade=.5-.5*np.cos(np.pi*fall[tail])
    out[np.ix_(tail,hand.position)]=q[np.ix_(tail,hand.position)]+fade[:,None]*end_delta
    out[np.ix_(tail,hand.quaternion)]=(root[tail]*Rotation.from_rotvec(fade[:,None]*body_offset.as_rotvec())).as_quat()
    out[weight==0]=q[weight==0];out[[0,-1]]=q[[0,-1]]
    return out,dict(reason='identified_supported_carry_response',ramp_frames=ramp,edit_bounds=[start,contact_end+ramp],
        requested_gain=gain,full_goal_position_change_m=peak_position,full_goal_orientation_change_rad=peak_angle)


def feasible_carry_candidate(current,hand,reference_palm,target_object,relation,*,project,**timing):
    """Contract a coupled pose response instead of distorting its components.

    Keep the original ramp duration while reducing gain; shortening the ramp
    again would undo the speed reduction. Projection is used as a feasibility
    oracle, and only numerically unchanged projected trajectories are returned.
    """
    requested=float(timing.get('gain',1.));gain=requested;ramp=0;attempts=[]
    for attempt in range(12):
        candidate,info=carry_candidate(current,hand,reference_palm,target_object,relation,
            **dict(timing,gain=gain,minimum_ramp_frames=ramp))
        if candidate is None:return None,info
        ramp=max(ramp,info['ramp_frames']);projected,scale=project(candidate)
        position=float(np.linalg.norm(projected[:,hand.position]-candidate[:,hand.position],axis=1).max())
        orientation=float((Rotation.from_quat(projected[:,hand.quaternion])*Rotation.from_quat(candidate[:,hand.quaternion]).inv()).magnitude().max())
        attempts.append(dict(gain=gain,projection_scale=float(scale),position_change_m=position,orientation_change_rad=orientation))
        if position<=1e-6 and orientation<=1e-5:
            info.update(requested_gain=requested,applied_gain=gain,projection_scale=float(scale),rate_backtracks=attempt,
                projection_position_change_m=position,projection_orientation_change_rad=orientation,feasibility_attempts=attempts)
            return projected,info
        gain*=min(.9,max(.1,float(scale)*.98))
    return None,dict(reason='no_coupled_feasible_carry_step',requested_gain=requested,last_gain=gain)
