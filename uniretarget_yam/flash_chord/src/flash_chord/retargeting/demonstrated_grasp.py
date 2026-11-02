"""Propose receiving references from an actually loaded grasp.

Keep achieved geometry and actuator-command geometry distinct. References
respect physical finger limits; remaining spring preload is an ordinary
residual target. No object command, simulation state or actor is edited here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_refinement import RefinementConfig,grasp_transition
from .grasp_feedback import ObjectRelativeGrasp,capture_object_relative_grasp


@dataclass(frozen=True)
class DemonstratedGrasp:
    geometry: ObjectRelativeGrasp
    command: ObjectRelativeGrasp
    frame: int
    diagnostic: dict


def capture_steady_grasp(wrist,objects,commands,supported,schema,*,interval,width_frames):
    """Choose one recorded sample from the steadiest independently loaded window.

    Window selection uses achieved hand/object geometry and discovery support.
    Commands at that exact sample preserve signed preload, including overdrive.
    This is a proposal source, not proof that the grasp can be reacquired.
    """
    h,o,c=np.asarray(wrist,float),np.asarray(objects,float),np.asarray(commands,float)
    support=np.asarray(supported)
    if (h.ndim!=2 or h.shape[-1]!=7 or o.shape!=h.shape or c.ndim!=2 or len(c)!=len(h)
            or support.shape!=(len(h),) or support.dtype!=bool
            or not isinstance(width_frames,int) or width_frames<3):
        raise ValueError('aligned achieved wrist/object/command frames and Boolean support required')
    a,b=interval
    if any(not isinstance(v,(int,np.integer)) for v in (a,b)) or not 0<=a<b<=len(h):
        raise ValueError('valid half-open discovery interval required')
    if (not all(np.isfinite(x[a:b]).all() for x in (h,o,c))
            or np.any(np.linalg.norm(h[a:b,3:],axis=1)<1e-8)
            or np.any(np.linalg.norm(o[a:b,3:],axis=1)<1e-8)):
        raise ValueError('finite supported poses and native commands required')
    ro=Rotation.from_quat(o[a:b,3:]);rh=Rotation.from_quat(h[a:b,3:])
    position=ro.inv().apply(h[a:b,:3]-o[a:b,:3]);rotation=ro.inv()*rh
    best=None
    for start in range(a,b-width_frames+1):
        stop=start+width_frames
        if not support[start:stop].all():continue
        p=position[start-a:stop-a];r=rotation[start-a:stop-a]
        ps=float(np.linalg.norm(p-np.median(p,axis=0),axis=1).max())
        rs=float((r*r.mean().inv()).magnitude().max())
        if ps>.02 or rs>np.pi/12:continue
        score=ps/.02+rs/(np.pi/12)
        if best is None or score<best[0]:best=(score,start,stop,ps,rs)
    if best is None:return None,dict(reason='no_steady_independently_loaded_grasp_window')
    _,start,stop,ps,rs=best;frame=(start+stop)//2
    ids=tuple(schema['wrist_pos_dof_ids']);rids=tuple(schema['wrist_orient_dof_ids']);fids=tuple(schema['finger_dof_ids'])
    if len(ids)!=3 or len(rids)!=3 or not fids or min((*ids,*rids,*fids))<0 or max((*ids,*rids,*fids))>=c.shape[1]:
        raise ValueError('native floating wrist/finger target schema required')
    control=np.r_[c[frame,list(ids)],Rotation.from_rotvec(c[frame,list(rids)]).as_quat()]
    fingers=c[frame,list(fids)]
    geometric=capture_object_relative_grasp(o[frame],h[frame],fingers)
    command=capture_object_relative_grasp(o[frame],control,fingers)
    report=dict(reason='captured_steady_loaded_grasp',frame=frame,window=[start,stop],
        achieved_position_spread_m=ps,achieved_rotation_spread_rad=rs,
        source_is_reacquisition_certificate=False)
    return DemonstratedGrasp(geometric,command,frame,report),report


def wrist_from_root(q,hand,baseline_q,baseline_wrist):
    """Exact floating-root to semantic wrist transform, verified on the source."""
    base=np.asarray(baseline_q,float);wrist=np.asarray(baseline_wrist,float)
    values=np.asarray(q,float)
    if values.shape!=base.shape or wrist.shape!=(len(base),7):raise ValueError('matching root/wrist trajectories required')
    old=Rotation.from_quat(base[:,hand.quaternion]);wr=Rotation.from_quat(wrist[:,3:])
    offset=old.inv().apply(wrist[:,:3]-base[:,hand.position]);local=old.inv()*wr
    if np.linalg.norm(offset-offset[0],axis=1).max()>1e-5 or (local*local[0].inv()).magnitude().max()>1e-5:
        raise ValueError('semantic wrist is not rigidly attached to the floating root')
    r=Rotation.from_quat(values[:,hand.quaternion])
    return np.c_[values[:,hand.position]+r.apply(np.broadcast_to(offset[0],(len(base),3)).copy()),
        (r*local[0]).as_quat()]


def render_demonstrated_grasp(current,hand,baseline_wrist,object_control,grasp,segment,segments,
                              lower,upper,*,fps,config=RefinementConfig(),return_config=None):
    """Realize a complete demonstrated posture; never accept a shrunken grasp.

    Contact motion follows the explicitly supplied object-control hypothesis.
    Approach/return use the hand's free intervals and measured added velocities.
    The object-control path is for HAND targets only, not an object command.
    """
    base=np.asarray(current,float);wrist=np.asarray(baseline_wrist,float);obj=np.asarray(object_control,float)
    lo,hi=np.asarray(lower,float),np.asarray(upper,float)
    if (obj.shape!=(len(base),7) or wrist.shape!=obj.shape or lo.shape!=(len(hand.fingers),) or hi.shape!=lo.shape
            or np.any(lo>hi) or not all(np.isfinite(x).all() for x in (base,wrist,obj,lo,hi))):
        raise ValueError('finite compatible grasp reference and finger limits required')
    ro=Rotation.from_quat(obj[:,3:]);goal=np.c_[obj[:,:3]+ro.apply(np.broadcast_to(grasp.wrist_position_o,(len(obj),3)).copy()),
        (ro*Rotation.from_quat(grasp.wrist_quaternion_o)).as_quat()]
    root=Rotation.from_quat(base[:,hand.quaternion]);wr=Rotation.from_quat(wrist[:,3:])
    delta=Rotation.from_quat(goal[:,3:])*wr.inv()
    template=base.copy();template[:,hand.position]=goal[:,:3]+delta.apply(base[:,hand.position]-wrist[:,:3])
    template[:,hand.quaternion]=(delta*root).as_quat()
    source=np.asarray(grasp.fingers,float)
    if source.shape!=lo.shape:raise ValueError('demonstrated finger targets do not match hand')
    bounded=np.clip(source,lo,hi);template[:,hand.fingers]=bounded
    lead=tail=0;attempts=[]
    return_limits=return_config or config
    for attempt in range(10):
        out,info=grasp_transition(base,template,hand,segment,segments,fps,config,
            minimum_approach_frames=lead,minimum_return_frames=tail,return_config=return_limits)
        if out is None:return None,None,info
        a,b=info['window'];t=np.arange(len(base));rise=np.clip((t-a)/(segment.start-a),0,1);fall=np.clip((b-t)/(b-segment.end),0,1)
        # A new fixed grip replaces the old finger choreography. Interpolate
        # absolute postures to/from the actual boundary values; adding a fixed
        # correction to a moving baseline can create unnecessary finger swings.
        smooth=lambda x:x*x*x*(10+x*(-15+6*x))
        before=(t>a)&(t<segment.start);after=(t>segment.end)&(t<b)
        out[np.ix_(before,hand.fingers)]=base[a,hand.fingers]+smooth(rise[before])[:,None]*(bounded-base[a,hand.fingers])
        out[segment.start:segment.end+1,hand.fingers]=bounded
        out[np.ix_(after,hand.fingers)]=bounded+smooth(1-fall[after])[:,None]*(base[b,hand.fingers]-bounded)
        out[:,hand.fingers]=np.clip(out[:,hand.fingers],lo,hi)
        out[(t<=a)|(t>=b)]=base[(t<=a)|(t>=b)]
        weight=(.5-.5*np.cos(np.pi*rise))*(.5-.5*np.cos(np.pi*fall))
        qrotation=Rotation.from_quat(out[:,hand.quaternion])
        ps=np.linalg.norm(np.diff(out[:,hand.position]-base[:,hand.position],axis=0),axis=1)*fps
        rs=np.linalg.norm((qrotation[1:]*qrotation[:-1].inv()).as_rotvec()-(root[1:]*root[:-1].inv()).as_rotvec(),axis=1)*fps
        fs=np.abs(np.diff(out[:,hand.fingers],axis=0)).max(1)*fps
        def ratio(start,stop,limits):
            return max(1.,float(ps[start:stop].max(initial=0))/limits.max_translation_speed_m_s,
                float(rs[start:stop].max(initial=0))/limits.max_rotation_speed_rad_s,
                float(fs[start:stop].max(initial=0))/limits.max_finger_speed_rad_s)
        before=ratio(a,segment.start,config);after=ratio(segment.end,b,return_limits)
        attempts.append(dict(approach_ratio=before,return_ratio=after,window=[a,b]))
        if max(before,after)<=1.+1e-5:
            actual=wrist_from_root(out,hand,base,wrist)
            contact=slice(segment.start,segment.end+1)
            pe=float(np.linalg.norm(actual[contact,:3]-goal[contact,:3],axis=1).max())
            re=float((Rotation.from_quat(actual[contact,3:])*Rotation.from_quat(goal[contact,3:]).inv()).magnitude().max())
            fe=float(np.abs(out[contact][:,hand.fingers]-bounded).max())
            if pe>1e-5 or re>1e-5 or fe>1e-6:raise ValueError('full demonstrated contact posture was not realized')
            preload=source-bounded
            commands=out[:,hand.fingers]+weight[:,None]*preload
            return out,dict(wrist=actual,fingers=commands),dict(info,reason='full_demonstrated_grasp',
                wrist_goal_position_error_m=pe,wrist_goal_rotation_error_rad=re,finger_goal_error_rad=fe,
                preload_max_rad=float(np.abs(preload).max()),transition_attempts=attempts)
        available=info['available_window']
        if (before>1.+1e-5 and a<=available[0]) or (after>1.+1e-5 and b>=available[1]):
            return None,None,dict(info,reason='insufficient_full_grasp_transition_time',transition_attempts=attempts)
        lead=max(lead,int(np.ceil(info['required_approach_frames']*before*1.05)))
        tail=max(tail,int(np.ceil(info['required_return_frames']*after*1.05)))
    return None,None,dict(reason='no_feasible_full_grasp_transition',transition_attempts=attempts)
