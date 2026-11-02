"""Supported incoming-grasp placement and coordinated withdrawal proposals.

Models predict only a local supported object pose, never ownership or task
success. A separately recorded repeat and fresh execution must validate them.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_refinement import (
    RefinementConfig, anchor_hand, event_time_map, grasp_transition, retime_hand,
)


@dataclass(frozen=True)
class SupportedAnchor:
    translation: np.ndarray
    rotation: np.ndarray

    def predict(self, target):
        target = np.asarray(target, float)
        return np.concatenate([target[..., :3] + self.translation,
            (Rotation.from_rotvec(self.rotation) * Rotation.from_quat(
                target[..., 3:].reshape(-1, 4))).as_quat().reshape(target[..., 3:].shape)], axis=-1)


def connect_supported_contacts(segment, segments, support_repeats, *, coverage=.8, world_fraction=.8,
                               attachment_repeats=None):
    """Propose one editing interval across physically supported label gaps.

    Only consecutive same-hand, same-object segments can connect. Every gap
    must show independently measured ownership in each complete repeat. The
    original contact labels are unchanged; the returned interval is a proposal
    hypothesis that still needs fresh full-horizon evaluation.
    """
    arrays=[np.asarray(x) for x in support_repeats]
    if (len(arrays)<2 or any(x.ndim!=2 or x.dtype!=bool for x in arrays)
            or len({len(x) for x in arrays})!=1 or any(x.shape[1]<2 for x in arrays)
            or not 0<coverage<=1 or not 0<world_fraction<=1 or segment not in segments):
        raise ValueError('two named-segment ownership repeats and valid coverage required')
    if attachment_repeats is not None:
        if len(attachment_repeats)!=len(arrays) or any(
                np.shape(h)!=(*x.shape,7) or np.shape(o)!=np.shape(h)
                for (h,o),x in zip(attachment_repeats,arrays)):
            raise ValueError('matching actual hand/object attachment repeats required')
    cluster=segment;removed=[segment];gaps=[]
    for following in sorted((s for s in segments if s.hand==segment.hand and s.start>segment.end),key=lambda s:s.start):
        if following.object_id!=segment.object_id:break
        start,stop=cluster.end+1,following.start
        if not 0<=start<=stop<=len(arrays[0]):break
        fractions=[float(((x[start:stop].mean(0)>=coverage)).mean()) if start<stop else 1. for x in arrays]
        if min(fractions)<world_fraction:break
        stability=[]
        if attachment_repeats is not None and start<stop:
            from .held_carry import fit_attachment
            if stop-start<3:break
            for (hand,obj),mask in zip(attachment_repeats,arrays):
                _,report=fit_attachment(np.asarray(hand)[start:stop],np.asarray(obj)[start:stop],mask[start:stop],coverage=coverage)
                # Ownership continuity requires each world's grasp to remain
                # steady, not an identical grasp transform across all worlds.
                stability.append(float(np.mean([r['stable'] for r in report['per_world']])))
            if min(stability)<world_fraction:break
        gaps.append(dict(interval=[start,stop],passed_world_fraction=fractions,
            independent_world_attachment_fraction=stability))
        removed.append(following);cluster=replace(cluster,end=following.end)
    editing=[s for s in segments if s not in removed]+[cluster]
    return cluster,editing,dict(reason='physically_supported_contact_cluster',
        source_segments=[dict(start=s.start,end=s.end,index=s.index) for s in removed],gaps=gaps,
        source_contact_labels_unchanged=True)


def anchor_validation(model, actual, target, supported, *, position_m=.03,
                      rotation_rad=np.pi/9, coverage=.8):
    """Every world is retained in the validation denominator."""
    actual, target, mask = _inputs(actual, target, supported)
    if (not np.isfinite([position_m,rotation_rad,coverage]).all() or
            min(position_m,rotation_rad)<=0 or not 0<coverage<=1):
        raise ValueError('positive tolerances and valid coverage required')
    prediction = model.predict(target)
    rows = []
    for world in range(actual.shape[1]):
        good = mask[:,world]
        row = dict(coverage=float(good.mean()), passed=False)
        if good.mean()>=coverage and good.sum()>=3:
            observed = actual[good,world]
            pe = float(np.quantile(np.linalg.norm(observed[:,:3]-prediction[good,:3],axis=1),.9))
            re = float(np.quantile((Rotation.from_quat(observed[:,3:]) *
                Rotation.from_quat(prediction[good,3:]).inv()).magnitude(),.9))
            row.update(position_p90_m=pe,rotation_p90_rad=re,passed=pe<=position_m and re<=rotation_rad)
        rows.append(row)
    return dict(worlds=len(rows),passed_fraction=float(np.mean([r['passed'] for r in rows])),per_world=rows)


def _inputs(actual, target, supported):
    actual, target = np.asarray(actual,float), np.asarray(target,float)
    mask = np.array(supported,bool,copy=True)
    if (actual.ndim!=3 or actual.shape[-1]!=7 or target.shape!=(len(actual),7)
            or mask.shape!=actual.shape[:2] or len(actual)<3 or actual.shape[1]<2
            or not np.isfinite(target).all() or np.any(np.linalg.norm(target[:,3:],axis=1)<1e-8)):
        raise ValueError('matched object poses, fixed targets and support worlds required')
    mask &= np.isfinite(actual).all(-1)&(np.linalg.norm(actual[...,3:],axis=-1)>1e-8)
    return actual, target, mask


def fit_supported_anchor(actual, target, supported, *, position_m=.03,
                         rotation_rad=np.pi/9, coverage=.8, world_fraction=.5):
    """Robust local actual-vs-target pose offset while a giver owns the object.

    The offset is world translation plus a rotation about the target object.
    It is an approach hypothesis over the recorded interval. Later frames are
    forecasts, not additional supported observations.
    """
    actual, target, mask = _inputs(actual,target,supported)
    if (not np.isfinite([world_fraction,position_m,rotation_rad,coverage]).all()
            or not 0<world_fraction<=1 or not 0<coverage<=1 or min(position_m,rotation_rad)<=0):
        raise ValueError('valid minimum world fraction, coverage and tolerances required')
    positions=[];rotations=[]
    for w in range(actual.shape[1]):
        good=mask[:,w]
        if good.mean()>=coverage and good.sum()>=3:
            positions.append(np.median(actual[good,w,:3]-target[good,:3],axis=0))
            rotations.append((Rotation.from_quat(actual[good,w,3:]) *
                Rotation.from_quat(target[good,3:]).inv()).mean().as_quat())
    required=max(2,int(np.ceil(actual.shape[1]*world_fraction)))
    if len(positions)<required:
        return None,dict(reason='insufficient_supported_worlds',worlds=actual.shape[1],required_worlds=required)
    positions=np.asarray(positions);rotations=Rotation.from_quat(rotations)
    distances=np.asarray([(rotations*r.inv()).magnitude() for r in rotations])
    medoid=rotations[int(np.argmin(distances.sum(1)))]
    position=np.median(positions,axis=0)
    keep=(np.linalg.norm(positions-position,axis=1)<=position_m)&((rotations*medoid.inv()).magnitude()<=rotation_rad)
    if keep.sum()<required:
        return None,dict(reason='inconsistent_supported_worlds',worlds=actual.shape[1],required_worlds=required)
    model=SupportedAnchor(np.median(positions[keep],axis=0),rotations[keep].mean().as_rotvec())
    report=anchor_validation(model,actual,target,mask,position_m=position_m,rotation_rad=rotation_rad,coverage=coverage)
    if report['passed_fraction']<world_fraction:
        return None,dict(report,reason='supported_pose_offset_not_locally_predictive')
    return model,dict(report,reason='supported_pose_offset',translation=model.translation.tolist(),rotation=model.rotation.tolist())


def incoming_approach(current, hand, object_target, anchor, segment, segments, fps,
                      *, project, gain=1., config=RefinementConfig()):
    """Prepare the incoming hand about a supported object; keep pose coupling.

    Projection is a feasibility oracle. Backtrack the coupled gain instead of
    accepting separately clipped translation and rotation. Keep ramp durations
    while backtracking so the speed reduction is real.
    """
    target=np.asarray(object_target,float);base=np.asarray(current,float)
    if target.shape!=(len(base),7) or not np.isfinite(target).all() or not 0<gain<=1:
        raise ValueError('finite object targets and valid gain required')
    requested=gain;lead=tail=0;attempts=[]
    for _ in range(12):
        template=anchor_hand(base,hand,gain*anchor.translation,gain*anchor.rotation,
            np.ones(len(base)),pivot=target[:,:3])
        proposed,info=grasp_transition(base,template,hand,segment,segments,fps,config,
            minimum_approach_frames=lead,minimum_return_frames=tail)
        if proposed is None:return None,info
        lead=max(lead,info['required_approach_frames']);tail=max(tail,info['required_return_frames'])
        projected,scale=project(proposed)
        pe=float(np.linalg.norm(projected[:,hand.position]-proposed[:,hand.position],axis=1).max())
        re=float((Rotation.from_quat(projected[:,hand.quaternion])*Rotation.from_quat(proposed[:,hand.quaternion]).inv()).magnitude().max())
        attempts.append(dict(gain=gain,scale=float(scale),position_distortion_m=pe,rotation_distortion_rad=re))
        if pe<=1e-6 and re<=1e-5:
            return projected,dict(info,requested_gain=requested,applied_gain=gain,feasibility_attempts=attempts)
        gain*=min(.9,max(.1,float(scale)*.98))
    return None,dict(reason='no_coupled_feasible_incoming_approach',feasibility_attempts=attempts)


def approach_motion_limits(current, proposed, hand, object_target, segment, fps,
                           config=RefinementConfig()):
    """Separate a rigid contact transport from extra approach/return speed.

    Rotating a whole grasp changes its world angular-velocity axis while keeping
    its angular speed. A bound on the difference of world velocity vectors
    alone would shrink a valid rigid grasp. Allow only the exactly verified
    constant object-pivot transform during contact; retain ordinary speed
    bounds in the free approach and return. No collision/dynamics claim follows.
    """
    base=np.asarray(current,float);q=np.asarray(proposed,float);target=np.asarray(object_target,float)
    if (q.shape!=base.shape or target.shape!=(len(base),7) or not np.isfinite(fps) or fps<=0
            or not 0<=segment.start<segment.end<len(base)
            or not all(np.isfinite(x).all() for x in (base,q,target))):
        raise ValueError('finite aligned references and contact interval required')
    old=Rotation.from_quat(base[:,hand.quaternion]);new=Rotation.from_quat(q[:,hand.quaternion])
    rotation=new*old.inv();contact=slice(segment.start,segment.end+1)
    if (rotation[contact]*rotation[segment.start].inv()).magnitude().max()>1e-5:
        raise ValueError('contact correction is not a constant rigid rotation')
    translation=q[:,hand.position]-target[:,:3]-rotation.apply(base[:,hand.position]-target[:,:3])
    if np.linalg.norm(translation[contact]-translation[segment.start],axis=1).max()>1e-6:
        raise ValueError('contact correction is not a constant object-pivot transport')
    ps=np.linalg.norm(np.diff(q[:,hand.position]-base[:,hand.position],axis=0),axis=1)*fps
    rs=np.linalg.norm((new[1:]*new[:-1].inv()).as_rotvec()-(old[1:]*old[:-1].inv()).as_rotvec(),axis=1)*fps
    inside=(np.arange(len(base)-1)>=segment.start)&(np.arange(len(base)-1)<segment.end)
    free_p=float(ps[~inside].max(initial=0));free_r=float(rs[~inside].max(initial=0))
    free_scale=min(1.,config.max_translation_speed_m_s/max(free_p,1e-12),
        config.max_rotation_speed_rad_s/max(free_r,1e-12))
    contact_p=float(ps[inside].max(initial=0));contact_r=float(rs[inside].max(initial=0))
    limits=replace(config,max_translation_speed_m_s=max(config.max_translation_speed_m_s,contact_p+1e-6),
        max_rotation_speed_rad_s=max(config.max_rotation_speed_rad_s,contact_r+1e-5))
    return limits,free_scale,dict(rigid_contact_transport=True,free_added_translation_peak_m_s=free_p,
        free_added_rotation_peak_rad_s=free_r,contact_added_translation_peak_m_s=contact_p,
        contact_added_rotation_peak_rad_s=contact_r,free_scale=free_scale)


def delay_withdrawal(current, hand, object_target, *, start, event, end, delay_frames, fps):
    """Stretch the outgoing contact in object coordinates, preserving the prefix.

    This timing proposal reports actual reference speeds. It is not a dynamics
    certificate; its composed release and incoming grasp need fresh rollouts.
    """
    base=np.asarray(current,float);target=np.asarray(object_target,float)
    if (target.shape!=(len(base),7) or not np.isfinite(target).all() or
            not isinstance(delay_frames,int) or delay_frames<=0 or not np.isfinite(fps) or fps<=0):
        raise ValueError('finite targets and positive integer delay/rate required')
    times=event_time_map(len(base),(start,end),event,delay_frames)
    weight=((np.arange(len(base))>start)&(np.arange(len(base))<end)).astype(float)
    out=retime_hand(base,hand,target[:,None,:3],target[:,None,3:],0,times,weight)
    # Exact prefix/suffix protection avoids quaternion round-off at untouched rows.
    out[:start+1]=base[:start+1];out[end:]=base[end:]
    ids=(*hand.position,*hand.quaternion,*hand.fingers)
    protected=[i for i in range(base.shape[1]) if i not in ids]
    np.testing.assert_array_equal(out[:,protected],base[:,protected])
    rotation=Rotation.from_quat(out[:,hand.quaternion])
    return out,dict(reason='object_relative_contact_extension',time_window=[start,end],
        original_withdrawal_frame=event,delayed_withdrawal_frame=event+delay_frames,
        peak_reference_translation_speed_m_s=float(np.linalg.norm(np.diff(out[:,hand.position],axis=0),axis=1).max()*fps),
        peak_reference_rotation_speed_rad_s=float((rotation[1:]*rotation[:-1].inv()).magnitude().max()*fps))
