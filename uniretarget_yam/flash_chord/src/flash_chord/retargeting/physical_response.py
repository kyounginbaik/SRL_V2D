"""Local identification of reference -> frozen-policy -> physical residuals.

This module estimates a response, not an analytic simulator gradient. No model
prediction is an acceptance decision. Contact-mode changes remain a separate
search, and all candidates require fresh full-horizon task evaluation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from .contact_refinement import event_time_map, find_giver, retime_hand, squeeze_hand
from .refinement_window import window_mask
from .contact_segment_window import aligned_pose_errors


def object_frame_vectors(quaternions, vector):
    """Rotate one object-frame vector per reference row.

    Some SciPy rotation backends require writable input buffers even though the
    operation is conceptually read-only; broadcast views must be materialized.
    """
    q=np.asarray(quaternions,float)
    if q.ndim!=2 or q.shape[1]!=4:raise ValueError('expected [frame,4] object quaternions')
    vectors=np.broadcast_to(np.asarray(vector,float),(len(q),3)).copy()
    return Rotation.from_quat(q).apply(vectors)


def projection_error(proposed, projected, hands):
    """Maximum normalized change imposed by feasibility projection.

    A single minimum projection factor cannot describe independently scaled
    translation, orientation and fingers. Compare the actual trajectories.
    Quaternion signs are immaterial.
    """
    a=np.asarray(proposed,float);b=np.asarray(projected,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite matching reference arrays required')
    errors=[0.]
    for hand in hands:
        errors.append(float(np.linalg.norm(a[:,hand.position]-b[:,hand.position],axis=-1).max())/.02)
        angle=(Rotation.from_quat(a[:,hand.quaternion])*Rotation.from_quat(b[:,hand.quaternion]).inv()).magnitude()
        errors.append(float(angle.max())/.2)
        errors.append(float(np.abs(a[:,hand.fingers]-b[:,hand.fingers]).max(initial=0))/.3)
    return max(errors)


def hand_action_indices(names,ranges,side,action_dim,*,component='all'):
    """Resolve a hand's residual controls from the runtime's named schema."""
    if side not in ('left','right') or len(names)!=len(ranges) or component not in ('all','wrist','fingers'):
        raise ValueError('valid hand, component and matched action schema required')
    selected=[]
    for name,(start,stop) in zip(names,ranges):
        if not 0<=start<stop<=action_dim:raise ValueError('invalid action block range')
        match=name.startswith(side+'_')
        if component=='wrist':match=match and name.startswith(side+'_wrist')
        if component=='fingers':match=match and name.startswith(side+'_finger')
        if match:selected.extend(range(start,stop))
    if not selected or len(set(selected))!=len(selected):raise ValueError('missing or overlapping hand action blocks')
    return np.asarray(selected,int)


def reference_action_schedule(names, ranges, action_dim, groups, candidates,
                              frames, receiver, intervention_window):
    """Explicit per-frame diagnostic residual gates, never a learned policy.

    Preserve the parent's named-hand gate and compose candidate-local gates.
    Invalid groups, channel schemas and intervals fail before any rollout.
    """
    groups=np.asarray(groups)
    if (groups.ndim!=1 or not np.issubdtype(groups.dtype,np.integer) or len(groups)==0
            or np.any(groups<0) or np.any(groups>=len(candidates))
            or not isinstance(frames,int) or frames<1):
        raise ValueError('valid per-world candidate groups and frame count required')
    schedule=np.zeros((frames,len(groups),action_dim),bool)
    for world,group in enumerate(groups):
        meta=candidates[int(group)];specifications=list(meta.get('extra_interventions',[]))
        if meta['component']!='actor':
            specifications.append(dict(hand=receiver,component=meta['component'],bounds=intervention_window))
        for specification in specifications:
            a,b=specification['bounds']
            if (any(not isinstance(v,(int,np.integer)) for v in (a,b))
                    or not 0<=a<=b<frames):
                raise ValueError('intervention bounds must be inclusive executed frames')
            ids=hand_action_indices(names,ranges,specification['hand'],action_dim,
                                    component=specification['component'])
            schedule[a:b+1,world,ids]=True
    return schedule


def supported_object_anchor(poses,supported,segment,info,*,minimum_support_fraction=.8):
    """Estimate the desired-to-actual object transform before transfer fails.

    Support is externally measured giver evidence, not inferred from a small
    tracking error. Later failed/falling poses cannot contribute to this grasp
    approach correction. Translation and world rotation match anchor_hand's
    pivot-at-desired-object convention.
    """
    x=np.asarray(poses,float);support=np.asarray(supported,bool)
    if x.ndim!=4 or x.shape[-1]!=14 or support.shape!=x.shape[:2]:
        raise ValueError('matching full object poses and per-world support required')
    if not 0<minimum_support_fraction<=1:raise ValueError('valid support fraction required')
    stop=min(len(x),int(info['event_frame']))
    if info.get('first_break') is not None:stop=min(stop,int(info['first_break']))
    start=max(0,segment.start,stop-max(2,round(.5*info['fps'])))
    report=dict(interval=[start,stop],object_id=segment.object_id)
    if start>=stop:return None,dict(report,reason='no_supported_pre_failure_interval')
    p=x[start:stop,:,segment.object_id]
    finite=(np.isfinite(p).all(-1)&(np.linalg.norm(p[...,3:7],axis=-1)>1e-8)
            &(np.linalg.norm(p[...,10:14],axis=-1)>1e-8))
    good=finite&support[start:stop]
    fraction=float(good.mean());report['supported_fraction']=fraction
    if fraction<minimum_support_fraction:return None,dict(report,reason='giver_support_unproven')
    values=p[good];delta=np.median(values[:,:3]-values[:,7:10],axis=0)
    rotations=Rotation.from_quat(values[:,3:7])*Rotation.from_quat(values[:,10:14]).inv()
    rotation=rotations.mean();rv=rotation.as_rotvec()
    report.update(reason='measured_giver_supported_anchor',samples=len(values),
        translation_m=delta.tolist(),rotation_rad=rv.tolist(),
        translation_spread_m=float(np.median(np.linalg.norm(values[:,:3]-values[:,7:10]-delta,axis=-1))),
        rotation_spread_rad=float(np.median((rotations*rotation.inv()).magnitude())))
    return (delta,rv),report


def measured_reference_tracking(actual, goals, world_groups, actual_sides, goal_sides, hand, interval):
    """Per-world palm tracking of the exact proposed reference, not task success.

    Actual poses are [executed frame,world,hand,7], goals are
    [reference group,reference frame,hand,7], both XYZW. The native rollout
    records the post-step state against reference row ``frame``; callers must
    provide that same alignment. Side names and world groups are independent
    mappings. Invalid interval samples invalidate the world instead of being
    dropped. No inference about finger tracking or grasp feasibility is made.
    """
    actual=np.asarray(actual,float);goals=np.asarray(goals,float)
    groups=np.asarray(world_groups)
    sides=list(actual_sides);reference_sides=list(goal_sides)
    if (actual.ndim!=4 or goals.ndim!=4 or actual.shape[-1]!=7 or goals.shape[-1]!=7
            or len(sides)!=actual.shape[2] or len(reference_sides)!=goals.shape[2]
            or len(set(sides))!=len(sides) or len(set(reference_sides))!=len(reference_sides)
            or hand not in sides or hand not in reference_sides
            or groups.shape!=(actual.shape[1],) or not np.issubdtype(groups.dtype,np.integer)
            or np.any(groups<0) or np.any(groups>=len(goals))):
        raise ValueError('named palms and valid per-world reference groups required')
    a,b=map(int,interval)
    if (a,b)!=tuple(interval) or not 0<=a<b<=min(len(actual),goals.shape[1]):
        raise ValueError('nonempty aligned tracking interval required')
    observed=actual[a:b,:,sides.index(hand)]
    desired=goals[groups,a:b,reference_sides.index(hand)].transpose(1,0,2)
    sample_valid=(np.isfinite(observed).all(-1)&np.isfinite(desired).all(-1)
        &(np.linalg.norm(observed[...,3:7],axis=-1)>1e-8)
        &(np.linalg.norm(desired[...,3:7],axis=-1)>1e-8))
    position=np.full(observed.shape[:2],np.inf);angle=position.copy()
    position[sample_valid]=np.linalg.norm(observed[sample_valid,:3]-desired[sample_valid,:3],axis=-1)
    angle[sample_valid]=(Rotation.from_quat(observed[sample_valid,3:7])*
        Rotation.from_quat(desired[sample_valid,3:7]).inv()).magnitude()
    valid=sample_valid.all(0)
    position[:,~valid]=0.;angle[:,~valid]=0.  # Avoid undefined quantiles of infinity.
    return dict(interval=[a,b],finite=valid,
        position_median_m=np.where(valid,np.median(position,axis=0),np.inf),
        orientation_median_rad=np.where(valid,np.median(angle,axis=0),np.inf),
        position_p90_m=np.where(valid,np.quantile(position,.9,axis=0),np.inf),
        orientation_p90_rad=np.where(valid,np.quantile(angle,.9,axis=0),np.inf))


def measured_transfer_statistics(poses,palms,contacts,segment,segments,fps):
    """Per-world ownership evidence after release, before intended recontact.

    Vertical load divided by weight is a diagnostic proxy, not a dynamic wrench
    certificate. Invalid stage worlds count as failures; later instability is
    reported separately so it does not erase an earlier established skill.
    """
    x=np.asarray(poses);h=np.asarray(palms);force=np.asarray(contacts['hand_object_force_w'])
    counts=np.asarray(contacts['hand_object_contact_count']);loads=np.asarray(contacts['hand_object_support_mg'])
    sides=list(contacts['physical_contact_sides']);worlds=x.shape[1]
    if (force.shape!=counts.shape+(3,) or loads.shape!=counts.shape or counts.shape[:2]!=x.shape[:2]
            or h.shape[:2]!=x.shape[:2] or not np.isfinite(fps) or fps<=0):
        raise ValueError('matching executed pose/contact frames and positive fps required')
    giver=find_giver(segment,segments,fps);event=giver.end if giver else segment.start
    a=max(segment.start, event)+max(1,round(.25*fps));b=min(len(x),segment.end+1)
    if giver is not None:
        b=min(b,min((s.start for s in segments if s.hand==giver.hand and s.object_id==segment.object_id
            and s.start>giver.end),default=b))
    global_finite=(np.isfinite(x).all((0,2,3))&np.isfinite(h).all((0,2,3))&np.isfinite(force).all((0,2,3,4)))
    if a>=b:raise ValueError('no loaded ownership interval before intended recontact')
    valid=(np.isfinite(x[a:b]).all((0,2,3))&np.isfinite(h[a:b]).all((0,2,3))&np.isfinite(force[a:b]).all((0,2,3,4)))
    receiver=sides.index(segment.hand);obj=segment.object_id
    receive=(counts[a:b,:,receiver,obj]>0).mean(0)
    give=(counts[a:b,:,sides.index(giver.hand),obj]>0).mean(0) if giver else np.zeros(worlds)
    load=loads[a:b,:,receiver,obj]
    valid&=np.isfinite(load).all(0)
    receive=np.where(valid,receive,0.);give=np.where(valid,give,1.)
    half_weight=np.where(valid,(load>.5).mean(0),0.)
    p,r=aligned_pose_errors(x)
    tracked=(np.mean(p[a:b,:,obj],axis=0)<.06)&(np.mean(r[a:b,:,obj],axis=0)<np.pi/4)
    clean=(receive>=.8)&(give<=.2)&valid
    proxy=clean&(half_weight>=.5)&tracked
    return dict(interval=[a,b],receiver_contact_fraction=receive,giver_contact_fraction=give,
        receiver_half_weight_fraction=half_weight,
        receiver_median_vertical_load_mg=np.where(valid,np.median(np.nan_to_num(load,nan=0.,posinf=0.,neginf=0.),axis=0),0.),
        clean_contact=clean,quasistatic_support_proxy=proxy,stage_finite=valid,full_rollout_finite=global_finite)


def calibrate_axis_scales(dim, error, *, tolerance=1e-4, backtracks=10):
    """Contract each signed axis until both probes execute without clipping.

    ``error(action)`` measures actual projection distortion. A channel that
    never becomes feasible is explicitly disabled instead of mislabeling the
    projected trajectory with an unexecuted action. Combined steps still need
    projection diagnostics and independent physical validation.
    """
    if dim<1 or backtracks<0 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('positive dimension/tolerance and nonnegative backtracks required')
    scales=np.zeros(dim);reports=[]
    for axis in range(dim):
        for trial in range(backtracks+1):
            scale=2.**(-trial);v=np.zeros(dim);v[axis]=scale
            distortion=np.asarray([error(v),error(-v)],float)
            if not np.isfinite(distortion).all() or (distortion<0).any():raise ValueError('invalid projection distortion')
            if distortion.max()<=tolerance:
                scales[axis]=scale;break
        reports.append(dict(axis=axis,scale=float(scales[axis]),enabled=bool(scales[axis]),
                            trials=trial+1,projection_error=distortion.tolist()))
    return scales,reports


def first_persistent(mask, frames):
    mask = np.asarray(mask, bool)
    if mask.ndim != 1 or frames < 1:
        raise ValueError('one-dimensional flags and positive duration required')
    if len(mask) < frames:
        return None
    ids = np.flatnonzero(np.convolve(mask.astype(int), np.ones(frames), 'valid') >= frames)
    return int(ids[0]) if len(ids) else None


def calibrate_axis_domain(dim, error, *, tolerance=1e-4, backtracks=10):
    """Identify signed feasible axes, retaining directions at joint limits.

    Scales map normalized coordinates to physical reference parameters. Bounds
    restrict each active coordinate to [-1,1], [0,1], or [-1,0]. Fully blocked
    coordinates have zero bounds and must be removed before identification.
    """
    if dim<1 or backtracks<0 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('positive dimension/tolerance and nonnegative backtracks required')
    scales=np.zeros(dim);lower=np.zeros(dim);upper=np.zeros(dim);reports=[]
    for axis in range(dim):
        signed=[]
        for sign in (-1,1):
            value=0.
            for trial in range(backtracks+1):
                v=np.zeros(dim);v[axis]=sign*2.**(-trial)
                distortion=float(error(v))
                if not np.isfinite(distortion) or distortion<0:raise ValueError('invalid projection distortion')
                if distortion<=tolerance:value=abs(v[axis]);break
            signed.append(dict(sign=sign,scale=value,trials=trial+1,projection_error=distortion))
        feasible=[s['scale'] for s in signed if s['scale']>0]
        if feasible:
            scales[axis]=min(feasible)
            lower[axis]=-float(signed[0]['scale']>0);upper[axis]=float(signed[1]['scale']>0)
            # Feasibility need not be monotone for retiming. Check the shared
            # physical scale explicitly rather than assuming contraction helps.
            while scales[axis]>0:
                checks=[]
                for sign in (-1,1):
                    if (sign<0 and lower[axis]<0) or (sign>0 and upper[axis]>0):
                        v=np.zeros(dim);v[axis]=sign*scales[axis];checks.append(float(error(v)))
                if not np.isfinite(checks).all() or min(checks)<0:raise ValueError('invalid projection distortion')
                if max(checks)<=tolerance:break
                scales[axis]*=.5
                if scales[axis]<2.**(-backtracks):scales[axis]=0.;lower[axis]=upper[axis]=0.
        reports.append(dict(axis=axis,scale=float(scales[axis]),lower=float(lower[axis]),upper=float(upper[axis]),signed=signed))
    return scales,lower,upper,reports


def select_causal_phase(poses, segments, fps, *, persistence_s=.25):
    """Locate a persistent break before optimizing borderline precision errors.

    A receiver around its giver's release owns the transfer failure, rather
    than attributing all later object error to the first acquisition segment.
    Strict task success thresholds are never changed by this diagnostic.
    """
    if not segments or not np.isfinite(fps) or fps <= 0:
        raise ValueError('contact segments and positive fps required')
    p, r = aligned_pose_errors(poses)
    p = np.median(np.nan_to_num(p, nan=1., posinf=1.), axis=1)
    r = np.median(np.nan_to_num(r, nan=np.pi, posinf=np.pi), axis=1)
    duration = max(1, round(persistence_s * fps))
    strict = first_persistent(((p > .03) | (r > np.pi/6)).any(1), duration)
    broken = first_persistent(((p > .06) | (r > np.pi/4)).any(1), duration)
    failure = broken if broken is not None else strict
    if failure is None:
        return None, dict(reason='no_persistent_pose_failure', first_strict_failure=None, first_break=None)
    objects = np.flatnonzero((p[failure] > (.06 if broken is not None else .03))
                             | (r[failure] > (np.pi/4 if broken is not None else np.pi/6)))
    transfers = []
    for segment in segments:
        giver = find_giver(segment, segments, fps)
        if (giver is not None and segment.object_id in objects
                and segment.start-round(.5*fps) <= failure <= giver.end+round(1.*fps)):
            transfers.append((abs(failure-giver.end), segment, giver))
    if transfers:
        _, chosen, giver = min(transfers, key=lambda x:x[0])
        event = giver.end
        mode = 'handover'
    else:
        eligible = [s for s in segments if s.object_id in objects
                    and s.start-round(.5*fps) <= failure <= s.end+round(.5*fps)]
        if not eligible:
            return None, dict(reason='failure_outside_contact_phases', first_strict_failure=strict, first_break=broken)
        chosen = min(eligible, key=lambda s:abs(s.start-failure))
        event, giver, mode = failure, None, 'tracking'
    return chosen, dict(reason='persistent_physical_break' if broken is not None else 'precision_failure',
        first_strict_failure=strict, first_break=broken, failure_frame=failure,
        event_frame=event, segment=chosen.key, giver=giver.hand if giver else None, mode=mode, fps=float(fps),
        score_window=[max(1,event-round(.25*fps)),min(len(poses),event+round(.75*fps))],
        edit_window=[max(1,chosen.start-round(1.*fps)),
                     min(len(poses)-1,max(event+round(1.*fps),chosen.end+round(.5*fps)))])


def reference_probe(base, hands, selected, segments, object_positions, object_quaternions, fps, info, action):
    """An object-relative edit that persists through the receiving contact.

    Timing changes stay local to their own closure/release event and restore
    before a later regrasp by that hand. Position/orientation offsets persist
    through carry instead of snapping back just after the scored transfer.
    Quaternions use XYZW. Caller applies joint/velocity trust-region projection.
    """
    hand=hands[selected.hand];giver=find_giver(selected,segments,fps) if info['mode']=='handover' else None
    expected=9 if giver is not None else 8
    v=np.asarray(action,float)
    if v.shape!=(expected,) or not np.isfinite(v).all():raise ValueError('finite action matching phase channels required')
    q=np.array(base,float,copy=True);obj=selected.object_id
    mask=window_mask(len(q),info['edit_window'],fps)
    op=np.asarray(object_positions,float)[:,obj]
    oq=np.asarray(object_quaternions,float)[:,obj]
    translation=object_frame_vectors(oq,.02*v[:3])
    if np.any(v[3:6]):
        rotation=Rotation.from_rotvec(mask[:,None]*object_frame_vectors(oq,.2*v[3:6]))
        q[:,hand.position]=op+rotation.apply((q[:,hand.position]-op).copy())
        q[:,hand.quaternion]=(rotation*Rotation.from_quat(q[:,hand.quaternion])).as_quat()
    if np.any(v[:3]):q[:,hand.position]+=mask[:,None]*translation
    if v[6]:q=squeeze_hand(q,hand,.3*v[6],mask)
    windows=[]
    events=[(7,selected.hand,selected.start)]+([(8,giver.hand,giver.end)] if giver else [])
    for index,side,event in events:
        if abs(v[index])<1e-12:continue
        seconds=.3*float(v[index]);shift=seconds*fps
        next_contact=min((s.start for s in segments if s.hand==side and s.start>event),default=len(q)-2)
        bounds=(max(info['edit_window'][0],event-round(.5*fps)),
                min(info['edit_window'][1],event+round((abs(seconds)+.75)*fps),next_contact))
        if not bounds[0]<event<bounds[1] or not bounds[0]<event+shift<bounds[1]:
            windows.append(dict(hand=side,event=event,applied=False));continue
        times=event_time_map(len(q),bounds,event,shift)
        local=window_mask(len(q),bounds,fps)
        q=retime_hand(q,hands[side],object_positions,object_quaternions,obj,times,local)
        windows.append(dict(hand=side,event=event,applied=True,bounds=list(bounds),shift_s=seconds))
    q[mask==0]=np.asarray(base)[mask==0]
    return q,dict(timing_windows=windows)


def aligned_signed_pose(poses):
    """Signed [position, rotation-vector] error with official first-pose alignment."""
    x = np.asarray(poses, float)
    if x.ndim != 4 or x.shape[-1] != 14:
        raise ValueError('expected [frame,world,object,14] poses')
    good = (np.isfinite(x).all(-1) & (np.linalg.norm(x[...,3:7],axis=-1)>1e-8)
            & (np.linalg.norm(x[...,10:14],axis=-1)>1e-8))
    safe = np.where(good[...,None],x,np.r_[np.zeros(3),[0,0,0,1],np.zeros(3),[0,0,0,1]])
    frames,worlds,objects = x.shape[:3]
    align = Rotation.from_quat(safe[0,:,0,10:14])*Rotation.from_quat(safe[0,:,0,3:7]).inv()
    offset = safe[0,:,0,7:10]-align.apply(safe[0,:,0,:3])
    rotations = Rotation.from_quat(np.broadcast_to(align.as_quat()[None,:,None],(frames,worlds,objects,4)).reshape(-1,4))
    delta_p = rotations.apply(safe[...,:3].reshape(-1,3)).reshape(frames,worlds,objects,3)+offset[None,:,None]-safe[...,7:10]
    delta_r = (Rotation.from_quat(safe[...,10:14].reshape(-1,4)).inv()*rotations
               *Rotation.from_quat(safe[...,3:7].reshape(-1,4))).as_rotvec().reshape(frames,worlds,objects,3)
    residual = np.concatenate([delta_p/.03,delta_r/(np.pi/6)],axis=-1)
    good &= good[:1,:,0:1]
    return np.where(good[...,None],residual,np.nan)


def binned_residual(values, start, end, bins=6):
    x = np.asarray(values,float)
    if x.ndim != 3 or not 0 <= start < end <= len(x) or bins < 1:
        raise ValueError('expected [frame,world,component] and nonempty bounds')
    indices = np.array_split(np.arange(start,end),min(bins,end-start))
    return np.concatenate([x[ids].mean(0) for ids in indices],axis=-1)


def physical_residuals(poses, palms, reference_palm, segment, info, hand_index, *, reach_weight=0., reach_mode='position',
                       fingertips=None,reference_fingertips=None,fingertip_weight=0.):
    """Object motion plus pre-release receiver position or pose in object axes.

    The reach goal comes from the chosen original grasp, held fixed across
    reference probes. Comparing each candidate to its own changed command would
    allow the loss to improve simply by moving the goalposts.
    Invalid/exploded worlds are explicitly excluded from model fitting; their
    rate and official failures must still be reported by the caller.
    """
    if reach_mode not in ('position','pose'):raise ValueError('unknown receiver reach mode')
    if not np.isfinite(reach_weight) or reach_weight<0:raise ValueError('nonnegative finite reach weight required')
    if not np.isfinite(fingertip_weight) or fingertip_weight<0:raise ValueError('nonnegative finite fingertip weight required')
    x = np.asarray(poses,float); h = np.asarray(palms,float)[:,:,hand_index]
    plan = np.asarray(reference_palm,float)[:len(x)]
    a,b = info['score_window']; obj = segment.object_id
    signed = aligned_signed_pose(x)
    task = binned_residual(signed[:,:,obj],a,b)
    result = task
    if reach_weight or fingertip_weight:
        lo = max(0,segment.start-round(.3*info['fps'])); hi = min(len(x),info['event_frame']+1)
        if hi <= lo: raise ValueError('receiver reach window is empty')
        actual = x[lo:hi,:,obj,:7]; target = x[lo:hi,:,obj,7:14]
        finite = (np.isfinite(actual).all(-1)&np.isfinite(target).all(-1)&np.isfinite(h[lo:hi]).all(-1)
                  & (np.linalg.norm(actual[...,3:],axis=-1)>1e-8)
                  & (np.linalg.norm(target[...,3:],axis=-1)>1e-8))
        actual = np.where(finite[...,None],actual,[0,0,0,0,0,0,1])
        target = np.where(finite[...,None],target,[0,0,0,0,0,0,1])
        ro = Rotation.from_quat(actual[...,3:].reshape(-1,4))
        rt = Rotation.from_quat(target[...,3:].reshape(-1,4))
        shape = actual.shape[:2]+(3,)
        achieved = ro.inv().apply((h[lo:hi,:,:3]-actual[...,:3]).reshape(-1,3)).reshape(shape)
        wanted = rt.inv().apply((plan[lo:hi,None,:3]-target[...,:3]).reshape(-1,3)).reshape(shape)
        reach = np.where(finite[...,None],(achieved-wanted)/.03,np.nan)
        if reach_mode=='pose':
            valid_q=(np.linalg.norm(h[lo:hi,:,3:7],axis=-1)>1e-8)
            valid_plan=(np.isfinite(plan[lo:hi]).all(-1)&(np.linalg.norm(plan[lo:hi,3:7],axis=-1)>1e-8))
            valid_q &= finite & valid_plan[:,None]
            hand_q=np.where(valid_q[...,None],h[lo:hi,:,3:7],[0,0,0,1])
            goal_q=np.where(valid_plan[:,None],plan[lo:hi,3:7],[0,0,0,1])
            goal_q=np.broadcast_to(goal_q[:,None],hand_q.shape).copy()
            achieved_rotation=ro.inv()*Rotation.from_quat(hand_q.reshape(-1,4))
            wanted_rotation=rt.inv()*Rotation.from_quat(goal_q.reshape(-1,4))
            orientation=(wanted_rotation.inv()*achieved_rotation).as_rotvec().reshape(shape)/(np.pi/6)
            reach=np.concatenate([reach,np.where(valid_q[...,None],orientation,np.nan)],axis=-1)
        if reach_weight:result = np.concatenate([task,np.sqrt(reach_weight)*binned_residual(reach,0,len(reach),bins=4)],axis=-1)
        if fingertip_weight:
            tips=np.asarray(fingertips,float);goals=np.asarray(reference_fingertips,float)[:len(x)]
            if (tips.ndim!=5 or tips.shape[:2]!=x.shape[:2] or tips.shape[-1]!=3 or tips.shape[3]<1
                    or goals.shape!=(len(plan),tips.shape[3],3)):
                raise ValueError('matching live [frame,world,hand,finger,3] and fixed [frame,finger,3] tips required')
            tips=tips[lo:hi,:,hand_index];goals=goals[lo:hi];nf=tips.shape[2]
            valid_tips=finite & np.isfinite(tips).all((2,3)) & np.isfinite(goals).all((1,2))[:,None]
            local_actual=Rotation.from_quat(np.repeat(actual[...,3:].reshape(-1,4),nf,axis=0))
            local_target=Rotation.from_quat(np.repeat(target[...,3:].reshape(-1,4),nf,axis=0))
            achieved_tips=local_actual.inv().apply((tips-actual[:,:,None,:3]).reshape(-1,3))
            wanted_tips=local_target.inv().apply((goals[:,None]-target[:,:,None,:3]).reshape(-1,3))
            tip_residual=(achieved_tips-wanted_tips).reshape(len(tips),tips.shape[1],nf*3)/.03
            tip_residual=np.where(valid_tips[...,None],tip_residual,np.nan)
            # Average over fingers so adding more landmarks does not silently
            # change this objective's relative weight. Goals never follow edits.
            result=np.concatenate([result,np.sqrt(fingertip_weight/nf)*binned_residual(tip_residual,0,len(tips),bins=4)],axis=-1)
    stable = np.isfinite(x).all((0,2,3)) & np.isfinite(result).all(1)
    position,_ = aligned_pose_errors(x)
    stable &= (position[a:b] < 1.).all((0,2))
    return result,stable


def moments(residuals):
    """Mean and population spread retain errors that cancel across worlds."""
    x = np.asarray(residuals,float)
    if x.ndim != 2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('at least two finite physical residual vectors required')
    return np.r_[x.mean(0),x.std(0)]


@dataclass(frozen=True)
class ResponseModel:
    origin: np.ndarray
    jacobian: np.ndarray
    condition: float

    def predict(self, action):
        return self.origin+np.asarray(action)@self.jacobian.T

    def step(self, radius=.5, damping=.03, *, lower=None, upper=None):
        if radius<=0 or damping<=0: raise ValueError('positive trust radius and damping required')
        j=self.jacobian; gram=j.T@j
        scale=max(float(np.trace(gram)/len(gram)),1e-9)
        if lower is None and upper is None:
            delta=np.linalg.solve(gram+damping*scale*np.eye(len(gram)),-j.T@self.origin)
        else:
            from scipy.optimize import lsq_linear
            lo=np.asarray(lower,float);hi=np.asarray(upper,float)
            if (lo.shape!=(j.shape[1],) or hi.shape!=lo.shape or np.isnan(lo).any() or np.isnan(hi).any()
                    or (lo>0).any() or (hi<0).any() or (lo>hi).any()):
                raise ValueError('step bounds must match channels and include zero')
            free=hi>lo;delta=np.zeros(j.shape[1])
            if free.any():
                a=np.vstack([j[:,free],np.sqrt(damping*scale)*np.eye(free.sum())])
                b=np.r_[-self.origin,np.zeros(free.sum())]
                solution=lsq_linear(a,b,bounds=(lo[free],hi[free]),tol=1e-10)
                if not solution.success:raise ValueError('bounded response step did not converge')
                delta[free]=solution.x
        delta*=min(1.,radius/max(np.linalg.norm(delta),1e-12))
        return delta


def fit_response(actions, groups, baseline, *, ridge=.03):
    a=np.asarray(actions,float); y=np.asarray([moments(group) for group in groups]); origin=moments(baseline)
    if a.ndim!=2 or len(a)!=len(y) or a.shape[1]<1 or not np.isfinite(a).all() or ridge<=0:
        raise ValueError('finite actions and matched residual groups required')
    gram=a.T@a
    if np.linalg.matrix_rank(a)<a.shape[1]:raise ValueError('probe actions do not identify all channels')
    regularizer=ridge*max(float(np.trace(gram)/len(gram)),1e-9)
    j=np.linalg.solve(gram+regularizer*np.eye(len(gram)),a.T@(y-origin)).T
    return ResponseModel(origin,j,float(np.linalg.cond(gram+regularizer*np.eye(len(gram)))))


def prediction_quality(model, actions, groups):
    prediction=np.sum(model.predict(actions)**2,axis=-1)
    measured=np.asarray([np.mean(np.sum(np.asarray(g)**2,axis=1)) for g in groups])
    correlation=spearmanr(prediction,measured).statistic if np.ptp(prediction)>1e-12 and np.ptp(measured)>1e-12 else None
    baseline=float(np.sum(model.origin**2))
    predicted_reduction=baseline-prediction;measured_reduction=baseline-measured
    predicted_improving=predicted_reduction>1e-9
    # Ranking can look good even if every proposed improvement is biased.
    # These are descriptive held-out checks, not acceptance or noise-adjusted
    # confidence tests; the final direction still needs independent rollouts.
    centered=np.sum((measured-measured.mean())**2)
    return dict(spearman=float(correlation) if correlation is not None and np.isfinite(correlation) else None,
        predicted_cost=prediction.tolist(),measured_cost=measured.tolist(),
        predicted_reduction=predicted_reduction.tolist(),measured_reduction=measured_reduction.tolist(),
        improvement_sign_agreement=float(((predicted_reduction>0)==(measured_reduction>0)).mean()),
        predicted_improvement_precision=float((measured_reduction[predicted_improving]>0).mean()) if predicted_improving.any() else None,
        predicted_improvement_count=int(predicted_improving.sum()),
        mean_cost_bias=float((prediction-measured).mean()),
        cost_rmse=float(np.sqrt(np.mean((prediction-measured)**2))),
        cost_r_squared=float(1-np.sum((prediction-measured)**2)/centered) if centered>1e-12 else None)
