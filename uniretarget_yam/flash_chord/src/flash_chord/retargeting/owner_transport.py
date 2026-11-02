"""Discover a physically observed single-owner alternative to a handover."""
import numpy as np

from .coupled_transport import smooth_task_correction,adaptive_position_correction,bounded_rotation_correction


def owner_interval_evidence(counts,support_mg,external_mg,invalid,hand,interval,*,minimum_fraction=.8):
    """Recheck a fixed owner interval; unsupported gaps remain in its denominator."""
    c=np.asarray(counts);load=np.asarray(support_mg,float);ext=np.asarray(external_mg,float);bad=np.asarray(invalid)
    a,b=interval
    if (c.ndim!=3 or load.shape!=c.shape or ext.shape!=c.shape[:2] or bad.shape!=ext.shape
            or not isinstance(hand,(int,np.integer)) or not 0<=hand<c.shape[-1]
            or not all(isinstance(i,(int,np.integer)) for i in (a,b)) or not 0<=a<b<=len(c)
            or not 0<minimum_fraction<=1):
        raise ValueError('aligned physical feedback and a fixed owner interval required')
    supported=(np.isfinite(load[:,:,hand])&np.isfinite(ext)&(bad==0)&(c[:,:,hand]>0)
               &((c>0).sum(-1)==1)&(load[:,:,hand]>.5)&(ext<.1))
    fractions=supported[a:b].mean(0)
    return dict(hand_index=int(hand),interval=[int(a),int(b)],world_support_fractions=fractions.tolist(),
                eligible_world_indices=np.flatnonzero(fractions>=minimum_fraction).tolist())


def dominant_owner_interval(counts, support_mg, external_mg, invalid, fps, *, population=.8, minimum_seconds=1., gap_seconds=.25):
    """Find the longest majority-supported free-holding interval, without task scores.

    Short gaps join intervals only for discovery. Their unsupported samples
    still count against each world's eligibility and are never erased.
    """
    c=np.asarray(counts);load=np.asarray(support_mg,float);ext=np.asarray(external_mg,float);bad=np.asarray(invalid)
    if (c.ndim!=3 or load.shape!=c.shape or ext.shape!=c.shape[:2] or bad.shape!=ext.shape
            or not np.isfinite(fps) or fps<=0 or not 0<population<=1
            or minimum_seconds<=0 or gap_seconds<0):
        raise ValueError('aligned [frame,world,hand] contacts/load and external support required')
    valid=np.isfinite(load)&np.isfinite(ext[...,None])&(bad[...,None]==0)
    active=c>0;alone=active.sum(-1)==1
    supported=valid&active&alone[...,None]&(load>.5)&(ext[...,None]<.1)
    choices=[];bridge=int(round(gap_seconds*fps));minimum=max(1,int(np.ceil(minimum_seconds*fps)))
    for hand in range(c.shape[-1]):
        mask=supported[:,:,hand].mean(1)>=population
        for group in np.split(np.flatnonzero(~mask),np.flatnonzero(np.diff(np.flatnonzero(~mask))>1)+1):
            if len(group) and len(group)<=bridge and group[0]>0 and group[-1]<len(mask)-1:
                mask[group]=True
        for group in np.split(np.flatnonzero(mask),np.flatnonzero(np.diff(np.flatnonzero(mask))>1)+1):
            if len(group)<minimum:continue
            a,b=int(group[0]),int(group[-1]+1);fractions=supported[a:b,:,hand].mean(0)
            eligible=np.flatnonzero(fractions>=population)
            if len(eligible):choices.append(dict(hand_index=hand,interval=[a,b],world_support_fractions=fractions.tolist(),eligible_world_indices=eligible.tolist()))
    if not choices:return None
    return min(choices,key=lambda item:(-(item['interval'][1]-item['interval'][0]),item['hand_index'],item['interval'][0]))


def held_correction_profile(source, desired, interval, fps, *, rotation=True,adaptive=False,position=True,chordal=False):
    """Fit only the observed held interval, fading corrections outside it."""
    source=np.asarray(source);desired=np.asarray(desired);a,b=interval
    if source.shape!=desired.shape or source.ndim!=2 or source.shape[1]!=7 or not 1<=a<b<=len(source)-1 or b-a<3 or not np.isfinite(fps) or fps<=0:
        raise ValueError('protected task paths and a nontrivial observed holding interval required')
    fit_target=desired[a:b].copy()
    if not rotation:fit_target[:,3:]=source[a:b,3:]
    if not position and not rotation:raise ValueError('at least one physical correction component is required')
    if adaptive and rotation and not chordal:raise ValueError('adaptive position basis cannot silently modify orientation')
    if chordal and rotation:
        if position:
            position_target=fit_target.copy();position_target[:,3:]=source[a:b,3:]
            fit,metadata=(adaptive_position_correction(source[a:b],position_target,fps=fps) if adaptive else
                          smooth_task_correction(source[a:b],position_target,knot_count=min(8,b-a)))
        else:
            fit=np.zeros((b-a,6));metadata=dict(method='rotation_only',position_residual_rms_m=float(np.sqrt(np.mean((source[a:b,:3]-desired[a:b,:3])**2))))
        rot,rotation_fit=bounded_rotation_correction(source[a:b],fit_target,knot_count=min(16,b-a))
        fit[:,3:]=rot[:,3:];metadata['rotation_fit']=rotation_fit
    else:
        if not position:fit_target[:,:3]=source[a:b,:3]
        fit,metadata=(adaptive_position_correction(source[a:b],fit_target,fps=fps) if adaptive else
                      smooth_task_correction(source[a:b],fit_target,knot_count=min(8,b-a)))
    result=np.zeros((len(source),6));result[a:b]=fit
    smooth=lambda x:x*x*x*(10+x*(-15+6*x))
    start=max(0,a-max(1,round(.5*fps)))
    result[start:a]=smooth(np.arange(a-start)/(a-start))[:,None]*fit[0]
    result[b:]=(1-smooth(np.arange(1,len(source)-b+1)/(len(source)-b)))[:,None]*fit[-1]
    result[0]=0.;result[-1]=0.
    return result,dict(metadata,fit_interval=[a,b],approach_start=start,components=('full_pose' if position else 'rotation_only') if rotation else 'position_only',tail='fade without fitting post-hold observations')
