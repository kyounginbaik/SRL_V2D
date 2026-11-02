"""Bounded temporal acquisition directions from measured object poses.

This is a native-tracking-preconditioned residual direction, not an identified
closed-loop Jacobian. Opposite/random physical controls must test its utility.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.acquisition_reference import _residuals


SCALES=np.array([.04,.04,.04,.35,.35,.35])


def acquisition_parent_indices(candidates):
    """Advance declared descent branches, never select a lucky random arm."""
    return [i for i,c in enumerate(candidates)
        if (c.get('operation')=='seated_closure_with_release' and c.get('alignment_gain')==0.)
        or (c.get('operation')=='temporal_acquisition_direction' and c.get('temporal_case')=='predicted_descent')]


def validated_response_weights(report):
    """Require held-out absolute and baseline-recentered response checks."""
    for key in ('held_out','baseline_recentered_held_out'):
        quality=report.get(key,{})
        if (quality.get('cost_r_squared') is None or quality['cost_r_squared']<.3
                or quality.get('improvement_sign_agreement',0)<.8
                or quality.get('predicted_improvement_precision') is None
                or quality['predicted_improvement_precision']<.8
                or quality.get('predicted_improvement_count',0)<2):
            return None
    weights=np.asarray(report['proposed_subspace_step'],float)
    if weights.shape!=(3,) or not np.isfinite(weights).all() or np.linalg.norm(weights)>1.+1e-6:
        raise ValueError('three bounded finite response weights required')
    return weights


def repeated_physical_centers(candidates,result,report,*,minimum_improvement=.1):
    """Select diagnostic centers from repeated physical joint-loss evidence.

    This is an empirical fallback when a response model fails calibration.
    It is not canonical acceptance; fresh old/new controls remain necessary.
    """
    if not 0<minimum_improvement<1:raise ValueError('a positive fractional improvement below one is required')
    choices=[]
    for model in report['models']:
        parent=model['parent'];members={c['temporal_case']:i for i,c in enumerate(candidates)
            if c.get('temporal_source_trial')==parent}
        if 'unchanged' not in members:continue
        baseline=members['unchanged'];best=[]
        for case,index in members.items():
            if case=='unchanged':continue
            ratios=[];safe=True
            for repeat,key in [(0,'discovery_joint_cost'),(1,'held_joint_cost')]:
                costs=model.get(key,{})
                if case not in costs or 'unchanged' not in costs:safe=False;break
                before,after=costs['unchanged'],costs[case]
                if not np.isfinite([before,after]).all() or before<=0 or after<0:safe=False;break
                ratio=after/before
                if ratio>1-minimum_improvement:safe=False;break
                old=next((r for r in result['trials'] if r['trial']==baseline and r['repeat']==repeat),None)
                new=next((r for r in result['trials'] if r['trial']==index and r['repeat']==repeat),None)
                cov=next((r for r in report.get('coverage',[]) if r['trial']==index and r['repeat']==repeat),None)
                if old is None or new is None or cov is None:safe=False;break
                if (new['mp_sr']<old['mp_sr'] or new['native_finite_fraction']<old['native_finite_fraction']
                        or new['clean_contact']<old['clean_contact']-.02
                        or cov['prefix_eligible']<4 or cov['used_for_fit']<.9*cov['prefix_eligible']):
                    safe=False;break
                ratios.append(float(ratio))
            if safe:best.append((max(ratios),index,case,ratios))
        if best:
            worst_ratio,index,case,ratios=min(best)
            choices.append(dict(trial=index,previous_center=baseline,case=case,parent=parent,
                worst_relative_joint_cost=worst_ratio,repeat_relative_joint_costs=ratios))
    return choices


@dataclass(frozen=True)
class AcquisitionDirection:
    knots: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        k=np.array(self.knots,float,copy=True);c=np.array(self.coefficients,float,copy=True)
        if k.ndim!=1 or len(k)<3 or c.shape!=(len(k),6) or not np.isfinite(k).all() or not np.isfinite(c).all() or np.any(np.diff(k)<=0):
            raise ValueError('ordered temporal knots and finite six-dimensional coefficients required')
        k.flags.writeable=False;c.flags.writeable=False;object.__setattr__(self,'knots',k);object.__setattr__(self,'coefficients',c)

    def values(self,frames):
        return CubicSpline(self.knots,self.coefficients,bc_type='clamped')(np.clip(frames,self.knots[0],self.knots[-1]))


def fit_acquisition_direction(planned,objects,grasp,mask,*,first_frame,knot_count=4):
    planned,objects,mask=np.asarray(planned,float),np.asarray(objects,float),np.asarray(mask)
    dp,dr,report=_residuals(planned,objects,grasp,mask)
    if knot_count<3 or len(planned)<knot_count+2 or report['used_samples']<4*knot_count or report['coverage']<.25:
        return None,dict(report,reason='insufficient_temporal_acquisition_feedback')
    finite=np.isfinite(objects).all(-1)&np.isfinite(planned).all(-1)[:,None]
    finite&=(np.linalg.norm(objects[...,3:],axis=-1)>1e-8)&(np.linalg.norm(planned[...,3:],axis=-1)>1e-8)[:,None]
    good=np.asarray(mask)&finite
    times=np.broadcast_to(np.arange(len(planned))[:,None]+first_frame,good.shape)[good]
    # A local principal chart cannot resolve an almost antipodal orientation.
    chart=dr.magnitude()<2.8;report['principal_branch_excluded']=int((~chart).sum())
    times=times[chart];y=np.c_[dp[chart],dr.as_rotvec()[chart]]/SCALES
    knots=np.linspace(first_frame,first_frame+len(planned)-1,knot_count)
    design=CubicSpline(knots,np.eye(knot_count),bc_type='clamped')(times)
    if len(times)<4*knot_count or np.linalg.matrix_rank(design)<knot_count:
        return None,dict(report,reason='temporal_direction_not_identifiable')
    prior=np.r_[.05*np.eye(knot_count),.15*np.diff(np.eye(knot_count),n=2,axis=0)]
    weights=np.ones(len(y));coefficients=None
    for _ in range(5):
        scale=np.sqrt(weights/weights.sum())[:,None]
        coefficients=np.linalg.lstsq(np.r_[scale*design,prior],np.r_[scale*y,np.zeros((len(prior),6))],rcond=None)[0]
        errors=np.linalg.norm(design@coefficients-y,axis=1)
        weights=np.minimum(1.,2.5/np.maximum(errors,1e-12))
    coefficients*=SCALES
    raw=AcquisitionDirection(knots,coefficients);dense=raw.values(np.linspace(knots[0],knots[-1],max(100,len(planned)*4)))
    contraction=max(1.,float(np.linalg.norm(dense[:,:3],axis=1).max()/.04),float(np.linalg.norm(dense[:,3:],axis=1).max()/.35))
    model=AcquisitionDirection(knots,coefficients/contraction)
    return model,dict(report,reason='bounded_temporal_object_relative_residual_direction',knots=knots.tolist(),
        contraction=contraction,temporal_condition=float(np.linalg.cond(design)),physical_direction_verified=False)


def validate_acquisition_direction(model,planned,objects,grasp,mask,*,first_frame):
    planned,objects,mask=np.asarray(planned,float),np.asarray(objects,float),np.asarray(mask)
    dp,dr,report=_residuals(planned,objects,grasp,mask)
    if not report['used_samples']:return dict(report,improved=False,reason='no_validation_samples')
    finite=np.isfinite(objects).all(-1)&np.isfinite(planned).all(-1)[:,None]
    finite&=(np.linalg.norm(objects[...,3:],axis=-1)>1e-8)&(np.linalg.norm(planned[...,3:],axis=-1)>1e-8)[:,None]
    times=np.broadcast_to(np.arange(len(planned))[:,None]+first_frame,mask.shape)[mask&finite]
    delta=model.values(times)
    before=np.linalg.norm(dp,axis=1)/.03+dr.magnitude()/np.deg2rad(30)
    after=np.linalg.norm(dp-delta[:,:3],axis=1)/.03+(dr*Rotation.from_rotvec(delta[:,3:]).inv()).magnitude()/np.deg2rad(30)
    return dict(report,conditional_mean_before=float(before.mean()),conditional_mean_after=float(after.mean()),
        conditional_median_before=float(np.median(before)),conditional_median_after=float(np.median(after)),
        improved=bool(report['coverage']>=.25 and after.mean()<.95*before.mean()))


def direction_profile(model,frames,*,bounds,hold_until,restore):
    a,b=bounds
    if not 0<=a<model.knots[0]<model.knots[-1]<=hold_until<restore<=b<frames:
        raise ValueError('protected approach, fitted interval and return timing required')
    t=np.arange(frames);u=np.clip((t-a)/(model.knots[0]-a),0,1);v=np.clip((restore-t)/(restore-hold_until),0,1)
    smooth=lambda x:x*x*x*(10+x*(-15+6*x))
    return model.values(t)*(smooth(u)*smooth(v))[:,None]


def apply_direction(wrist,profile,*,strength=1.):
    base=np.asarray(wrist,float);delta=np.asarray(profile,float)
    if base.ndim!=2 or base.shape[1]!=7 or delta.shape!=(len(base),6) or not np.isfinite(delta).all() or not np.isfinite(base).all() or not np.isfinite(strength):
        raise ValueError('matching finite geometric goals and temporal directions required')
    out=base.copy();out[:,:3]+=strength*delta[:,:3]
    out[:,3:]=(Rotation.from_rotvec(strength*delta[:,3:])*Rotation.from_quat(base[:,3:])).as_quat()
    unchanged=np.all(delta==0,axis=1);out[unchanged]=base[unchanged]
    return out


def matched_random_direction(model,frames,*,seed,settings,fps=None):
    rng=np.random.default_rng(seed);target=direction_profile(model,frames,**settings)
    target_norm=np.linalg.norm(target/SCALES)
    if target_norm<1e-8:raise ValueError('cannot compare random controls with a zero direction')
    for _ in range(1000):
        random=AcquisitionDirection(model.knots,rng.normal(size=model.coefficients.shape)*SCALES)
        profile=direction_profile(random,frames,**settings);ratio=target_norm/np.linalg.norm(profile/SCALES)
        values=random.values(np.linspace(model.knots[0],model.knots[-1],200))*ratio
        if fps is not None:
            rates=np.linalg.norm(np.diff(profile*ratio,axis=0).reshape(-1,2,3),axis=-1).max(0)*fps
            if rates[0]>.3+1e-8 or rates[1]>2.+1e-8:continue
        if np.linalg.norm(values[:,:3],axis=1).max()<=.04+1e-10 and np.linalg.norm(values[:,3:],axis=1).max()<=.35+1e-10:
            return AcquisitionDirection(model.knots,random.coefficients*ratio)
    raise ValueError('no equal-trajectory-norm random control satisfies the same trust bounds')


def compose_response_direction(basis,weights,frames,*,settings,fps):
    weights=np.asarray(weights,float)
    if len(basis)!=3 or weights.shape!=(3,) or not np.isfinite(weights).all() or fps<=0:
        raise ValueError('three finite response-basis weights and positive frame rate required')
    if any(not np.array_equal(model.knots,basis[0].knots) for model in basis):
        raise ValueError('response directions must share the same temporal chart')
    model=AcquisitionDirection(basis[0].knots,sum(w*d.coefficients for w,d in zip(weights,basis)))
    dense=model.values(np.linspace(model.knots[0],model.knots[-1],200));profile=direction_profile(model,frames,**settings)
    rates=np.linalg.norm(np.diff(profile,axis=0).reshape(-1,2,3),axis=-1).max(0)*fps
    factor=max(1.,float(np.linalg.norm(dense[:,:3],axis=1).max()/.04),
        float(np.linalg.norm(dense[:,3:],axis=1).max()/.35),float(rates[0]/.3),float(rates[1]/2.))
    return AcquisitionDirection(model.knots,model.coefficients/factor),factor


def observed_acquisition_cost(objects,palms,grasp):
    """All-world actual object-relative residual cost, never task success.

    Any invalid phase sample makes that world's diagnostic cost 100. Large
    errors are capped only in this local search diagnostic; official metrics
    must still evaluate their original, uncapped full-episode errors.
    """
    objects,palms=np.asarray(objects,float),np.asarray(palms,float)
    if objects.ndim!=3 or objects.shape[-1]!=7 or palms.shape!=objects.shape or not len(objects):
        raise ValueError('matching actual time/world XYZ/XYZW poses required')
    finite=np.isfinite(objects).all(-1)&np.isfinite(palms).all(-1)
    finite&=(np.linalg.norm(objects[...,3:],axis=-1)>1e-8)&(np.linalg.norm(palms[...,3:],axis=-1)>1e-8)
    o=np.where(finite[...,None],objects,[0,0,0,0,0,0,1]);h=np.where(finite[...,None],palms,[0,0,0,0,0,0,1])
    ro=Rotation.from_quat(o[...,3:].reshape(-1,4));rh=Rotation.from_quat(h[...,3:].reshape(-1,4))
    wanted=o[...,:3]+ro.apply(np.broadcast_to(grasp.wrist_position_o,o[...,:3].shape).reshape(-1,3).copy()).reshape(o[...,:3].shape)
    dp=np.linalg.norm(wanted-h[...,:3],axis=-1)/.03
    dr=((ro*Rotation.from_quat(grasp.wrist_quaternion_o))*rh.inv()).magnitude().reshape(finite.shape)/np.deg2rad(30)
    valid=finite.all(0)
    cost=np.minimum(dp+dr,100.).mean(0);cost[~valid]=100.
    return cost,valid
