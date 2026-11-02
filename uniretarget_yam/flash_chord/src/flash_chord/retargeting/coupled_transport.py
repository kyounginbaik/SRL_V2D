"""Transport all demonstrated hand poses and spring targets with one object.

The actual object path and actual hand/control paths come from physical
feedback. A shared rigid transform preserves time-varying inter-hand geometry
and wrist preload; it does not certify dynamics under changed task motion.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def transport_poses(source_object, control_object, poses):
    source=np.asarray(source_object,float);control=np.asarray(control_object,float);value=np.asarray(poses,float)
    if (source.ndim!=2 or source.shape[1]!=7 or control.shape!=source.shape
            or value.ndim!=3 or value.shape[0]!=len(source) or value.shape[-1]!=7
            or not all(np.isfinite(x).all() for x in [source,control,value])
            or any(np.any(np.linalg.norm(x[...,3:],axis=-1)<1e-8) for x in [source,control,value])):
        raise ValueError('matching finite object paths and named hand/control poses required')
    if np.array_equal(source,control):return value.copy()
    delta=Rotation.from_quat(control[:,3:].copy())*Rotation.from_quat(source[:,3:].copy()).inv()
    p=np.empty_like(value)
    for h in range(value.shape[1]):
        p[:,h,:3]=control[:,:3]+delta.apply(value[:,h,:3]-source[:,:3])
        p[:,h,3:]=(delta*Rotation.from_quat(value[:,h,3:].copy())).as_quat()
    return p


def smooth_task_correction(source_object, desired_object, *, knot_count=8, regularization=.1):
    """Fit low-frequency signed object error, not contact-jitter/falling tails."""
    source=np.asarray(source_object,float);desired=np.asarray(desired_object,float)
    if (source.ndim!=2 or source.shape[1]!=7 or desired.shape!=source.shape
            or not np.isfinite(source).all() or not np.isfinite(desired).all()
            or not 3<=knot_count<=len(source) or regularization<0 or not np.isfinite(regularization)):
        raise ValueError('finite object paths and valid temporal fit required')
    rotation=(Rotation.from_quat(desired[:,3:].copy())*Rotation.from_quat(source[:,3:].copy()).inv()).as_rotvec()
    if np.any(np.linalg.norm(rotation,axis=1)>2.8):raise ValueError('object correction outside local rotation chart')
    values=np.c_[desired[:,:3]-source[:,:3],rotation]
    knots=np.linspace(0,len(source)-1,knot_count);time=np.arange(len(source))
    design=CubicSpline(knots,np.eye(knot_count),bc_type='clamped')(time)
    penalty=np.diff(np.eye(knot_count),n=2,axis=0)
    coefficients=np.linalg.lstsq(np.r_[design,np.sqrt(regularization)*penalty],
        np.r_[values,np.zeros((len(penalty),6))],rcond=None)[0]
    return design@coefficients,dict(knots=knots.tolist(),coefficients=coefficients.tolist(),
        regularization=regularization,position_residual_rms_m=float(np.sqrt(np.mean((design@coefficients-values)[:,:3]**2))),
        rotation_residual_rms_rad=float(np.sqrt(np.mean((design@coefficients-values)[:,3:]**2))))


def correction_control(source_object, correction, *, gain, ramp_frames, random_seed=None):
    source=np.asarray(source_object,float);profile=np.array(correction,float,copy=True)
    if (profile.shape!=(len(source),6) or not np.isfinite(profile).all() or not np.isfinite(gain)
            or not isinstance(ramp_frames,(int,np.integer)) or not 1<=ramp_frames<len(source)):
        raise ValueError('finite correction and protected initial ramp required')
    if gain==0:return source.copy()
    if random_seed is not None:
        rng=np.random.default_rng(random_seed)
        profile[:,:3]=Rotation.random(random_state=rng).apply(profile[:,:3])
        profile[:,3:]=Rotation.random(random_state=rng).apply(profile[:,3:])
    u=np.clip(np.arange(len(source))/ramp_frames,0,1);weight=gain*u*u*u*(10+u*(-15+6*u))
    control=source.copy();control[:,:3]+=weight[:,None]*profile[:,:3]
    control[:,3:]=(Rotation.from_rotvec(weight[:,None]*profile[:,3:])*Rotation.from_quat(source[:,3:].copy())).as_quat()
    # Quaternion normalization must not perturb rows with no requested edit.
    # This also preserves endpoint rows of an explicitly faded correction.
    unchanged=(weight==0)|np.all(profile==0,axis=1)
    control[unchanged]=source[unchanged]
    return control


def adaptive_position_correction(source_object,desired_object,*,fps,target_rms_m=.005,max_knots=32,min_spacing_s=.25,regularization=.1):
    """Resolve persistent physical position error with a bounded spline basis.

    Rotation is deliberately unchanged. Added knots follow smoothed fit
    residuals; this is a proposal representation, not physical acceptance.
    """
    source=np.asarray(source_object,float);desired=np.asarray(desired_object,float)
    if (source.ndim!=2 or source.shape[1]!=7 or desired.shape!=source.shape or len(source)<8
            or not np.isfinite(source).all() or not np.isfinite(desired).all()
            or not np.isfinite([fps,target_rms_m,min_spacing_s,regularization]).all()
            or min(fps,target_rms_m,min_spacing_s)<=0 or regularization<0
            or not isinstance(max_knots,(int,np.integer)) or max_knots<8):
        raise ValueError('finite task paths and bounded positive adaptive spline settings required')
    values=desired[:,:3]-source[:,:3];clock=np.arange(len(source));spacing=max(1,int(np.ceil(min_spacing_s*fps)))
    knots=np.linspace(0,len(source)-1,8);history=[]
    def fit(nodes):
        spline=CubicSpline(nodes,np.eye(len(nodes)),bc_type='clamped');design=spline(clock)
        # Scale curvature by mean knot spacing to keep the regularizer useful
        # when knots are nonuniform, instead of treating all intervals alike.
        penalty=spline(clock,2)*((len(source)-1)/(len(nodes)-1))**2/np.sqrt(len(source))
        coefficients=np.linalg.lstsq(np.r_[design,np.sqrt(regularization)*penalty],np.r_[values,np.zeros((len(penalty),3))],rcond=None)[0]
        prediction=design@coefficients
        return prediction,coefficients,float(np.sqrt(np.mean((prediction-values)**2)))
    prediction,coefficients,rms=fit(knots)
    while True:
        history.append(dict(knots=len(knots),position_residual_rms_m=rms))
        if rms<=target_rms_m or len(knots)>=min(max_knots,len(source)):break
        residual=np.linalg.norm(prediction-values,axis=-1)
        width=min(len(source),spacing);kernel=np.ones(width)
        persistent=np.convolve(residual,kernel,'same')/np.convolve(np.ones(len(source)),kernel,'same')
        eligible=np.min(np.abs(clock[:,None]-knots[None]),axis=1)>=spacing
        if not eligible.any():break
        score=np.where(eligible,persistent,-np.inf);where=int(np.argmax(score))
        nodes=np.sort(np.r_[knots,where]);new_prediction,new_coefficients,new_rms=fit(nodes)
        if new_rms>=rms-1e-12:break
        knots,prediction,coefficients,rms=nodes,new_prediction,new_coefficients,new_rms
    profile=np.c_[prediction,np.zeros_like(prediction)]
    return profile,dict(method='adaptive_position_residual',knots=knots.tolist(),coefficients=np.c_[coefficients,np.zeros_like(coefficients)].tolist(),
        regularization=regularization,min_spacing_frames=spacing,target_rms_m=target_rms_m,max_knots=int(max_knots),
        position_residual_rms_m=rms,rotation_residual_rms_rad=0.,knot_history=history)


def bounded_rotation_correction(source_object,desired_object,*,max_angle_rad=.25,knot_count=16,regularization=.1):
    """Smooth a bounded chordal SO(3) descent direction in world axes.

    The antisymmetric rotation error is continuous across the principal-log
    branch at pi. Exactly pi is stationary; this does not certify global
    convergence or replace physical evaluation with an analytic gradient.
    """
    source=np.asarray(source_object,float);desired=np.asarray(desired_object,float)
    if (source.ndim!=2 or source.shape[1]!=7 or desired.shape!=source.shape
            or not np.isfinite(source).all() or not np.isfinite(desired).all()
            or np.any(np.linalg.norm(source[:,3:],axis=-1)<1e-8) or np.any(np.linalg.norm(desired[:,3:],axis=-1)<1e-8)
            or not np.isfinite([max_angle_rad,regularization]).all() or not 0<max_angle_rad<=.5 or regularization<0
            or not isinstance(knot_count,(int,np.integer)) or not 3<=knot_count<=len(source)):
        raise ValueError('finite pose paths and a bounded orientation trust step required')
    error=(Rotation.from_quat(desired[:,3:].copy())*Rotation.from_quat(source[:,3:].copy()).inv()).as_matrix()
    gradient=.5*np.c_[error[:,2,1]-error[:,1,2],error[:,0,2]-error[:,2,0],error[:,1,0]-error[:,0,1]]
    knots=np.linspace(0,len(source)-1,knot_count);clock=np.arange(len(source));design=CubicSpline(knots,np.eye(knot_count),bc_type='clamped')(clock)
    penalty=np.diff(np.eye(knot_count),n=2,axis=0)
    coefficients=np.linalg.lstsq(np.r_[design,np.sqrt(regularization)*penalty],np.r_[gradient,np.zeros((len(penalty),3))],rcond=None)[0]
    fitted=design@coefficients;norm=np.linalg.norm(fitted,axis=-1)
    step=max_angle_rad*fitted/np.maximum(1.,norm)[:,None]
    result=np.c_[np.zeros_like(step),step]
    return result,dict(method='bounded_chordal_gradient',knots=knots.tolist(),coefficients=coefficients.tolist(),
        max_angle_rad=max_angle_rad,regularization=regularization,rotation_step_peak_rad=float(np.linalg.norm(step,axis=-1).max()),
        gradient_fit_rms=float(np.sqrt(np.mean((fitted-gradient)**2))),stationary_pi_samples=int(((np.linalg.norm(gradient,axis=-1)<1e-8)&(np.trace(error,axis1=1,axis2=2)<-.999999)).sum()))
