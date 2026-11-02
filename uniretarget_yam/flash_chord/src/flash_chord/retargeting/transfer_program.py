"""Compose demonstrated contact choreography with a new fixed object task.

Time reversal is a physical hypothesis: these functions preserve measured
geometry/preload but never assert that friction or contact dynamics reverse.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .coupled_transport import transport_poses


def _rotation(values):
    return Rotation.from_quat(np.array(values, float, copy=True))


def relative_paths(objects, poses):
    identity=np.zeros_like(np.asarray(objects,float));identity[:,6]=1
    return transport_poses(objects,identity,poses)


def world_paths(objects, poses):
    identity=np.zeros_like(np.asarray(objects,float));identity[:,6]=1
    return transport_poses(identity,objects,poses)


def sample_poses(poses, times):
    poses=np.asarray(poses,float);times=np.asarray(times,float)
    if (poses.ndim!=3 or poses.shape[-1]!=7 or len(poses)<2 or times.ndim!=1
            or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or np.any(times<0) or np.any(times>len(poses)-1)):
        raise ValueError('finite pose paths and in-range source times required')
    output=np.empty((len(times),poses.shape[1],7));clock=np.arange(len(poses))
    for h in range(poses.shape[1]):
        for axis in range(3):output[:,h,axis]=np.interp(times,clock,poses[:,h,axis])
        output[:,h,3:]=Slerp(clock,_rotation(poses[:,h,3:]))(times).as_quat()
    return output


def blend_poses(start,stop,weight):
    a,b=np.broadcast_arrays(np.asarray(start,float),np.asarray(stop,float));w=np.asarray(weight,float)
    if a.ndim!=3 or a.shape[-1]!=7 or w.shape!=(len(a),) or not np.isfinite(w).all() or np.any(w<0) or np.any(w>1):
        raise ValueError('matching pose paths and weights in [0,1] required')
    result=a.copy()
    for h in range(a.shape[1]):
        ar=_rotation(a[:,h,3:]);br=_rotation(b[:,h,3:])
        result[:,h,:3]=a[:,h,:3]+w[:,None]*(b[:,h,:3]-a[:,h,:3])
        result[:,h,3:]=(Rotation.from_rotvec(w[:,None]*(br*ar.inv()).as_rotvec())*ar).as_quat()
    result[w==0]=a[w==0];result[w==1]=b[w==1]
    return result


def reserve_free_hand(path, source, hand, start, end):
    """Keep a released hand near the demonstrated next-transfer start pose.

    Both paths are object relative. The held hand and all rows outside the
    declared free interval remain exact. This changes a robot reference, not
    the object task; physical clearance and support still require a rollout.
    """
    values=np.asarray(path,float);observed=np.asarray(source,float)
    if (values.ndim!=3 or values.shape[-1]!=7 or observed.ndim!=3
            or observed.shape[1:]!=values.shape[1:] or not isinstance(hand,(int,np.integer))
            or not 0<=hand<values.shape[1] or not all(isinstance(i,(int,np.integer)) for i in (start,end))
            or not 1<=start<end<len(values) or start>=len(observed)
            or not np.isfinite(values).all() or not np.isfinite(observed).all()):
        raise ValueError('aligned finite object-relative paths and a protected free-hand interval required')
    anchor=observed[start,hand]
    if (not np.allclose(values[start,hand,:3],anchor[:3],atol=1e-9)
            or not np.allclose(values[end,hand,:3],anchor[:3],atol=1e-9)
            or (_rotation(values[[start,end],hand,3:])*_rotation(np.repeat(anchor[None,3:],2,axis=0)).inv()).magnitude().max()>1e-8):
        raise ValueError('free interval must start and end at the demonstrated transfer pose')
    result=values.copy();result[start:end+1,hand]=anchor
    return result


def inverse_path(poses,model,position_model):
    p=np.asarray(poses,float);output=p.copy()
    if p.ndim!=2 or p.shape[1]!=7 or len(p)<3:raise ValueError('three actual/desired pose rows required')
    output[1:]=model.inverse(p[:-1],p[1:])
    if position_model is not None:output[2:,:3]=position_model.inverse(p[:-2,:3],p[1:-1,:3],p[2:,:3])
    return output


def object_preload_residual(objects,actual,commands,models,position_models):
    """Subtract forward-motion feedforward; store remaining preload in object axes."""
    obj=np.asarray(objects,float);h=np.asarray(actual,float);c=np.asarray(commands,float)
    if h.shape!=c.shape or h.shape[:1]!=obj.shape[:1] or len(models)!=h.shape[1] or len(position_models)!=len(models):
        raise ValueError('aligned source geometry/native spring poses and models required')
    residual=np.zeros_like(h);residual[...,6]=1;ro=_rotation(obj[:,3:])
    for hand,(model,position) in enumerate(zip(models,position_models)):
        predicted=inverse_path(h[:,hand],model,position)
        residual[:,hand,:3]=ro.inv().apply(c[:,hand,:3]-predicted[:,:3])
        delta=_rotation(c[:,hand,3:])*_rotation(predicted[:,3:]).inv()
        residual[:,hand,3:]=(ro.inv()*delta*ro).as_quat()
    return residual


def commands_with_preload(objects,goals,residual,models,position_models):
    """New motion lead plus transported source preload; no physics/control writes."""
    obj=np.asarray(objects,float);h=np.asarray(goals,float);bias=np.asarray(residual,float)
    if h.shape!=bias.shape or len(h)!=len(obj) or len(models)!=h.shape[1]:raise ValueError('aligned target poses/preload/models required')
    ro=_rotation(obj[:,3:]);result=h.copy()
    for hand,(model,position) in enumerate(zip(models,position_models)):
        inverse=inverse_path(h[:,hand],model,position)
        result[:,hand,:3]=inverse[:,:3]+ro.apply(bias[:,hand,:3])
        delta=ro*_rotation(bias[:,hand,3:])*ro.inv()
        result[:,hand,3:]=(delta*_rotation(inverse[:,3:])).as_quat()
    return result
