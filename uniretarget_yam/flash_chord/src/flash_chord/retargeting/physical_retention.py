"""Physical retention diagnostics, separate from full-task acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation


def retention_evidence(receiver_contact,giver_contact,load,objects,palms):
    """Per-world evidence on an explicitly selected post-release interval.

    Contact, load and constant attachment are separate proxies; none replaces
    the immutable full-task metric. Invalid worlds remain in every denominator.
    """
    rc,gc,load=[np.asarray(x) for x in (receiver_contact,giver_contact,load)]
    objects,palms=[np.asarray(x,float) for x in (objects,palms)]
    if rc.ndim!=2 or gc.shape!=rc.shape or load.shape!=rc.shape or objects.shape!=rc.shape+(7,) or palms.shape!=objects.shape or len(rc)<2:
        raise ValueError('matching post-release time/world samples required')
    finite=np.isfinite(load)&np.isfinite(objects).all(-1)&np.isfinite(palms).all(-1)
    finite&=(np.linalg.norm(objects[...,3:],axis=-1)>1e-8)&(np.linalg.norm(palms[...,3:],axis=-1)>1e-8)
    valid=finite.all(0)
    clean=(rc.mean(0)>=.8)&(gc.mean(0)<=.2)&valid
    half=(load>.5).mean(0)
    safe_object=np.where(finite[...,None],objects,[0,0,0,0,0,0,1]);safe_palm=np.where(finite[...,None],palms,[0,0,0,0,0,0,1])
    ro=Rotation.from_quat(safe_object[...,3:].reshape(-1,4));rh=Rotation.from_quat(safe_palm[...,3:].reshape(-1,4))
    relative=ro.inv().apply((safe_palm[...,:3]-safe_object[...,:3]).reshape(-1,3)).reshape(*rc.shape,3)
    motion=np.linalg.norm(relative-np.median(relative,axis=0),axis=-1)
    rq=(ro.inv()*rh).as_quat().reshape(*rc.shape,4)
    rotations=(Rotation.from_quat(rq.reshape(-1,4))*Rotation.from_quat(np.broadcast_to(rq[0:1],rq.shape).reshape(-1,4)).inv()).magnitude().reshape(rc.shape)
    stable=(np.percentile(motion,90,axis=0)<.03)&(np.percentile(rotations,90,axis=0)<np.deg2rad(30))&valid
    return dict(worlds=int(rc.shape[1]),samples=int(len(rc)),finite_world_fraction=float(valid.mean()),
        receiver_contact_fraction=float(np.where(valid,rc.mean(0),0).mean()),giver_contact_fraction=float(np.where(valid,gc.mean(0),1).mean()),
        clean_contact_fraction=float(clean.mean()),stable_attachment_fraction=float(stable.mean()),
        supported_retention_proxy_fraction=float((clean&stable&(half>=.5)).mean()),
        per_world_retention_proxy=(clean&stable&(half>=.5)).tolist())


def elevated_retention_windows(receiver_contact,giver_contact,load,objects,palms,targets,gravity,
        previous_owner,position_error,rotation_error,*,bounds,window_frames,height_margin=.03):
    """Find retention before intended placement, requiring an earlier giver.

    Elevation relative to each initial pose is only a phase proxy, not a native
    no-table-contact certificate. Object/task errors retain official alignment;
    this short-window diagnostic never replaces whole-episode official scoring.
    """
    objects,palms,targets=[np.asarray(x,float) for x in (objects,palms,targets)]
    rc,gc,load=[np.asarray(x) for x in (receiver_contact,giver_contact,load)]
    gravity=np.asarray(gravity,float);previous=np.asarray(previous_owner)
    pe,re=np.asarray(position_error,float),np.asarray(rotation_error,float)
    if (objects.ndim!=3 or objects.shape[-1]!=7 or palms.shape!=objects.shape or targets.shape!=objects.shape
            or any(x.shape!=objects.shape[:2] for x in (rc,gc,load,pe,re))
            or gravity.shape!=(objects.shape[1],3) or previous.shape!=(objects.shape[1],) or previous.dtype!=bool
            or not np.isfinite(gravity).all() or np.any(np.linalg.norm(gravity,axis=1)<1e-8)
            or not np.isfinite(height_margin) or height_margin<=0):
        raise ValueError('matching phase poses, errors, contacts, gravity and Boolean previous ownership required')
    a,b=bounds
    if not 0<=a<b<=len(objects) or not isinstance(window_frames,(int,np.integer)) or window_frames<2:
        raise ValueError('nonempty bounds and a sustained integer window required')
    up=-gravity/np.linalg.norm(gravity,axis=1)[:,None]
    actual_height=np.einsum('twj,wj->tw',objects[...,:3]-objects[:1,:,:3],up)
    wanted_height=np.einsum('twj,wj->tw',targets[...,:3]-targets[:1,:,:3],up)
    elevated=(actual_height>height_margin)&(wanted_height>height_margin)
    retained=np.zeros(objects.shape[1],bool);strict=retained.copy();first=np.full(objects.shape[1],-1,int)
    for start in range(a,b-window_frames+1):
        stop=start+window_frames;eligible=elevated[start:stop].all(0)&previous
        if not eligible.any():continue
        evidence=retention_evidence(rc[start:stop],gc[start:stop],load[start:stop],objects[start:stop],palms[start:stop])
        good=np.asarray(evidence['per_world_retention_proxy'])&eligible
        first[good&(first<0)]=start;retained|=good
        strict|=good&(pe[start:stop].mean(0)<.03)&(re[start:stop].mean(0)<np.pi/6)
    return dict(window_frames=int(window_frames),bounds=[a,b],height_margin_m=float(height_margin),
        elevated_retention_proxy_fraction=float(retained.mean()),strict_tracked_retention_fraction=float(strict.mean()),
        per_world_retention_proxy=retained.tolist(),per_world_strict_tracking=strict.tolist(),first_retained_frame=first.tolist(),
        support_contact_absence_verified=False)

