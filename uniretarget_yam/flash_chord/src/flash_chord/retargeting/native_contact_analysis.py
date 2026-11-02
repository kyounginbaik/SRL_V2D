"""Interpret pinned native contact snapshots with explicit force/frame semantics."""
from dataclasses import dataclass
import numpy as np
from .contact_wrench import ContactPatch


@dataclass(frozen=True)
class ObjectContactSnapshot:
    native_body_id: int
    origin_world: np.ndarray
    quaternion_wxyz: np.ndarray
    patches: tuple
    forces_on_object: np.ndarray
    intrinsic_torques_on_object: np.ndarray
    counterpart_native_bodies: np.ndarray
    counterpart_newton_bodies: np.ndarray
    wrench_world: np.ndarray


def object_contact_snapshot(data, frame, world_id, object_id=0):
    """Use cached force-evaluation geometry; retain support and hand identities.

    Newton stores WORLD [force, intrinsic torque] on shape0. Native frame row0
    points from geom0 to geom1, hence positive normal force on geom1. Only
    native condim=3 is supported by ContactPatch's four-edge force model.
    Unsupported contact dimensions fail rather than losing torsional moments.
    """
    worlds=np.asarray(data['native_capture_world_ids'])
    found=np.flatnonzero(worlds==world_id)
    if len(found)!=1: raise ValueError('world was not uniquely captured')
    wi=int(found[0]); frames=np.asarray(data['native_capture_frame_ids'])
    if frame not in range(len(frames)) or frames[frame]!=frame: raise ValueError('invalid capture frame')
    object_bodies=np.asarray(data['native_object_body_ids'])
    if object_id not in range(object_bodies.shape[1]): raise ValueError('invalid object ID')
    bodymap=np.asarray(data['native_body_to_newton_body'])
    if bodymap.ndim!=2: raise ValueError('pinned world/body-to-global-Newton map required')
    native=np.flatnonzero(bodymap[world_id]==object_bodies[wi,object_id])
    if len(native)!=1: raise ValueError('object body binding is ambiguous')
    native=int(native[0]); geom_bodies=np.asarray(data['native_geom_bodyid'])
    a,b=np.asarray(data['native_contact_offsets'])[frame:frame+2]
    rows=np.arange(a,b)
    rows=rows[np.asarray(data['native_contact_worldid'])[rows]==world_id]
    geoms=np.asarray(data['native_contact_geom'])[rows]
    body_pairs=geom_bodies[geoms]
    on0=body_pairs[:,0]==native; on1=body_pairs[:,1]==native
    keep=on0 ^ on1; rows=rows[keep]; on0=on0[keep]; body_pairs=body_pairs[keep]
    if np.any(np.asarray(data['native_contact_dim'])[rows]!=3):
        raise ValueError('only native condim=3 contact force patches are supported')
    positions=np.asarray(data['native_contact_pos'])[rows]
    frames_w=np.asarray(data['native_contact_frame'])[rows]
    friction=np.asarray(data['native_contact_friction'])[rows,:2]
    raw=np.asarray(data['native_contact_newton_force'])[rows]
    sign=np.where(on0,1.,-1.)
    force=sign[:,None]*raw[:,:3]; torque=sign[:,None]*raw[:,3:]
    # Native normal is the direction of force on geom1, the opposite of
    # Newton shape0's force direction. Tangent signs are symmetric in the cone.
    normal=-sign[:,None]*frames_w[:,0]
    other=np.where(on0,body_pairs[:,1],body_pairs[:,0])
    newton_other=bodymap[world_id,other]
    patches=tuple(ContactPatch(tuple(p),tuple(n),tuple(t[1]),tuple(t[2]),tuple(mu),int(group))
        for p,n,t,mu,group in zip(positions,normal,frames_w,friction,other))
    origin=np.asarray(data['native_xpos'])[frame,wi,native]
    quat=np.asarray(data['native_xquat'])[frame,wi,native]
    wrench=np.r_[force.sum(0),(np.cross(positions-origin,force)+torque).sum(0)]
    return ObjectContactSnapshot(native,origin.copy(),quat.copy(),patches,force,torque,
                                 other,newton_other,wrench)
