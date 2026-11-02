"""Rigid-body contact demand from the fixed object trajectory (planning only)."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def rigid_contact_wrench_targets(position, quaternion_xyzw, fps, mass, inertia_principal,
        inertial_quaternion_xyzw=(0.,0.,0.,1.), com_body=(0.,0.,0.), gravity=(0.,0.,-9.81),
        smoothing_seconds=.25):
    """Return WORLD contact [N,Nm] about the desired body origin per frame.

    Gravity is subtracted, not credited to the hand. COM offsets and rotated
    inertia are included. Smoothing estimates derivatives only; it does not
    change the official task trajectory. This inverse-dynamics demand does
    not specify which hand, surface, or controller should provide it.
    """
    p=np.asarray(position,float);q=np.asarray(quaternion_xyzw,float)
    inertia=np.asarray(inertia_principal,float);com=np.asarray(com_body,float);g=np.asarray(gravity,float)
    iq=np.asarray(inertial_quaternion_xyzw,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or q.shape!=(len(p),4)
            or inertia.shape!=(3,) or com.shape!=(3,) or g.shape!=(3,) or iq.shape!=(4,)
            or not all(np.isfinite(v).all() for v in (p,q,inertia,com,g,iq))
            or np.any(inertia<0) or np.any(np.linalg.norm(q,axis=1)<1e-8) or np.linalg.norm(iq)<1e-8
            or not np.isfinite(mass) or mass<=0 or not np.isfinite(fps) or fps<=0
            or not np.isfinite(smoothing_seconds) or smoothing_seconds<0):
        raise ValueError('finite rigid trajectory, positive mass/FPS and physical inertia required')
    r=Rotation.from_quat(q.copy()); offset=r.apply(np.broadcast_to(com,p.shape).copy())
    world_com=p+offset; dt=1./fps
    window=max(3,int(round(smoothing_seconds*fps))|1);window=min(window,len(p) if len(p)%2 else len(p)-1)
    if window>=5:
        acceleration=savgol_filter(world_com,window,min(3,window-1),deriv=2,delta=dt,axis=0)
    else:
        acceleration=np.gradient(np.gradient(world_com,dt,axis=0,edge_order=2),dt,axis=0,edge_order=2)
    edges=(r[1:]*r[:-1].inv()).as_rotvec()/dt
    omega=np.vstack([edges[0],.5*(edges[:-1]+edges[1:]),edges[-1]])
    if window>=5:
        alpha=savgol_filter(omega,window,min(3,window-1),deriv=1,delta=dt,axis=0)
    else:
        alpha=np.gradient(omega,dt,axis=0,edge_order=2)
    world_from_inertia=(r*Rotation.from_quat(iq)).as_matrix()
    iw=np.einsum('fij,j,fkj->fik',world_from_inertia,inertia,world_from_inertia)
    momentum=np.einsum('fij,fj->fi',iw,omega)
    force=mass*(acceleration-g)
    torque=np.einsum('fij,fj->fi',iw,alpha)+np.cross(omega,momentum)+np.cross(offset,force)
    return np.concatenate([force,torque],axis=1)
