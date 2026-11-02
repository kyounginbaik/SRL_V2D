"""Analytic point Jacobians from the SAME cached native force evaluation."""
import numpy as np


def cached_point_jacobian(data, frame, world_id, native_body, point_world):
    """Return [3,native_nv] Jacobian; native angular DOFs use native axes.

    Implements the pinned MuJoCo-Warp support.jac_dof formula using cdof and
    subtree_com caches, rather than mixing pre-integration forces with a
    post-integration FK configuration. Scalar joints can be mapped directly
    to reference coordinates. Ball/free rotational DOFs require a separate
    coordinate differential when editing a rotvec or quaternion reference.
    """
    captured=np.asarray(data['native_capture_world_ids'])
    rows=np.flatnonzero(captured==world_id)
    if len(rows)!=1: raise ValueError('world was not uniquely captured')
    wi=int(rows[0]); parent=np.asarray(data['native_model_body_parentid'])
    root=np.asarray(data['native_model_body_rootid'])
    dof_body=np.asarray(data['native_model_dof_bodyid'])
    point=np.asarray(point_world,float)
    if (native_body not in range(len(parent)) or point.shape!=(3,) or not np.isfinite(point).all()
            or frame not in range(len(data['native_capture_frame_ids']))):
        raise ValueError('valid body, captured frame and finite WORLD point required')
    ancestors=set(); current=int(native_body)
    while current>0:
        if current in ancestors: raise ValueError('cyclic native body tree')
        ancestors.add(current); current=int(parent[current])
    active=np.isin(dof_body,list(ancestors))
    cdof=np.asarray(data['native_cdof'])[frame,wi]
    com=np.asarray(data['native_subtree_com'])[frame,wi,root[native_body]]
    if cdof.shape!=(len(dof_body),6): raise ValueError('invalid native motion-axis cache')
    jac=cdof[:,3:]+np.cross(cdof[:,:3],point-com)
    return np.where(active[:,None],jac,0.).T


def scalar_reference_channels(data, world_id=0):
    """Map native hinge/slide DOFs to robot geometry and actuator coordinates.

    No ball/free rotation linearization is guessed. Static/support/object
    joints are excluded by the explicit robot coordinate range.
    """
    qstart=np.asarray(data['newton_joint_q_start']); tstart=np.asarray(data['newton_joint_qd_start'])
    mapping=np.asarray(data['native_joint_to_newton_joint'])
    nq=int(data['newton_robot_nq']); worlds=len(mapping)
    if qstart[-1]%worlds or tstart[-1]%worlds: raise ValueError('nonreplicated joint-coordinate layout')
    qoffset=world_id*(qstart[-1]//worlds); toffset=world_id*(tstart[-1]//worlds)
    native_dof=[]; geometry=[]; target=[]
    for j,kind in enumerate(np.asarray(data['native_model_jnt_type'])):
        nj=int(mapping[world_id,j])
        if kind not in (2,3) or nj<0: continue
        qi=int(qstart[nj]-qoffset); ti=int(tstart[nj]-toffset)
        if 0<=qi<nq:
            if qstart[nj+1]-qstart[nj]!=1 or tstart[nj+1]-tstart[nj]!=1:
                raise ValueError('native scalar joint maps to a nonscalar robot coordinate')
            native_dof.append(int(data['native_model_jnt_dofadr'][j]));geometry.append(qi);target.append(ti)
    return np.asarray(native_dof,int),np.asarray(geometry,int),np.asarray(target,int)
