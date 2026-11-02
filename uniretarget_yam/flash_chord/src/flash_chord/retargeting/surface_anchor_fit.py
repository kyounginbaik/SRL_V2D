"""Fit actual robot material points to force-feasible object surface anchors."""
from dataclasses import dataclass,replace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from .contact_wrench import ContactPatch,project_contact_wrench,solve_contact_wrench
from .contact_invariant_fit import ContactInvariantFit


@dataclass(frozen=True)
class SurfaceAnchor:
    hand_geom: int
    point_geom_local: tuple
    normal_geom_local: tuple
    patch: ContactPatch
    triangle_world: tuple | None = None

    def __post_init__(self):
        p,n=np.asarray(self.point_geom_local,float),np.asarray(self.normal_geom_local,float)
        if (p.shape!=(3,) or n.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(n).all()
                or abs(np.linalg.norm(n)-1)>1e-5 or not isinstance(self.patch,ContactPatch)):
            raise ValueError('finite robot material point, unit normal hint and object ContactPatch required')
        if self.triangle_world is not None:
            tri=np.asarray(self.triangle_world,float)
            if tri.shape!=(3,3) or not np.isfinite(tri).all() or np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))<1e-12:
                raise ValueError('finite nondegenerate object surface triangle required')


def closest_triangle_point(point,triangle):
    p=np.asarray(point,float);tri=np.asarray(triangle,float);a,b,c=tri;e=b-a;f=c-a
    normal=np.cross(e,f);normal/=np.linalg.norm(normal)
    plane=p-normal*np.dot(p-a,normal)
    gram=np.array([[e@e,e@f],[e@f,f@f]])
    alpha,beta=np.linalg.solve(gram,np.array([e@(plane-a),f@(plane-a)]))
    if alpha>=0 and beta>=0 and alpha+beta<=1:return plane
    candidates=[]
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;t=np.clip(np.dot(p-start,edge)/np.dot(edge,edge),0.,1.)
        candidates.append(start+t*edge)
    return min(candidates,key=lambda q:np.sum((q-p)**2)).copy()


def _material_support(model,geom,normal_local,mesh_vertices):
    import mujoco
    kind=int(model.geom_type[geom]);size=model.geom_size[geom];n=np.asarray(normal_local,float)
    if kind==mujoco.mjtGeom.mjGEOM_MESH:return float(np.max(mesh_vertices[geom]@n))
    if kind==mujoco.mjtGeom.mjGEOM_BOX:return float(np.abs(n)@size)
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:return float(size[0]*np.linalg.norm(n))
    if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:return float(size[0]*np.linalg.norm(n)+size[1]*abs(n[2]))
    if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:return float(size[0]*np.linalg.norm(n[:2])+size[1]*abs(n[2]))
    if kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:return float(np.linalg.norm(size*n))
    raise ValueError('unsupported hand material geometry for a contact support query')


def _validate_material_point(model,geom,point,mesh_vertices):
    import mujoco
    p=np.asarray(point,float);kind=int(model.geom_type[geom]);size=model.geom_size[geom];tol=1e-6
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        hull=ConvexHull(mesh_vertices[geom]);values=hull.equations[:,:3]@p+hull.equations[:,3]
        valid=values.max()<=tol and values.max()>=-tol
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:valid=np.all(np.abs(p)<=size+tol) and np.min(np.abs(np.abs(p)-size))<=tol
    elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:valid=abs(np.linalg.norm(p)-size[0])<=tol
    elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
        center=np.array([0.,0.,np.clip(p[2],-size[1],size[1])]);valid=abs(np.linalg.norm(p-center)-size[0])<=tol
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(p[:2]);valid=radial<=size[0]+tol and abs(p[2])<=size[1]+tol and min(abs(radial-size[0]),abs(abs(p[2])-size[1]))<=tol
    elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:valid=abs(np.linalg.norm(p/size)-1)<1e-5
    else:valid=False
    if not valid:raise ValueError('contact material point is not on the actual hand collider surface')


def fit_native_surface_anchors(model,seed_qpos,qpos_ids,lower,upper,scales,anchors,
        robot_body_ids,object_body_id,desired_wrench,normal_force_limits,*,
        rotation_qpos_starts=(),maximum_rotation_rad=.6,point_scale_m=.002,
        max_evaluations=150,normal_force_minima=None):
    """Use assigned surfaces to avoid a flat nearest-face contact-mode gradient.

    The robot normal is an initialization hint; contact validity uses the
    material point's actual convex support cone, allowing vertex/edge contacts.
    Anchor selection remains the caller's contact-arrangement proposal. No
    kinematic site is substituted for a robot collision-surface material point.
    Wrench projection precedes IK; actual physical replay still decides whether
    these reachable contacts and the controller can realize the predicted load.
    """
    import mujoco
    anchors=tuple(anchors);seed=np.asarray(seed_qpos,float);ids=np.asarray(qpos_ids,int)
    lo,hi,scale=(np.asarray(v,float) for v in (lower,upper,scales));robot=np.asarray(robot_body_ids,int)
    if (not anchors or seed.shape!=(model.nq,) or ids.ndim!=1 or not len(ids)
            or len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=model.nq)
            or any(a.hand_geom not in range(model.ngeom) or model.geom_bodyid[a.hand_geom] not in robot for a in anchors)
            or any(v.shape!=ids.shape for v in (lo,hi,scale)) or np.any(lo>=hi) or np.any(scale<=0)
            or not all(np.isfinite(v).all() for v in (seed,lo,hi,scale))):
        raise ValueError('explicit robot material anchors, coordinate bounds and scales required')
    for j in np.flatnonzero(model.jnt_bodyid==object_body_id):
        qi=int(model.jnt_qposadr[j]);n={0:7,1:4,2:1,3:1}[int(model.jnt_type[j])]
        if np.isin(ids,np.arange(qi,qi+n)).any():raise ValueError('object coordinates are immutable')
    rotations=tuple(int(qi) for qi in rotation_qpos_starts)
    valid_rotations={int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.jnt_type[j]==1 and model.jnt_bodyid[j] in robot}
    if any(qi not in valid_rotations for qi in rotations):raise ValueError('robot ball-joint starts required')
    mesh_vertices={}
    for a in anchors:
        g=a.hand_geom
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and g not in mesh_vertices:
            mesh=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
            mesh_vertices[g]=np.asarray(model.mesh_vert[start:start+count],float).copy()
        _validate_material_point(model,g,a.point_geom_local,mesh_vertices)
    base=seed.copy();base[ids]=np.clip(base[ids],lo,hi);d=mujoco.MjData(model)
    d.qpos[:]=base;mujoco.mj_kinematics(model,d)
    patches=tuple(a.patch for a in anchors)
    if normal_force_minima is None:
        projection=project_contact_wrench(patches,desired_wrench,center=d.xpos[object_body_id],normal_force_limits=normal_force_limits)
        force_converged=projection.converged;forces=projection.contact_forces;missing=projection.missing_wrench
    else:
        projection=solve_contact_wrench(patches,desired_wrench,center=d.xpos[object_body_id],
            normal_force_limits=normal_force_limits,normal_force_minima=normal_force_minima)
        force_converged=projection.feasible;forces=projection.contact_forces;missing=-projection.wrench_residual
    units=np.r_[np.ones(3),np.full(3,.05)]
    if not force_converged or np.linalg.norm(missing/units)>.1*max(.1,np.linalg.norm(np.asarray(desired_wrench)/units)):
        raise ValueError('selected object surface anchors cannot supply the required wrench')
    ns=len(ids);nv=ns+3*len(rotations);bounds=(np.r_[(lo-base[ids])/scale,np.full(nv-ns,-maximum_rotation_rad/(.2*np.sqrt(3)))],
        np.r_[(hi-base[ids])/scale,np.full(nv-ns,maximum_rotation_rad/(.2*np.sqrt(3)))])
    latest={}
    def evaluate(x):
        q=base.copy();q[ids]+=x[:ns]*scale
        for k,qi in enumerate(rotations):
            rotation=Rotation.from_rotvec(.2*x[ns+3*k:ns+3*k+3])*Rotation.from_quat(q[qi:qi+4][[1,2,3,0]])
            q[qi:qi+4]=rotation.as_quat()[[3,0,1,2]]
        d.qpos[:]=q;mujoco.mj_kinematics(model,d);mujoco.mj_collision(model,d)
        offsets=[];support_gaps=[];contact_points=[]
        for a in anchors:
            r=d.geom_xmat[a.hand_geom].reshape(3,3)
            material=d.geom_xpos[a.hand_geom]+r@np.asarray(a.point_geom_local)
            point=np.asarray(a.patch.position) if a.triangle_world is None else closest_triangle_point(material,a.triangle_world)
            offsets.append(material-point);contact_points.append(point)
            local_normal=r.T@np.asarray(a.patch.normal)
            support_gaps.append(max(0.,_material_support(model,a.hand_geom,local_normal,mesh_vertices)-np.asarray(a.point_geom_local)@local_normal))
        offsets=np.asarray(offsets);support_gaps=np.asarray(support_gaps);depth=np.zeros(len(robot))
        for c in d.contact[:d.ncon]:
            bodies=model.geom_bodyid[c.geom]
            for i,body in enumerate(robot):
                if body in bodies:depth[i]=max(depth[i],-float(c.dist))
        latest.update(q=q,offsets=offsets,support_gaps=support_gaps,depth=depth,contact_points=np.asarray(contact_points))
        return np.r_[(offsets/point_scale_m).ravel(),(support_gaps/.001).ravel(),
                     2*np.maximum(depth-.0005,0.)/.001,.02*x]
    solution=least_squares(evaluate,np.zeros(nv),bounds=bounds,max_nfev=max_evaluations,
                           ftol=1e-8,xtol=1e-8,gtol=1e-8,diff_step=.001)
    evaluate(solution.x);gaps=np.linalg.norm(latest['offsets'],axis=1)
    feasible=bool(np.all(gaps<.001) and np.all(latest['support_gaps']<.0002)
                  and latest['depth'].max(initial=0.)<.0006)
    final_patches=tuple(replace(a.patch,position=tuple(point)) for a,point in zip(anchors,latest['contact_points']))
    if normal_force_minima is None:
        final=project_contact_wrench(final_patches,desired_wrench,center=d.xpos[object_body_id],normal_force_limits=normal_force_limits)
        force_ok=final.converged;forces=final.contact_forces;missing=final.missing_wrench
    else:
        final=solve_contact_wrench(final_patches,desired_wrench,center=d.xpos[object_body_id],normal_force_limits=normal_force_limits,normal_force_minima=normal_force_minima)
        force_ok=final.feasible;forces=final.contact_forces;missing=-final.wrench_residual
    feasible=bool(feasible and force_ok and np.linalg.norm(missing/units)<.1*max(.1,np.linalg.norm(np.asarray(desired_wrench)/units)))
    return ContactInvariantFit(latest['q'].copy(),final_patches,np.array([a.hand_geom for a in anchors]),
        forces,gaps,missing,float(latest['depth'].max(initial=0.)),
        bool(solution.success),feasible,int(solution.nfev),float(solution.cost),str(solution.message),
        float(max(normal_force_limits.values())))
