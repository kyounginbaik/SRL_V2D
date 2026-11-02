"""Jointly fit reachable native contact geometry and finite friction forces.

An offline geometry/force proposal, using the unchanged native MuJoCo model.
No simulator state from this fitting model is injected into physical evaluation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax
from scipy.spatial.transform import Rotation
from .contact_wrench import ContactPatch, _wrench_columns
from .convex_distance import ConvexDistanceQuery


@dataclass(frozen=True)
class ContactInvariantFit:
    qpos: np.ndarray
    patches: tuple
    hand_geom_ids: np.ndarray
    forces_on_object: np.ndarray
    contact_gaps: np.ndarray
    wrench_residual: np.ndarray
    max_collision_penetration: float
    converged: bool
    model_feasible: bool
    evaluations: int
    cost: float
    message: str
    normal_force_budget: float = float("inf")
    distance_repairs: int = 0
    distance_fallbacks: int = 0


def _contact_geometry(m, d, hand_geoms, object_geoms, group, mujoco, query):
    best=None
    pairs=sorted((query.lower_bound(int(h),int(o)),int(h),int(o)) for h in hand_geoms for o in object_geoms)
    for bound,h,o in pairs:
        if best is not None and bound>max(best[0],0.)+1e-9:continue
        distance,points=query.distance(h,o)
        if best is None or distance<best[0]:best=(distance,h,o,points.copy())
    distance,h,o,points=best
    direction=points[3:]-points[:3]
    if abs(distance)>1e-9 and np.linalg.norm(direction)>1e-10:
        normal=direction/distance;normal/=np.linalg.norm(normal)
    else:
        normal=d.geom_xpos[o]-d.geom_xpos[h];normal/=max(np.linalg.norm(normal),1e-12)
    axis=np.eye(3)[np.argmin(np.abs(normal))]
    t1=np.cross(normal,axis);t1/=max(np.linalg.norm(t1),1e-12);t2=np.cross(normal,t1)
    priority=m.geom_priority
    mu=(m.geom_friction[h,0] if priority[h]>priority[o] else m.geom_friction[o,0]
        if priority[o]>priority[h] else max(m.geom_friction[h,0],m.geom_friction[o,0]))
    # Contract the nominal cone for a load margin; use native friction, never
    # a material change. condim>3 needs a richer wrench model and fails here.
    dim=m.geom_condim[h] if priority[h]>priority[o] else m.geom_condim[o] if priority[o]>priority[h] else max(m.geom_condim[h],m.geom_condim[o])
    if dim!=3:raise ValueError('contact-invariant force model requires native condim=3')
    patch=ContactPatch(tuple(points[3:]),tuple(normal),tuple(t1),tuple(t2),(.7*mu,.7*mu),group)
    return distance,h,patch


def fit_native_contact_pose(model, seed_qpos, qpos_ids, lower, upper, scales,
        contact_geom_groups, robot_body_ids, object_body_id, desired_wrench, *,
        normal_force_budget, length_scale_m=.05, gap_scale_m=.001,
        maximum_penetration_m=.0005, max_evaluations=200,
        rotation_qpos_starts=(), maximum_rotation_rad=.35):
    """Optimize scalar robot coordinates and contact-cone weights together.

    Groups are embodiment collision-geometry chains, not named manipulation
    skills. Positive latent forces let geometry gradients exist before a good
    grasp forms. A force-times-gap penalty makes a force expensive at separated
    surfaces; final checks still require actual near-contact and wrench balance.
    Only the explicitly listed robot qpos coordinates may change. Bounds/scales
    are supplied by the embodiment adapter. No object or material is edited.
    """
    import mujoco
    seed=np.asarray(seed_qpos,float);ids=np.asarray(qpos_ids,int)
    lo,hi,scale=(np.asarray(v,float) for v in (lower,upper,scales))
    groups=tuple(tuple(int(g) for g in group) for group in contact_geom_groups)
    robot=np.asarray(robot_body_ids,int);target=np.asarray(desired_wrench,float)
    if (seed.shape!=(model.nq,) or ids.ndim!=1 or not len(ids) or len(set(ids))!=len(ids)
            or np.any(ids<0) or np.any(ids>=model.nq) or lo.shape!=ids.shape or hi.shape!=ids.shape
            or scale.shape!=ids.shape or np.any(lo>=hi) or np.any(scale<=0)
            or target.shape!=(6,) or not groups or any(not group for group in groups)
            or not all(np.isfinite(v).all() for v in (seed,lo,hi,scale,target))
            or not np.isfinite(normal_force_budget) or normal_force_budget<=0):
        raise ValueError('finite robot coordinate bounds/scales, contact chains and force budget required')
    object_geoms=np.flatnonzero(model.geom_bodyid==object_body_id)
    if not len(object_geoms):raise ValueError('object has no native collision geometry')
    for group in groups:
        if any(g<0 or g>=model.ngeom or model.geom_bodyid[g] not in robot for g in group):
            raise ValueError('every contact geom must belong to an explicit robot body')
    # Reject an attempted object-coordinate edit before any fitting operation.
    for j in np.flatnonzero(model.jnt_bodyid==object_body_id):
        start=model.jnt_qposadr[j];n={0:7,1:4,2:1,3:1}[int(model.jnt_type[j])]
        if np.isin(ids,np.arange(start,start+n)).any():raise ValueError('object coordinates are immutable')
    rotations=tuple(int(qi) for qi in rotation_qpos_starts)
    ball_starts=set(int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.jnt_type[j]==1)
    if (any(qi not in ball_starts for qi in rotations) or len(set(rotations))!=len(rotations)
            or not np.isfinite(maximum_rotation_rad) or maximum_rotation_rad<=0
            or any(np.isin(ids,np.arange(qi,qi+4)).any() for qi in rotations)):
        raise ValueError('unique native robot ball-joint rotations required')
    base=seed.copy();base[ids]=np.clip(base[ids],lo,hi)
    n=len(groups);ns=len(ids);nv=ns+3*len(rotations);d=mujoco.MjData(model)
    force_scale=max(float(np.linalg.norm(target[:3])),float(np.linalg.norm(target[3:]))/length_scale_m,.1)
    row_scale=force_scale*np.r_[np.ones(3),np.full(3,length_scale_m)]
    # Logits plus a slack category bound every patch's total normal force.
    initial_weight=min(.8,force_scale/(n*normal_force_budget))
    initial_logit=np.log(initial_weight/(4*(1-initial_weight)))
    initial=np.r_[np.zeros(nv),np.full(4*n,initial_logit)]
    bounds=(np.r_[(lo-base[ids])/scale,np.full(nv-ns,-maximum_rotation_rad/(.2*np.sqrt(3))),np.full(4*n,-16.)],
            np.r_[(hi-base[ids])/scale,np.full(nv-ns,maximum_rotation_rad/(.2*np.sqrt(3))),np.full(4*n,16.)])
    latest={};query=ConvexDistanceQuery(model);geometry_cache={}
    def evaluate(x):
        if 'x' not in geometry_cache or not np.array_equal(geometry_cache['x'],x[:nv]):
            q=base.copy();q[ids]+=x[:ns]*scale
            for k,qi in enumerate(rotations):
                quat=q[qi:qi+4][[1,2,3,0]]
                moved=Rotation.from_rotvec(.2*x[ns+3*k:ns+3*k+3])*Rotation.from_quat(quat)
                q[qi:qi+4]=moved.as_quat()[[3,0,1,2]]
            d.qpos[:]=q;d.qvel[:]=0
            mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d);mujoco.mj_collision(model,d)
            query.update(d)
            geometry=[_contact_geometry(model,d,group,object_geoms,i,mujoco,query) for i,group in enumerate(groups)]
            gaps=np.array([g[0] for g in geometry]);patches=tuple(g[2] for g in geometry)
            columns,matrix,_=_wrench_columns(patches,d.xpos[object_body_id])
            depth=np.zeros(len(robot))
            for c in d.contact[:d.ncon]:
                bodies=model.geom_bodyid[c.geom]
                for k,body in enumerate(robot):
                    if body in bodies:depth[k]=max(depth[k],-float(c.dist))
            geometry_cache.update(x=x[:nv].copy(),q=q,gaps=gaps,patches=patches,
                columns=columns,matrix=matrix,depth=depth,geometry=geometry)
        q,gaps,patches,columns,matrix,depth,geometry=(geometry_cache[k] for k in
            ('q','gaps','patches','columns','matrix','depth','geometry'))
        logits=x[nv:].reshape(n,4)
        weights=normal_force_budget*softmax(np.c_[logits,np.zeros(n)],axis=1)[:,:4]
        forces=(columns*weights.reshape(-1,1)).reshape(n,4,3).sum(1)
        residual=matrix@weights.ravel()-target
        vector=np.r_[residual/row_scale,
            np.maximum(gaps,0.)*weights.sum(1)/(gap_scale_m*force_scale),
            np.maximum(-gaps-maximum_penetration_m,0.)/gap_scale_m,
            2*np.maximum(depth-maximum_penetration_m,0.)/gap_scale_m,.03*x[:nv],.01*weights.ravel()/force_scale]
        latest.update(q=q,patches=patches,forces=forces,gaps=gaps,residual=residual,
            max_depth=float(depth.max(initial=0.)),hand_geoms=np.array([g[1] for g in geometry]),weights=weights)
        return vector
    def jacobian(x):
        # Mesh-distance certificates have finite precision. A fixed physical
        # difference scale avoids asking an optimizer for sub-nm derivatives.
        columns=[]
        for k in range(len(x)):
            plus=x.copy();minus=x.copy()
            plus[k]=min(x[k]+.001,bounds[1][k]);minus[k]=max(x[k]-.001,bounds[0][k])
            if plus[k]<=minus[k]:raise ValueError('empty fitting differentiation interval')
            columns.append((evaluate(plus)-evaluate(minus))/(plus[k]-minus[k]))
        return np.stack(columns,axis=1)
    solution=least_squares(evaluate,initial,jac=jacobian,bounds=bounds,max_nfev=max_evaluations,
                           ftol=1e-7,xtol=1e-7,gtol=1e-7)
    evaluate(solution.x)
    active=np.linalg.norm(latest['forces'],axis=1)>force_scale*.01
    feasible=bool(np.linalg.norm(latest['residual']/row_scale)<.1
        and np.all(latest['gaps'][active]<=gap_scale_m)
        and np.all(latest['gaps']>=-maximum_penetration_m-gap_scale_m*.1)
        and latest['max_depth']<=maximum_penetration_m+gap_scale_m*.1)
    return ContactInvariantFit(latest['q'].copy(),latest['patches'],latest['hand_geoms'],
        latest['forces'].copy(),latest['gaps'].copy(),latest['residual'].copy(),
        latest['max_depth'],bool(solution.success),feasible,int(solution.nfev),
        float(solution.cost),str(solution.message),float(normal_force_budget),query.repairs,query.fallbacks)


def scalar_force_preload(model, fit, *, qpos_seed=None):
    """Convert fitted contact loads to ordinary scalar position-PD targets.

    Requires native gear-one affine position actuators. Other actuator types
    remain explicit missing controls; no invented gain or force is supplied.
    Return ctrl targets, generalized load, and unmapped controlled DOFs.
    """
    import mujoco
    d=mujoco.MjData(model);d.qpos[:]=fit.qpos if qpos_seed is None else qpos_seed
    d.qvel[:]=0;mujoco.mj_forward(model,d)
    # Native gravity compensation already cancels this part of the bias force.
    load=d.qfrc_bias-d.qfrc_gravcomp;jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    for geom,patch,force in zip(fit.hand_geom_ids,fit.patches,fit.forces_on_object):
        mujoco.mj_jac(model,d,jp,jr,np.asarray(patch.position),int(model.geom_bodyid[geom]))
        load+=jp.T@force
    controls={};covered=set()
    for a in range(model.nu):
        j=int(model.actuator_trnid[a,0]);gear=float(model.actuator_gear[a,0]);gain=float(model.actuator_gainprm[a,0])
        if (model.actuator_trntype[a]!=mujoco.mjtTrn.mjTRN_JOINT or j<0
                or model.jnt_type[j] not in (2,3) or abs(gear-1)>1e-8 or gain<=0
                or abs(model.actuator_biasprm[a,1]+gain)>1e-5):continue
        dof=int(model.jnt_dofadr[j]);qadr=int(model.jnt_qposadr[j]);tau=float(load[dof])
        if model.actuator_forcelimited[a] and not model.actuator_forcerange[a,0]<=tau<=model.actuator_forcerange[a,1]:
            raise ValueError('required fitted effort exceeds native scalar actuator force limit')
        control=float(d.qpos[qadr]+tau/gain)
        if model.actuator_ctrllimited[a] and not model.actuator_ctrlrange[a,0]<=control<=model.actuator_ctrlrange[a,1]:
            raise ValueError('required fitted preload exceeds native actuator control limit')
        controls[a]=control;covered.add(dof)
    missing=np.flatnonzero((np.abs(load)>1e-6)&~np.isin(np.arange(model.nv),list(covered)))
    return controls,load,missing


def calibrate_native_normal_preload(model, fit, initial_controls, measure, *, iterations=10,
                                     gain=.8, relative_tolerance=.05):
    """Use fresh physical-prefix measurements to correct compliant normal load.

    ``measure(controls)`` must replay the REAL reset/prefix and return WORLD
    force on the object for each fitted patch. It may not teleport the object
    into a grasp. Contact correspondence/changed-mode checks belong to that
    evaluator. Calibration treats normals as minimum loads within the fitted force budget,
    not exact equality targets. It uses normals, not tangential load from a table that
    may presently support the object. A final fresh task replay is required.
    """
    import mujoco
    if iterations<1 or not 0<gain<=1 or relative_tolerance<=0:
        raise ValueError('positive calibration budget, gain and tolerance required')
    d=mujoco.MjData(model);d.qpos[:]=fit.qpos;mujoco.mj_forward(model,d)
    normals=np.asarray([p.normal for p in fit.patches]);desired=np.einsum('ij,ij->i',fit.forces_on_object,normals)
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);jac=[]
    for geom,patch in zip(fit.hand_geom_ids,fit.patches):
        mujoco.mj_jac(model,d,jp,jr,np.asarray(patch.position),int(model.geom_bodyid[geom]));jac.append(jp.copy())
    jac=np.asarray(jac);controls=dict(initial_controls);history=[]
    actuators=[]
    for a in controls:
        j=int(model.actuator_trnid[a,0]);kp=float(model.actuator_gainprm[a,0])
        if (model.actuator_trntype[a]!=mujoco.mjtTrn.mjTRN_JOINT or j<0
                or model.jnt_type[j] not in (2,3) or abs(model.actuator_gear[a,0]-1)>1e-8
                or kp<=0 or abs(model.actuator_biasprm[a,1]+kp)>1e-5):
            raise ValueError('calibration requires verified gear-one scalar position actuators')
        actuators.append((a,int(model.jnt_dofadr[j]),kp))
    converged=False
    for iteration in range(iterations):
        observed=np.asarray(measure(dict(controls)),float)
        if observed.shape!=fit.forces_on_object.shape or not np.isfinite(observed).all():
            raise ValueError('invalid physical contact force measurements')
        actual=np.einsum('ij,ij->i',observed,normals)
        # Allocated normals are minimum required loads, not equality targets.
        # Removing a useful force margin can cross a contact-loss bifurcation.
        error=np.maximum(desired-actual,0.)
        active=desired>max(1e-5,float(desired.max(initial=0.))*.01)
        relative=float(np.max(np.abs(error[active])/np.maximum(desired[active],1e-5),initial=0.))
        history.append(dict(iteration=iteration,measured_normal=actual.tolist(),
                            minimum_desired_normal=desired.tolist(),maximum_relative_deficit=relative))
        if np.any(actual>fit.normal_force_budget+1e-6):
            history[-1]['normal_force_budget_exceeded']=True
            break
        if relative<=relative_tolerance:converged=True;break
        if iteration==iterations-1:break
        tau=np.einsum('nkd,nk->d',jac,error[:,None]*normals)
        for a,dof,kp in actuators:
            controls[a]+=gain*float(tau[dof])/kp
            if model.actuator_ctrllimited[a]:
                controls[a]=float(np.clip(controls[a],*model.actuator_ctrlrange[a]))
    return controls,dict(converged=converged,history=history,
                         acceptance_performed=False,final_fresh_replay_required=True)
