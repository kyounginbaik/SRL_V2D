"""Deterministic native-surface contact proposals ranked by force and reach.

This module selects geometric contact arrangements, not named skill sequences.
It currently requires convex native mesh collision geoms and material-point
contacts; unsupported geometry is reported rather than silently approximated.
"""
import itertools
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .convex_distance import ConvexDistanceQuery
from .contact_wrench import ContactPatch,solve_contact_wrench,contact_wrench_support,_wrench_columns
from .surface_anchor_fit import SurfaceAnchor,fit_native_surface_anchors,closest_triangle_point


def _faces(model,data,query,geom):
    if geom not in query.local:raise ValueError('surface proposals currently require native convex mesh geoms')
    v=query.local[geom];hull=ConvexHull(v);r=data.geom_xmat[geom].reshape(3,3);p=data.geom_xpos[geom]
    result=[]
    for face,equation in zip(hull.simplices,hull.equations):
        tri=v[face];area=.5*np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
        if area<1e-10:continue
        local=tri.mean(0);normal=equation[:3]
        result.append(dict(geom=int(geom),local=local,local_normal=normal,
                           point=p+r@local,outward=r@normal,area=area,triangle=tuple(map(tuple,tri@r.T+p))))
    # Planar facet centers provide physically useful zero-moment contact
    # candidates instead of making every square face inherit a triangle bias.
    planes={}
    for face,eq in zip(hull.simplices,hull.equations):
        key=tuple(np.round(eq/1e-7).astype(np.int64));planes.setdefault(key,[]).append(face)
    centers=[]
    for key,faces in planes.items():
        vertices=np.unique(np.asarray(faces).ravel());local=v[vertices].mean(0)
        triangles=[v[face] for face in faces]
        triangle=min(triangles,key=lambda tri:np.linalg.norm(closest_triangle_point(local,tri)-local))
        normal=np.asarray(key[:3],float);normal/=np.linalg.norm(normal)
        centers.append(dict(geom=int(geom),local=local,local_normal=normal,point=p+r@local,
            outward=r@normal,area=sum(.5*np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0])) for t in triangles),
            triangle=tuple(map(tuple,triangle@r.T+p))))
    return centers+result


def demand_coverage(patches,limits,minima,demands,center):
    """Fraction of motion wrenches this rigidly held contact set can supply."""
    if not len(demands):return 1.
    return float(np.mean([solve_contact_wrench(patches,demand,center=center,normal_force_limits=limits,
                                               normal_force_minima=minima).feasible for demand in demands]))


def search_native_surface_contacts(model,seed_qpos,qpos_ids,lower,upper,scales,
        contact_geom_groups,robot_body_ids,object_body_id,desired_wrench,*,
        rotation_qpos_starts=(),maximum_fits=12,max_evaluations=100,require_internal_grip=True,
        progress=None,on_candidate=None,faces_per_direction=2,face_spacing_m=.002,demand_wrenches=(),
        demand_ranked=400):
    """Find force-feasible object faces, assign robot surfaces, then fit pose.

    Self-balanced minimum grip forces reject one-sided pressure without
    changing the required net object wrench. A full directional feasibility
    check remains distinct from a six-dimensional span-rank test. demand_wrenches
    are object-rest-frame motion demands; arrangements supporting more of them
    rank first, because one frame's feasibility says little about a carry. Contacts are
    surface material points on actual native colliders, not FK fingertip sites.
    Final physical-prefix execution and task metrics are mandatory downstream.
    """
    import mujoco
    seed=np.asarray(seed_qpos,float);d=mujoco.MjData(model);d.qpos[:]=seed
    mujoco.mj_kinematics(model,d);query=ConvexDistanceQuery(model);query.update(d)
    groups=tuple(tuple(group) for group in contact_geom_groups)
    if not groups:raise ValueError('at least one robot collision-surface chain required')
    contact_count=3 if require_internal_grip else 1
    object_geoms=np.flatnonzero(model.geom_bodyid==object_body_id)
    com=d.xpos[object_body_id]+d.xmat[object_body_id].reshape(3,3)@model.body_ipos[object_body_id]
    object_faces=[]
    for geom in object_geoms:
        for face in _faces(model,d,query,int(geom)):
            # Reject decomposition-interior faces by testing a point just
            # outside that face against every other object hull.
            outside=face['point']+1e-5*face['outward'];hidden=False
            for other in object_geoms:
                if other==geom:continue
                r=d.geom_xmat[other].reshape(3,3);local=r.T@(outside-d.geom_xpos[other])
                eq=query.hulls[int(other)]
                if np.max(eq[:,:3]@local+eq[:,3])<-1e-6:hidden=True;break
            if not hidden:object_faces.append(face)
    if not object_faces:raise ValueError('no exposed object surface faces found')
    # A digit cannot reach a face looking down at a support closer than the
    # thinnest contact link. Applies only when the object rests on static geometry.
    up=-np.asarray(model.opt.gravity,float);up/=np.linalg.norm(up)
    mujoco.mj_collision(model,d)
    resting=any(object_body_id in model.geom_bodyid[c.geom] and 0 in model.body_weldid[model.geom_bodyid[c.geom]] for c in d.contact[:d.ncon])
    low=min(float((d.geom_xpos[g]+query.local[int(g)]@d.geom_xmat[g].reshape(3,3).T).dot(up).min()) for g in object_geoms)
    thickness=min(float(np.ptp(query.local[int(g)],axis=0).min()) for group in groups for g in group if int(g) in query.local)
    exposed=len(object_faces)
    if resting:object_faces=[f for f in object_faces if not (f['outward']@up<-.5 and f['point']@up-low<thickness)]
    blocked=exposed-len(object_faces)
    material=[sum((_faces(model,d,query,int(g)) for g in group),[]) for group in groups]
    for group,geoms in enumerate(groups):
        # Vertices and edges have a normal cone, not one compulsory face
        # normal. They are essential contact features on curved finger meshes.
        for geom in geoms:
            vertices=query.local[int(geom)];hull=ConvexHull(vertices)
            r=d.geom_xmat[geom].reshape(3,3);position=d.geom_xpos[geom]
            for vertex in hull.vertices:
                adjacent=(hull.simplices==vertex).any(1)
                normal=hull.equations[adjacent,:3].sum(0);normal/=np.linalg.norm(normal)
                local=vertices[vertex]
                material[group].append(dict(geom=int(geom),local=local,local_normal=normal,
                    point=position+r@local,outward=r@normal,area=0.))
    for group,features in enumerate(material):
        for feature in features:feature['chain']=group
    all_material=sum(material,[])
    if any(not group for group in material):raise ValueError('empty robot collision-surface chain')
    object_frame=Rotation.from_matrix(d.xmat[object_body_id].reshape(3,3))*Rotation.from_quat(model.body_iquat[object_body_id][[1,2,3,0]])
    directions=np.r_[object_frame.as_matrix().T,-object_frame.as_matrix().T]
    selected=[]
    for direction in directions:
        choices=[face for face in object_faces if -face['outward']@direction>.6]
        choices.sort(key=lambda face:np.linalg.norm(face['point']-com)/.04+2*(1+face['outward']@direction))
        count=0
        for face in choices:
            if any(np.linalg.norm(face['point']-old['point'])<face_spacing_m and face['outward']@old['outward']>.95 for old in selected):continue
            selected.append(face);count+=1
            if count==faces_per_direction:break
    mg=max(.01,float(model.body_mass[object_body_id]*np.linalg.norm(model.opt.gravity)))
    budget=8*mg;minimum=mg if require_internal_grip else 0.
    minimum_mu=min(float(model.geom_friction[g,0]) for group in groups for g in group)
    minimum_mu=min(minimum_mu,min(float(model.geom_friction[g,0]) for g in object_geoms))
    mu=.7*minimum_mu
    ranked=[]
    for faces in itertools.combinations(selected,contact_count):
        patches=[]
        for index,face in enumerate(faces):
            n=-face['outward'];axis=np.eye(3)[np.argmin(np.abs(n))];t1=np.cross(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
            patches.append(ContactPatch(tuple(face['point']),tuple(n),tuple(t1),tuple(t2),(mu,mu),index))
        _,matrix,_=_wrench_columns(patches,d.xpos[object_body_id])
        if require_internal_grip and np.linalg.matrix_rank(matrix/np.r_[np.ones(3),np.full(3,.05)][:,None],tol=1e-9)<6:continue
        limits={i:budget for i in range(contact_count)};minima={i:minimum for i in range(contact_count)}
        if require_internal_grip:
            internal=solve_contact_wrench(patches,np.zeros(6),center=d.xpos[object_body_id],
                normal_force_limits=limits,normal_force_minima=minima)
            if not internal.feasible:continue
        load=solve_contact_wrench(patches,desired_wrench,center=d.xpos[object_body_id],
            normal_force_limits=limits,normal_force_minima=minima)
        if not load.feasible:continue
        # Assign actual material points, allowing multiple distinct points on
        # one digit. Requiring three different fingers would exclude ordinary
        # two-digit grasps with a finite contact patch.
        hand_points=np.asarray([f['point'] for f in all_material]);hand_normals=np.asarray([f['outward'] for f in all_material])
        costs=np.array([np.linalg.norm(hand_points-face['point'],axis=1)/.04+
                        .5*(1-hand_normals@np.asarray(patch.normal)) for face,patch in zip(faces,patches)])
        # Preserve distinct physical material assignments. A single cheapest
        # unconstrained assignment can put every contact on one finger and
        # starve a feasible two-finger patch of any optimization budget.
        assignments=[]
        rows,free_assignment=linear_sum_assignment(costs)
        assignments.append(tuple(map(int,free_assignment)))
        feature_groups=[np.array([k for k,f in enumerate(all_material) if f['chain']==group],int) for group in range(len(groups))]
        if contact_count==3:
            # One candidate with three different chains.
            reduced=np.array([[costs[i,idx].min() for idx in feature_groups] for i in range(3)])
            if len(groups)>=3:
                _,choices=linear_sum_assignment(reduced)
                assignments.append(tuple(int(feature_groups[g][np.argmin(costs[i,feature_groups[g]])]) for i,g in enumerate(choices)))
            # Three alternatives cover which material contact is on the second
            # chain; the other two can share a finite patch on the first.
            for singleton in range(3):
                repeated=[i for i in range(3) if i!=singleton];best_assignment=None
                for primary,indices in enumerate(feature_groups):
                    if len(indices)<2:continue
                    rr,cc=linear_sum_assignment(costs[np.ix_(repeated,indices)])
                    chosen={repeated[int(r)]:int(indices[c]) for r,c in zip(rr,cc)}
                    for secondary,other in enumerate(feature_groups):
                        if secondary==primary:continue
                        feature_index=int(other[np.argmin(costs[singleton,other])])
                        combined=dict(chosen);combined[singleton]=feature_index
                        score=sum(costs[i,combined[i]] for i in range(3))
                        candidate=tuple(combined[i] for i in range(3))
                        if best_assignment is None or score<best_assignment[0]:best_assignment=(score,candidate)
                if best_assignment is not None:assignments.append(best_assignment[1])
        for assignment in dict.fromkeys(assignments):
            assigned=[]
            for i,index in enumerate(assignment):
                feature=all_material[index];patch=replace(patches[i],group=feature['chain'])
                assigned.append(SurfaceAnchor(feature['geom'],tuple(feature['local']),tuple(feature['local_normal']),patch,faces[i]['triangle']))
            actual_limits={a.patch.group:budget for a in assigned};actual_minima={a.patch.group:minimum for a in assigned}
            actual_patches=tuple(a.patch for a in assigned)
            actual_load=solve_contact_wrench(actual_patches,desired_wrench,center=d.xpos[object_body_id],
                normal_force_limits=actual_limits,normal_force_minima=actual_minima)
            if not actual_load.feasible:continue
            score=float(sum(costs[i,index] for i,index in enumerate(assignment))+.03*sum(actual_load.normal_force_per_group.values())/mg)
            ranked.append((score,assigned,actual_load,actual_limits,actual_minima))
    ranked.sort(key=lambda item:item[0]);attempts=[];best=None
    coverage=[demand_coverage(tuple(a.patch for a in item[1]),item[3],item[4],demand_wrenches,d.xpos[object_body_id]) for item in ranked[:demand_ranked]]
    coverage+=[0.]*(len(ranked)-len(coverage)) if len(demand_wrenches) else [1.]*(len(ranked)-len(coverage))
    order=sorted(range(len(ranked)),key=lambda i:(-round(coverage[i],2),ranked[i][0]))
    ranked=[ranked[i]+(coverage[i],) for i in order]
    buckets={}
    for item in ranked:
        count=len(set(a.patch.group for a in item[1]));buckets.setdefault(count,[]).append(item)
    scheduled=[]
    for depth in range(maximum_fits):
        for count in sorted(buckets,reverse=True):
            if depth<len(buckets[count]):scheduled.append(buckets[count][depth])
        if len(scheduled)>=maximum_fits:break
    for index,(score,anchors,load,limits,minima,covered) in enumerate(scheduled[:maximum_fits]):
        patches=tuple(a.patch for a in anchors)
        if require_internal_grip:
            support=contact_wrench_support(patches,np.zeros(6),mg*np.r_[np.full(3,.1),np.full(3,.005)],
                center=d.xpos[object_body_id],normal_force_limits=limits)
            if not support['all_directional_probes_feasible']:
                attempts.append(dict(index=index,reason='directional_wrench_probe_failed'));continue
        guess=seed.copy();alignment_angle=0.
        if len(rotation_qpos_starts)==1:
            # A rigid registration supplies a useful floating-wrist seed;
            # finger IK then resolves the remaining nonrigid mismatch. This
            # uses the selected surface correspondence, not random rotation.
            hp=np.array([d.geom_xpos[a.hand_geom]+d.geom_xmat[a.hand_geom].reshape(3,3)@np.asarray(a.point_geom_local) for a in anchors])
            hn=np.array([d.geom_xmat[a.hand_geom].reshape(3,3)@np.asarray(a.normal_geom_local) for a in anchors])
            op=np.array([a.patch.position for a in anchors]);on=np.array([a.patch.normal for a in anchors])
            hc,oc=hp.mean(0),op.mean(0)
            delta,_=Rotation.align_vectors(np.r_[op-oc,.02*on],np.r_[hp-hc,.02*hn])
            alignment_angle=float(delta.magnitude())
            qi=int(rotation_qpos_starts[0]);j=int(np.flatnonzero(model.jnt_qposadr==qi)[0]);body=int(model.jnt_bodyid[j])
            old_rotation=Rotation.from_matrix(d.xmat[body].reshape(3,3).copy())
            ball=Rotation.from_quat(seed[qi:qi+4][[1,2,3,0]]);fixed=old_rotation*ball.inv()
            moved=(fixed.inv()*delta*old_rotation).as_quat();guess[qi:qi+4]=moved[[3,0,1,2]]
            slides=[j for j in range(model.njnt) if model.jnt_type[j]==2 and int(model.jnt_qposadr[j]) in qpos_ids]
            if slides:
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_comPos(model,d)
                mujoco.mj_jac(model,d,jp,jr,d.xpos[body].copy(),body)
                qids=np.array([model.jnt_qposadr[j] for j in slides],int);dids=np.array([model.jnt_dofadr[j] for j in slides],int)
                desired=oc+delta.apply(d.xpos[body]-hc)
                guess[qids]+=np.linalg.lstsq(jp[:,dids],desired-d.xpos[body],rcond=None)[0]
        fit=fit_native_surface_anchors(model,guess,qpos_ids,lower,upper,scales,anchors,robot_body_ids,
            object_body_id,desired_wrench,limits,rotation_qpos_starts=rotation_qpos_starts,
            normal_force_minima=minima,max_evaluations=max_evaluations)
        if require_internal_grip and fit.model_feasible:
            supported=contact_wrench_support(fit.patches,np.zeros(6),mg*np.r_[np.full(3,.1),np.full(3,.005)],
                center=d.xpos[object_body_id],normal_force_limits=limits)
            if not supported['all_directional_probes_feasible']:
                fit=replace(fit,model_feasible=False,message='fitted surface cloud lost directional wrench support')
        row=dict(index=index,rank_cost=score,model_feasible=fit.model_feasible,cost=fit.cost,
            maximum_material_point_error_m=float(fit.contact_gaps.max()),
            maximum_collision_penetration_m=fit.max_collision_penetration,independent_chains=len(set(a.patch.group for a in anchors)),
            initial_rigid_alignment_angle_rad=alignment_angle,demand_coverage=covered)
        attempts.append(row)
        if on_candidate is not None:on_candidate(row,fit)
        if progress is not None:progress(row)
        if best is None or (not fit.model_feasible,fit.cost)<(not best.model_feasible,best.cost):best=fit
    if best is None:raise ValueError('no force-feasible and directionally supported object-surface contact set')
    return best,dict(exposed_object_faces=exposed,faces_blocked_by_support=blocked,selected_surface_features=len(selected),
        best_demand_coverage=max((item[5] for item in ranked),default=0.),
        nominally_force_feasible_arrangements=len(ranked),attempts=attempts,
        require_internal_grip=require_internal_grip,contact_material_model=f'{contact_count} actual convex-mesh surface points; chain reuse allowed',
        physical_validation_performed=False)
