"""Fit native collision contacts using a physical discovery trace, no task edits."""
import argparse,json,time,itertools
import mujoco
from pathlib import Path
import numpy as np
import warp as wp
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv,RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.retargeting.contact_invariant_fit import fit_native_contact_pose,scalar_force_preload
from flash_chord.retargeting.evaluation_task import apply_task_protocol
from flash_chord.retargeting.native_contact_analysis import object_contact_snapshot
from flash_chord.retargeting.rigid_wrench_targets import rigid_contact_wrench_targets
from flash_chord.retargeting.surface_contact_search import search_native_surface_contacts
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovery',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-evaluations',type=int,default=100)
    parser.add_argument('--branches',type=int,default=25)
    parser.add_argument('--surface-search',action='store_true')
    parser.add_argument('--faces-per-direction',type=int,default=2)
    parser.add_argument('--face-spacing',type=float,default=.002)
    parser.add_argument('--demand-seconds',type=float,default=0.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.plan/'manifest.json').read_text())
    cfg=json.loads(read_checkpoint_metadata(manifest['checkpoint'])[CONFIG_METADATA_KEY])
    resolved=apply_task_protocol(cfg,manifest);apply_evaluation_composition(resolved,world_count=1)
    resolved['env']['action']['reference_source']='native_targets'
    config=instantiate_typed(OmegaConf.create(resolved['env']),RLEnvConfig)
    embodiment=instantiate_typed(OmegaConf.create(resolved['embodiment']),Embodiment)
    collision=instantiate_typed(OmegaConf.create(resolved['collision']),CollisionPolicy)
    diagnosis=json.loads((args.discovery.parent/'force_direction_plan_v1/diagnosis.json').read_text())
    world=0;failure=int(diagnosis['failure_frame']);keyframe=int(diagnosis['contact_frame'])
    with np.load(args.discovery/'physics.npz') as z: data={k:z[k].copy() for k in z.files}
    with wp.ScopedDevice('cpu'):
        task,scene=resolved['task'],resolved['scene']
        setup=setup_scene(parquet=task['parquet'],control_fps=config.sim.fps,motion_speed=task['motion_speed'],
            embodiment=embodiment,collision=collision,world_count=1,include_support=scene['support'],
            decompose_objects=scene['decompose_objects'],contact_friction=scene.get('contact_friction',1.),
            object_free_joint_damping=scene.get('object_free_joint_damping',0.),
            robot_reference_path=manifest['reference'],robot_reference_sha256=manifest['reference_sha256'])
        env=RLEnv(setup.scene,setup.reference,config=config);model=env.solver.mj_model
        mujoco.mj_saveModel(model,str(args.output/'native_model.mjb'),None)
        gmap=env.solver.mjc_geom_to_newton_shape.numpy()[0]
        bmap=env.solver.mjc_body_to_newton.numpy()[0]
        snapshot=object_contact_snapshot(data,keyframe,world)
        owner={}
        for hand in setup.scene.layout.hands:
            geoms=np.flatnonzero(np.isin(gmap,hand.collision_shape_ids))
            owner[hand.side]=(hand,geoms,set(int(b) for b in model.geom_bodyid[geoms]))
        scores={side:sum(int(b in bodies) for b in snapshot.counterpart_native_bodies)
                for side,(_,geoms,bodies) in owner.items()}
        side=max(scores,key=scores.get)
        if scores[side]==0:raise ValueError('no embodiment hand owns the observed contacts')
        hand,hand_geoms,bodies=owner[side]
        palm=np.flatnonzero(bmap==hand.palm_frame.body_id)
        if len(palm)!=1:raise ValueError('palm body mapping is ambiguous')
        palm=int(palm[0]);groups=[];ancestors=set()
        for fingertip in hand.fingertip_frames:
            native=np.flatnonzero(bmap==fingertip.body_id)
            if len(native)!=1:raise ValueError('fingertip body mapping is ambiguous')
            body=int(native[0]);chain=set()
            while body>0:
                ancestors.add(body)
                if body==palm:break
                chain.add(body);body=int(model.body_parentid[body])
            group=tuple(int(g) for g in hand_geoms if model.geom_bodyid[g] in chain)
            if group:groups.append(group)
        body=palm
        while body>0:ancestors.add(body);body=int(model.body_parentid[body])
        robot=np.array(sorted(ancestors|bodies),int)
        ids=[];lower=[];upper=[];scales=[];rotations=[]
        seed=data['native_qpos'][keyframe,0].copy()
        for j,body in enumerate(model.jnt_bodyid):
            if body not in ancestors:continue
            kind=int(model.jnt_type[j]);qi=int(model.jnt_qposadr[j])
            if kind==1:rotations.append(qi);continue
            if kind not in (2,3):continue
            radius=(.3*max(.1,(failure-1)/float(data['fps'])) if args.surface_search else .04) if kind==2 else .7
            lo,hi=seed[qi]-radius,seed[qi]+radius
            if model.jnt_limited[j]:
                lo=max(lo,model.jnt_range[j,0]);hi=min(hi,model.jnt_range[j,1])
                if args.surface_search and kind==3:lo,hi=map(float,model.jnt_range[j])
            if hi-lo<1e-7:continue
            ids.append(qi);lower.append(lo);upper.append(hi);scales.append(.02 if kind==2 else .35)
        first_contact=None
        for frame in range(keyframe+1):
            s=object_contact_snapshot(data,frame,world)
            if any(int(body) in bodies for body in s.counterpart_native_bodies):first_contact=frame;break
        rest=max(0,(first_contact if first_contact is not None else keyframe)-1)
        resting=object_contact_snapshot(data,rest,world)
        obj=resting.native_body_id
        joint=int(model.body_jntadr[obj]);qobj=int(model.jnt_qposadr[joint])
        if model.jnt_type[joint]!=0:raise ValueError('current prototype requires a rigid free object')
        seed[qobj:qobj+7]=np.r_[resting.origin_world,resting.quaternion_wxyz]
        poses=data['object_pose'];initial=poses[0,0,0]
        align=Rotation.from_quat(initial[3:7])*Rotation.from_quat(initial[10:14]).inv()
        offset=initial[:3]-align.apply(initial[7:10]);goalp=align.apply(poses[:,0,0,7:10])+offset
        goalr=align*Rotation.from_quat(poses[:,0,0,10:14].copy())
        gravity=data['newton_gravity'][0];mass=float(model.body_mass[obj]);mg=mass*np.linalg.norm(gravity)
        demands=rigid_contact_wrench_targets(goalp,goalr.as_quat(),float(data['fps']),mass,
            model.body_inertia[obj],np.asarray(model.body_iquat[obj])[[1,2,3,0]],model.body_ipos[obj],gravity)
        wrench=demands[failure].copy();wrench[3:]+=np.cross(goalp[failure]-resting.origin_world,wrench[:3])
        # Motion demands, re-expressed in the resting frame a rigid grasp would hold.
        stride=max(1,int(round(.2*float(data['fps']))));count=int(round(args.demand_seconds*float(data['fps'])))
        rest_rotation=Rotation.from_quat(resting.quaternion_wxyz[[1,2,3,0]])
        frames=np.arange(failure,min(failure+count,len(demands)),stride)
        motion=np.empty((len(frames),6))
        for row,frame in enumerate(frames):
            into_rest=rest_rotation*goalr[int(frame)].inv()
            motion[row]=np.r_[into_rest.apply(demands[frame,:3]),into_rest.apply(demands[frame,3:])]
        print(json.dumps(dict(stage='demand',frames=len(frames),peak_force_n=float(np.linalg.norm(motion[:,:3],axis=1).max(initial=0.)),
            peak_torque_nm=float(np.linalg.norm(motion[:,3:],axis=1).max(initial=0.)))),flush=True)
        print(json.dumps(dict(stage='fitting',side=side,scalar_variables=len(ids),rotation_variables=len(rotations)*3,
            contact_chains=len(groups),rest_frame=rest,keyframe=keyframe,desired_wrench=wrench.tolist())),flush=True)
        started=time.monotonic()
        def save_candidate(row,candidate):
            path=args.output/f"candidate_{row['index']:03d}"
            np.savez_compressed(path.with_suffix('.npz'),qpos=candidate.qpos,hand_geom_ids=candidate.hand_geom_ids,
                forces_on_object=candidate.forces_on_object,patch_positions=np.asarray([p.position for p in candidate.patches]),
                patch_normals=np.asarray([p.normal for p in candidate.patches]))
            path.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n')
        if args.surface_search:
            fit,search_report=search_native_surface_contacts(model,seed,ids,lower,upper,scales,groups,robot,obj,wrench,
                rotation_qpos_starts=rotations,maximum_fits=args.branches,max_evaluations=args.max_evaluations,
                progress=lambda row:print(json.dumps(row),flush=True),on_candidate=save_candidate,
                faces_per_direction=args.faces_per_direction,face_spacing_m=args.face_spacing,demand_wrenches=motion)
            label='force_feasible_surface_anchors'
            (args.output/'attempts.json').write_text(json.dumps(search_report,indent=2)+'\n')
        else:
            # Deterministic object-axis orientations supply alternative contact
            # arrangements when the local force/geometry fit gets stuck. Every
            # branch still solves the SAME constrained physical objective.
            query=mujoco.MjData(model);query.qpos[:]=seed;mujoco.mj_kinematics(model,query)
            old_palm=Rotation.from_matrix(query.xmat[palm].reshape(3,3).copy())
            old_position=query.xpos[palm].copy()
            pivot=np.mean([p.position for p,b in zip(snapshot.patches,snapshot.counterpart_native_bodies)
                           if int(b) in bodies],axis=0)
            object_axes=Rotation.from_quat(resting.quaternion_wxyz[[1,2,3,0]])*Rotation.from_quat(np.asarray(model.body_iquat[obj])[[1,2,3,0]])
            orientation=[]
            for perm in itertools.permutations(range(3)):
                for sign in itertools.product((-1.,1.),repeat=3):
                    matrix=np.eye(3)[:,perm]*np.asarray(sign)[None,:]
                    if np.linalg.det(matrix)<0:continue
                    desired=object_axes*Rotation.from_matrix(matrix)
                    delta=desired*old_palm.inv()
                    orientation.append((float(delta.magnitude()),delta))
            orientation.sort(key=lambda item:item[0])
            seeds=[('original',seed.copy())]
            if len(rotations)==1:
                qi=rotations[0];ball_old=Rotation.from_quat(seed[qi:qi+4][[1,2,3,0]])
                parent_fixed=old_palm*ball_old.inv()
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mujoco.mj_comPos(model,query);mujoco.mj_jac(model,query,jp,jr,old_position,palm)
                slides=[j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in ancestors and model.jnt_type[j]==2]
                slide_q=np.array([model.jnt_qposadr[j] for j in slides],int)
                slide_d=np.array([model.jnt_dofadr[j] for j in slides],int)
                for branch,(angle,delta) in enumerate(orientation):
                    variant=seed.copy()
                    moved=(parent_fixed.inv()*delta*old_palm).as_quat()
                    variant[qi:qi+4]=moved[[3,0,1,2]]
                    position=pivot+delta.apply(old_position-pivot)
                    variant[slide_q]+=np.linalg.lstsq(jp[:,slide_d],position-old_position,rcond=None)[0]
                    seeds.append((f'object_axes_{branch}',variant))
            attempts=[];best=None
            widths=np.asarray(upper)-np.asarray(lower)
            for label,guess in seeds[:args.branches]:
                lo=np.maximum(np.asarray(lower),guess[ids]-.5*widths)
                hi=np.minimum(np.asarray(upper),guess[ids]+.5*widths)
                # Translation alternatives retain a local 4 cm fitting region;
                # selected references still require full-path physical replay.
                for k,qi in enumerate(ids):
                    j=int(np.flatnonzero(model.jnt_qposadr==qi)[0])
                    if model.jnt_type[j]==2:lo[k]=guess[qi]-.04;hi[k]=guess[qi]+.04
                candidate=fit_native_contact_pose(model,guess,ids,lo,hi,scales,groups,robot,obj,wrench,
                    normal_force_budget=8*mg,max_evaluations=args.max_evaluations,rotation_qpos_starts=rotations)
                row=dict(branch=label,model_feasible=candidate.model_feasible,cost=candidate.cost,
                    maximum_penetration=candidate.max_collision_penetration,wrench_residual=candidate.wrench_residual.tolist(),
                    distance_repairs=candidate.distance_repairs,distance_fallbacks=candidate.distance_fallbacks)
                attempts.append(row);print(json.dumps(row),flush=True)
                if best is None or (not candidate.model_feasible,candidate.cost)<(not best[1].model_feasible,best[1].cost):
                    best=(label,candidate)
            label,fit=best
            (args.output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
        controls,load,missing=scalar_force_preload(model,fit)
        report=dict(selected_branch=label,side=side,rest_frame=rest,keyframe=keyframe,failure_frame=failure,
            converged=fit.converged,model_feasible=fit.model_feasible,
            contact_gaps=fit.contact_gaps.tolist(),wrench_residual=fit.wrench_residual.tolist(),
            maximum_collision_penetration=fit.max_collision_penetration,
            forces_on_object=fit.forces_on_object.tolist(),cost=fit.cost,evaluations=fit.evaluations,
            elapsed_seconds=time.monotonic()-started,scalar_pd_targets=controls,
            unmapped_generalized_load_dofs=missing.tolist(),physical_validation_performed=False,
            no_task_or_object_physics_edits=True,offline_geometry_query_only=True,
            prototype_rigid_object_only=True,rotation_qpos_starts=rotations)
        np.savez_compressed(args.output/'fit.npz',qpos=fit.qpos,seed_qpos=seed,
            hand_geom_ids=fit.hand_geom_ids,forces_on_object=fit.forces_on_object,
            scalar_qpos_ids=np.array(ids),patch_positions=np.array([p.position for p in fit.patches]),
            patch_normals=np.array([p.normal for p in fit.patches]),
            object_anchor=np.r_[resting.origin_world,resting.quaternion_wxyz])
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
