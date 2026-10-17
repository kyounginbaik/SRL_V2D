#!/usr/bin/env python3
"""Generate geometric full-finger grasp priors; never certify physical success.

Source contact labels select intervals. Opposing surface points guide bounded
28-DOF hand IK; a chosen hand/object transform and finger shape are maintained
inside that interval. All object targets, timings and initial states stay exact.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing

B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from tool_library.common import tracking as TR
from tool_library.common import hand as H

FINGERS=['thumb','index','middle','ring','pinky']


def nearest_euler(e,previous):
    e=np.asarray(e,float);previous=np.asarray(previous,float)
    candidates=[e,np.array([e[0]+np.pi,np.pi-e[1],e[2]+np.pi])]
    candidates=[c+2*np.pi*np.round((previous-c)/(2*np.pi)) for c in candidates]
    return min(candidates,key=lambda c:np.linalg.norm(c-previous))


def intervals(mask,min_frames=8):
    padded=np.r_[False,mask,False].astype(int);starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
    return [(int(a),int(b-1)) for a,b in zip(starts,ends) if b-a>=min_frames]


def surface(path):
    mesh=trimesh.load(path,process=False)
    if mesh.volume<0:mesh.invert()
    points,faces=trimesh.sample.sample_surface(mesh,24000,seed=43)
    return points,mesh.face_normals[faces],cKDTree(points)


def opposing_targets(tips,points,normals,tree):
    """Find an outer-surface opposing thumb/finger pair near the given hand."""
    thumb_ids=np.asarray(tree.query(tips[0],k=800)[1]);best=None
    for finger in [1,2,3]:
        ids=np.asarray(tree.query(tips[finger],k=1500)[1])
        a=points[thumb_ids];b=points[ids]
        dots=normals[thumb_ids]@normals[ids].T
        sep=np.linalg.norm(a[:,None]-b[None],axis=-1)
        # A pair must oppose across the object, not the two faces of a thin wall.
        normal_along=np.einsum('ijk,ik->ij',a[:,None]-b[None],normals[thumb_ids])
        other_normal_along=-np.einsum('ijk,jk->ij',a[:,None]-b[None],normals[ids])
        # Both contact-to-contact directions must fit inside the friction
        # cones (mu=1 in the fixed source scene). A mere negative normal dot
        # product permits a tangential pinch that cannot sustain an object.
        valid=(dots<-.7)&(sep>.009)&(sep<.14)&(normal_along>np.sqrt(.5)*sep)&(other_normal_along>np.sqrt(.5)*sep)
        cost=np.linalg.norm(a-tips[0],axis=-1)[:,None]**2+np.linalg.norm(b-tips[finger],axis=-1)[None]**2
        cost[~valid]=np.inf
        i,j=np.unravel_index(np.argmin(cost),cost.shape)
        if np.isfinite(cost[i,j]) and (best is None or cost[i,j]<best[0]):
            best=(float(cost[i,j]),finger,int(thumb_ids[i]),int(ids[j]))
    if best is None:return None
    _,finger,ti,fi=best
    target={0:points[ti]+.0015*normals[ti],finger:points[fi]+.0015*normals[fi]}
    # Another nearby nonthumb finger may join on the same side, without forcing
    # ring/pinky pads to touch a narrow spoon that cannot accommodate them.
    for other in []:
        if other==finger:continue
        ids=np.asarray(tree.query(tips[other],k=150)[1]);valid=(normals[ids]@normals[fi]>.3)
        if np.any(valid):
            choices=ids[valid];distance=np.linalg.norm(points[choices]-tips[other],axis=-1);k=choices[np.argmin(distance)]
            if distance.min()<.025:target[other]=points[k]+.0015*normals[k]
    return target,{'opposing_fingers':['thumb',FINGERS[finger]],'normal_dot':float(normals[ti]@normals[fi]),
                   'contact_separation_m':float(np.linalg.norm(points[ti]-points[fi]))}


def bounded_ik(model,data,rig,q0,targets,target_body):
    cols=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]+list(rig.finger_qpos.values()))
    jointids=np.array([int(model.actuator_trnid[rig.wrist_act[k],0]) for k in ['x','y','z','rx','ry','rz']]+
                      [int(model.actuator_trnid[rig.finger_act[k],0]) for k in rig.finger_qpos])
    dofs=model.jnt_dofadr[jointids];limits=model.jnt_range[jointids]
    initial=q0[cols].copy();span=np.r_[np.full(3,.06),np.full(3,1.0),np.full(22,1.4)]
    lo=np.maximum(limits[:,0],initial-span);hi=np.minimum(limits[:,1],initial+span)
    scale=np.r_[np.full(3,.035),np.full(3,.4),np.full(22,.8)]
    sites=[rig.fingertip_sites[FINGERS[k]] for k in targets];desired=np.stack(list(targets.values()))
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    self_pairs={}
    for g1,g2 in zip(model.pair_geom1,model.pair_geom2):
        if int(model.geom_bodyid[g1]) in rig.body_ids and int(model.geom_bodyid[g2]) in rig.body_ids:
            key=tuple(sorted((int(g1),int(g2))))
            if key not in self_pairs:self_pairs[key]=len(self_pairs)
    last_contacts={}
    external_rows={(g,k):i for i,(g,k) in enumerate((g,k) for g in rig.collision_gids for k in ['table','target','other'])}
    external_contacts={}
    allowances={'table':0.,'target':.0015,'other':.001}
    def place(x):
        data.qpos[:]=q0;data.qpos[cols]=x;mujoco.mj_fwdPosition(model,data)
        last_contacts.clear();external_contacts.clear()
        for c in data.contact:
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            key=tuple(sorted((g1,g2)))
            if key in self_pairs and c.dist<-.001:
                if key not in last_contacts or c.dist<last_contacts[key][0]:
                    last_contacts[key]=(float(c.dist),int(c.geom1),int(c.geom2),c.pos.copy(),c.frame[:3].copy())
            if (b1 in rig.body_ids)==(b2 in rig.body_ids):continue
            handgeom,other=(g1,b2) if b1 in rig.body_ids else (g2,b1)
            kind='table' if model.body(other).name=='table' else 'target' if other==target_body else 'other'
            key=(handgeom,kind)
            if key in external_rows and c.dist < -allowances[kind]:
                if key not in external_contacts or c.dist<external_contacts[key][0]:
                    external_contacts[key]=(float(c.dist),g1,g2,c.pos.copy(),c.frame[:3].copy())
    def fun(x):
        place(x)
        pen=np.zeros(len(self_pairs))
        for key,(dist,*_) in last_contacts.items():pen[self_pairs[key]]=(-dist-.001)/.002
        external_pen=np.zeros(len(external_rows))
        for key,(dist,*_) in external_contacts.items():external_pen[external_rows[key]]=(-dist-allowances[key[1]])/.0015
        return np.r_[(data.site_xpos[sites]-desired).ravel()/.004,.12*(x-initial)/scale,pen,external_pen]
    def jac(x):
        place(x);rows=[]
        for sid in sites:
            mujoco.mj_jacSite(model,data,jp,jr,sid);rows.append(jp[:,dofs]/.004)
        penalty_jac=np.zeros((len(self_pairs),len(cols)))
        for key,(dist,g1,g2,pos,normal) in last_contacts.items():
            mujoco.mj_jac(model,data,jp,jr,pos,int(model.geom_bodyid[g1]));a=jp[:,dofs].copy()
            mujoco.mj_jac(model,data,jp,jr,pos,int(model.geom_bodyid[g2]));b=jp[:,dofs].copy()
            penalty_jac[self_pairs[key]]=-(normal@(b-a))/.002
        external_jac=np.zeros((len(external_rows),len(cols)))
        for key,(dist,g1,g2,pos,normal) in external_contacts.items():
            mujoco.mj_jac(model,data,jp,jr,pos,int(model.geom_bodyid[g1]));a=jp[:,dofs].copy()
            mujoco.mj_jac(model,data,jp,jr,pos,int(model.geom_bodyid[g2]));b=jp[:,dofs].copy()
            external_jac[external_rows[key]]=-(normal@(b-a))/.0015
        return np.vstack([*rows,.12*np.diag(1/scale),penalty_jac,external_jac])
    sol=least_squares(fun,np.clip(initial,lo+1e-8,hi-1e-8),jac=jac,bounds=(lo,hi),max_nfev=120,
                      ftol=1e-6,gtol=1e-6,xtol=1e-6)
    place(sol.x);q=data.qpos.copy();res=np.linalg.norm(data.site_xpos[sites]-desired,axis=-1)
    return q,{'target_residual_m':res.tolist(),'max_target_residual_m':float(res.max()),'nfev':sol.nfev,
              'wrist_translation_change_m':float(np.linalg.norm(sol.x[:3]-initial[:3])),
              'wrist_euler_change_rad':float(np.linalg.norm(sol.x[3:6]-initial[3:6])),
              'finger_change_rms_rad':float(np.sqrt(np.mean((sol.x[6:]-initial[6:])**2)))}


def main(episode,mode='rigid_grasp'):
    ep=f'episode_{episode:06d}';e=B/'episodes'/ep;out=B/'refinement_20261017/geometry_v6'/ep;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'challenge_reconstruction/final_recon'/ep
    row=pq.read_table(source/'reference.parquet').to_pylist()[0]
    # Frozen physical scene provides exactly the contacts used in the pilot.
    scene=B/'runs/pilot_v2'/ep/'seed_44/baseline/scene_physics.xml'
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    ref=TR.load_reference(e/'tracking_reference.npz',model);Q=ref['qpos'].copy();fps=ref['fps'];N=len(Q)
    repaired_euler_samples=0
    for side in ['left','right']:
        cols=[int(model.joint(f'robot_{side}_{side}_rot_{axis}').qposadr[0]) for axis in 'xyz']
        for t in range(1,N):
            fixed=nearest_euler(Q[t,cols],Q[t-1,cols]);repaired_euler_samples+=int(np.max(abs(fixed-Q[t,cols]))>1e-6)
            Q[t,cols]=fixed
    original=Q.copy();result=Q.copy()
    from floating_runtime import servo_rollout
    passive=servo_rollout(model,np.repeat(Q[:1],round(3.5*fps),axis=0),fps,qpos_start=Q[0],record_substeps=False)
    if np.max(passive['hand_object_force_n'])>1e-6:
        raise ValueError('passive forecast was disturbed by hand/object contact')
    np.savez_compressed(out/'passive_forecast.npz',qpos=passive['qpos'],time_s=passive['time_s'],hand_object_force_n=passive['hand_object_force_n'])
    names=np.load(e/'reference_trajectory.npz')['object_names'].tolist()
    meshes={name:surface(source/'assets'/f'{name}.obj') for name in names}
    reports=[];seed_poses=[];role_notes=[]
    for side in ['left','right']:
        if side!='right':continue
        rig=H.HandRig.build(model,f'robot_{side}_',side)
        labels=np.array(row[f'mano_{side}_object_contact_part_ids']);valid=np.array(row.get(f'mano_{side}_valid',[True]*N))
        evidence=np.array([np.count_nonzero(labels==k+1) for k in range(len(names))])
        ranking=np.argsort(evidence)[::-1]
        dominant=int(ranking[0]) if evidence[ranking[0]]>2*max(1,evidence[ranking[1]]) else None
        role_notes.append({'side':side,'contact_label_counts':dict(zip(names,evidence.tolist())),
                           'dominant_object':names[dominant] if dominant is not None else None,
                           'scope':'minor proximity labels on other objects are not promoted to separate grasp intentions'})
        qcols=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]+list(rig.finger_qpos.values()))
        for objindex,name in enumerate(names):
            if dominant is not None and objindex!=dominant:continue
            contact_count=np.sum(labels==objindex+1,axis=1)
            mask=binary_closing((contact_count>=2)&valid,structure=np.ones(7,dtype=bool))
            for start,end in intervals(mask):
                bid=model.body(name).id;points,normals,tree=meshes[name]
                close_frame=min(end,start+round(.30*fps));approach_begin=max(1,start-round(.1*fps))
                if close_frame>=len(passive['qpos']):continue
                # Predict where the unactuated object actually rests. This
                # temporary object path is used only for hand IK; exported
                # object targets remain bitwise identical to the source.
                oc=int(model.joint(name+'_free').qposadr[0]);settled=passive['qpos'][close_frame,oc:oc+7]
                Q=original.copy();times=np.arange(N)/fps
                fade=np.clip((times-(close_frame/fps+.15))/1.0,0.,1.);fade=fade*fade*(3-2*fade)
                settledR=Rotation.from_quat(settled[3:],scalar_first=True)
                for ti in range(N):
                    goalR=Rotation.from_quat(original[ti,oc+3:oc+7],scalar_first=True)
                    Q[ti,oc:oc+3]=(1-fade[ti])*settled[:3]+fade[ti]*original[ti,oc:oc+3]
                    Q[ti,oc+3:oc+7]=(settledR*Rotation.from_rotvec((settledR.inv()*goalR).as_rotvec()*fade[ti])).as_quat(scalar_first=True)
                # Several early actual contact frames give different finger shapes.
                frames=np.unique(np.linspace(close_frame,min(end,close_frame+int(fps*.35)),8).astype(int))
                candidates=[]
                for frame in frames:
                    data.qpos[:]=Q[frame];mujoco.mj_kinematics(model,data)
                    object_pos=data.xpos[bid].copy();object_rot=data.xmat[bid].reshape(3,3).copy()
                    sites=[rig.fingertip_sites[f] for f in FINGERS]
                    tips_local=(data.site_xpos[sites]-object_pos)@object_rot
                    before=tree.query(tips_local)[0]
                    chosen=opposing_targets(tips_local,points,normals,tree)
                    if chosen is None:continue
                    local_targets,opposition=chosen
                    world_targets={k:object_rot@v+object_pos for k,v in local_targets.items()}
                    q,ik=bounded_ik(model,data,rig,Q[frame],world_targets,bid)
                    mujoco.mj_forward(model,data)
                    worst=0.;self_pen=0.;floor_pen=0.;other_pen=0.
                    for c in data.contact:
                        g1,g2=int(c.geom1),int(c.geom2)
                        b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
                        if (g1 in rig.collision_gids and b2==bid) or (g2 in rig.collision_gids and b1==bid):
                            worst=max(worst,-float(c.dist))
                        if b1 in rig.body_ids and b2 in rig.body_ids:self_pen=max(self_pen,-float(c.dist))
                        elif b1 in rig.body_ids or b2 in rig.body_ids:
                            other=b2 if b1 in rig.body_ids else b1
                            if model.body(other).name=='table':floor_pen=max(floor_pen,-float(c.dist))
                            elif other!=bid and not model.body(other).name.startswith('robot_'):other_pen=max(other_pen,-float(c.dist))
                    after=tree.query((data.site_xpos[sites]-object_pos)@object_rot)[0]
                    entry={'frame':int(frame),'seconds':float(frame/fps),'tip_gaps_before_m':dict(zip(FINGERS,before.tolist())),
                           'tip_gaps_after_m':dict(zip(FINGERS,after.tolist())),'object_penetration_m':worst,
                           'self_penetration_m':self_pen,'support_penetration_m':floor_pen,'bystander_penetration_m':other_pen,**opposition,**ik}
                    # A geometry prior is rejected if it needs obvious penetration.
                    # A later lifted pose must also fit at the initial pickup;
                    # otherwise transporting it creates a palm/table collision.
                    localp=object_rot.T@(rig.wrist_pos(data)-object_pos)
                    localR=object_rot.T@H.euler_to_mat(rig.wrist_euler(data))
                    pickup_support=0.;pickup_other=0.;pickup_target=0.;prev=q[qcols[3:6]]
                    for check_frame in range(approach_begin,min(end,close_frame+int(.5*fps))+1):
                        data.qpos[:]=Q[check_frame];mujoco.mj_kinematics(model,data)
                        pp=data.xpos[bid].copy();rr=data.xmat[bid].reshape(3,3).copy()
                        prev=nearest_euler(H.mat_to_euler(rr@localR),prev)
                        target=np.r_[pp+rr@localp,prev,q[qcols[6:]]]
                        a=np.clip((check_frame-approach_begin)/max(1,close_frame-approach_begin),0.,1.);alpha=a*a*(3-2*a)
                        data.qpos[qcols]=(1-alpha)*Q[check_frame,qcols]+alpha*target
                        data.qpos[qcols[2]]+=.015*np.sin(np.pi*alpha)
                        mujoco.mj_fwdPosition(model,data)
                        for c in data.contact:
                            b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
                            if (b1 in rig.body_ids)==(b2 in rig.body_ids):continue
                            other=b2 if b1 in rig.body_ids else b1
                            if model.body(other).name=='table':pickup_support=max(pickup_support,-float(c.dist))
                            elif other==bid:pickup_target=max(pickup_target,-float(c.dist))
                            else:pickup_other=max(pickup_other,-float(c.dist))
                    entry.update(pickup_support_penetration_m=pickup_support,pickup_other_penetration_m=pickup_other,pickup_target_penetration_m=pickup_target)
                    entry['geometry_valid']=bool(worst<.004 and self_pen<.003 and floor_pen<.0015 and other_pen<.003 and pickup_support<.002 and pickup_other<.004 and pickup_target<.005 and ik['max_target_residual_m']<.012)
                    entry['score']=float(ik['max_target_residual_m']+3*max(0,worst-.002)+.05*ik['wrist_translation_change_m'])
                    candidates.append((entry['score'],q.copy(),entry))
                feasible=[c for c in candidates if c[2]['geometry_valid']]
                report={'side':side,'object':name,'start_frame':start,'end_frame':end,
                        'start_seconds':start/fps,'end_seconds':end/fps,'candidates':[c[2] for c in candidates],
                        'applied':bool(feasible),'physical_validation':'required; geometric seed only',
                        'predicted_settled_object_pose':settled.tolist(),'closure_frame':close_frame,
                        'prediction_scope':'hands held at original initial pose; zero hand/object contact verified'}
                if feasible:
                    _,qbest,best=min(feasible,key=lambda x:x[0]);anchor=best['frame'];data.qpos[:]=qbest;mujoco.mj_kinematics(model,data)
                    opos=data.xpos[bid].copy();orot=data.xmat[bid].reshape(3,3).copy()
                    wrist=rig.wrist_pos(data);wrot=H.euler_to_mat(rig.wrist_euler(data));localp=orot.T@(wrist-opos);localR=orot.T@wrot
                    fingers=qbest[qcols[6:]].copy();blend_frames=max(2,int(.25*fps));begin=approach_begin;last_euler=Q[begin,qcols[3:6]]
                    for t in range(begin,end+1):
                        data.qpos[:]=Q[t];mujoco.mj_kinematics(model,data);op=data.xpos[bid];orr=data.xmat[bid].reshape(3,3)
                        target_pos=op+orr@localp;target_euler=nearest_euler(H.mat_to_euler(orr@localR),last_euler);last_euler=target_euler
                        target=np.r_[target_pos,target_euler,fingers]
                        alpha=min(1.,(t-begin)/max(1,close_frame-begin),(end-t)/blend_frames);alpha=max(0.,alpha);alpha=alpha*alpha*(3-2*alpha)
                        if mode=='delta':
                            target=Q[t,qcols]+qbest[qcols]-Q[anchor,qcols]
                        result[t,qcols]=(1-alpha)*Q[t,qcols]+alpha*target
                        if t<close_frame:result[t,qcols[2]]+=.015*np.sin(np.pi*alpha)
                    report.update(anchor_frame=anchor,approach_start_frame=begin,fully_closed_frame=close_frame,selected=best,object_from_wrist_position=localp.tolist(),
                                  object_from_wrist_rotation=localR.tolist(),finger_targets=dict(zip(rig.finger_qpos,fingers.tolist())))
                    data.qpos[:]=qbest;mujoco.mj_kinematics(model,data)
                    pose=H.pose_from_data(model,data,rig,side);pose.update(frame=anchor,object=name,source='bounded opposing-surface IK prior; not physics validated',
                           grip_fingers=best['opposing_fingers'],press_depth_rad=.12)
                    seed_poses.append(pose)
                reports.append(report)
    # Respect exact actuator ranges, retain every nonrobot/object coordinate.
    clipped=0
    for j in range(model.njnt):
        if not model.joint(j).name.startswith('robot_'):continue
        col=int(model.jnt_qposadr[j]);old=result[:,col].copy();result[:,col]=np.clip(result[:,col],*model.jnt_range[j]);clipped+=int(np.count_nonzero(abs(result[:,col]-old)>1e-8))
    assert np.array_equal(result[0],original[0])
    for name in names:
        col=int(model.joint(name+'_free').qposadr[0]);assert np.array_equal(result[:,col:col+7],original[:,col:col+7])
    path=out/'tracking_reference_geometry.npz'
    TR.save_reference(path,model,result,ref['labels'],fps,ref['segment'],ref['segments'],
                      source=f'grasp geometry {mode} seed from exact reference; not physical execution',arms_coupled=False)
    report={'episode':ep,'scene':str(scene),'input_reference':str(e/'tracking_reference.npz'),'output_reference':str(path),
            'mode':mode,'objects_unchanged':True,'timing_unchanged':True,'initial_state_unchanged':True,
            'clipped_joint_samples':clipped,'euler_equivalent_branch_repaired_frames':repaired_euler_samples,
            'role_inference':role_notes,'grasp_intervals':reports,'physical_validation':False}
    (out/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'grasp_intervals.json').write_text(json.dumps(reports,indent=2)+'\n')
    (out/'native_grasp_pose_seeds.json').write_text(json.dumps(seed_poses,indent=2)+'\n')
    print(json.dumps({'episode':ep,'reference':str(path),'intervals':len(reports),'applied':sum(r['applied'] for r in reports),
                       'clipped_samples':clipped}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,nargs='+',default=[3,5,8]);p.add_argument('--mode',choices=['rigid_grasp','delta'],default='rigid_grasp');a=p.parse_args()
    for ep in a.episodes:main(ep,a.mode)
