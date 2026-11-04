"""Reflect an achieved right Sharpa spoon grasp into a left receiving grasp.

Mirroring uses the same finger angles: the supplied Sharpa models differ by
local-Y reflection, including their joint axes. No object state is changed in
execution; measured object poses below are planning forecasts only.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,argparse
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from prepare_spoon36_handover import init,solve,open_fingers
from spoon_collision_grasps import collision_surface
from physics_pickup_search import B,H,TR,dump
from wrist_reference import continuous_euler_xyz

def main(workers=8,curl_unused=False):
    source=B/'refinement_20261018/spoon36_long_grasp/search';out=B/'refinement_20261018/episode_000036'/('handover_mirrored_curled_inputs' if curl_unused else 'handover_mirrored_inputs');out.mkdir(parents=True,exist_ok=True)
    inp=B/'episodes/episode_000036';prev=json.loads((source/'result.json').read_text());scene=prev['execution']['scene']
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m);left=H.HandRig.build(m,'robot_left_','left');right=H.HandRig.build(m,'robot_right_','right')
    prior=TR.load_reference(source/'reference.npz',m);original=TR.load_reference(inp/'tracking_reference.npz',m);record=np.load(source/'execution/trajectory.npz')
    oc=int(m.joint('wooden_spoon_free').qposadr[0]);lc=np.array([left.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);rc=np.array([right.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
    # Forecast receiver contact from the actually held donor object at9.9s.
    q0=record['qpos'][198].copy();q0[np.r_[lc,list(left.finger_qpos.values())]]=original['qpos'][200,np.r_[lc,list(left.finger_qpos.values())]]
    curl={}
    if curl_unused:
        for finger in ['middle','ring','pinky']:
            for suffix,angle in [('MCP_FE',45.),('PIP',35.),('DIP',25.)]:
                key=finger+'_'+suffix;curl[key]=float(np.deg2rad(angle));q0[right.finger_qpos[key]]=curl[key]
    d.qpos[:]=q0;mujoco.mj_forward(m,d);body=m.body('wooden_spoon').id;op=d.xpos[body].copy();OR=Rotation.from_matrix(d.xmat[body].reshape(3,3))
    points,normals,surface=collision_surface(m,d,body);_,axes=np.linalg.eigh(np.cov(points.T));long=axes[:,-1];width=axes[:,-2]
    donor=record['qpos'][160].copy()
    for key,value in curl.items():donor[right.finger_qpos[key]]=value
    DR=Rotation.from_quat(donor[oc+3:oc+7],scalar_first=True);DP=donor[oc:oc+3]
    localp=DR.inv().apply(donor[rc[:3]]-DP);localR=DR.inv()*Rotation.from_euler('XYZ',donor[rc[3:]])
    df=mujoco.MjData(m);df.qpos[:]=donor;mujoco.mj_kinematics(m,df)
    pinch=DR.inv().apply(df.site_xpos[[right.fingertip_sites[f] for f in ['thumb','index']]].mean(0)-DP)
    if long@(-pinch)<0:long=-long
    local_reflection=np.diag([1.,-1.,1.]);jobs=[]
    for angle in [0.,np.pi/2]:
        normal=Rotation.from_rotvec(long*angle).apply(width);mirror=np.eye(3)-2*np.outer(normal,normal)
        for shift in [.035,.055,.075]:
            initial=q0.copy();initial[lc[:3]]=op+OR.apply(mirror@localp+shift*long)
            wr=OR*Rotation.from_matrix(mirror@localR.as_matrix()@local_reflection)
            initial[lc[3:]]=continuous_euler_xyz(Rotation.from_matrix(wr.as_matrix()[None]),initial[lc[3:]])[0]
            for key,col in left.finger_qpos.items():initial[col]=donor[right.finger_qpos[key]]
            d.qpos[:]=initial;mujoco.mj_forward(m,d)
            tips=OR.inv().apply(d.site_xpos[[left.fingertip_sites[f] for f in ['thumb','index']]]-op)
            ids=[np.argsort(np.linalg.norm(points-tip,axis=1))[:600] for tip in tips];a,b=points[ids[0]],points[ids[1]];na,nb=normals[ids[0]],normals[ids[1]]
            delta=a[:,None]-b[None];dist=np.linalg.norm(delta,axis=-1);dots=na@nb.T
            valid=(dist>.007)&(dist<.06)&(dots<-.8)&(np.einsum('ijk,ik->ij',delta,na)>.65*dist)&(np.einsum('ijk,jk->ij',-delta,nb)>.65*dist)
            cost=np.linalg.norm(a-tips[0],axis=1)[:,None]**2+np.linalg.norm(b-tips[1],axis=1)[None]**2;cost[~valid]=np.inf
            chosen=[]
            for flat in np.argsort(cost.ravel()):
                i,j=np.unravel_index(flat,cost.shape)
                if not np.isfinite(cost[i,j]):break
                pair=np.array([a[i],b[j]]);n=np.array([na[i],nb[j]])
                if any(np.linalg.norm(pair-old)<.006 for old in chosen):continue
                chosen.append(pair);target=OR.apply(pair+.0005*n)+op
                jobs.append(dict(q0=initial.tolist(),targets=target.tolist(),normal_world=OR.apply(n).tolist(),pair_local=pair.tolist(),mirror_angle_rad=angle,shift_toward_bowl_m=shift,mode='same_angle_mirrored_achieved_grasp'))
                if len(chosen)==2:break
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(scene,)) as pool:
        for row,q in pool.map(solve,jobs):results.append((row,q));print(json.dumps(row),flush=True)
    usable=sorted([x for x in results if x[0]['valid']],key=lambda x:x[0]['score']);exports=[]
    for rank,(row,q) in enumerate(usable[:4]):
        pose=out/f'pose_{rank:02d}.npz';np.savez_compressed(pose,qpos=q)
        opened,opening=open_fingers(m,left,q,row['normal_world'],opening=.008)
        spec=dict(episode='episode_000036',scene=scene,reference=str(source/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),manifest=str(inp/'input_manifest.json'),source=json.loads((inp/'input_manifest.json').read_text())['source'],geometry=str(inp/'metric_geometry.json'),anchor_pose=str(pose),open_fingers=opened.tolist(),side='left',object_name='wooden_spoon',horizon_s=13.3,times=dict(start=9.,hover=9.5,contact=9.9,closed=10.1,transport=10.35,release=13.6),hover_offset=[0.,0.,0.],original_right_release_start_s=10.25,original_right_release_end_s=10.75,opening=opening,seed_geometry=row,planning_forecast_s=9.9,donor_s=8.,donor_trajectory=str(source/'execution/trajectory.npz'))
        spec['giver_unused_curl']=curl
        path=out/f'pose_{rank:02d}.json';dump(path,spec);exports.append(dict(spec=str(path),pose=str(pose),geometry=row,opening=opening))
    dump(out/'report.json',dict(surface=surface,donor_s=8.,planning_forecast_s=9.9,reflection_joint_rule='same22angles; localYreflection',source_initial_state_changed=False,object_target_changed=False,results=[x[0] for x in results],exported=exports,physical_success_claim=False))
    print('EXPORTED',len(exports),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--curl-unused',action='store_true');main(**vars(p.parse_args()))
