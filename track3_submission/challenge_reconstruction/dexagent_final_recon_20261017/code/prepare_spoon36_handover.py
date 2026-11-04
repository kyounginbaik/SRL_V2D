"""Collision-aware left-hand receiving grasps for episode36's first transfer.

The donor right grip and every object target remain immutable. Static IK is
only a seed; the exported poses must subsequently acquire in real simulation.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from spoon_collision_grasps import collision_surface,contact_opposition
from grasp_geometry_seeds_v8 import bounded_ik
from lid_grasp_seed import contact_penetration
from physics_pickup_search import B,TR,H,dump

OUT=B/'refinement_20261018/episode_000036/handover_inputs'
SOURCE=B/'refinement_20261018/spoon36_long_grasp/search'
WORK={}

def init(scene):
    m=mujoco.MjModel.from_xml_path(scene)
    WORK.update(m=m,d=mujoco.MjData(m),rig=H.HandRig.build(m,'robot_left_','left'))

def solve(job):
    m,d,rig=WORK['m'],WORK['d'],WORK['rig'];ob=m.body('wooden_spoon').id
    q,ik=bounded_ik(m,d,rig,np.array(job['q0']),{0:np.array(job['targets'][0]),1:np.array(job['targets'][1])},ob)
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    pen=contact_penetration(m,d,rig,ob);opp=contact_opposition(m,d,rig,ob)
    row={k:v for k,v in job.items() if k not in ['q0','targets']}
    row.update(ik=ik,penetration=pen,opposition=opp)
    row['valid']=bool(ik['max_target_residual_m']<.006 and max(pen.values())<.004 and opp['min_normal_dot'] is not None and opp['min_normal_dot']<-.5)
    row['score']=float(ik['max_target_residual_m']+2*max(pen.values()))
    return row,q

def open_fingers(m,rig,q,normals,opening=.004):
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
    keys=list(rig.finger_qpos);cols=np.array(list(rig.finger_qpos.values()));jids=m.actuator_trnid[list(rig.finger_act.values()),0];dofs=m.jnt_dofadr[jids]
    sites=[rig.fingertip_sites[f] for f in ['thumb','index']]
    target=d.site_xpos[sites].copy()+opening*np.asarray(normals)
    lo=np.maximum(m.jnt_range[jids,0],q[cols]-.5);hi=np.minimum(m.jnt_range[jids,1],q[cols]+.5)
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def put(x):d.qpos[:]=q;d.qpos[cols]=x;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    def fun(x):put(x);return np.r_[(d.site_xpos[sites]-target).ravel()/.002,.04*(x-q[cols])]
    def jac(x):
        put(x);rows=[]
        for site in sites:mujoco.mj_jacSite(m,d,jp,jr,site);rows.append(jp[:,dofs]/.002)
        return np.vstack([*rows,.04*np.eye(22)])
    result=least_squares(fun,np.clip(q[cols],lo+1e-8,hi-1e-8),jac=jac,bounds=(lo,hi),max_nfev=80)
    put(result.x)
    return result.x,dict(requested_m=opening,achieved_along_normals_m=np.einsum('ij,ij->i',d.site_xpos[sites]-(target-opening*np.asarray(normals)),normals).tolist())

def main(workers=8):
    OUT.mkdir(parents=True,exist_ok=True);prev=json.loads((SOURCE/'result.json').read_text());scene=prev['execution']['scene']
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m);rig=H.HandRig.build(m,'robot_left_','left');right=H.HandRig.build(m,'robot_right_','right')
    inp=B/'episodes/episode_000036';original=TR.load_reference(inp/'tracking_reference.npz',m);prior=TR.load_reference(SOURCE/'reference.npz',m)
    q0=original['qpos'][200].copy();rc=np.array(list(right.wrist_qpos.values())+list(right.finger_qpos.values()));q0[rc]=prior['qpos'][200,rc]
    d.qpos[:]=q0;mujoco.mj_forward(m,d);ob=m.body('wooden_spoon').id;op=d.xpos[ob].copy();R=d.xmat[ob].reshape(3,3).copy()
    points,normals,surface=collision_surface(m,d,ob)
    tips=(d.site_xpos[[rig.fingertip_sites['thumb'],rig.fingertip_sites['index']]]-op)@R
    ids=[np.argsort(np.linalg.norm(points-tip,axis=1))[:900] for tip in tips]
    a,b=points[ids[0]],points[ids[1]];delta=a[:,None]-b[None];dist=np.linalg.norm(delta,axis=-1)
    na,nb=normals[ids[0]],normals[ids[1]];dots=na@nb.T
    valid=(dist>.007)&(dist<.07)&(dots<-.8)&(np.einsum('ijk,ik->ij',delta,na)>.65*dist)&(np.einsum('ijk,jk->ij',-delta,nb)>.65*dist)
    cost=np.linalg.norm(a-tips[0],axis=1)[:,None]**2+np.linalg.norm(b-tips[1],axis=1)[None]**2;cost[~valid]=np.inf
    pairs=[]
    for flat in np.argsort(cost.ravel()):
        i,j=np.unravel_index(flat,cost.shape)
        if not np.isfinite(cost[i,j]):break
        pair=np.array([a[i],b[j]])
        if any(np.linalg.norm(pair-old[0])<.006 for old in pairs):continue
        pairs.append((pair,np.array([na[i],nb[j]])))
        if len(pairs)==8:break
    jobs=[]
    for k,(pair,normal) in enumerate(pairs):
        for mode in ['source','donor_flexion']:
            initial=q0.copy()
            if mode=='donor_flexion':
                for key,c in rig.finger_qpos.items():
                    val=q0[right.finger_qpos[key]]*(-1 if key.endswith('_AA') else 1)
                    jid=m.actuator_trnid[rig.finger_act[key],0];initial[c]=np.clip(val,*m.jnt_range[jid])
            target=(pair+.0005*normal)@R.T+op
            jobs.append(dict(pair=k,mode=mode,q0=initial.tolist(),targets=target.tolist(),normal_world=(normal@R.T).tolist(),pair_local=pair.tolist()))
    results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(scene,)) as pool:
        for row,q in pool.map(solve,jobs):results.append((row,q));print(json.dumps(row),flush=True)
    usable=sorted([v for v in results if v[0]['valid']],key=lambda item:item[0]['score']);exports=[]
    for rank,(row,q) in enumerate(usable[:4]):
        pose=OUT/f'pose_{rank:02d}.npz';np.savez_compressed(pose,qpos=q)
        opened,opening=open_fingers(m,rig,q,row['normal_world'])
        spec=dict(episode='episode_000036',scene=scene,reference=str(SOURCE/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),
            manifest=str(inp/'input_manifest.json'),source=json.loads((inp/'input_manifest.json').read_text())['source'],geometry=str(inp/'metric_geometry.json'),
            anchor_pose=str(pose),open_fingers=opened.tolist(),side='left',object_name='wooden_spoon',horizon_s=13.3,
            times=dict(start=9.,hover=9.5,contact=9.8,closed=10.05,transport=10.25,release=13.6),hover_offset=[0.,0.,0.],
            original_right_release_start_s=10.1,original_right_release_end_s=10.65,opening=opening,seed_geometry=row)
        path=OUT/f'pose_{rank:02d}.json';dump(path,spec);exports.append(dict(spec=str(path),pose=str(pose),geometry=row,opening=opening))
    dump(OUT/'report.json',dict(scene=scene,surface=surface,results=[x[0] for x in results],exported=exports,physical_success_claim=False))
    print('EXPORTED',len(exports),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);main(**vars(p.parse_args()))
